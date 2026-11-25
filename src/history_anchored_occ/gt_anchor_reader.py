"""Confirmed equal-parameter GT-F/Mcore token readers, no GT future information."""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from .gt_forecaster import RawGTAnchorBank,FUTURE_OFFSETS_S


def masked_softmax(score,valid):
    # All-invalid queries return explicit zeros, never NaNs or uniform/free reads.
    weights=torch.softmax(score.masked_fill(~valid,torch.finfo(score.dtype).min),dim=-1)*valid
    return weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)


def sample_qualified(features,qualification,positions_xy_m):
    """Mask every bilinear corner before aggregation and renormalize valid mass.

    features B,K,C,50,50; positions B,K,Xq,Yq,L,2, source ego meters.
    qualification is inherited GT token eligibility, not voxel support.
    """
    b,k,c,h,w=features.shape
    if (h,w)!=(50,50) or qualification.shape!=(b,k,h,w) or qualification.dtype!=torch.bool:
        raise ValueError('Native source token layout/qualification differs')
    if not torch.isfinite(positions_xy_m).all():raise FloatingPointError('Nonfinite learned sampling position')
    xy=(positions_xy_m+40.)/1.6-.5
    lower=torch.floor(xy).long();frac=xy-lower.to(xy.dtype)
    flat=features.reshape(b*k,c,h*w).transpose(1,2)
    qual=qualification.reshape(b*k,h*w)
    shape=positions_xy_m.shape[:-1]
    value=features.new_zeros((*shape,c));rho=features.new_zeros(shape)
    for dx,dy in ((0,0),(0,1),(1,0),(1,1)):
        ix,iy=lower[...,0]+dx,lower[...,1]+dy
        inside=(ix>=0)&(ix<h)&(iy>=0)&(iy<w)
        index=ix.clamp(0,h-1)*w+iy.clamp(0,w-1)
        index_flat=index.reshape(b*k,-1)
        eligible=qual.gather(1,index_flat).reshape(shape)&inside
        weight=(frac[...,0] if dx else 1-frac[...,0])*(frac[...,1] if dy else 1-frac[...,1])
        weight=weight*eligible.to(weight.dtype)
        corner=flat.gather(1,index_flat[...,None].expand(-1,-1,c)).reshape(*shape,c)
        # Hard exclusion before corner aggregation; invalid values must not leak.
        corner=torch.where(eligible[...,None],corner,torch.zeros_like(corner))
        value=value+weight[...,None]*corner;rho=rho+weight
    return value/rho.clamp_min(1e-12)[...,None],rho


class GTAnchorReader(nn.Module):
    def __init__(self,variant,seed=2026):
        super().__init__()
        if variant not in ('GT-F','GT-Mcore'):raise ValueError('Only confirmed F/Mcore readers')
        self.variant=variant
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.norm=nn.LayerNorm(128)
            self.query=nn.Linear(128,128);self.key=nn.Linear(128,128)
            self.value=nn.Linear(128,128);self.project=nn.Linear(128,128)
            self.offset=nn.Sequential(nn.Linear(136,64),nn.ReLU(),nn.Linear(64,8))
            nn.init.zeros_(self.offset[-1].weight);nn.init.zeros_(self.offset[-1].bias)
            self.metadata=nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,1))
            self.gate=nn.Sequential(nn.Linear(258,64),nn.ReLU(),nn.Linear(64,1))
            self.w_age=nn.Parameter(torch.zeros(()));self.w_gate_age=nn.Parameter(torch.zeros(()))
        self.last_diagnostic=None
        self.read_calls=0

    def forward(self,query,bank:RawGTAnchorBank,step):
        if not isinstance(bank,RawGTAnchorBank) or query.shape[1:]!=(128,50,50) or step not in range(1,7):
            raise ValueError('Confirmed token reader requires native query and immutable original bank')
        b,c,xq,yq=query.shape;k=5;l=4
        features=bank.features.to(query.device,query.dtype)
        qualification=bank.reading_qualification.to(query.device)
        if features.shape!=(b,k,128,50,50):raise ValueError('Exactly five source slots required')
        source=bank.world_from_source_grid_ego.to(query.device,dtype=torch.float64)
        source_from_t0=torch.linalg.inv(source)@source[:,-1:, :, :]
        xx,yy=torch.meshgrid(torch.arange(50,device=query.device,dtype=torch.float64),
                             torch.arange(50,device=query.device,dtype=torch.float64),indexing='ij')
        point=torch.stack((-40+(xx+.5)*1.6,-40+(yy+.5)*1.6,torch.zeros_like(xx),torch.ones_like(xx)),-1)
        base=torch.einsum('bkij,xyj->bkxyi',source_from_t0,point)[...,:2].to(query.dtype)
        translation=source_from_t0[...,:2,3].to(query.dtype)
        yaw=torch.atan2(source_from_t0[...,1,0],source_from_t0[...,0,0]).to(query.dtype)
        ages=bank.ages_s(FUTURE_OFFSETS_S[step-1]).to(query.device,query.dtype)
        h=query.new_tensor(FUTURE_OFFSETS_S[step-1]/3)
        u=query.permute(0,2,3,1)
        us=self.norm(u)[:,None].expand(-1,k,-1,-1,-1)
        context=torch.cat((us,ages[:,:,None,None,None].expand(-1,-1,50,50,1)/5,
            h.expand(b,k,50,50,1),base/40,
            translation[:,:,None,None].expand(-1,-1,50,50,-1)/40,
            yaw.sin()[:,:,None,None,None].expand(-1,-1,50,50,1),
            yaw.cos()[:,:,None,None,None].expand(-1,-1,50,50,1)),-1)
        offset=self.offset(context).reshape(b,k,xq,yq,l,2)
        position=base[...,None,:]+offset
        sampled,rho=sample_qualified(features,qualification,position)
        q=self.query(u)[:,None,:,:,None,:]
        scores=(q*self.key(sampled)).sum(-1)/(128**.5)
        metadata=torch.cat((rho[...,None],h.expand(b,k,xq,yq,l,1),base[...,None,:].expand(-1,-1,-1,-1,l,-1)/40),-1)
        scores=scores+self.metadata(metadata).squeeze(-1)
        ages_sites=ages[:,:,None,None,None].expand(-1,-1,xq,yq,l)
        age_coefficient=self.w_age if self.variant=='GT-F' else -F.softplus(self.w_age)
        scores=scores+age_coefficient*ages_sites/5
        valid=rho>0
        scores_flat=scores.permute(0,2,3,1,4).reshape(b,xq,yq,k*l)
        valid_flat=valid.permute(0,2,3,1,4).reshape(b,xq,yq,k*l)
        alpha=masked_softmax(scores_flat,valid_flat).reshape(b,xq,yq,k,l).permute(0,3,1,2,4)
        read=(alpha[...,None]*self.value(sampled)).sum((1,4))
        age_bar=(alpha*ages_sites).sum((1,4));rho_bar=(alpha*rho).sum((1,4))
        gate_logit=self.gate(torch.cat((u,read,h.expand(b,xq,yq,1),rho_bar[...,None]),-1)).squeeze(-1)
        if self.variant=='GT-F':
            gate=torch.sigmoid(gate_logit+self.w_gate_age*age_bar/5)
            confidence=query.new_ones((b,xq,yq))
        else:
            confidence=(alpha*rho*torch.exp(-F.softplus(self.w_gate_age)*ages_sites/5)).sum((1,4))
            gate=torch.sigmoid(gate_logit)*confidence
        any_source=valid_flat.any(-1)
        gate=torch.where(any_source,gate,torch.zeros_like(gate))
        read=torch.where(any_source[...,None],read,torch.zeros_like(read))
        residual=gate[...,None]*self.project(read)
        fused=(u+residual).permute(0,3,1,2).contiguous()
        self.read_calls+=1
        self.last_diagnostic=dict(step=step,valid_samples=valid.detach(),alpha=alpha.detach(),
            gate=gate.detach(),confidence=confidence.detach(),source_age_s=ages.detach(),
            any_source=any_source.detach(),qualification_source='historical_GT_only',
            sampling_positions_xy_m=position.detach(),future_GT_pose_used=False)
        return fused
