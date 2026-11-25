"""Independent quality-aware and equal-parameter capacity-control readers."""
import torch
from torch.nn import functional as F
from .gt_anchor_variants import GTAnchorReaderV2
from .gt_anchor_reader import sample_qualified,masked_softmax
from .gt_forecaster import RawGTAnchorBank,FUTURE_OFFSETS_S
from .motion_prior import ReadGeometry


class QualityGTAnchorReader(GTAnchorReaderV2):
    def __init__(self,variant,information,seed=2026):
        super().__init__(variant,seed)
        if information not in ("cap","quality"):raise ValueError("Explicit gate information required")
        self.information=information
        old=self.gate[0]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            new=torch.nn.Linear(266,64)
            projection=(torch.randint(0,2,(258,8))*2-1).float()/(258**.5)
        with torch.no_grad():
            new.weight[:,:258].copy_(old.weight);new.bias.copy_(old.bias)
        self.gate[0]=new
        self.register_buffer("capacity_projection",projection)

    def forward(self,query,bank:RawGTAnchorBank,step,*,geometry:ReadGeometry,pre_logits):
        if (not isinstance(bank,RawGTAnchorBank) or not isinstance(geometry,ReadGeometry)
                or query.shape[1:]!=(128,50,50) or step not in range(1,7) or geometry.step!=step):
            raise ValueError('Reviewed bank, native query and corresponding geometry required')
        b,c,xq,yq=query.shape;k=5;l=4
        features=bank.features.to(query.device,query.dtype)
        qualification=bank.reading_qualification.to(query.device)
        source=bank.world_from_source_grid_ego.to(query.device,dtype=torch.float64)
        if (features.shape!=(b,k,128,50,50)
                or geometry.source_from_query.shape!=(b,k,4,4)
                or not torch.equal(source,geometry.world_from_source_ego)
                or not torch.equal(bank.input_frame_timestamps_us.to(query.device),geometry.source_timestamps_us)):
            raise ValueError('Geometry source provenance differs from immutable original slots')
        source_from_t0=torch.linalg.inv(source)@source[:,-1:,:,:]
        if not torch.equal(source_from_t0,geometry.source_from_t0):
            raise ValueError('Past reference mapping differs')
        xx,yy=torch.meshgrid(torch.arange(50,device=query.device,dtype=torch.float64),
                             torch.arange(50,device=query.device,dtype=torch.float64),indexing='ij')
        point=torch.stack((-40+(xx+.5)*1.6,-40+(yy+.5)*1.6,torch.zeros_like(xx),torch.ones_like(xx)),-1)
        base=torch.einsum('bkij,xyj->bkxyi',geometry.source_from_query,point)[...,:2].to(query.dtype)
        # These two metadata fields continue to describe observed source vs t0.
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
        gate_input=torch.cat((u,read,h.expand(b,xq,yq,1),rho_bar[...,None]),-1)
        detail=quality_statistics(pre_logits,q,self.key(sampled),self.value(sampled),alpha,valid)
        extra=detail if self.information=='quality' else torch.tanh(gate_input.detach()@self.capacity_projection)
        gate_logit=self.gate(torch.cat((gate_input,extra.detach()),-1)).squeeze(-1)
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
            quality_statistics=detail.detach(),extra_gate_inputs=extra.detach(),residual_norm=residual.detach().norm(dim=-1),
            any_source=any_source.detach(),sampling_positions_xy_m=position.detach(),
            sampling_base_xy_m=base.detach(),future_GT_pose_used=False,
            geometry_source='observed yaw rate and student-selected predicted XY prefix')
        return fused


def quality_statistics(pre_logits,q,keys,values,alpha,valid):
    """Eight detached causal features; K/L axes are candidate slots/points, not spatial axes."""
    with torch.no_grad():
        if pre_logits.shape!=(q.shape[0],512,50,50):raise ValueError('Same-prefix native logits required')
        probs=pre_logits.detach().softmax(1).permute(0,2,3,1)
        entropy=-(probs*probs.clamp_min(1e-12).log()).sum(-1)/__import__('math').log(512)
        top=probs.topk(2,dim=-1).values; margin=top[...,0]-top[...,1]
        cosine=F.cosine_similarity(q.detach(),keys.detach(),dim=-1,eps=1e-8)
        aa=alpha.detach(); vv=valid.detach(); n=vv.sum((1,4)); any_valid=n>0
        cosmax=cosine.masked_fill(~vv,-2).amax((1,4));cosmax=torch.where(any_valid,cosmax,torch.zeros_like(cosmax))
        cosmean=(aa*torch.where(vv,cosine,torch.zeros_like(cosine))).sum((1,4))
        attent_entropy=-(aa*aa.clamp_min(1e-12).log()).sum((1,4))/n.clamp_min(2).float().log()
        a_top=aa.permute(0,2,3,1,4).flatten(-2).topk(2,dim=-1).values
        a_gap=a_top[...,0]-a_top[...,1]
        v=torch.where(vv[...,None],values.detach(),torch.zeros_like(values))
        mean=(aa[...,None]*v).sum((1,4))
        variance=(aa*(v-mean[:,None,:,:,None,:]).square().sum(-1)).sum((1,4))
        energy=(aa*v.square().sum(-1)).sum((1,4))
        dispersion=(variance/(energy+1e-12)).clamp(0,1)
        d=torch.stack((entropy,margin,cosmax,cosmean,attent_entropy,a_gap,n/20.,dispersion),-1)
        if not torch.isfinite(d).all():raise FloatingPointError('Nonfinite quality inputs')
        return d.detach()
