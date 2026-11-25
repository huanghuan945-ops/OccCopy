"""Versioned independent quality-gate predictor; native and old entries unchanged."""
import torch
from torch.nn import functional as F
from .motion_forecaster import MotionAlignedGTPredictor
from .quality_reader import QualityGTAnchorReader


def native_head_logits(head,query):
    if type(head) is not torch.nn.Conv2d or head.in_channels!=128 or head.out_channels!=512 or head.padding_mode!='zeros':
        raise ValueError('Reviewed native Conv2d head required')
    with torch.no_grad():
        return F.conv2d(query.detach(),head.weight,head.bias,head.stride,head.padding,head.dilation,head.groups)


class QualityGTPredictor(MotionAlignedGTPredictor):
    capture_quality=False
    quality_records=()
    def _future_query_step(self,work,poses,bank,step):
        if self._active_prior is None:raise RuntimeError('No active causal motion prefix')
        geometry=self._active_prior.read_geometry(step)
        if self.query_extension is None:
            return super()._future_query_step(work,poses,bank,step)
        if not isinstance(self.query_extension,QualityGTAnchorReader):
            raise TypeError('Motion geometry requires the independent matching reader')
        b,context=work.shape[:2]
        def before_head(module,arguments):
            tensor=arguments[0]
            if tensor.shape!=(b*context,128,50,50):raise ValueError('Native pre-head shape changed')
            views=tensor.reshape(b,context,128,50,50)
            # Functional call bypasses only the active top-level fusion hook. Native head is Conv2d.
            pre=native_head_logits(module,views[:,-1])
            fused=self.query_extension(views[:,-1],bank,step,geometry=geometry,pre_logits=pre)
            if self.capture_quality:
                self.quality_records.append((pre.detach().argmax(1).cpu(),
                    {k:v.detach().cpu() for k,v in self.query_extension.last_diagnostic.items() if isinstance(v,torch.Tensor)}))
            if fused.shape!=views[:,-1].shape or not torch.isfinite(fused).all():
                raise ValueError('Invalid aligned reader output')
            replacement=views.clone();replacement[:,-1]=fused
            return (replacement.reshape_as(tensor),*arguments[1:])
        handle=self.native.transformer.conv_out.register_forward_pre_hook(before_head)
        try:
            return self.native.transformer.forward_autoreg_step(work,pose_tokens=poses,
                                                               start_frame=0,mid_frame=context)
        finally:handle.remove()

    def forward(self,history):
        self.quality_records=[]
        try:
            return super().forward(history)
        except BaseException:
            self.quality_records=[]
            raise
