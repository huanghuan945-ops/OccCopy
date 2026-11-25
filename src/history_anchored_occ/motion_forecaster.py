"""Independent six-step integration reusing the original native/VQ/ego rollout."""
import torch
from .gt_forecaster import GTHistory,GTCausalPredictor
from .motion_prior import TorchCausalPlanarPrior
from .motion_reader import MotionAlignedGTAnchorReader


class MotionAlignedGTPredictor(GTCausalPredictor):
    def __init__(self,native,seed=2026,query_extension=None,geometry_gradient_policy=None):
        super().__init__(native,seed,query_extension)
        if geometry_gradient_policy not in (None,'connected','detached'):
            raise ValueError('Explicit connected/detached engineering policy required')
        self.geometry_gradient_policy=geometry_gradient_policy
        self._active_prior=None
        self._geometry_trace=[]

    def _select_motion(self,query):
        result=super()._select_motion(query)
        if self._active_prior is None:raise RuntimeError('Motion called outside an active history-only rollout')
        selected=result[3]
        if torch.is_grad_enabled() and self.geometry_gradient_policy is None:
            raise RuntimeError('Geometry training gradient contract not selected; use no_grad for inference')
        delta=selected.detach() if self.geometry_gradient_policy=='detached' else selected
        pose=self._active_prior.append_predicted_delta(delta)
        self._geometry_trace.append(dict(query=self._active_prior.completed_steps-1,
            serves_occupancy_step=self._active_prior.completed_steps,
            student_selected_delta_XY=selected.detach().cpu().tolist(),
            ego0_from_predicted_ego=pose.detach().cpu().tolist(),future_GT_used=False))
        return result

    def _future_query_step(self,work,poses,bank,step):
        if self._active_prior is None:raise RuntimeError('No active causal motion prefix')
        geometry=self._active_prior.read_geometry(step)
        if self.query_extension is None:
            return super()._future_query_step(work,poses,bank,step)
        if not isinstance(self.query_extension,MotionAlignedGTAnchorReader):
            raise TypeError('Motion geometry requires the independent matching reader')
        b,context=work.shape[:2]
        def before_head(module,arguments):
            tensor=arguments[0]
            if tensor.shape!=(b*context,128,50,50):raise ValueError('Native pre-head shape changed')
            views=tensor.reshape(b,context,128,50,50)
            fused=self.query_extension(views[:,-1],bank,step,geometry=geometry)
            if fused.shape!=views[:,-1].shape or not torch.isfinite(fused).all():
                raise ValueError('Invalid aligned reader output')
            replacement=views.clone();replacement[:,-1]=fused
            return (replacement.reshape_as(tensor),*arguments[1:])
        handle=self.native.transformer.conv_out.register_forward_pre_hook(before_head)
        try:
            return self.native.transformer.forward_autoreg_step(work,pose_tokens=poses,
                                                               start_frame=0,mid_frame=context)
        finally:handle.remove()

    def forward(self,history:GTHistory):
        if not isinstance(history,GTHistory):raise TypeError('Student API accepts GTHistory only')
        if self._active_prior is not None:raise RuntimeError('Concurrent/reentrant forwards are unsupported')
        if torch.is_grad_enabled() and self.geometry_gradient_policy is None:
            raise RuntimeError('Inference-only default; explicit engineering gradient policy required')
        history.validate();self._geometry_trace=[]
        self._active_prior=TorchCausalPlanarPrior(history.world_from_grid_ego,history.world_from_lidar,
            history.timestamps_us,device=next(self.native.parameters()).device)
        try:
            output=super().forward(history)
            if self._active_prior.completed_steps!=6:raise AssertionError('Exactly q0..q5 motion prefixes required')
            output.update(motion_prior_trace=tuple(self._geometry_trace),
                motion_prior_yaw_rate_rad_s=self._active_prior.yaw_rate.detach().cpu(),
                geometry_gradient_policy=self.geometry_gradient_policy,
                motion_prior_kind='past yaw rate plus selected predicted LiDAR XY; planar ego',
                geometry_is_full_future_SE3=False)
            return output
        finally:
            self._active_prior=None
            self._geometry_trace=[]
