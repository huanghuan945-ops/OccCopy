"""Differentiable, past-only planar geometry; no trainable parameters or GT API."""
from dataclasses import dataclass
import torch
from .c0_model import validate_rigid


def rotation_z(angle):
    c,s=angle.cos(),angle.sin();z=torch.zeros_like(c);one=torch.ones_like(c)
    return torch.stack((c,-s,z,s,c,z,z,z,one),-1).reshape(*angle.shape,3,3)


def ego_step_from_lidar_delta(ego_from_lidar,delta_xy,yaw_increment):
    """Planar ego motion consistent with the same LiDAR reference point XY.

    delta_Lxy=Pi R_EL^T [g_Exy,0]+Pi R_EL^T (Rz(dyaw)-I)c_EL.
    Parameters retain autograd; policy at the caller must be explicit for any
    gradient-enabled entry forward. Past observed geometry itself is fixed.
    """
    if (ego_from_lidar.ndim!=3 or ego_from_lidar.shape[1:]!=(4,4)
            or delta_xy.shape!=(len(ego_from_lidar),2)
            or yaw_increment.shape!=(len(ego_from_lidar),)):
        raise ValueError('Expected batched4x4 calibration, Bx2 delta and B yaw')
    if not torch.isfinite(delta_xy).all() or not torch.isfinite(yaw_increment).all():
        raise FloatingPointError('Nonfinite predicted motion')
    x=ego_from_lidar;d=delta_xy.to(device=x.device,dtype=x.dtype)
    r=rotation_z(yaw_increment.to(device=x.device,dtype=x.dtype))
    rt=x[:,:3,:3].transpose(-1,-2);a=rt[:,:2,:2]
    if torch.any(torch.linalg.cond(a.detach())>1e10):
        raise ValueError('Planar XY projection numerically singular')
    lever=(rt@((r-torch.eye(3,device=x.device,dtype=x.dtype))@x[:,:3,3,None]))[:,:2,0]
    g=torch.linalg.solve(a,(d-lever)[...,None]).squeeze(-1)
    translation=torch.cat((g,torch.zeros_like(g[:,:1])),-1)
    top=torch.cat((r,translation[...,None]),-1)
    bottom=x.new_tensor([0,0,0,1]).expand(len(x),1,4)
    return torch.cat((top,bottom),-2)


@dataclass(frozen=True)
class ReadGeometry:
    step:int
    source_from_query:torch.Tensor
    source_from_t0:torch.Tensor
    world_from_source_ego:torch.Tensor
    source_timestamps_us:torch.Tensor


class TorchCausalPlanarPrior:
    """Per-forward state; append exactly one already-predicted delta at a time."""
    def __init__(self,world_from_ego,world_from_lidar,timestamps_us,device=None):
        if (world_from_ego.ndim!=4 or world_from_ego.shape[1:]!=(5,4,4)
                or world_from_lidar.shape!=world_from_ego.shape
                or timestamps_us.shape!=world_from_ego.shape[:2] or timestamps_us.dtype!=torch.int64):
            raise ValueError('Exactly five observed batched poses and int64 times required')
        device=world_from_ego.device if device is None else device
        self.source=world_from_ego.detach().to(device=device,dtype=torch.float64).clone()
        lidar=world_from_lidar.detach().to(device=device,dtype=torch.float64)
        self.timestamps=timestamps_us.detach().to(device=device).clone()
        validate_rigid(self.source);validate_rigid(lidar)
        dt=self.timestamps[:,1:]-self.timestamps[:,:-1]
        if torch.any(dt <= 0):
            raise ValueError("Observed timestamps must increase")
        relative=self.source[:,-2,:3,:3].transpose(-1,-2)@self.source[:,-1,:3,:3]
        self.yaw_rate=torch.atan2(relative[:,1,0],relative[:,0,0])/(dt[:,-1].double()*1e-6)
        self.extrinsic=torch.linalg.inv(self.source[:,-1])@lidar[:,-1]
        self.source_from_t0=torch.linalg.inv(self.source)@self.source[:,-1:]
        self.pose=torch.eye(4,device=device,dtype=torch.float64).expand(len(self.source),4,4).clone()
        self.completed_steps=0

    def append_predicted_delta(self,delta_xy):
        if self.completed_steps>=6:raise ValueError('No3->3.5s outgoing step')
        self.pose=self.pose@ego_step_from_lidar_delta(self.extrinsic,delta_xy,self.yaw_rate*.5)
        self.completed_steps+=1
        return self.pose

    def read_geometry(self,step):
        if step not in range(1,7) or step!=self.completed_steps:
            raise ValueError('Geometry must consume exactly the predicted prefix for this step')
        return ReadGeometry(step,self.source_from_t0@self.pose[:,None],self.source_from_t0,
                            self.source,self.timestamps)

    def sampling_base(self):
        xx,yy=torch.meshgrid(torch.arange(50,device=self.pose.device,dtype=self.pose.dtype),
                             torch.arange(50,device=self.pose.device,dtype=self.pose.dtype),indexing='ij')
        point=torch.stack((-40+(xx+.5)*1.6,-40+(yy+.5)*1.6,torch.zeros_like(xx),torch.ones_like(xx)),-1)
        return torch.einsum('bkij,xyj->bkxyi',self.read_geometry(self.completed_steps).source_from_query,point)[...,:2]
