import torch.nn as nn
from einops import rearrange
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class FlowHead2D(BaseModule):
    def __init__(
            self,
            in_channels,
            hidden_channels=128,
            z_dim=16,
            init_cfg=None):
        super().__init__(init_cfg)
        self.z_dim = z_dim
        self.flow_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, z_dim * 2, 1),
        )

    def forward(self, x, batch_size, num_frames):
        # x: (B*T), (Z*feature_dim), X, Y
        flow = self.flow_head(x)
        flow = rearrange(
            flow,
            '(b t) (z c) x y -> b t x y z c',
            b=batch_size,
            t=num_frames,
            z=self.z_dim,
            c=2)
        return flow
