import torch
import torch.nn as nn
from einops import rearrange
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class TransportGateHead3D(BaseModule):
    """Predict whether each transported dynamic voxel should be accepted.

    The head only gates existing transport candidates. It does not create new
    dynamic voxels by itself, which keeps the branch conservative.
    """

    def __init__(
            self,
            in_channels=12,
            hidden_channels=32,
            init_bias=-2.0,
            init_cfg=None):
        super().__init__(init_cfg)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(init_bias))

    def forward(self, gate_features):
        if gate_features.ndim != 6:
            raise ValueError(
                'gate_features must have shape [B,T,X,Y,Z,C], got '
                f'{tuple(gate_features.shape)}')
        batch_size, num_frames, _, _, _, channels = gate_features.shape
        if channels != self.net[0].in_channels:
            raise ValueError(
                f'Expected {self.net[0].in_channels} gate feature channels, '
                f'got {channels}')
        feature = rearrange(gate_features, 'b t x y z c -> (b t) c x y z')
        logits = self.net(feature).squeeze(1)
        return rearrange(
            logits,
            '(b t) x y z -> b t x y z',
            b=batch_size,
            t=num_frames)
