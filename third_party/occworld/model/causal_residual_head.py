import torch
import torch.nn as nn
from einops import rearrange
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class CausalResidualFlowHead2D(BaseModule):
    """Predict object-motion residual after analytic ego compensation."""

    def __init__(
            self,
            in_channels=128,
            hidden_channels=128,
            z_dim=16,
            init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = int(in_channels)
        self.z_dim = int(z_dim)
        context_channels = self.in_channels * 2 + 3
        self.context = nn.Sequential(
            nn.Conv2d(context_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(hidden_channels, self.z_dim * 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
            self, future_feature, source_feature, ego_displacement_xy,
            batch_size, num_frames):
        if future_feature.shape[0] != batch_size * num_frames:
            raise ValueError(
                f'future feature batch {future_feature.shape[0]} does not '
                f'match {batch_size}x{num_frames}')
        if source_feature.shape[:2] != (batch_size, self.in_channels):
            raise ValueError(
                f'expected source feature {(batch_size, self.in_channels)}, '
                f'got {tuple(source_feature.shape[:2])}')
        if ego_displacement_xy.shape != (batch_size, num_frames, 2):
            raise ValueError(
                'ego_displacement_xy must have shape '
                f'{(batch_size, num_frames, 2)}, got '
                f'{tuple(ego_displacement_xy.shape)}')

        _, _, size_x, size_y = future_feature.shape
        source = source_feature[:, None].expand(
            -1, num_frames, -1, -1, -1)
        source = source.reshape(
            batch_size * num_frames, self.in_channels, size_x, size_y)
        ego = ego_displacement_xy.reshape(
            batch_size * num_frames, 2, 1, 1).expand(
                -1, -1, size_x, size_y)
        horizon = torch.linspace(
            1.0 / num_frames,
            1.0,
            num_frames,
            device=future_feature.device,
            dtype=future_feature.dtype)
        horizon = horizon[None, :, None, None, None].expand(
            batch_size, -1, -1, size_x, size_y)
        horizon = horizon.reshape(
            batch_size * num_frames, 1, size_x, size_y)
        context = torch.cat([
            future_feature,
            source.to(future_feature),
            ego.to(future_feature),
            horizon,
        ], dim=1)
        residual = self.output(self.context(context))
        return rearrange(
            residual,
            '(b t) (z c) x y -> b t x y z c',
            b=batch_size,
            t=num_frames,
            z=self.z_dim,
            c=2)
