import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class EgoSE2Head(BaseModule):
    """Predict cumulative source-relative [dx, dy, dyaw] per horizon."""

    def __init__(self, in_channels=128, hidden_channels=128, init_cfg=None):
        super().__init__(init_cfg)
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 3),
        )

    def forward(self, pose_features):
        return self.net(pose_features)


@MODELS.register_module()
class EgoSE2ResidualHead(BaseModule):
    """Predict a bounded correction to OccWorld's cumulative ego XY."""

    def __init__(
            self,
            in_channels=128,
            hidden_channels=128,
            translation_scale=20.0,
            max_translation_residual=2.0,
            max_yaw_residual=0.2,
            init_cfg=None):
        super().__init__(init_cfg)
        self.translation_scale = float(translation_scale)
        self.max_translation_residual = float(max_translation_residual)
        self.max_yaw_residual = float(max_yaw_residual)
        self.feature_norm = nn.LayerNorm(in_channels)
        self.net = nn.Sequential(
            nn.Linear(in_channels + 4, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Linear(hidden_channels, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, pose_features, base_ego_se2):
        if pose_features.shape[:-1] != base_ego_se2.shape[:-1]:
            raise ValueError(
                'pose feature/base SE2 shape mismatch: '
                f'{pose_features.shape} vs {base_ego_se2.shape}')
        if base_ego_se2.shape[-1] != 3:
            raise ValueError(
                f'base_ego_se2 must end in 3, got {base_ego_se2.shape}')
        num_frames = pose_features.shape[1]
        horizon = torch.linspace(
            1.0 / num_frames,
            1.0,
            num_frames,
            device=pose_features.device,
            dtype=pose_features.dtype)
        horizon = horizon[None, :, None].expand(
            pose_features.shape[0], -1, -1)
        normalized_base = base_ego_se2.to(pose_features).clone()
        normalized_base[..., :2] /= self.translation_scale
        context = torch.cat([
            self.feature_norm(pose_features),
            normalized_base,
            horizon,
        ], dim=-1)
        raw = torch.tanh(self.output(self.net(context)))
        scale = raw.new_tensor([
            self.max_translation_residual,
            self.max_translation_residual,
            self.max_yaw_residual,
        ])
        return raw * scale
