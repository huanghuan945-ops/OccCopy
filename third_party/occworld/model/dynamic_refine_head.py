import torch
import torch.nn as nn
from einops import rearrange
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class DynamicResidualHead2D(BaseModule):
    """Predict additive logits for dynamic classes only.

    The dense decoder feature remains the primary input. Predicted planar flow
    is flattened over height bins and used as motion conditioning. The final
    layer starts at zero, so enabling an untrained adapter exactly preserves
    the original semantic logits.
    """

    def __init__(
            self,
            in_channels=128,
            hidden_channels=128,
            z_dim=16,
            num_dynamic_classes=8,
            use_flow_condition=True,
            use_transport_condition=False,
            residual_scale=1.0,
            init_cfg=None):
        super().__init__(init_cfg)
        self.z_dim = int(z_dim)
        self.num_dynamic_classes = int(num_dynamic_classes)
        self.use_flow_condition = bool(use_flow_condition)
        self.use_transport_condition = bool(use_transport_condition)
        self.residual_scale = float(residual_scale)
        flow_channels = self.z_dim * 2 if self.use_flow_condition else 0
        transport_channels = (
            self.z_dim * (self.num_dynamic_classes + 1)
            if self.use_transport_condition else 0)
        self.pre = nn.Sequential(
            nn.Conv2d(
                in_channels + flow_channels + transport_channels,
                hidden_channels,
                kernel_size=3,
                padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.out = nn.Conv2d(
            hidden_channels,
            self.z_dim * self.num_dynamic_classes,
            kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
            self,
            dense_feature,
            pred_flow,
            batch_size,
            num_frames,
            transport_condition=None):
        features = [dense_feature]
        if self.use_flow_condition:
            if pred_flow is None:
                raise ValueError(
                    'DynamicResidualHead2D requires pred_flow when '
                    'use_flow_condition=True')
            flow_feature = rearrange(
                pred_flow,
                'b t x y z c -> (b t) (z c) x y',
                b=batch_size,
                t=num_frames,
                z=self.z_dim,
                c=2)
            features.append(flow_feature)
        if self.use_transport_condition:
            if transport_condition is None:
                raise ValueError(
                    'DynamicResidualHead2D requires transport_condition when '
                    'use_transport_condition=True')
            expected = (
                batch_size,
                num_frames,
                dense_feature.shape[-2],
                dense_feature.shape[-1],
                self.z_dim,
                self.num_dynamic_classes + 1,
            )
            if tuple(transport_condition.shape) != expected:
                raise ValueError(
                    'transport_condition shape mismatch: expected '
                    f'{expected}, got {tuple(transport_condition.shape)}')
            transport_feature = rearrange(
                transport_condition,
                'b t x y z c -> (b t) (z c) x y',
                b=batch_size,
                t=num_frames,
                z=self.z_dim,
                c=self.num_dynamic_classes + 1)
            features.append(transport_feature)
        feature = torch.cat(features, dim=1)

        residual = self.out(self.pre(feature)) * self.residual_scale
        return rearrange(
            residual,
            '(b t) (z c) x y -> b t x y z c',
            b=batch_size,
            t=num_frames,
            z=self.z_dim,
            c=self.num_dynamic_classes)
