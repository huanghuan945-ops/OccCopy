import torch
import torch.nn as nn
from einops import rearrange
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class CausalHorizonAlignmentHead3D(BaseModule):
    """Predict a bounded, causal residual for cumulative transport flow."""

    def __init__(
            self,
            dynamic_label_ids=(2, 3, 4, 5, 6, 7, 9, 10),
            num_horizons=6,
            horizon_embedding_dim=4,
            hidden_channels=16,
            max_residual_m=2.0,
            init_cfg=None):
        super().__init__(init_cfg)
        self.dynamic_label_ids = tuple(int(v) for v in dynamic_label_ids)
        self.num_horizons = int(num_horizons)
        self.horizon_embedding_dim = int(horizon_embedding_dim)
        self.hidden_channels = int(hidden_channels)
        self.max_residual_m = float(max_residual_m)
        if len(set(self.dynamic_label_ids)) != len(self.dynamic_label_ids):
            raise ValueError('dynamic_label_ids must be unique')
        if not self.dynamic_label_ids:
            raise ValueError('dynamic_label_ids must not be empty')
        if self.num_horizons <= 0 or self.horizon_embedding_dim <= 0:
            raise ValueError('horizon dimensions must be positive')
        if self.hidden_channels <= 0 or self.hidden_channels % 4:
            raise ValueError('hidden_channels must be positive and divisible by 4')
        if self.max_residual_m <= 0.0:
            raise ValueError('max_residual_m must be positive')

        self.horizon_embedding = nn.Embedding(
            self.num_horizons, self.horizon_embedding_dim)
        in_channels = (
            2 + len(self.dynamic_label_ids) + self.horizon_embedding_dim)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, self.hidden_channels, 3, padding=1),
            nn.GroupNorm(4, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.GroupNorm(4, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(self.hidden_channels, 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, source_semantics, base_cumulative_flow):
        if base_cumulative_flow.ndim != 6 or (
                base_cumulative_flow.shape[-1] != 2):
            raise ValueError(
                'base_cumulative_flow must be [B,T,X,Y,Z,2], got '
                f'{tuple(base_cumulative_flow.shape)}')
        batch_size, num_horizons, size_x, size_y, size_z, _ = (
            base_cumulative_flow.shape)
        if num_horizons != self.num_horizons:
            raise ValueError(
                f'expected {self.num_horizons} horizons, got {num_horizons}')
        if tuple(source_semantics.shape) != (
                batch_size, size_x, size_y, size_z):
            raise ValueError(
                'source_semantics must align with the flow source grid, got '
                f'{tuple(source_semantics.shape)} versus '
                f'{tuple(base_cumulative_flow.shape)}')

        source_semantics = source_semantics.to(
            device=base_cumulative_flow.device, dtype=torch.long)
        dynamic_one_hot = torch.stack([
            source_semantics == label_id
            for label_id in self.dynamic_label_ids
        ], dim=-1).to(dtype=base_cumulative_flow.dtype)
        dynamic_one_hot = dynamic_one_hot[:, None].expand(
            -1, num_horizons, -1, -1, -1, -1)
        dynamic_mask = dynamic_one_hot.sum(dim=-1, keepdim=True) > 0

        horizon_ids = torch.arange(
            num_horizons, device=base_cumulative_flow.device)
        horizon_features = self.horizon_embedding(horizon_ids).to(
            dtype=base_cumulative_flow.dtype)
        horizon_features = horizon_features.view(
            1, num_horizons, 1, 1, 1, self.horizon_embedding_dim).expand(
                batch_size, -1, size_x, size_y, size_z, -1)
        features = torch.cat([
            base_cumulative_flow, dynamic_one_hot, horizon_features,
        ], dim=-1)
        features = rearrange(
            features, 'b t x y z c -> (b t) c x y z')
        raw_residual = self.net(features)
        raw_residual = rearrange(
            raw_residual, '(b t) c x y z -> b t x y z c',
            b=batch_size, t=num_horizons)
        bounded = self.max_residual_m * torch.tanh(raw_residual)
        return bounded * dynamic_mask.to(dtype=bounded.dtype)


@MODELS.register_module()
class CausalEgoTargetAlignmentHead3D(BaseModule):
    """Predict bounded flow residuals with explicit target ego coordinates."""

    def __init__(
            self,
            dynamic_label_ids=(2, 3, 4, 5, 6, 7, 9, 10),
            num_horizons=6,
            horizon_embedding_dim=4,
            hidden_channels=16,
            max_residual_m=2.0,
            point_cloud_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
            voxel_size=(0.4, 0.4, 0.4),
            coordinate_scale_m=40.0,
            init_cfg=None):
        super().__init__(init_cfg)
        self.dynamic_label_ids = tuple(int(v) for v in dynamic_label_ids)
        self.num_horizons = int(num_horizons)
        self.horizon_embedding_dim = int(horizon_embedding_dim)
        self.hidden_channels = int(hidden_channels)
        self.max_residual_m = float(max_residual_m)
        self.point_cloud_range = tuple(float(v) for v in point_cloud_range)
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.coordinate_scale_m = float(coordinate_scale_m)
        if len(set(self.dynamic_label_ids)) != len(self.dynamic_label_ids):
            raise ValueError('dynamic_label_ids must be unique')
        if not self.dynamic_label_ids:
            raise ValueError('dynamic_label_ids must not be empty')
        if self.num_horizons <= 0 or self.horizon_embedding_dim <= 0:
            raise ValueError('horizon dimensions must be positive')
        if self.hidden_channels <= 0 or self.hidden_channels % 4:
            raise ValueError(
                'hidden_channels must be positive and divisible by 4')
        if self.max_residual_m <= 0.0:
            raise ValueError('max_residual_m must be positive')
        if len(self.point_cloud_range) != 6:
            raise ValueError('point_cloud_range must contain six values')
        if len(self.voxel_size) != 3 or any(v <= 0.0 for v in self.voxel_size):
            raise ValueError('voxel_size must contain three positive values')
        if any(
                self.point_cloud_range[index + 3]
                <= self.point_cloud_range[index]
                for index in range(3)):
            raise ValueError('point_cloud_range upper bounds must be larger')
        if self.coordinate_scale_m <= 0.0:
            raise ValueError('coordinate_scale_m must be positive')

        self.horizon_embedding = nn.Embedding(
            self.num_horizons, self.horizon_embedding_dim)
        in_channels = (
            2 + len(self.dynamic_label_ids)
            + self.horizon_embedding_dim + 2)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, self.hidden_channels, 3, padding=1),
            nn.GroupNorm(4, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.GroupNorm(4, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(self.hidden_channels, 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def target_coordinate_features(self, base_cumulative_flow):
        """Return normalized target-grid landing coordinates [B,T,X,Y,Z,2]."""
        if base_cumulative_flow.ndim != 6 or (
                base_cumulative_flow.shape[-1] != 2):
            raise ValueError(
                'base_cumulative_flow must be [B,T,X,Y,Z,2], got '
                f'{tuple(base_cumulative_flow.shape)}')
        _, _, size_x, size_y, size_z, _ = base_cumulative_flow.shape
        expected_sizes = []
        for axis in range(3):
            extent = (
                self.point_cloud_range[axis + 3]
                - self.point_cloud_range[axis])
            expected_sizes.append(int(round(extent / self.voxel_size[axis])))
        if (size_x, size_y, size_z) != tuple(expected_sizes):
            raise ValueError(
                'flow grid does not match point_cloud_range/voxel_size: '
                f'{(size_x, size_y, size_z)} versus {tuple(expected_sizes)}')

        dtype = base_cumulative_flow.dtype
        device = base_cumulative_flow.device
        source_x = (
            self.point_cloud_range[0]
            + (torch.arange(size_x, device=device, dtype=dtype) + 0.5)
            * self.voxel_size[0])
        source_y = (
            self.point_cloud_range[1]
            + (torch.arange(size_y, device=device, dtype=dtype) + 0.5)
            * self.voxel_size[1])
        target_x = (
            source_x.view(1, 1, size_x, 1, 1, 1)
            + base_cumulative_flow[..., 0:1])
        target_y = (
            source_y.view(1, 1, 1, size_y, 1, 1)
            + base_cumulative_flow[..., 1:2])
        target_x = target_x.expand(-1, -1, -1, size_y, size_z, -1)
        target_y = target_y.expand(-1, -1, size_x, -1, size_z, -1)
        return torch.cat([target_x, target_y], dim=-1) / (
            self.coordinate_scale_m)

    def forward(self, source_semantics, base_cumulative_flow):
        if base_cumulative_flow.ndim != 6 or (
                base_cumulative_flow.shape[-1] != 2):
            raise ValueError(
                'base_cumulative_flow must be [B,T,X,Y,Z,2], got '
                f'{tuple(base_cumulative_flow.shape)}')
        batch_size, num_horizons, size_x, size_y, size_z, _ = (
            base_cumulative_flow.shape)
        if num_horizons != self.num_horizons:
            raise ValueError(
                f'expected {self.num_horizons} horizons, got {num_horizons}')
        if tuple(source_semantics.shape) != (
                batch_size, size_x, size_y, size_z):
            raise ValueError(
                'source_semantics must align with the flow source grid, got '
                f'{tuple(source_semantics.shape)} versus '
                f'{tuple(base_cumulative_flow.shape)}')

        base_cumulative_flow = base_cumulative_flow.detach()
        source_semantics = source_semantics.to(
            device=base_cumulative_flow.device, dtype=torch.long)
        dynamic_one_hot = torch.stack([
            source_semantics == label_id
            for label_id in self.dynamic_label_ids
        ], dim=-1).to(dtype=base_cumulative_flow.dtype)
        dynamic_one_hot = dynamic_one_hot[:, None].expand(
            -1, num_horizons, -1, -1, -1, -1)
        dynamic_mask = dynamic_one_hot.sum(dim=-1, keepdim=True) > 0

        horizon_ids = torch.arange(
            num_horizons, device=base_cumulative_flow.device)
        horizon_features = self.horizon_embedding(horizon_ids).to(
            dtype=base_cumulative_flow.dtype)
        horizon_features = horizon_features.view(
            1, num_horizons, 1, 1, 1, self.horizon_embedding_dim).expand(
                batch_size, -1, size_x, size_y, size_z, -1)
        target_coordinates = self.target_coordinate_features(
            base_cumulative_flow)
        features = torch.cat([
            base_cumulative_flow,
            dynamic_one_hot,
            horizon_features,
            target_coordinates,
        ], dim=-1)
        features = rearrange(
            features, 'b t x y z c -> (b t) c x y z')
        raw_residual = self.net(features)
        raw_residual = rearrange(
            raw_residual, '(b t) c x y z -> b t x y z c',
            b=batch_size, t=num_horizons)
        bounded = self.max_residual_m * torch.tanh(raw_residual)
        return bounded * dynamic_mask.to(dtype=bounded.dtype)
