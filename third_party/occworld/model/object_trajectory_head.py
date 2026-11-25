import os

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class ObjectTrajectoryHead2D(BaseModule):
    """Predict causal residual trajectories per occupancy component.

    With ``num_hypotheses=1`` this is the original deterministic head.  For
    ``num_hypotheses>1`` the existing deterministic prediction is kept as a
    shared base and a learned behavior anchor conditions a small FiLM decoder
    for each hypothesis.  The added decoder is zero initialized, so loading a
    deterministic checkpoint initially gives K identical copies of the
    checkpoint prediction without changing the tokenizer or codebook.
    """

    def __init__(
            self,
            in_channels=128,
            hidden_channels=128,
            z_dim=16,
            dynamic_label_ids=(2, 3, 4, 5, 6, 7, 9, 10),
            max_residual_m=12.0,
            num_hypotheses=1,
            behavior_anchor_dim=16,
            behavior_anchor_file=None,
            behavior_anchor_key='scene_centers_normalized',
            behavior_anchors_trainable=True,
            mode_score_pooling='global',
            mode_score_type='scene_joint',
            conditional_latent_enabled=False,
            conditional_latent_dim=1,
            conditional_latent_descriptor_dim=12,
            conditional_latent_stats_file=None,
            conditional_latent_delta_scale=0.25,
            conditional_latent_apply_delta=True,
            conditional_latent_use_dense_future=False,
            conditional_future_frames=6,
            conditional_latent_direct_film=False,
            conditional_latent_direct_film_scale=0.25,
            conditional_latent_force_posterior=False,
            freeze_deterministic_base=False,
            init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = int(in_channels)
        self.z_dim = int(z_dim)
        self.dynamic_label_ids = [int(v) for v in dynamic_label_ids]
        self.max_residual_m = float(max_residual_m)
        self.num_hypotheses = int(num_hypotheses)
        self.behavior_anchor_dim = int(behavior_anchor_dim)
        self.mode_score_pooling = str(mode_score_pooling)
        if self.mode_score_pooling not in ('global', 'dynamic'):
            raise ValueError(
                'mode_score_pooling must be "global" or "dynamic", got '
                f'{self.mode_score_pooling!r}')
        self.mode_score_type = str(mode_score_type)
        if self.mode_score_type not in ('scene_joint', 'conditioned_shared'):
            raise ValueError(
                'mode_score_type must be "scene_joint" or '
                f'"conditioned_shared", got {self.mode_score_type!r}')
        self.conditional_latent_enabled = bool(
            conditional_latent_enabled)
        self.conditional_latent_dim = int(conditional_latent_dim)
        self.conditional_latent_descriptor_dim = int(
            conditional_latent_descriptor_dim)
        self.conditional_latent_delta_scale = float(
            conditional_latent_delta_scale)
        self.conditional_latent_apply_delta = bool(
            conditional_latent_apply_delta)
        self.conditional_latent_use_dense_future = bool(
            conditional_latent_use_dense_future)
        self.conditional_future_frames = int(conditional_future_frames)
        self.conditional_latent_direct_film = bool(
            conditional_latent_direct_film)
        self.conditional_latent_direct_film_scale = float(
            conditional_latent_direct_film_scale)
        self.conditional_latent_force_posterior = bool(
            conditional_latent_force_posterior)
        if self.num_hypotheses < 1:
            raise ValueError('num_hypotheses must be positive')
        if self.behavior_anchor_dim < 1:
            raise ValueError('behavior_anchor_dim must be positive')
        if self.conditional_latent_enabled:
            if self.num_hypotheses != 3:
                raise ValueError(
                    'The conditional-latent pilot uses the three-point '
                    'Gauss-Hermite rule and therefore requires K=3')
            if (not self.conditional_latent_use_dense_future
                    and self.conditional_latent_dim != 1):
                raise ValueError(
                    'The probability-grounded K=3 pilot requires a scalar '
                    'conditional latent (conditional_latent_dim=1)')
            if self.conditional_latent_dim < 1:
                raise ValueError('conditional_latent_dim must be positive')
            if self.conditional_latent_descriptor_dim < 1:
                raise ValueError(
                    'conditional_latent_descriptor_dim must be positive')
            if self.conditional_latent_delta_scale <= 0:
                raise ValueError(
                    'conditional_latent_delta_scale must be positive')
            if self.conditional_future_frames < 1:
                raise ValueError('conditional_future_frames must be positive')
            if self.conditional_latent_direct_film_scale <= 0:
                raise ValueError(
                    'conditional_latent_direct_film_scale must be positive')
        context_channels = (
            self.in_channels * 2 + 3 + len(self.dynamic_label_ids))
        self.context = nn.Sequential(
            nn.Conv2d(context_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(hidden_channels, self.z_dim * 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        if self.num_hypotheses > 1:
            if behavior_anchor_file:
                if not os.path.isfile(behavior_anchor_file):
                    raise FileNotFoundError(
                        f'Missing behavior anchor file: '
                        f'{behavior_anchor_file}')
                anchor_data = np.load(behavior_anchor_file)
                anchor_values = np.asarray(
                    anchor_data[behavior_anchor_key], dtype=np.float32)
                anchor_values = anchor_values.reshape(
                    anchor_values.shape[0], -1)
                expected_shape = (
                    self.num_hypotheses, self.behavior_anchor_dim)
                if anchor_values.shape != expected_shape:
                    raise ValueError(
                        f'{behavior_anchor_key} has shape '
                        f'{anchor_values.shape}, expected {expected_shape}')
                anchor_tensor = torch.from_numpy(anchor_values)
            else:
                anchor_tensor = torch.empty(
                    self.num_hypotheses, self.behavior_anchor_dim)
                nn.init.normal_(anchor_tensor, std=0.02)
            self.behavior_anchors = nn.Parameter(
                anchor_tensor,
                requires_grad=bool(behavior_anchors_trainable))
            self.anchor_to_film = nn.Linear(
                self.behavior_anchor_dim, hidden_channels * 2)
            self.mode_output = nn.Conv2d(
                hidden_channels, self.z_dim * 2, 1)
            # Predict all K probabilities jointly from the shared scene
            # context.  A shared scalar scorer would have cancelling
            # cross-entropy gradients when zero initialization makes the
            # initial per-mode features identical.
            self.mode_score_head = nn.Linear(
                hidden_channels, self.num_hypotheses)
            if self.mode_score_type == 'conditioned_shared':
                # Score each hypothesis from its own conditioned feature with
                # the same scalar MLP.  This exposes candidate quality while
                # keeping the comparison rule permutation-equivariant across
                # modes, following the query-wise scoring pattern used by
                # multimodal trajectory predictors.
                self.mode_quality_head = nn.Sequential(
                    nn.Linear(
                        hidden_channels + self.behavior_anchor_dim,
                        hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_channels, 1),
                )
            nn.init.zeros_(self.mode_output.weight)
            nn.init.zeros_(self.mode_output.bias)
            nn.init.zeros_(self.mode_score_head.weight)
            nn.init.zeros_(self.mode_score_head.bias)
            if self.mode_score_type == 'conditioned_shared':
                nn.init.zeros_(self.mode_quality_head[-1].weight)
                nn.init.zeros_(self.mode_quality_head[-1].bias)

            if self.conditional_latent_enabled:
                if (not self.conditional_latent_use_dense_future
                        and not conditional_latent_stats_file):
                    raise ValueError(
                        'conditional_latent_stats_file is required when '
                        'conditional_latent_enabled=True')
                if (conditional_latent_stats_file
                        and not os.path.isfile(conditional_latent_stats_file)):
                    raise FileNotFoundError(
                        'Missing conditional-latent descriptor statistics: '
                        f'{conditional_latent_stats_file}')
                if not self.conditional_latent_use_dense_future:
                    statistics = np.load(conditional_latent_stats_file)
                    descriptor_mean = np.asarray(
                        statistics['scene_feature_mean'], dtype=np.float32
                    ).reshape(-1)
                    descriptor_std = np.asarray(
                        statistics['scene_feature_std'], dtype=np.float32
                    ).reshape(-1)
                    expected_descriptor_shape = (
                        self.conditional_latent_descriptor_dim,)
                    if descriptor_mean.shape != expected_descriptor_shape:
                        raise ValueError(
                            'scene_feature_mean has shape '
                            f'{descriptor_mean.shape}, expected '
                            f'{expected_descriptor_shape}')
                    if descriptor_std.shape != expected_descriptor_shape:
                        raise ValueError(
                            'scene_feature_std has shape '
                            f'{descriptor_std.shape}, expected '
                            f'{expected_descriptor_shape}')
                    self.register_buffer(
                        'conditional_descriptor_mean',
                        torch.from_numpy(descriptor_mean))
                    self.register_buffer(
                        'conditional_descriptor_std',
                        torch.from_numpy(descriptor_std).clamp_min(1e-3))
                self.latent_prior_head = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(
                        hidden_channels,
                        self.conditional_latent_dim * 2),
                )
                posterior_condition_dim = (
                    hidden_channels
                    if self.conditional_latent_use_dense_future
                    else self.conditional_latent_descriptor_dim)
                if self.conditional_latent_use_dense_future:
                    self.future_dynamic_encoder = nn.Sequential(
                        nn.Conv2d(
                            self.conditional_future_frames, 32,
                            5, stride=2, padding=2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 64, 3, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, hidden_channels, 3, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                    )
                self.latent_posterior_head = nn.Sequential(
                    nn.Linear(
                        hidden_channels
                        + posterior_condition_dim,
                        hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(
                        hidden_channels,
                        self.conditional_latent_dim * 2),
                )
                self.latent_to_anchor_delta = nn.Sequential(
                    nn.Linear(self.conditional_latent_dim, 32),
                    nn.Tanh(),
                    nn.Linear(32, self.behavior_anchor_dim),
                )
                if self.conditional_latent_direct_film:
                    self.latent_to_direct_film = nn.Sequential(
                        nn.Linear(
                            self.conditional_latent_dim, hidden_channels),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_channels, hidden_channels * 2),
                    )
                    nn.init.normal_(
                        self.latent_to_direct_film[-1].weight, std=0.02)
                    nn.init.zeros_(self.latent_to_direct_film[-1].bias)
                # Start p(z|past) and q(z|past,future) as N(0, I).  A small
                # decoder initialization preserves the completed K generator
                # while keeping a reconstruction gradient path into q.
                nn.init.zeros_(self.latent_prior_head[-1].weight)
                nn.init.zeros_(self.latent_prior_head[-1].bias)
                nn.init.zeros_(self.latent_posterior_head[-1].weight)
                nn.init.zeros_(self.latent_posterior_head[-1].bias)
                nn.init.normal_(
                    self.latent_to_anchor_delta[-1].weight, std=1e-3)
                nn.init.zeros_(self.latent_to_anchor_delta[-1].bias)
                # Three-point Gauss-Hermite quadrature for N(mu, sigma^2).
                # Mode 0 is the deployable prior/posterior mean (MAP).
                latent_nodes = torch.zeros(
                    self.num_hypotheses, self.conditional_latent_dim)
                direction = torch.ones(self.conditional_latent_dim)
                direction = direction / direction.norm().clamp_min(1.0)
                latent_nodes[1] = -(3.0 ** 0.5) * direction
                latent_nodes[2] = (3.0 ** 0.5) * direction
                self.register_buffer(
                    'conditional_latent_nodes', latent_nodes)
                self.register_buffer(
                    'conditional_latent_probabilities',
                    torch.tensor([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]))

        if freeze_deterministic_base:
            for module in (self.context, self.output):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    @staticmethod
    def _pool_components(dense_residual, component_ids):
        pooled_batches = []
        for batch_index in range(dense_residual.shape[0]):
            labels = component_ids[batch_index].reshape(-1)
            valid = labels >= 0
            if not valid.any():
                pooled_batches.append(dense_residual[batch_index] * 0.0)
                continue
            valid_labels = labels[valid]
            num_components = int(valid_labels.max().item()) + 1
            values = dense_residual[batch_index].reshape(
                dense_residual.shape[1], -1, 2)[:, valid]
            sums = values.new_zeros(
                dense_residual.shape[1], num_components, 2)
            sums.index_add_(1, valid_labels, values)
            counts = torch.bincount(
                valid_labels, minlength=num_components).to(values)
            means = sums / counts[None, :, None].clamp_min(1.0)
            safe_labels = labels.clamp_min(0)
            pooled_values = means[:, safe_labels]
            pooled_values = pooled_values * valid[None, :, None].to(values)
            pooled_batches.append(pooled_values.reshape_as(
                dense_residual[batch_index]))
        return torch.stack(pooled_batches, dim=0)

    def forward(
            self,
            future_feature,
            source_feature,
            source_semantics,
            component_ids,
            ego_displacement_xy,
            batch_size,
            num_frames,
            future_behavior_descriptor=None,
            future_behavior_valid=None,
            future_target_semantics=None):
        if future_feature.shape[0] != batch_size * num_frames:
            raise ValueError('future feature batch/time shape mismatch')
        if source_feature.shape[:2] != (batch_size, self.in_channels):
            raise ValueError('source feature shape mismatch')
        if source_semantics.shape != component_ids.shape:
            raise ValueError(
                'source semantics/component shape mismatch: '
                f'{source_semantics.shape} vs {component_ids.shape}')
        if ego_displacement_xy.shape != (batch_size, num_frames, 2):
            raise ValueError('ego displacement shape mismatch')

        _, _, size_x, size_y = future_feature.shape
        expected_grid = (batch_size, size_x, size_y, self.z_dim)
        if source_semantics.shape != expected_grid:
            raise ValueError(
                f'expected source grid {expected_grid}, got '
                f'{tuple(source_semantics.shape)}')
        source = source_feature[:, None].expand(
            -1, num_frames, -1, -1, -1).reshape(
                batch_size * num_frames,
                self.in_channels,
                size_x,
                size_y)
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
            batch_size, -1, -1, size_x, size_y).reshape(
                batch_size * num_frames, 1, size_x, size_y)
        class_bev = torch.stack([
            (source_semantics == label_id).any(dim=-1)
            for label_id in self.dynamic_label_ids
        ], dim=1).to(future_feature.dtype)
        class_bev = class_bev[:, None].expand(
            -1, num_frames, -1, -1, -1).reshape(
                batch_size * num_frames,
                len(self.dynamic_label_ids),
                size_x,
                size_y)
        context = torch.cat([
            future_feature,
            source.to(future_feature),
            ego.to(future_feature),
            horizon,
            class_bev,
        ], dim=1)
        context_feature = self.context(context)
        base_logits = self.output(context_feature)
        component_ids = component_ids.to(
            device=base_logits.device, dtype=torch.long)

        if self.num_hypotheses == 1:
            dense = torch.tanh(base_logits)
            dense = rearrange(
                dense,
                '(b t) (z c) x y -> b t x y z c',
                b=batch_size,
                t=num_frames,
                z=self.z_dim,
                c=2)
            dense = dense * self.max_residual_m
            return self._pool_components(dense, component_ids)

        score_feature_grid = rearrange(
            context_feature,
            '(b t) c x y -> b t c x y',
            b=batch_size,
            t=num_frames)
        global_score_feature = score_feature_grid.mean(dim=(1, 3, 4))
        if self.mode_score_pooling == 'dynamic':
            # Dynamic objects occupy only a small fraction of the scene.  A
            # global spatial mean therefore mostly exposes static context to
            # the selector.  Pool over inference-visible source-frame dynamic
            # cells, falling back to the global feature for empty scenes.
            dynamic_bev = torch.stack([
                (source_semantics == label_id).any(dim=-1)
                for label_id in self.dynamic_label_ids
            ], dim=1).any(dim=1)
            dynamic_weights = dynamic_bev[:, None, None].to(
                score_feature_grid)
            dynamic_sum = (
                score_feature_grid * dynamic_weights
            ).sum(dim=(1, 3, 4))
            dynamic_count = dynamic_bev.sum(dim=(1, 2)).to(
                score_feature_grid) * num_frames
            dynamic_score_feature = (
                dynamic_sum / dynamic_count[:, None].clamp_min(1.0))
            has_dynamic = dynamic_count > 0
            scene_score_feature = torch.where(
                has_dynamic[:, None],
                dynamic_score_feature,
                global_score_feature)
        else:
            scene_score_feature = global_score_feature

        latent_statistics = None
        if self.conditional_latent_enabled:
            prior_parameters = self.latent_prior_head(scene_score_feature)
            prior_mu, prior_log_sigma = prior_parameters.chunk(2, dim=-1)
            prior_log_sigma = prior_log_sigma.clamp(-4.0, 2.0)
            posterior_mu = prior_mu
            posterior_log_sigma = prior_log_sigma
            posterior_valid = torch.zeros(
                batch_size, dtype=torch.bool,
                device=scene_score_feature.device)
            if (self.conditional_latent_use_dense_future
                    and future_target_semantics is not None):
                expected_target_shape = (
                    batch_size, self.conditional_future_frames,
                    size_x, size_y, self.z_dim)
                if future_target_semantics.shape != expected_target_shape:
                    raise ValueError(
                        'future_target_semantics has shape '
                        f'{tuple(future_target_semantics.shape)}, expected '
                        f'{expected_target_shape}')
                target_dynamic = torch.zeros_like(
                    future_target_semantics, dtype=torch.bool)
                for label_id in self.dynamic_label_ids:
                    target_dynamic |= future_target_semantics == label_id
                target_dynamic_bev = target_dynamic.any(dim=-1).to(
                    scene_score_feature)
                future_condition = self.future_dynamic_encoder(
                    target_dynamic_bev)
                posterior_parameters = self.latent_posterior_head(torch.cat([
                    scene_score_feature, future_condition], dim=-1))
                candidate_mu, candidate_log_sigma = (
                    posterior_parameters.chunk(2, dim=-1))
                candidate_log_sigma = candidate_log_sigma.clamp(-4.0, 2.0)
                posterior_valid = torch.ones(
                    batch_size, dtype=torch.bool,
                    device=scene_score_feature.device)
                posterior_mu = candidate_mu
                posterior_log_sigma = candidate_log_sigma
            elif future_behavior_descriptor is not None:
                if future_behavior_descriptor.shape != (
                        batch_size,
                        self.conditional_latent_descriptor_dim):
                    raise ValueError(
                        'future_behavior_descriptor has shape '
                        f'{tuple(future_behavior_descriptor.shape)}, '
                        'expected '
                        f'{(batch_size, self.conditional_latent_descriptor_dim)}')
                normalized_descriptor = (
                    future_behavior_descriptor.to(scene_score_feature)
                    - self.conditional_descriptor_mean.to(
                        scene_score_feature)) / self.conditional_descriptor_std.to(
                            scene_score_feature)
                posterior_parameters = self.latent_posterior_head(torch.cat([
                    scene_score_feature, normalized_descriptor], dim=-1))
                candidate_mu, candidate_log_sigma = (
                    posterior_parameters.chunk(2, dim=-1))
                candidate_log_sigma = candidate_log_sigma.clamp(-4.0, 2.0)
                if future_behavior_valid is None:
                    posterior_valid = torch.isfinite(
                        future_behavior_descriptor).all(dim=-1)
                else:
                    posterior_valid = future_behavior_valid.to(
                        device=scene_score_feature.device,
                        dtype=torch.bool)
                posterior_mu = torch.where(
                    posterior_valid[:, None], candidate_mu, prior_mu)
                posterior_log_sigma = torch.where(
                    posterior_valid[:, None], candidate_log_sigma,
                    prior_log_sigma)
            use_posterior = (
                self.training or self.conditional_latent_force_posterior
            ) & posterior_valid
            latent_mu = torch.where(
                use_posterior[:, None], posterior_mu, prior_mu)
            latent_log_sigma = torch.where(
                use_posterior[:, None], posterior_log_sigma,
                prior_log_sigma)
            latent = (
                latent_mu[:, None]
                + latent_log_sigma.exp()[:, None]
                * self.conditional_latent_nodes[None].to(latent_mu))
            anchor_delta = self.conditional_latent_delta_scale * torch.tanh(
                self.latent_to_anchor_delta(latent))
            if not self.conditional_latent_apply_delta:
                anchor_delta = anchor_delta * 0.0
            anchors = self.behavior_anchors[None].to(anchor_delta) + anchor_delta
            behavior_mode_logits = self.conditional_latent_probabilities.to(
                scene_score_feature).log()[None].expand(batch_size, -1)
            latent_statistics = {
                'behavior_latent_prior_mu': prior_mu,
                'behavior_latent_prior_log_sigma': prior_log_sigma,
                'behavior_latent_posterior_mu': posterior_mu,
                'behavior_latent_posterior_log_sigma': posterior_log_sigma,
                'behavior_latent_posterior_valid': posterior_valid,
                'behavior_latent_mode_probabilities': (
                    self.conditional_latent_probabilities.to(
                        scene_score_feature)[None].expand(batch_size, -1)),
                'behavior_latent_scene_feature': scene_score_feature,
            }
        else:
            anchors = self.behavior_anchors[None].expand(
                batch_size, -1, -1)

        if (self.conditional_latent_enabled
                and self.conditional_latent_direct_film):
            film = self.latent_to_direct_film(latent)
            film_scale = self.conditional_latent_direct_film_scale
        else:
            film = self.anchor_to_film(anchors)
            film_scale = 0.1
        scale, shift = film.chunk(2, dim=-1)
        conditioned = (
            context_feature[:, None]
            * (1.0 + film_scale * torch.tanh(scale).repeat_interleave(
                num_frames, dim=0)[:, :, :, None, None])
            + film_scale * shift.repeat_interleave(
                num_frames, dim=0)[:, :, :, None, None])
        conditioned_score_grid = rearrange(
            conditioned,
            '(b t) k c x y -> b t k c x y',
            b=batch_size,
            t=num_frames)
        global_conditioned_feature = conditioned_score_grid.mean(
            dim=(1, 4, 5))
        if self.mode_score_pooling == 'dynamic':
            conditioned_dynamic_weights = dynamic_bev[
                :, None, None, None].to(conditioned_score_grid)
            conditioned_dynamic_sum = (
                conditioned_score_grid * conditioned_dynamic_weights
            ).sum(dim=(1, 4, 5))
            conditioned_dynamic_feature = (
                conditioned_dynamic_sum
                / dynamic_count[:, None, None].clamp_min(1.0))
            mode_score_feature = torch.where(
                has_dynamic[:, None, None],
                conditioned_dynamic_feature,
                global_conditioned_feature)
        else:
            mode_score_feature = global_conditioned_feature
        if not self.conditional_latent_enabled:
            if self.mode_score_type == 'conditioned_shared':
                behavior_mode_logits = self.mode_quality_head(torch.cat(
                    [mode_score_feature, anchors.to(mode_score_feature)],
                    dim=-1)).squeeze(-1)
            else:
                behavior_mode_logits = self.mode_score_head(
                    scene_score_feature)
        mode_logits = self.mode_output(rearrange(
            conditioned, 'bt k c x y -> (bt k) c x y'))
        mode_logits = rearrange(
            mode_logits,
            '(b t k) (z c) x y -> b k t x y z c',
            b=batch_size,
            t=num_frames,
            k=self.num_hypotheses,
            z=self.z_dim,
            c=2)
        base_logits = rearrange(
            base_logits,
            '(b t) (z c) x y -> b 1 t x y z c',
            b=batch_size,
            t=num_frames,
            z=self.z_dim,
            c=2)
        dense = torch.tanh(base_logits + mode_logits) * self.max_residual_m

        dense_flat = rearrange(dense, 'b k t x y z c -> (b k) t x y z c')
        component_flat = component_ids[:, None].expand(
            -1, self.num_hypotheses, -1, -1, -1)
        component_flat = rearrange(component_flat, 'b k x y z -> (b k) x y z')
        pooled = self._pool_components(dense_flat, component_flat)
        pooled = rearrange(
            pooled,
            '(b k) t x y z c -> b k t x y z c',
            b=batch_size,
            k=self.num_hypotheses)
        if latent_statistics is not None:
            return pooled, behavior_mode_logits, latent_statistics
        return pooled, behavior_mode_logits
