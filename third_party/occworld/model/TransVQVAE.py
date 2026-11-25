from mmengine.registry import MODELS
from mmengine.model import BaseModule
import numpy as np
import torch.nn as nn, torch
import torch.nn.functional as F
from einops import rearrange
from copy import deepcopy
from contextlib import nullcontext
import torch.distributions as dist
from utils.metric_stp3 import PlanningMetric
import time

@MODELS.register_module()
class TransVQVAE(BaseModule):
    def __init__(self, vae, transformer, num_frames=10, offset=1,
                 pose_encoder=None, pose_decoder=None,
                 pose_actor=None, give_hiddens=False, delta_input=False,
                 without_all=False, use_flow=False, flow_head=None,
                 return_flow_logits=False, return_pred_flow=True,
                 use_dynamic_refine=False, dynamic_refine_head=None,
                 use_transport_gate=False, transport_gate_head=None,
                 transport_gate_detach_base=True,
                 transport_gate_threshold=0.5,
                 use_transport_alignment=False,
                 transport_alignment_head=None,
                 dynamic_label_ids=(2, 3, 4, 5, 6, 7, 9, 10),
                 dynamic_refine_detach_base=True,
                 use_causal_transport=False,
                 causal_source_index=4,
                 causal_target_start=5,
                 causal_target_end=11,
                 causal_detach_features=True,
                 causal_support_threshold=0.05,
                 causal_flow_mode='learned_total',
                 causal_merge_mode='replace',
                 causal_static_label_ids=(0, 1, 8, 11, 12, 13, 14, 15, 16),
                 causal_static_confidence_threshold=0.6,
                 causal_transport_start_index=0,
                 causal_apply_transport=True,
                 causal_residual_head=None,
                 object_trajectory_head=None,
                 object_trajectory_residual_gate='none',
                 object_trajectory_residual_gate_threshold=0.0,
                 object_trajectory_residual_gate_softness=0.25,
                 use_ego_se2=False,
                 ego_se2_head=None,
                 ego_se2_residual=False,
                 ego_se2_target_start=5,
                 ego_se2_target_end=11):
        super().__init__()
        self.num_frames = num_frames
        self.offset = offset
        self.use_flow = use_flow
        self.vae = MODELS.build(vae)
        self.transformer = MODELS.build(transformer)
        if pose_encoder is not None:
            self.pose_encoder = MODELS.build(pose_encoder)
        if pose_decoder is not None:
            self.pose_decoder = MODELS.build(pose_decoder)
        if pose_actor is not None:
            self.pose_actor = MODELS.build(pose_actor)
        self.give_hiddens = give_hiddens
        self.delta_input = delta_input
        self.planning_metric = None
        self.without_all = without_all
        self.return_flow_logits = return_flow_logits
        self.return_pred_flow = return_pred_flow
        self.use_dynamic_refine = use_dynamic_refine
        self.use_transport_gate = bool(use_transport_gate)
        self.transport_gate_detach_base = bool(transport_gate_detach_base)
        self.transport_gate_threshold = float(transport_gate_threshold)
        self.use_transport_alignment = bool(use_transport_alignment)
        self.dynamic_label_ids = [int(v) for v in dynamic_label_ids]
        self.dynamic_refine_detach_base = dynamic_refine_detach_base
        self.use_causal_transport = use_causal_transport
        self.causal_source_index = int(causal_source_index)
        self.causal_target_start = int(causal_target_start)
        self.causal_target_end = int(causal_target_end)
        self.causal_detach_features = causal_detach_features
        self.causal_support_threshold = float(causal_support_threshold)
        self.causal_flow_mode = causal_flow_mode
        self.causal_merge_mode = causal_merge_mode
        self.causal_static_label_ids = [
            int(v) for v in causal_static_label_ids]
        self.causal_static_confidence_threshold = float(
            causal_static_confidence_threshold)
        self.causal_transport_start_index = int(
            causal_transport_start_index)
        self.causal_apply_transport = bool(causal_apply_transport)
        self.object_trajectory_residual_gate = (
            object_trajectory_residual_gate)
        self.object_trajectory_residual_gate_threshold = float(
            object_trajectory_residual_gate_threshold)
        self.object_trajectory_residual_gate_softness = float(
            object_trajectory_residual_gate_softness)
        self.use_ego_se2 = bool(use_ego_se2)
        self.ego_se2_residual = bool(ego_se2_residual)
        self.ego_se2_target_start = int(ego_se2_target_start)
        self.ego_se2_target_end = int(ego_se2_target_end)
        if self.causal_flow_mode not in (
                'learned_total', 'predicted_ego', 'ego_residual',
                'object_trajectory'):
            raise ValueError(
                'causal_flow_mode must be one of "learned_total", '
                '"predicted_ego", "ego_residual", or '
                f'"object_trajectory", got '
                f'{self.causal_flow_mode}')
        if self.causal_merge_mode not in (
                'replace', 'union', 'free_only', 'static_confidence'):
            raise ValueError(
                'causal_merge_mode must be one of "replace", "union", '
                '"free_only", or "static_confidence", got '
                f'{self.causal_merge_mode}')
        if self.object_trajectory_residual_gate not in (
                'none', 'norm', 'sigmoid_norm'):
            raise ValueError(
                'object_trajectory_residual_gate must be one of "none", '
                f'"norm", or "sigmoid_norm", got '
                f'{self.object_trajectory_residual_gate}')
        if self.object_trajectory_residual_gate_softness <= 0:
            raise ValueError(
                'object_trajectory_residual_gate_softness must be positive')
        num_causal_frames = self.causal_target_end - self.causal_target_start
        if not 0 <= self.causal_transport_start_index < num_causal_frames:
            raise ValueError(
                'causal_transport_start_index must be in '
                f'[0, {num_causal_frames}), got '
                f'{self.causal_transport_start_index}')
        if self.use_flow:
            assert flow_head is not None, 'flow_head must be set when use_flow=True'
            self.flow_head = MODELS.build(flow_head)
        if self.use_dynamic_refine:
            assert self.use_flow, (
                'use_dynamic_refine=True requires the existing flow head')
            assert dynamic_refine_head is not None, (
                'dynamic_refine_head must be set when '
                'use_dynamic_refine=True')
            self.dynamic_refine_head = MODELS.build(dynamic_refine_head)
        if self.use_transport_gate:
            assert self.use_causal_transport, (
                'use_transport_gate=True requires use_causal_transport=True')
            assert transport_gate_head is not None, (
                'transport_gate_head must be set when '
                'use_transport_gate=True')
            self.transport_gate_head = MODELS.build(transport_gate_head)
        if self.use_transport_alignment:
            if not self.use_transport_gate:
                raise ValueError(
                    'use_transport_alignment=True requires the transport gate')
            if transport_alignment_head is None:
                raise ValueError(
                    'use_transport_alignment=True requires '
                    'transport_alignment_head')
            self.transport_alignment_head = MODELS.build(
                transport_alignment_head)
        if self.use_causal_transport:
            assert self.use_flow, (
                'use_causal_transport=True requires use_flow=True')
            if self.causal_flow_mode != 'learned_total':
                assert pose_encoder is not None and pose_decoder is not None, (
                    f'{self.causal_flow_mode} requires OccWorld pose modules')
            if self.causal_flow_mode == 'ego_residual':
                assert causal_residual_head is not None, (
                    'ego_residual requires causal_residual_head')
                self.causal_residual_head = MODELS.build(
                    causal_residual_head)
            if self.causal_flow_mode == 'object_trajectory':
                assert object_trajectory_head is not None, (
                    'object_trajectory requires object_trajectory_head')
                self.object_trajectory_head = MODELS.build(
                    object_trajectory_head)
        if self.use_ego_se2:
            if ego_se2_head is None:
                raise ValueError('use_ego_se2=True requires ego_se2_head')
            if not hasattr(self, 'pose_encoder'):
                raise ValueError('ego SE2 head requires OccWorld pose modules')
            if self.ego_se2_target_end <= self.ego_se2_target_start:
                raise ValueError('ego_se2_target_end must exceed start')
            self.ego_se2_head = MODELS.build(ego_se2_head)
    def forward(self, x, metas=None, target_occs=None):
        if hasattr(self, 'pose_encoder'):
            if self.training:
                return self.forward_train_with_plan(
                    x, metas, target_occs=target_occs)
            else:
                return self.forward_inference_with_plan(
                    x, metas, target_occs=target_occs)
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_inference(x)

    def _semantic_logits_from_dense_feature(self, dense_feature, target_shape):
        bs, num_frames, _, _, depth = target_shape
        dense_feature = dense_feature.permute(0, 2, 3, 1)
        dense_feature = dense_feature.reshape(-1, depth, self.vae.expansion)
        template = self.vae.class_embeds.weight.T.unsqueeze(0)
        semantic_logits = torch.matmul(dense_feature, template)
        return semantic_logits.reshape(
            bs,
            num_frames,
            target_shape[2],
            target_shape[3],
            depth,
            self.vae.num_cls)

    def _add_ego_se2_output(
            self, output_dict, pose_features, autoregressive=False,
            base_ego_se2=None):
        expected_frames = (
            self.ego_se2_target_end - self.ego_se2_target_start)
        if autoregressive:
            selected = pose_features
        else:
            start = self.ego_se2_target_start - self.offset
            end = self.ego_se2_target_end - self.offset
            selected = pose_features[:, start:end]
        if selected.shape[1] != expected_frames:
            raise ValueError(
                f'Ego SE2 selected {selected.shape[1]} frames, expected '
                f'{expected_frames}')
        if self.ego_se2_residual:
            if base_ego_se2 is None:
                raise ValueError(
                    'ego_se2_residual requires OccWorld base ego SE2')
            if base_ego_se2.shape != (
                    selected.shape[0], expected_frames, 3):
                raise ValueError(
                    'base ego SE2 shape mismatch: expected '
                    f'{(selected.shape[0], expected_frames, 3)}, got '
                    f'{tuple(base_ego_se2.shape)}')
            residual = self.ego_se2_head(selected, base_ego_se2)
            output_dict['base_ego_se2'] = base_ego_se2
            output_dict['pred_ego_se2_residual'] = residual
            output_dict['pred_ego_se2'] = base_ego_se2 + residual
        else:
            output_dict['pred_ego_se2'] = self.ego_se2_head(selected)

    def _build_occworld_ego_se2(
            self, x, metas, future_pose_increments, num_frames):
        """Convert OccWorld pose increments to source-relative ego SE(2).

        OccWorld's dense ego flow maps source points into each future ego
        frame. Ego translation has the opposite sign; yaw is unavailable in
        the original pose decoder and starts at zero for the residual head.
        """
        source_index = self.ego_se2_target_start - 1
        known_increment = x.new_tensor(np.asarray([
            meta['rel_poses'][source_index] for meta in metas
        ]), dtype=torch.float32)
        expected = (x.shape[0], max(num_frames - 1, 0), 2)
        if future_pose_increments.shape != expected:
            raise ValueError(
                'future pose increments must have shape '
                f'{expected}, got {tuple(future_pose_increments.shape)}')
        increments = torch.cat([
            known_increment[:, None],
            future_pose_increments.to(known_increment),
        ], dim=1)
        displacement = torch.cumsum(increments, dim=1)
        translation = torch.stack([
            displacement[..., 1],
            -displacement[..., 0],
        ], dim=-1)
        yaw = translation.new_zeros((*translation.shape[:-1], 1))
        return torch.cat([translation, yaw], dim=-1)

    def _decode_dense_feature_from_token_logits(
            self, token_logits, shape, target_shape):
        bs, num_frames, _, _, _ = token_logits.shape
        codebook_weight = self.vae.vqvae.embedding.weight.detach()
        token_probs = F.softmax(token_logits, dim=2)
        z_q_predict = torch.einsum(
            'btkhw,kc->btchw', token_probs, codebook_weight)
        z_q_predict = rearrange(
            z_q_predict, 'b t c h w -> (b t) c h w')
        z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)
        dense_feature = self.vae.forward_decoder(
            z_q_predict,
            deepcopy(shape),
            target_shape,
            return_feature_only=True)
        if self.use_causal_transport and self.causal_detach_features:
            dense_feature = dense_feature.detach()
        return dense_feature

    def _predict_flow_from_token_logits(
            self, token_logits, shape, target_shape, return_logits=False):
        assert self.use_flow
        bs, num_frames = token_logits.shape[:2]
        dense_feature = self._decode_dense_feature_from_token_logits(
            token_logits, shape, target_shape)
        pred_flow = self.flow_head(dense_feature, bs, num_frames)
        if not return_logits:
            return pred_flow
        semantic_logits = self._semantic_logits_from_dense_feature(
            dense_feature, target_shape)
        return pred_flow, semantic_logits

    def _decode_source_dense_feature(
            self, source_quantized, shape, target_shape):
        source_quantized = self.vae.vqvae.post_quant_conv(
            source_quantized)
        source_target_shape = (
            source_quantized.shape[0],
            1,
            target_shape[2],
            target_shape[3],
            target_shape[4],
        )
        source_feature = self.vae.forward_decoder(
            source_quantized,
            deepcopy(shape),
            source_target_shape,
            return_feature_only=True)
        if self.causal_detach_features:
            source_feature = source_feature.detach()
        return source_feature

    def _add_flow_outputs(self, output_dict, token_logits, shape, target_shape):
        flow_output = self._predict_flow_from_token_logits(
            token_logits,
            shape,
            target_shape,
            return_logits=self.return_flow_logits)
        if self.return_flow_logits:
            pred_flow, flow_logits = flow_output
            output_dict['pred_flow'] = pred_flow
            output_dict['flow_logits'] = flow_logits
        else:
            output_dict['pred_flow'] = flow_output

    def _add_causal_flow_outputs(
            self, output_dict, token_logits, shape, target_shape,
            autoregressive=False):
        if autoregressive:
            selected_logits = token_logits
        else:
            start = self.causal_target_start - self.offset
            end = self.causal_target_end - self.offset
            selected_logits = token_logits[:, start:end]
        expected_frames = self.causal_target_end - self.causal_target_start
        if selected_logits.shape[1] != expected_frames:
            raise ValueError(
                f'Causal flow selected {selected_logits.shape[1]} frames, '
                f'expected {expected_frames}')
        output_dict['pred_flow'] = self._predict_flow_from_token_logits(
            selected_logits, shape, target_shape, return_logits=False)

    def _build_predicted_ego_flow(
            self, x, metas, future_pose_increments, num_frames):
        """Build source-grid to future-ego translation in meters.

        OccWorld pose increments use [lateral, longitudinal]. Occupancy ego
        coordinates use [forward, left], so a moving ego maps a static source
        point by [-longitudinal, lateral]. The source frame's first future
        increment is already available in the observed pose sequence; later
        increments come from the autoregressive pose prediction.
        """
        if len(metas) != x.shape[0]:
            raise ValueError(
                f'{len(metas)} metadata entries for batch {x.shape[0]}')
        known_increment = x.new_tensor(np.asarray([
            meta['rel_poses'][self.causal_source_index]
            for meta in metas
        ]), dtype=torch.float32)
        if future_pose_increments.shape != (
                x.shape[0], max(num_frames - 1, 0), 2):
            raise ValueError(
                'future pose increments must have shape '
                f'{(x.shape[0], max(num_frames - 1, 0), 2)}, got '
                f'{tuple(future_pose_increments.shape)}')
        increments = torch.cat([
            known_increment[:, None],
            future_pose_increments.to(known_increment),
        ], dim=1)
        displacement = torch.cumsum(increments, dim=1)
        flow_xy = torch.stack([
            -displacement[..., 1],
            displacement[..., 0],
        ], dim=-1)
        return flow_xy[:, :, None, None, None].expand(
            -1, -1, x.shape[2], x.shape[3], x.shape[4], -1)

    def _gate_object_trajectory_residual(self, residual):
        gate_mode = self.object_trajectory_residual_gate
        if gate_mode == 'none':
            return residual, None
        residual_norm = torch.linalg.vector_norm(
            residual, dim=-1, keepdim=True)
        threshold = self.object_trajectory_residual_gate_threshold
        if gate_mode == 'norm':
            gate = (residual_norm >= threshold).to(residual)
        elif gate_mode == 'sigmoid_norm':
            softness = self.object_trajectory_residual_gate_softness
            gate = torch.sigmoid((residual_norm - threshold) / softness)
        else:
            raise AssertionError(f'Unhandled gate mode {gate_mode}')
        return residual * gate, gate

    @staticmethod
    def _select_pose_increments(pose_decoded, output_metas, start, count):
        if count == 0:
            return pose_decoded.new_empty((pose_decoded.shape[0], 0, 2))
        pose_slice = pose_decoded[:, start:start + count]
        mode = pose_decoded.new_tensor(np.asarray([
            meta['gt_mode'][start:start + count]
            for meta in output_metas
        ]), dtype=torch.bool)
        return pose_slice[mode].reshape(pose_decoded.shape[0], count, 2)

    def _apply_causal_ego_flow(
            self, output_dict, x, metas, future_pose_increments,
            token_logits=None, source_quantized=None, shape=None,
            target_shape=None, autoregressive=False,
            future_target_occs=None,
            conditional_future_mode='correct'):
        if self.causal_flow_mode == 'learned_total':
            return
        num_frames = self.causal_target_end - self.causal_target_start
        ego_flow = self._build_predicted_ego_flow(
            x, metas, future_pose_increments, num_frames)
        output_dict['predicted_ego_flow'] = ego_flow
        if self.causal_flow_mode == 'predicted_ego':
            output_dict['pred_flow'] = ego_flow
        else:
            if any(value is None for value in (
                    token_logits, source_quantized, shape, target_shape)):
                raise ValueError(
                    f'{self.causal_flow_mode} requires token logits, source '
                    'quantized feature, decoder shape, and target shape')
            if autoregressive:
                selected_logits = token_logits
            else:
                start = self.causal_target_start - self.offset
                end = self.causal_target_end - self.offset
                selected_logits = token_logits[:, start:end]
            if selected_logits.shape[1] != num_frames:
                raise ValueError(
                    f'ego residual selected {selected_logits.shape[1]} '
                    f'frames, expected {num_frames}')
            future_feature = self._decode_dense_feature_from_token_logits(
                selected_logits, shape, target_shape)
            source_feature = self._decode_source_dense_feature(
                source_quantized, shape, target_shape)
            if self.causal_flow_mode == 'ego_residual':
                residual = self.causal_residual_head(
                    future_feature,
                    source_feature,
                    ego_flow[:, :, 0, 0, 0],
                    x.shape[0],
                    num_frames)
                output_dict['pred_residual_flow'] = residual
            elif self.causal_flow_mode == 'object_trajectory':
                missing_components = [
                    index for index, meta in enumerate(metas)
                    if 'dynamic_component_ids' not in meta]
                if missing_components:
                    raise KeyError(
                        'object_trajectory requires occupancy-derived '
                        'dynamic_component_ids for samples '
                        f'{missing_components}')
                component_ids = torch.as_tensor(
                    np.stack([
                        meta['dynamic_component_ids'] for meta in metas
                    ], axis=0),
                    dtype=torch.long,
                    device=x.device)
                future_behavior_descriptor = None
                future_behavior_valid = None
                future_target_semantics = None
                conditional_enabled = getattr(
                    self.object_trajectory_head,
                    'conditional_latent_enabled', False)
                dense_future = getattr(
                    self.object_trajectory_head,
                    'conditional_latent_use_dense_future', False)
                force_posterior = getattr(
                    self.object_trajectory_head,
                    'conditional_latent_force_posterior', False)
                if conditional_enabled and dense_future:
                    if future_target_occs is not None:
                        future_target_semantics = future_target_occs[
                            :, self.causal_target_start:self.causal_target_end]
                        if conditional_future_mode == 'permuted':
                            # Preserve class counts and occupancy density while
                            # destroying temporal/spatial alignment with the
                            # current scene.  This is an explicit oracle
                            # diagnostic, never a deployable inference path.
                            future_target_semantics = torch.flip(
                                future_target_semantics, dims=(1, 2, 3))
                        elif conditional_future_mode != 'correct':
                            raise ValueError(
                                'conditional_future_mode must be "correct" '
                                f'or "permuted", got '
                                f'{conditional_future_mode!r}')
                elif conditional_enabled and (
                        self.training or force_posterior):
                    (future_behavior_descriptor,
                     future_behavior_valid) = (
                        self._future_behavior_descriptor(metas, x.device))
                residual = self.object_trajectory_head(
                    future_feature,
                    source_feature,
                    x[:, self.causal_source_index],
                    component_ids,
                    ego_flow[:, :, 0, 0, 0],
                    x.shape[0],
                    num_frames,
                    future_behavior_descriptor=future_behavior_descriptor,
                    future_behavior_valid=future_behavior_valid,
                    future_target_semantics=future_target_semantics)
                if isinstance(residual, tuple):
                    if len(residual) == 3:
                        residual, behavior_mode_logits, latent_stats = residual
                        output_dict.update(latent_stats)
                    elif len(residual) == 2:
                        residual, behavior_mode_logits = residual
                    else:
                        raise ValueError(
                            'Unexpected object trajectory head output tuple '
                            f'length: {len(residual)}')
                    output_dict['behavior_mode_logits'] = (
                        behavior_mode_logits)
                raw_residual = residual
                residual, residual_gate = (
                    self._gate_object_trajectory_residual(residual))
                output_dict['pred_object_trajectory_residual_raw'] = (
                    raw_residual)
                output_dict['pred_object_trajectory_residual'] = residual
                if residual_gate is not None:
                    output_dict['pred_object_trajectory_residual_gate'] = (
                        residual_gate)
                output_dict['num_object_components'] = torch.as_tensor(
                    [meta.get('num_dynamic_components', 0) for meta in metas],
                    dtype=torch.float32,
                    device=x.device)
            else:
                raise AssertionError(
                    f'Unhandled causal flow mode {self.causal_flow_mode}')
            if residual.ndim == ego_flow.ndim + 1:
                flow_hypotheses = residual + ego_flow[:, None]
                output_dict['pred_object_trajectory_residual_hypotheses'] = (
                    residual)
                output_dict['pred_flow_hypotheses'] = flow_hypotheses
                # Keep the historical single-prediction interface available
                # for strict STP3 and legacy diagnostics.  Mode 0 is a real
                # learned hypothesis, not an oracle choice.
                output_dict['pred_flow'] = flow_hypotheses[:, 0]
            elif residual.ndim == ego_flow.ndim:
                output_dict['pred_flow'] = residual + ego_flow
            else:
                raise ValueError(
                    'Object trajectory residual must be [B,T,X,Y,Z,2] or '
                    '[B,K,T,X,Y,Z,2], got '
                    f'{tuple(residual.shape)}')

    def _future_behavior_descriptor(self, metas, device):
        """Build the training-only future descriptor used by q(z|past,future)."""
        descriptor_dim = int(getattr(
            self.object_trajectory_head,
            'conditional_latent_descriptor_dim', 12))
        descriptor = torch.zeros(
            len(metas), descriptor_dim, dtype=torch.float32, device=device)
        valid = torch.zeros(len(metas), dtype=torch.bool, device=device)
        for index, meta in enumerate(metas):
            value = meta.get('behavior_anchor_descriptor')
            if value is None:
                continue
            value = np.asarray(value, dtype=np.float32).reshape(-1)
            if value.shape != (descriptor_dim,) or not np.isfinite(value).all():
                continue
            descriptor[index] = torch.as_tensor(value, device=device)
            valid[index] = True
        return descriptor, valid

    def _decode_hard_token_logits(self, token_logits, shape, target_shape):
        """Decode argmax tokens exactly as the original inference path."""
        context = (
            torch.no_grad()
            if self.dynamic_refine_detach_base else nullcontext())
        with context:
            hard_tokens = token_logits.flatten(0, 1).argmax(dim=1)
            hard_tokens = self.vae.vqvae.get_codebook_entry(
                hard_tokens, shape=None)
            hard_tokens = rearrange(
                hard_tokens, 'bf h w c -> bf c h w')
            hard_tokens = self.vae.vqvae.post_quant_conv(hard_tokens)
            base_logits, dense_feature = self.vae.forward_decoder(
                hard_tokens,
                deepcopy(shape),
                target_shape,
                return_feature=True)
        return base_logits, dense_feature

    def _add_dynamic_refine_outputs(
            self,
            output_dict,
            token_logits,
            shape,
            target_shape,
            source_semantics=None,
            transport_start_index=0):
        """Add a separate residual to dynamic semantic logits only."""
        if not self.use_dynamic_refine:
            return
        batch_size, num_frames = token_logits.shape[:2]
        base_logits, dense_feature = self._decode_hard_token_logits(
            token_logits, shape, target_shape)
        pred_flow = self.flow_head(
            dense_feature, batch_size, num_frames)
        if self.dynamic_refine_detach_base:
            base_logits = base_logits.detach()
            dense_feature = dense_feature.detach()
            pred_flow = pred_flow.detach()
        transport_condition = None
        if getattr(
                self.dynamic_refine_head,
                'use_transport_condition',
                False):
            if source_semantics is None:
                raise ValueError(
                    'transport-conditioned dynamic refine requires '
                    'source_semantics')
            if 'pred_flow' not in output_dict:
                raise ValueError(
                    'transport-conditioned dynamic refine requires '
                    'output_dict["pred_flow"]')
            from utils.causal_flow_transport import (
                splat_cumulative_dynamic_flow,
            )
            class_mass, support = splat_cumulative_dynamic_flow(
                source_semantics,
                output_dict['pred_flow'].detach(),
                self.dynamic_label_ids)
            transport_condition = base_logits.new_zeros(
                batch_size,
                num_frames,
                target_shape[2],
                target_shape[3],
                target_shape[4],
                len(self.dynamic_label_ids) + 1)
            start = int(transport_start_index)
            if start < 0:
                raise ValueError(
                    f'transport_start_index must be non-negative, got {start}')
            end = min(start + class_mass.shape[1], num_frames)
            if end > start:
                source_end = end - start
                transport_condition[:, start:end] = torch.cat([
                    support[:, :source_end, :, :, :, None],
                    class_mass[:, :source_end],
                ], dim=-1).to(transport_condition)
            output_dict['dynamic_refine_transport_condition'] = (
                transport_condition)
        residual = self.dynamic_refine_head(
            dense_feature,
            pred_flow,
            batch_size,
            num_frames,
            transport_condition=transport_condition)
        dynamic_ids = torch.as_tensor(
            self.dynamic_label_ids,
            dtype=torch.long,
            device=base_logits.device)
        if residual.shape[:-1] != base_logits.shape[:-1]:
            raise ValueError(
                f'dynamic residual shape {residual.shape} does not match '
                f'base logits shape {base_logits.shape}')
        if residual.shape[-1] != len(self.dynamic_label_ids):
            raise ValueError(
                f'dynamic residual has {residual.shape[-1]} channels, '
                f'expected {len(self.dynamic_label_ids)}')
        delta = base_logits.new_zeros(base_logits.shape)
        delta = delta.index_copy(-1, dynamic_ids, residual)
        output_dict['dynamic_base_logits'] = base_logits
        output_dict['dynamic_residual'] = residual
        output_dict['dynamic_refined_logits'] = base_logits + delta
        output_dict['dynamic_refine_pred_flow'] = pred_flow

    def _add_transport_gate_outputs(
            self,
            output_dict,
            token_logits,
            shape,
            target_shape,
            source_semantics,
            autoregressive=False,
            semantic_logits=None):
        """Predict a conservative accept/reject gate for transport candidates."""
        if not self.use_transport_gate:
            return
        if 'pred_flow' not in output_dict:
            raise ValueError(
                'use_transport_gate=True requires output_dict["pred_flow"]')
        if autoregressive:
            selected_logits = token_logits
        else:
            start = self.causal_target_start - self.offset
            end = self.causal_target_end - self.offset
            selected_logits = token_logits[:, start:end]
        expected_frames = self.causal_target_end - self.causal_target_start
        if selected_logits.shape[1] != expected_frames:
            raise ValueError(
                f'transport gate selected {selected_logits.shape[1]} frames, '
                f'expected {expected_frames}')

        if semantic_logits is None:
            dense_feature = self._decode_dense_feature_from_token_logits(
                selected_logits, shape, target_shape)
            semantic_logits = self._semantic_logits_from_dense_feature(
                dense_feature, target_shape)
        else:
            if semantic_logits.shape[:-1] != target_shape:
                raise ValueError(
                    f'semantic_logits {semantic_logits.shape} do not align '
                    f'with target_shape {target_shape}')
        pred_flow = output_dict.get(
            'pred_flow_hypotheses', output_dict['pred_flow'])
        multimodal = pred_flow.ndim == 7
        if multimodal:
            batch_size, num_hypotheses = pred_flow.shape[:2]
            pred_flow = rearrange(
                pred_flow, 'b k t x y z c -> (b k) t x y z c')
            source_semantics = rearrange(
                source_semantics[:, None].expand(
                    -1, num_hypotheses, -1, -1, -1),
                'b k x y z -> (b k) x y z')
            semantic_logits = rearrange(
                semantic_logits[:, None].expand(
                    -1, num_hypotheses, -1, -1, -1, -1, -1),
                'b k t x y z c -> (b k) t x y z c')
        elif pred_flow.ndim != 6:
            raise ValueError(
                'Transport flow must be [B,T,X,Y,Z,2] or '
                f'[B,K,T,X,Y,Z,2], got {tuple(pred_flow.shape)}')
        if self.transport_gate_detach_base:
            semantic_logits = semantic_logits.detach()
            pred_flow = pred_flow.detach()

        from utils.causal_flow_transport import (
            splat_cumulative_dynamic_flow,
            static_confidence_from_logits,
            transport_candidate_mask,
        )
        alignment_base_flow = pred_flow
        if self.use_transport_alignment:
            alignment_residual = self.transport_alignment_head(
                source_semantics, alignment_base_flow)
            pred_flow = alignment_base_flow + alignment_residual
        else:
            alignment_residual = None
        class_mass, support = splat_cumulative_dynamic_flow(
            source_semantics,
            pred_flow,
            self.dynamic_label_ids)
        baseline_prediction = semantic_logits.argmax(dim=-1)
        candidate_mask = transport_candidate_mask(
            baseline_prediction,
            support,
            baseline_logits=semantic_logits,
            static_label_ids=self.causal_static_label_ids,
            support_threshold=self.causal_support_threshold,
            static_confidence_threshold=(
                self.causal_static_confidence_threshold),
            merge_mode=self.causal_merge_mode)

        static_confidence = static_confidence_from_logits(
            semantic_logits, self.causal_static_label_ids)
        dynamic_ids = torch.as_tensor(
            self.dynamic_label_ids,
            device=semantic_logits.device,
            dtype=torch.long)
        dynamic_logsumexp = torch.logsumexp(
            semantic_logits.index_select(-1, dynamic_ids), dim=-1)
        all_logsumexp = torch.logsumexp(semantic_logits, dim=-1)
        dynamic_confidence = torch.exp(dynamic_logsumexp - all_logsumexp)
        free_label = min(17, semantic_logits.shape[-1] - 1)
        free_confidence = torch.softmax(
            semantic_logits, dim=-1)[..., free_label]
        gate_features = torch.cat([
            support[..., None],
            class_mass,
            static_confidence[..., None],
            dynamic_confidence[..., None],
            free_confidence[..., None],
        ], dim=-1)
        gate_logits = self.transport_gate_head(gate_features)
        class_ids = torch.as_tensor(
            self.dynamic_label_ids,
            device=gate_logits.device,
            dtype=torch.long)
        transported_labels = class_ids[class_mass.argmax(dim=-1)]

        if multimodal:
            gate_logits = rearrange(
                gate_logits,
                '(b k) t x y z -> b k t x y z',
                b=batch_size,
                k=num_hypotheses)
            candidate_mask = rearrange(
                candidate_mask,
                '(b k) t x y z -> b k t x y z',
                b=batch_size,
                k=num_hypotheses)
            transported_labels = rearrange(
                transported_labels,
                '(b k) t x y z -> b k t x y z',
                b=batch_size,
                k=num_hypotheses)
            support = rearrange(
                support,
                '(b k) t x y z -> b k t x y z',
                b=batch_size,
                k=num_hypotheses)
            class_mass = rearrange(
                class_mass,
                '(b k) t x y z c -> b k t x y z c',
                b=batch_size,
                k=num_hypotheses)
            if alignment_residual is not None:
                alignment_base_flow = rearrange(
                    alignment_base_flow,
                    '(b k) t x y z c -> b k t x y z c',
                    b=batch_size,
                    k=num_hypotheses)
                alignment_residual = rearrange(
                    alignment_residual,
                    '(b k) t x y z c -> b k t x y z c',
                    b=batch_size,
                    k=num_hypotheses)
                pred_flow = rearrange(
                    pred_flow,
                    '(b k) t x y z c -> b k t x y z c',
                    b=batch_size,
                    k=num_hypotheses)

        output_dict['transport_gate_logits'] = gate_logits
        output_dict['transport_gate_prob'] = torch.sigmoid(gate_logits)
        output_dict['transport_gate_candidate_mask'] = candidate_mask
        output_dict['transport_gate_labels'] = transported_labels
        output_dict['transport_gate_support'] = support
        output_dict['transport_gate_class_mass'] = class_mass
        # Expose the already-computed frozen parent probability for the
        # Phase-2S1 deterministic planning loss.  This is not a new model
        # branch and remains detached under the registered gate protocol.
        output_dict['transport_gate_base_dynamic_prob'] = dynamic_confidence
        if alignment_residual is not None:
            output_dict['transport_alignment_base_flow'] = alignment_base_flow
            output_dict['transport_alignment_residual'] = alignment_residual
            output_dict['transport_alignment_corrected_flow'] = pred_flow

    def _transport_mass_for_semantic_merge(
            self, output_dict, source_semantics):
        """Return the same transport mass consumed by the alignment gate."""
        from utils.causal_flow_transport import (
            splat_cumulative_dynamic_flow,
        )

        flow_hypotheses = output_dict.get('pred_flow_hypotheses')
        if flow_hypotheses is not None:
            batch_size, num_hypotheses = flow_hypotheses.shape[:2]
            if self.use_transport_alignment:
                class_mass = rearrange(
                    output_dict['transport_gate_class_mass'],
                    'b k t x y z c -> (b k) t x y z c')
                support = rearrange(
                    output_dict['transport_gate_support'],
                    'b k t x y z -> (b k) t x y z')
            else:
                flat_flow = rearrange(
                    flow_hypotheses,
                    'b k t x y z c -> (b k) t x y z c')
                flat_source = rearrange(
                    source_semantics[:, None].expand(
                        -1, num_hypotheses, -1, -1, -1),
                    'b k x y z -> (b k) x y z')
                class_mass, support = splat_cumulative_dynamic_flow(
                    flat_source, flat_flow, self.dynamic_label_ids)
            return class_mass, support, (batch_size, num_hypotheses)

        if self.use_transport_alignment:
            return (
                output_dict['transport_gate_class_mass'],
                output_dict['transport_gate_support'],
                None,
            )
        class_mass, support = splat_cumulative_dynamic_flow(
            source_semantics, output_dict['pred_flow'],
            self.dynamic_label_ids)
        return class_mass, support, None

    def forward_train(self, x):
        # given x: bs, f, h, w, d where f == num_frames + offset
        # output : ce_inputs: logits for the codebook 
        # output : ce_labels: labels for the ce_inputs
        assert hasattr(self.vae, 'vqvae')
        bs, F, H, W, D = x.shape
        assert F == self.num_frames + self.offset
        output_dict = {}
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]
        z_q_predict = self.transformer(z_q[:, :self.num_frames], hidden=hidden)
        if self.use_causal_transport:
            self._add_causal_flow_outputs(
                output_dict,
                z_q_predict,
                shape,
                x[:, self.causal_target_start:self.causal_target_end].shape)
            if self.use_transport_gate:
                self._add_transport_gate_outputs(
                    output_dict,
                    z_q_predict,
                    shape,
                    x[:, self.causal_target_start:self.causal_target_end].shape,
                    x[:, self.causal_source_index])
        elif self.use_flow and self.return_pred_flow:
            self._add_flow_outputs(
                output_dict, z_q_predict, shape, x[:, self.offset:].shape)
        if self.use_dynamic_refine:
            self._add_dynamic_refine_outputs(
                output_dict, z_q_predict, shape, x[:, self.offset:].shape)
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        # z: bs*f, c, h, w 

        # z: bs*f, h, w
        return output_dict
        
    
    def forward_inference(self, x):
        bs, F, H, W, D = x.shape
        output_dict = {}
        output_dict['target_occs'] = x[:, self.offset:]
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]
        z_q_predict = self.transformer(z_q[:, :self.num_frames], hidden=hidden)
        if self.use_causal_transport:
            self._add_causal_flow_outputs(
                output_dict,
                z_q_predict,
                shape,
                output_dict['target_occs'][
                    :, self.causal_target_start:self.causal_target_end].shape)
            if self.use_transport_gate:
                self._add_transport_gate_outputs(
                    output_dict,
                    z_q_predict,
                    shape,
                    output_dict['target_occs'][
                        :, self.causal_target_start:self.causal_target_end].shape,
                    x[:, self.causal_source_index])
        elif self.use_flow and self.return_pred_flow:
            self._add_flow_outputs(
                output_dict, z_q_predict, shape,
                output_dict['target_occs'].shape)
        if self.use_dynamic_refine:
            self._add_dynamic_refine_outputs(
                output_dict, z_q_predict, shape,
                output_dict['target_occs'].shape)
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        if self.use_dynamic_refine:
            semantic_logits = output_dict['dynamic_refined_logits']
        else:
            z_q_predict = z_q_predict.argmax(dim=1)
            z_q_predict = self.vae.vqvae.get_codebook_entry(
                z_q_predict, shape=None)
            z_q_predict = rearrange(
                z_q_predict, 'bf h w c-> bf c h w')
            z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)
            semantic_logits = self.vae.forward_decoder(
                z_q_predict, shape, output_dict['target_occs'].shape)
        output_dict['logits'] = semantic_logits
        pred = semantic_logits.argmax(dim=-1).detach().cuda()
        output_dict['sem_pred'] = pred
        pred_iou = deepcopy(pred)
        
        pred_iou[pred_iou!=17] = 1
        pred_iou[pred_iou==17] = 0
        output_dict['iou_pred'] = pred_iou
    
        return output_dict
    
    
    def forward_train_with_plan(self, x, metas, target_occs=None):
        assert hasattr(self.vae, 'vqvae')
        assert hasattr(self, 'pose_encoder')
        bs, F, H, W, D = x.shape
        assert F == self.num_frames + self.offset
        output_dict = {}
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]
            
            
        rel_poses, output_metas = self._get_pose_feature(metas, F-self.offset)
        
        z_q_predict, rel_poses = self.transformer(z_q[:, :self.num_frames], pose_tokens=rel_poses)
        if self.use_ego_se2 and not self.ego_se2_residual:
            self._add_ego_se2_output(output_dict, rel_poses)
        
        pose_decoded = self.pose_decoder(rel_poses)
        output_dict['pose_decoded'] = pose_decoded
        output_dict['output_metas'] = output_metas
        if self.use_ego_se2 and self.ego_se2_residual:
            num_ego_frames = (
                self.ego_se2_target_end - self.ego_se2_target_start)
            future_pose_increments = self._select_pose_increments(
                pose_decoded,
                output_metas,
                self.ego_se2_target_start - 1,
                num_ego_frames - 1)
            base_ego_se2 = self._build_occworld_ego_se2(
                x, metas, future_pose_increments, num_ego_frames)
            self._add_ego_se2_output(
                output_dict, rel_poses, base_ego_se2=base_ego_se2)
        
        if self.use_causal_transport:
            causal_target_shape = x[
                :, self.causal_target_start:self.causal_target_end].shape
            if self.causal_flow_mode == 'learned_total':
                self._add_causal_flow_outputs(
                    output_dict, z_q_predict, shape, causal_target_shape)
            else:
                num_causal_frames = (
                    self.causal_target_end - self.causal_target_start)
                future_pose_increments = self._select_pose_increments(
                    pose_decoded,
                    output_metas,
                    self.causal_source_index,
                    num_causal_frames - 1)
                self._apply_causal_ego_flow(
                    output_dict,
                    x,
                    metas,
                    future_pose_increments,
                    token_logits=z_q_predict,
                    source_quantized=z_q[:, self.causal_source_index],
                    shape=shape,
                    target_shape=causal_target_shape,
                    future_target_occs=target_occs)
            if self.use_transport_gate:
                self._add_transport_gate_outputs(
                    output_dict,
                    z_q_predict,
                    shape,
                    causal_target_shape,
                    x[:, self.causal_source_index])
        elif self.use_flow and self.return_pred_flow:
            self._add_flow_outputs(
                output_dict, z_q_predict, shape, x[:, self.offset:].shape)
        if self.use_dynamic_refine:
            transport_start = max(self.causal_target_start - self.offset, 0)
            self._add_dynamic_refine_outputs(
                output_dict,
                z_q_predict,
                shape,
                x[:, self.offset:].shape,
                source_semantics=(
                    x[:, self.causal_source_index]
                    if self.use_causal_transport else None),
                transport_start_index=transport_start)
        
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        # z: bs*f, c, h, w 
        
        # z: bs*f, h, w
        return output_dict
    def forward_inference_with_plan(self, x, metas, target_occs=None):
        bs, F, H, W, D = x.shape
        output_dict = {}
        output_dict['target_occs'] = x[:, self.offset:]
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]
            
        
        rel_poses, output_metas = self._get_pose_feature(metas, F-self.offset)
        
        z_q_predict, rel_poses = self.transformer(z_q[:, :self.num_frames], pose_tokens=rel_poses)
        if self.use_ego_se2 and not self.ego_se2_residual:
            self._add_ego_se2_output(output_dict, rel_poses)
        
        pose_decoded = self.pose_decoder(rel_poses)
        output_dict['pose_decoded'] = pose_decoded
        output_dict['output_metas'] = output_metas
        if self.use_ego_se2 and self.ego_se2_residual:
            num_ego_frames = (
                self.ego_se2_target_end - self.ego_se2_target_start)
            future_pose_increments = self._select_pose_increments(
                pose_decoded,
                output_metas,
                self.ego_se2_target_start - 1,
                num_ego_frames - 1)
            base_ego_se2 = self._build_occworld_ego_se2(
                x, metas, future_pose_increments, num_ego_frames)
            self._add_ego_se2_output(
                output_dict, rel_poses, base_ego_se2=base_ego_se2)
        
        
        
        if self.use_causal_transport:
            causal_target_shape = output_dict['target_occs'][
                :, self.causal_target_start:self.causal_target_end].shape
            if self.causal_flow_mode == 'learned_total':
                self._add_causal_flow_outputs(
                    output_dict, z_q_predict, shape, causal_target_shape)
            else:
                num_causal_frames = (
                    self.causal_target_end - self.causal_target_start)
                future_pose_increments = self._select_pose_increments(
                    pose_decoded,
                    output_metas,
                    self.causal_source_index,
                    num_causal_frames - 1)
                self._apply_causal_ego_flow(
                    output_dict,
                    x,
                    metas,
                    future_pose_increments,
                    token_logits=z_q_predict,
                    source_quantized=z_q[:, self.causal_source_index],
                    shape=shape,
                    target_shape=causal_target_shape,
                    future_target_occs=target_occs)
            if self.use_transport_gate:
                self._add_transport_gate_outputs(
                    output_dict,
                    z_q_predict,
                    shape,
                    causal_target_shape,
                    x[:, self.causal_source_index])
        elif self.use_flow and self.return_pred_flow:
            self._add_flow_outputs(
                output_dict, z_q_predict, shape,
                output_dict['target_occs'].shape)
        if self.use_dynamic_refine:
            transport_start = max(self.causal_target_start - self.offset, 0)
            self._add_dynamic_refine_outputs(
                output_dict,
                z_q_predict,
                shape,
                output_dict['target_occs'].shape,
                source_semantics=(
                    x[:, self.causal_source_index]
                    if self.use_causal_transport else None),
                transport_start_index=transport_start)

        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        if self.use_dynamic_refine:
            semantic_logits = output_dict['dynamic_refined_logits']
        else:
            z_q_predict = z_q_predict.argmax(dim=1)
            z_q_predict = self.vae.vqvae.get_codebook_entry(
                z_q_predict, shape=None)
            z_q_predict = rearrange(
                z_q_predict, 'bf h w c-> bf c h w')
            z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)
            semantic_logits = self.vae.forward_decoder(
                z_q_predict, shape, output_dict['target_occs'].shape)
        output_dict['logits'] = semantic_logits
        pred = semantic_logits.argmax(dim=-1).detach().cuda()
        output_dict['sem_pred'] = pred
        pred_iou = deepcopy(pred)
        
        pred_iou[pred_iou!=17] = 1
        pred_iou[pred_iou==17] = 0
        output_dict['iou_pred'] = pred_iou
    
        return output_dict

    def _get_pose_feature(self, metas=None, F=None):
        rel_poses, output_metas = None, None
        if hasattr(self, 'pose_encoder'):
            assert hasattr(self, 'pose_decoder')
            assert metas is not None
            output_metas = []
            for meta in metas:
                output_meta = dict()
                output_meta['rel_poses'] = meta['rel_poses'][self.offset:]
                output_meta['gt_mode'] = meta['gt_mode'][self.offset:]
                output_metas.append(output_meta)
                
                
            rel_poses = np.array([meta['rel_poses'] for meta in metas])
            gt_mode = np.array([meta['gt_mode'] for meta in metas])
            
            
            
            
            gt_mode = torch.tensor(gt_mode).cuda()
            rel_poses = torch.tensor(rel_poses).cuda()# list of (num_frames+offsets, 2)
            if self.delta_input:
                rel_poses_pre = torch.cat([torch.zeros_like(rel_poses[:, :1]), rel_poses[:, :-1]], dim=1)
                rel_poses = rel_poses - rel_poses_pre
            if F>self.num_frames:
                assert F == self.num_frames + self.offset
            else:
                assert F == self.num_frames
                gt_mode = gt_mode[:, :-self.offset, :] 
                rel_poses = rel_poses[:, :-self.offset, :]
                
            rel_poses = torch.cat([rel_poses, gt_mode], dim=-1)
            #rel_poses = rearrange(rel_poses, 'b f d -> b f 1 d')
            rel_poses = self.pose_encoder(rel_poses.float())
        return rel_poses, output_metas
    
    def forward_autoreg_with_pose(
            self, x, metas, start_frame=0, mid_frame=6, end_frame=12,
            conditional_future_mode='off'):
        t0 = time.time()
        bs, F, H, W, D = x.shape
        output_dict = {}
        output_dict['input_occs'] = x[:, mid_frame-1:end_frame]
        output_dict['target_occs'] = x[:, mid_frame:end_frame]
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, mid_frame:end_frame].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        source_quantized = z_q[:, mid_frame - 1]
        z_q_predict = z_q[:, start_frame:mid_frame]
        t1 = time.time()
        output_metas = []
        input_metas = []
        for meta in metas:
            input_meta = dict()
            input_meta['rel_poses'] = meta['rel_poses'][start_frame:mid_frame]
            input_meta['gt_mode'] = meta['gt_mode'][start_frame:mid_frame]
            input_metas.append(input_meta)
        output_dict['input_metas'] = input_metas
        for meta in metas:
            output_meta = dict()
            output_meta['rel_poses'] = meta['rel_poses'][mid_frame:end_frame]#-meta['rel_poses'][mid_frame-1]
            output_meta['gt_mode'] = meta['gt_mode'][mid_frame:end_frame]
            output_metas.append(output_meta)
        output_dict['gt_poses_'] = np.array([meta['rel_poses'] for meta in output_metas])
        rel_poses = np.array([meta['rel_poses'] for meta in metas])
        gt_mode = np.array([meta['gt_mode'] for meta in metas])
        gt_mode = torch.tensor(gt_mode).cuda()
        
        rel_poses = torch.tensor(rel_poses).cuda()
        if self.delta_input:
            rel_poses_pre = torch.cat(torch.zeros_like(rel_poses[:, :1]), rel_poses[:, :-1], dim=1)
            rel_poses = rel_poses - rel_poses_pre
        rel_poses_sumed = rel_poses[:, start_frame:mid_frame]
        rel_poses = torch.cat([rel_poses, gt_mode], dim=-1)
        rel_poses = rel_poses[:, start_frame:mid_frame]
        
        rel_poses = self.pose_encoder(rel_poses.float())
        rel_poses_state = rel_poses
        z_q_list = []
        t2 = time.time()
        poses_ = []
        for i in range(mid_frame, end_frame):
            z_q_, rel_poses_ = self.transformer.forward_autoreg_step(
                z_q_predict, pose_tokens=rel_poses_state,
                start_frame=start_frame, mid_frame=i)
            z_q_list.append(z_q_[:, -1:])
            #print(z_q_.shape)
            z_q_ = z_q_[:, -1:].clone().detach().argmax(dim=2)
            #print(z_q_.shape)
            z_q_ = self.vae.vqvae.get_codebook_entry(z_q_, shape=None)
            z_q_ = rearrange(z_q_, 'b f h w c-> b f c h w')
            z_q_predict = torch.cat([z_q_predict, z_q_], dim=1)
            rel_poses = torch.cat([rel_poses, rel_poses_[:, -1:]], dim=1)
            rel_poses_state_, rel_poses_sumed, pose_ = self.decode_pose(rel_poses_[:, -1:], gt_mode[:,i:i+1], rel_poses_sumed)
            poses_.append(pose_)
            rel_poses_state = torch.cat([rel_poses_state, rel_poses_state_], dim=1)
        poses_ = torch.cat(poses_, dim=1)
        output_dict['poses_'] = poses_
        t3 = time.time()
        z_q_predict = z_q_predict[:, mid_frame:end_frame]
        rel_poses = rel_poses[:, mid_frame:end_frame]
        if self.use_ego_se2:
            if self.ego_se2_residual:
                num_ego_frames = (
                    self.ego_se2_target_end - self.ego_se2_target_start)
                base_ego_se2 = self._build_occworld_ego_se2(
                    x,
                    metas,
                    poses_[:, :num_ego_frames - 1],
                    num_ego_frames)
                self._add_ego_se2_output(
                    output_dict,
                    rel_poses,
                    autoregressive=True,
                    base_ego_se2=base_ego_se2)
            else:
                self._add_ego_se2_output(
                    output_dict, rel_poses, autoregressive=True)
        #assert False, f'z_q_predict.shape: {z_q_predict.shape}, rel_poses.shape: {rel_poses.shape}, {output_dict["target_occs"].shape}'
        # print(z_q_predict.shape, rel_poses.shape)
        pose_decoded = self.pose_decoder(rel_poses)
        output_dict['pose_decoded'] = pose_decoded
        output_dict['output_metas'] = output_metas
        
        z_q = torch.cat(z_q_list, dim=1)
        #print(z_q.shape)
        output_dict['ce_inputs'] = z_q.flatten(0, 1)
        if self.use_causal_transport:
            if self.causal_flow_mode == 'learned_total':
                self._add_causal_flow_outputs(
                    output_dict,
                    z_q,
                    shape,
                    output_dict['target_occs'].shape,
                    autoregressive=True)
            else:
                num_causal_frames = (
                    self.causal_target_end - self.causal_target_start)
                self._apply_causal_ego_flow(
                    output_dict,
                    x,
                    metas,
                    poses_[:, :num_causal_frames - 1],
                    token_logits=z_q,
                    source_quantized=source_quantized,
                    shape=shape,
                    target_shape=output_dict['target_occs'].shape,
                    autoregressive=True,
                    future_target_occs=(
                        x if conditional_future_mode != 'off' else None),
                    conditional_future_mode=conditional_future_mode)
        if self.use_dynamic_refine:
            self._add_dynamic_refine_outputs(
                output_dict,
                z_q,
                shape,
                output_dict['target_occs'].shape,
                source_semantics=(
                    x[:, mid_frame - 1]
                    if self.use_causal_transport else None),
                transport_start_index=0)
            semantic_logits = output_dict['dynamic_refined_logits']
        else:
            z_q_predict = z_q_predict.flatten(0, 1)
            z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)
            semantic_logits = self.vae.forward_decoder(
                z_q_predict, shape, output_dict['target_occs'].shape)
        output_dict['logits'] = semantic_logits
        if self.use_transport_gate:
            self._add_transport_gate_outputs(
                output_dict,
                z_q,
                shape,
                output_dict['target_occs'].shape,
                x[:, mid_frame - 1],
                autoregressive=True,
                semantic_logits=semantic_logits)
        pred = semantic_logits.argmax(dim=-1).detach().cuda()
        if self.use_causal_transport and self.causal_apply_transport:
            from utils.causal_flow_transport import (
                merge_transport_with_baseline,
            )
            class_mass, support, hypothesis_shape = (
                self._transport_mass_for_semantic_merge(
                    output_dict, x[:, mid_frame - 1]))
            if hypothesis_shape is not None:
                batch_size, num_hypotheses = hypothesis_shape
                flat_pred = rearrange(
                    pred[:, None].expand(
                        -1, num_hypotheses, -1, -1, -1, -1),
                    'b k t x y z -> (b k) t x y z')
                flat_logits = rearrange(
                    semantic_logits[:, None].expand(
                        -1, num_hypotheses, -1, -1, -1, -1, -1),
                    'b k t x y z c -> (b k) t x y z c')
                gate_prob = (
                    rearrange(
                        output_dict['transport_gate_prob'],
                        'b k t x y z -> (b k) t x y z')
                    if self.use_transport_gate else None)
            else:
                flat_pred = pred
                flat_logits = semantic_logits
                gate_prob = (
                    output_dict['transport_gate_prob']
                    if self.use_transport_gate else None)
            start = self.causal_transport_start_index
            if start:
                flat_pred[:, start:] = merge_transport_with_baseline(
                    flat_pred[:, start:],
                    class_mass[:, start:],
                    support[:, start:],
                    self.dynamic_label_ids,
                    baseline_logits=flat_logits[:, start:],
                    static_label_ids=self.causal_static_label_ids,
                    support_threshold=self.causal_support_threshold,
                    static_confidence_threshold=(
                        self.causal_static_confidence_threshold),
                    merge_mode=self.causal_merge_mode,
                    gate_prob=(
                        gate_prob[:, start:]
                        if gate_prob is not None else None),
                    gate_threshold=self.transport_gate_threshold)
            else:
                flat_pred = merge_transport_with_baseline(
                    flat_pred,
                    class_mass,
                    support,
                    self.dynamic_label_ids,
                    baseline_logits=flat_logits,
                    static_label_ids=self.causal_static_label_ids,
                    support_threshold=self.causal_support_threshold,
                    static_confidence_threshold=(
                        self.causal_static_confidence_threshold),
                    merge_mode=self.causal_merge_mode,
                    gate_prob=gate_prob,
                    gate_threshold=self.transport_gate_threshold)
            if hypothesis_shape is not None:
                hypotheses = rearrange(
                    flat_pred,
                    '(b k) t x y z -> b k t x y z',
                    b=batch_size,
                    k=num_hypotheses)
                support = rearrange(
                    support,
                    '(b k) t x y z -> b k t x y z',
                    b=batch_size,
                    k=num_hypotheses)
                output_dict['sem_pred_hypotheses'] = hypotheses
                output_dict['causal_transport_support_hypotheses'] = support
                output_dict['causal_transport_support'] = support[:, 0]
                pred = hypotheses[:, 0]
            else:
                pred = flat_pred
                output_dict['causal_transport_support'] = support
        output_dict['sem_pred'] = pred
        pred_iou = deepcopy(pred)
        
        pred_iou[pred_iou!=17] = 1
        pred_iou[pred_iou==17] = 0
        output_dict['iou_pred'] = pred_iou

        
        if self.without_all:
            #output_dict['pose_decoded'] = 
            output_dict['sem_pred'] = output_dict['input_occs'][:, 0:1].repeat(1, end_frame-mid_frame, 1, 1, 1)
            pred_iou = deepcopy(output_dict['sem_pred'])
            pred_iou[pred_iou!=17] = 1
            pred_iou[pred_iou==17] = 0
            output_dict['iou_pred'] = pred_iou
            output_dict['pose_decoded'] = torch.tensor([meta['rel_poses'] for meta in input_metas])[:,-1:].unsqueeze(2).repeat(1, end_frame-mid_frame, 3, 1)
        output_dict['time'] = {'encode':t1-t0, 'mid':t2-t1, 'autoreg':t3-t2, 'total':t3-t0, 'per_frame':t1-t0+(t3-t2)/(end_frame-mid_frame)}
        return output_dict
        
        
        
    def decode_pose(self, pose, gt_mode, rel_poses_sumed):
        pose = self.pose_decoder(pose)
        # pose:b, f, 3, 2
        # mode:b, f, 3
        # b, f, 2
        bs, num_frames, num_modes, _ = pose.shape
        #gt_mode_ = gt_mode.unsqueeze(-1).repeat(1, 1, 1, 2)
        pose = pose[gt_mode.bool()].reshape(bs, num_frames, 2)
        pose_decoded = pose.clone().detach()
        '''if not self.delta_input:
            pose = pose+rel_poses_sumed[:, -1:]
            rel_poses_sumed = torch.cat([rel_poses_sumed, pose], dim=1)'''
        pose = torch.cat([pose, gt_mode], dim=-1)
        pose = self.pose_encoder(pose.float())
        return pose, rel_poses_sumed, pose_decoded
    def forward_autoreg(self, x, metas=None, start_frame=0, mid_frame=6,end_frame=12):
        
        pass
    def generate_inference(self, x):
        #import pdb; pdb.set_trace()
        bs, F, H, W, D = x.shape
        output_dict = {}
        output_dict['target_occs'] = x[:, self.offset:]
        z, shape = self.vae.forward_encoder(x)
        z = self.vae.vqvae.quant_conv(z)
        z_q, loss, (perplexity, min_encodings, min_encoding_indices) = self.vae.vqvae.forward_quantizer(z, is_voxel=False)
        min_encoding_indices = rearrange(min_encoding_indices, '(b f) h w -> b f h w', b=bs)
        output_dict['ce_labels'] = min_encoding_indices[:, self.offset:].detach().flatten(0,1)
        z_q = rearrange(z_q, '(b f) c h w -> b f c h w', b=bs)
        hidden = None
        if self.give_hiddens:
            hidden = z_q[:, :self.offset]
        z_q_predict = self.transformer(z_q[:, :self.num_frames], hidden=hidden)
        z_q_predict = z_q_predict.flatten(0, 1)
        output_dict['ce_inputs'] = z_q_predict
        z_q_predict = z_q_predict.permute(0, 2, 3, 1)
        cata_distribution = dist.Categorical(logits=(z_q_predict-z_q_predict.min())/(z_q_predict.max()-z_q_predict.min()))
        import pdb;pdb.set_trace()
        z_q_predict = cata_distribution.sample()
        z_q_predict = self.vae.vqvae.get_codebook_entry(z_q_predict, shape=None)
        z_q_predict = rearrange(z_q_predict, 'bf h w c-> bf c h w')
        z_q_predict = self.vae.vqvae.post_quant_conv(z_q_predict)
        
        z_q_predict = self.vae.forward_decoder(z_q_predict, shape, output_dict['target_occs'].shape)
        output_dict['logits'] = z_q_predict
        pred = z_q_predict.argmax(dim=-1).detach().cuda()
        output_dict['sem_pred'] = pred
        pred_iou = deepcopy(pred)
        
        pred_iou[pred_iou!=17] = 1
        pred_iou[pred_iou==17] = 0
        output_dict['iou_pred'] = pred_iou
    
        return output_dict
    
    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3"""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
            'plan_L2_1s_single':0,
            'plan_L2_2s_single':0,
            'plan_L2_3s_single':0,
            'plan_obj_col_1s_single':0,
            'plan_obj_col_2s_single':0,
            'plan_obj_col_3s_single':0,
            'plan_obj_box_col_1s_single':0,
            'plan_obj_box_col_2s_single':0,
            'plan_obj_box_col_3s_single':0,
            
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)
        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                traj_L2_single = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, cur_time-1:cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, cur_time-1:cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                obj_coll_single, obj_box_coll_single = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, cur_time-1:cur_time].detach(),
                    gt_ego_fut_trajs[:, cur_time-1:cur_time],
                    occupancy[:, cur_time-1:cur_time])
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_L2_{}s_single'.format(i+1)] = traj_L2_single
                metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.mean().item()
                metric_dict['plan_obj_col_{}s_single'.format(i+1)] = obj_coll_single.item()
                metric_dict['plan_obj_box_col_{}s_single'.format(i+1)] = obj_box_coll_single.item()
                
                
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_L2_{}s_single'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
            
        return metric_dict
    
    def autoreg_for_stp3_metric(
            self, x, metas, start_frame=0, mid_frame=6, end_frame=12,
            conditional_future_mode='off'):
        output_dict = self.forward_autoreg_with_pose(
            x, metas, start_frame, mid_frame, end_frame,
            conditional_future_mode=conditional_future_mode)
        pred_ego_fut_trajs = output_dict['pose_decoded']
        gt_mode = torch.tensor([meta['gt_mode'] for meta in output_dict['output_metas']])
        bs, num_frames, num_modes, _ = pred_ego_fut_trajs.shape
        pred_ego_fut_trajs = pred_ego_fut_trajs[gt_mode.bool()].reshape(bs, num_frames, 2)
        pred_ego_fut_trajs = torch.cumsum(pred_ego_fut_trajs, dim=1).cpu()
        gt_ego_fut_trajs = torch.tensor([meta['rel_poses'] for meta in output_dict['output_metas']])
        gt_ego_fut_trajs = torch.cumsum(gt_ego_fut_trajs, dim=1).cpu()
        assert len(metas) == 1, f'len(metas): {len(metas)}'
        gt_bbox = metas[0]['gt_bboxes_3d']
        gt_attr_labels = torch.tensor(metas[0]['attr_labels'])
        fut_valid_flag = torch.tensor(metas[0]['fut_valid_flag'])
        # import pdb;pdb.set_trace()
        metric_stp3 = self.compute_planner_metric_stp3(
            pred_ego_fut_trajs, gt_ego_fut_trajs, 
            gt_bbox, gt_attr_labels[None], True)
        
        output_dict['metric_stp3'] = metric_stp3
        
        return output_dict
