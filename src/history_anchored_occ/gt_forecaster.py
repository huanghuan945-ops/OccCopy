"""OccWorld rollout from observed occupancy and ego motion."""
from dataclasses import dataclass
from pathlib import Path
import torch
from .c0_model import (CausalC0Predictor, observed_motion, validate_classes,
                               validate_rigid, GRID_SHAPE, FUTURE_OFFSETS_S)


@dataclass(frozen=True)
class GTHistory:
    require_nominal_history = True
    occupancy: torch.Tensor                 # int64 B,5,X,Y,Z, legal historical GT
    timestamps_us: torch.Tensor             # actual GT input-frame times, never camera observations
    world_from_lidar: torch.Tensor          # past only; shared causal ego contract
    world_from_grid_ego: torch.Tensor       # past only; native Occ3D grid reference
    sample_tokens: tuple[tuple[str, ...], ...]

    def validate(self):
        validate_classes(self.occupancy, 5)
        b = len(self.occupancy)
        t = self.timestamps_us
        if t.shape != (b, 5) or t.dtype != torch.int64:
            raise ValueError('Five actual GT frame timestamps required')
        delta = t[:, 1:] - t[:, :-1]
        if torch.any(delta <= 0):
            raise ValueError("Observed timestamps must increase")
        if self.require_nominal_history:
            if torch.any((delta < 400000) | (delta > 600000)):
                raise ValueError("Training history intervals must be between 0.4 and 0.6 seconds")
            offsets = t.new_tensor([-2000000, -1500000, -1000000, -500000, 0])
            if torch.any(torch.abs(t - t[:, -1:] - offsets) > 100000):
                raise ValueError("Training history exceeds the 100 ms offset tolerance")
        if any(x.shape != (b, 5, 4, 4) for x in (self.world_from_lidar, self.world_from_grid_ego)):
            raise ValueError('Past ego and LiDAR transforms required')
        validate_rigid(self.world_from_lidar)
        validate_rigid(self.world_from_grid_ego)
        if len(self.sample_tokens) != b or any(len(x) != 5 or len(set(x)) != 5 for x in self.sample_tokens):
            raise ValueError('Exactly five distinct source frame IDs required')


@dataclass(frozen=True)
class RawGTAnchorBank:
    """Single-frame features with token-level GT qualification, not camera support."""
    features: torch.Tensor                  # frozen pre-quantizer B,5,128,50,50
    input_frame_timestamps_us: torch.Tensor
    world_from_source_grid_ego: torch.Tensor
    sample_tokens: tuple[tuple[str, ...], ...]
    reading_qualification: torch.Tensor     # bool B,5,50,50; GT token eligibility
    coordinate_reference: str = 'each source ego at its LIDAR_TOP keyframe time'
    qualification_status: str = 'confirmed_token_level_GT_occupied_input_not_camera_support'

    def ages_s(self, nominal_prediction_offset_s):
        if nominal_prediction_offset_s not in FUTURE_OFFSETS_S:
            raise ValueError('Only the six allowed prediction horizons')
        t = self.input_frame_timestamps_us
        return (t[:, -1:] + round(nominal_prediction_offset_s * 1e6) - t).double() * 1e-6


def encode_gt_history(vae, history):
    """Expose the reviewed quant_conv output immediately before quantization."""
    history.validate()
    vae.requires_grad_(False).eval()
    device = next(vae.parameters()).device
    x = history.occupancy.to(device)
    with torch.no_grad():
        latent, shapes = vae.forward_encoder(x)
        continuous = vae.vqvae.quant_conv(latent)
        quantized, _, info = vae.vqvae.forward_quantizer(continuous, is_voxel=False)
    expected = (len(x) * 5, 128, 50, 50)
    if continuous.shape != expected or quantized.shape != expected:
        raise ValueError('Reviewed VQ feature resolution differs')
    if not torch.isfinite(continuous).all() or not torch.isfinite(quantized).all():
        raise FloatingPointError('Nonfinite frozen GT encoding')
    occupied=(x!=17).any(-1).float().reshape(len(x)*5,1,200,200)
    qualification=torch.nn.functional.max_pool2d(occupied,4,4).bool().reshape(len(x),5,50,50)
    bank = RawGTAnchorBank(continuous.reshape(len(x), 5, 128, 50, 50).detach().clone(),
        history.timestamps_us.detach().clone(), history.world_from_grid_ego.detach().clone(),
        history.sample_tokens,qualification.detach().clone())
    return quantized.reshape(len(x), 5, 128, 50, 50), info[2].reshape(len(x), 5, 50, 50), tuple(shapes), bank


class GTCausalPredictor(CausalC0Predictor):
    def __init__(self, native, seed=2026, query_extension=None):
        super().__init__(native, seed=seed)
        self.query_extension = query_extension

    def _future_query_step(self, work, poses, bank, step):
        """Only the final future query is changed before native conv_out.

        The native pose query of the current step precedes conv_out; subsequent
        scene AND ego computations see the fused predicted token embedding.
        No claim of full BPTT through argmax feedback is made.
        """
        b, context = work.shape[:2]
        handle = None
        if self.query_extension is not None:
            def before_head(module, arguments):
                tensor = arguments[0]
                if tensor.shape != (b * context, 128, 50, 50):
                    raise ValueError('Native pre-token-classifier shape changed')
                views = tensor.reshape(b, context, 128, 50, 50)
                query = views[:, -1]
                fused = self.query_extension(query, bank, step)
                if fused.shape != query.shape or not torch.isfinite(fused).all():
                    raise ValueError('Invalid continuous query extension')
                replacement = views.clone()
                replacement[:, -1] = fused
                return (replacement.reshape_as(tensor), *arguments[1:])
            handle = self.native.transformer.conv_out.register_forward_pre_hook(before_head)
        try:
            return self.native.transformer.forward_autoreg_step(work, pose_tokens=poses,
                                                               start_frame=0, mid_frame=context)
        finally:
            if handle is not None:
                handle.remove()

    def forward(self, history: GTHistory):
        if not isinstance(history, GTHistory):
            raise TypeError('GT student accepts GTHistory only; no future GT container')
        history.validate()
        quantized, history_indices, shapes, bank = encode_gt_history(self.native.vae, history)
        # No source alias to generated work. Snapshot checks accidental writes.
        bank_before = (bank.features.clone(), bank.input_frame_timestamps_us.clone(),
                       bank.world_from_source_grid_ego.clone(),bank.reading_qualification.clone())
        delta_past, velocity = observed_motion(history)
        device, dtype = quantized.device, quantized.dtype
        q0 = self.bootstrap(torch.cat((quantized.mean((1,3,4)), velocity.to(device, dtype)), -1))
        logits0, mode0, candidates0, delta0, _ = self._select_motion(q0)
        initial_delta = torch.cat((delta_past.to(device, dtype), delta0[:, None]), 1)
        initial_mode = torch.nn.functional.one_hot(mode0, 3).to(dtype)[:, None].expand(-1, 5, -1)
        poses = self.native.pose_encoder(torch.cat((initial_delta, initial_mode), -1))
        work = quantized
        token_logits, ids_all, semantic_logits, traces = [], [], [], []
        modes, mode_logits, candidates, selected = [mode0], [logits0], [candidates0], [delta0]
        for step in range(1, 7):
            predicted, pose_queries = self._future_query_step(work, poses, bank, step)
            current = predicted[:, -1]
            if current.shape != (len(quantized), 512, 50, 50):
                raise ValueError('Native six-step token classifier differs')
            ids = current.argmax(1)
            embedding = self.native.vae.vqvae.get_codebook_entry(ids, shape=None).permute(0,3,1,2).contiguous().detach()
            decoded = self.native.vae.forward_decoder(self.native.vae.vqvae.post_quant_conv(embedding),
                list(shapes), (len(quantized), 1, *GRID_SHAPE))
            token_logits.append(current); ids_all.append(ids); semantic_logits.append(decoded[:, 0])
            if step < 6:
                ml, mode, c, delta, next_pose = self._select_motion(pose_queries[:, -1])
                mode_logits.append(ml); modes.append(mode); candidates.append(c); selected.append(delta)
                work = torch.cat((work, embedding[:, None]), 1)
                poses = torch.cat((poses, next_pose[:, None]), 1)
            traces.append(dict(query=step, context_frames=5+step-1, future_occupancy_index=step-1,
                offset_s=FUTURE_OFFSETS_S[step-1], predicts_outgoing_ego=step<6,
                ego_target_interval_s=[.5*step,.5*(step+1)] if step<6 else None,
                feedback='actual post-extension token argmax -> frozen codebook -> detached embedding',
                GT_input_frame_ages_s=bank.ages_s(FUTURE_OFFSETS_S[step-1]).tolist()))
        for old, now in zip(bank_before, (bank.features, bank.input_frame_timestamps_us,
                                        bank.world_from_source_grid_ego,bank.reading_qualification)):
            if not torch.equal(old, now):
                raise AssertionError('Read-only original GT anchor content/time/coordinates were overwritten')
        output = dict(token_logits=torch.stack(token_logits,1), predicted_tokens=torch.stack(ids_all,1),
            occupancy_logits=torch.stack(semantic_logits,1), history_token_ids=history_indices,
            mode_logits=torch.stack(mode_logits,1), predicted_modes=torch.stack(modes,1),
            motion_candidates=torch.stack(candidates,1), predicted_motion=torch.stack(selected,1),
            rollout_trace=tuple(traces), future_offsets_s=FUTURE_OFFSETS_S,
            occupancy_query_indices=(1,2,3,4,5,6), motion_query_indices=(0,1,2,3,4,5),
            motion_target_intervals_s=((0.,.5),(.5,1.),(1.,1.5),(1.5,2.),(2.,2.5),(2.5,3.)),
            original_anchor_bank_unchanged=True, historical_input_kind='ideal_GT_occupancy',
            bridge_loss_enabled=False, support_accepted=False)
        output['occupancy'] = output['occupancy_logits'].argmax(-1)
        if any(not torch.isfinite(v).all() for v in output.values() if isinstance(v, torch.Tensor)):
            raise FloatingPointError('Nonfinite actual GT prediction')
        return output


class GTStageEntry(torch.nn.Module):
    def __init__(self, predictor):
        super().__init__(); self.predictor = predictor

    def forward(self, history: GTHistory):
        return self.predictor(history)


def load_gt_c0_initialization(root, device='cpu'):
    """Load the verified OccWorld checkpoint and initialize the ego-motion heads."""
    import json
    from .reader_model import sha
    root = Path(root).resolve()
    pair = json.loads((root / 'configs/occworld/checkpoint.json').read_text())
    path = root / 'checkpoints/reference_only/dynamics_fulltrain_epoch50.pth'
    if sha(path) != pair['dynamics_checkpoint']['sha256'] or not pair['actual_embedded_vae_all_tensors_equal']:
        raise RuntimeError('Reviewed actual native pair changed')
    from .occworld_backend import build_occworld_model
    config = root / 'configs/occworld/model.py'
    native = build_occworld_model(root, config)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    keys = native.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    entry = GTStageEntry(GTCausalPredictor(native, seed=2026)).to(device).eval()
    return entry, dict(dynamics_path=str(path.relative_to(root)), dynamics_sha256=sha(path),
        native_config_path=config.relative_to(root).as_posix(), native_config_sha256=sha(config),
        missing_keys=list(keys.missing_keys), unexpected_keys=list(keys.unexpected_keys),
        VQ_pair_provenance_reused=True, seed=2026, camera_C0_weights_loaded=False,
        old_optimizer_loaded=False, old_sampler_cursor_loaded=False,
        new_ego_parameters=sum(p.numel() for m in (entry.predictor.bootstrap,entry.predictor.mode_selector) for p in m.parameters()),
        model_parameters=sum(p.numel() for p in entry.parameters()),
        trainable_parameters=sum(p.numel() for p in entry.parameters() if p.requires_grad))
