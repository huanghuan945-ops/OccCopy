"""Causal occupancy and ego-motion prediction with the OccWorld backbone."""
from dataclasses import dataclass
import torch
from torch import nn
from .voxel_warp import GRID_SHAPE

HISTORY_OFFSETS_US = (-2000000, -1500000, -1000000, -500000, 0)
FUTURE_OFFSETS_S = (.5, 1., 1.5, 2., 2.5, 3.)


def validate_rigid(transforms):
    if transforms.shape[-2:] != (4, 4) or not torch.isfinite(transforms).all():
        raise ValueError('Expected finite rigid transforms')
    r = transforms[..., :3, :3]
    identity = torch.eye(3, dtype=r.dtype, device=r.device)
    if (not torch.allclose(r.transpose(-1, -2) @ r, identity.expand_as(r), atol=1e-7, rtol=0)
            or not torch.allclose(torch.linalg.det(r), torch.ones_like(torch.linalg.det(r)), atol=1e-7, rtol=0)
            or not torch.allclose(transforms[..., 3, :], transforms.new_tensor([0, 0, 0, 1]).expand_as(transforms[..., 3, :]), atol=1e-10, rtol=0)):
        raise ValueError('Invalid rotation or homogeneous row')


def points_in_frame(points, target_from_source):
    return torch.einsum('...ij,...j->...i', target_from_source[..., :3, :3], points) + target_from_source[..., :3, 3]


def displacement_in_frame(displacement, target_from_source):
    """Same physical endpoints, changing expression frame: NO translation."""
    return torch.einsum('...ij,...j->...i', target_from_source[..., :3, :3], displacement)


@dataclass(frozen=True)
class PredictedHistory:
    occupancy: torch.Tensor                # B,5,X,Y,Z; actual camera argmax
    timestamps_us: torch.Tensor            # B,5 actual observed timestamps
    world_from_lidar: torch.Tensor          # B,5,4,4; observed poses only
    sample_tokens: tuple[tuple[str, ...], ...]
    camera_timestamps_us: torch.Tensor      # B,5,6, all <= t0

    def validate(self):
        x = self.occupancy
        if x.dtype != torch.int64 or x.ndim != 5 or tuple(x.shape[1:]) != (5, *GRID_SHAPE):
            raise ValueError('Expected int64 camera predictions B,5,200,200,16')
        if torch.any(x < 0) or torch.any(x >= 18):
            raise ValueError('Ignore/unknown values cannot enter class embedding')
        b = len(x)
        if self.timestamps_us.shape != (b, 5) or self.timestamps_us.dtype != torch.int64:
            raise ValueError('Expected five actual int64 history timestamps')
        if self.camera_timestamps_us.shape != (b, 5, 6) or self.camera_timestamps_us.dtype != torch.int64:
            raise ValueError('Expected actual six-camera times for every history frame')
        t = self.timestamps_us
        if torch.any(t[:, 1:] <= t[:, :-1]):
            raise ValueError('Nonmonotonic history times')
        intervals = t[:, 1:] - t[:, :-1]
        if torch.any((intervals < 400000) | (intervals > 600000)):
            raise ValueError('History interval exceeds frozen tolerance')
        expected = t.new_tensor(HISTORY_OFFSETS_US)
        if torch.any(torch.abs(t - t[:, -1:] - expected) > 100000):
            raise ValueError('History exceeds frozen 100ms offset tolerance')
        if torch.any(self.camera_timestamps_us > t[:, -1:, None]):
            raise ValueError('Camera after t0')
        if self.world_from_lidar.shape != (b, 5, 4, 4):
            raise ValueError('Expected observed world-from-LiDAR poses')
        validate_rigid(self.world_from_lidar)
        if len(self.sample_tokens) != b or any(len(s) != 5 or len(set(s)) != 5 for s in self.sample_tokens):
            raise ValueError('Expected distinct five history tokens per batch item')


def observed_motion(history):
    """Four outgoing observed increments + latest velocity in fixed L0 axes."""
    transforms = history.world_from_lidar
    positions, rotation = transforms[..., :3, 3], transforms[..., :3, :3]
    differences = positions[:, 1:] - positions[:, :-1]
    delta = torch.einsum('btji,btj->bti', rotation[:, :-1], differences)[..., :2]
    dt = (history.timestamps_us[:, -1] - history.timestamps_us[:, -2]).to(device=transforms.device, dtype=transforms.dtype) * 1e-6
    velocity = torch.einsum('bji,bj->bi', rotation[:, -1], differences[:, -1])[:, :2] / dt[:, None]
    return delta, velocity


def validate_classes(x, frames):
    if x.dtype != torch.int64 or x.ndim != 5 or tuple(x.shape[1:]) != (frames, *GRID_SHAPE):
        raise ValueError('Invalid native XYZ discrete class grid')
    if torch.any((x < 0) | (x >= 18)):
        raise ValueError('Classes must be in [0,18), never supervision ignore')


def encode_classes(vae, occupancy):
    """Frozen native class embedding/encoder/quantizer; no top-level CUDA wrapper."""
    vae.eval()
    with torch.no_grad():
        latent, shapes = vae.forward_encoder(occupancy)
        projected = vae.vqvae.quant_conv(latent)
        quantized, _, info = vae.vqvae.forward_quantizer(projected, is_voxel=False)
        b, f = occupancy.shape[:2]
        if quantized.shape != (b * f, 128, 50, 50):
            raise ValueError('Native quantized layout differs')
        indices = info[2].reshape(b, f, 50, 50)
    return quantized.reshape(b, f, 128, 50, 50), indices, tuple(shapes)


class CausalC0Predictor(nn.Module):
    """Student signature accepts ONLY actual predicted history and past poses."""
    def __init__(self, native_dynamics, seed=2026):
        super().__init__()
        self.native = native_dynamics
        if any(getattr(native_dynamics, flag, False) for flag in
               ('use_flow', 'use_causal_transport', 'use_dynamic_refine', 'use_transport_gate', 'use_ego_se2')):
            raise ValueError('C0 uses reviewed base dynamics only')
        native_dynamics.vae.requires_grad_(False).eval()
        # Preserve caller RNG; fixed new engineering parameters only.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.bootstrap = nn.Sequential(nn.Linear(130, 128), nn.ReLU())
            self.mode_selector = nn.Linear(128, 3)
        self.new_parameters_trained = False

    def train(self, mode=True):
        super().train(mode)
        self.native.vae.eval()
        return self

    def _select_motion(self, query):
        logits = self.mode_selector(query)
        modes = logits.argmax(-1)  # torch chooses first index on exact tie
        candidates = self.native.pose_decoder(query)
        chosen = candidates.gather(-2, modes[..., None, None].expand(*modes.shape, 1, 2)).squeeze(-2)
        onehot = torch.nn.functional.one_hot(modes, 3).to(query.dtype)
        feedback = self.native.pose_encoder(torch.cat((chosen, onehot), -1))
        return logits, modes, candidates, chosen, feedback

    def forward(self, history: PredictedHistory):
        if not isinstance(history, PredictedHistory):
            raise TypeError('Student only accepts PredictedHistory; no GT/metas payload')
        history.validate()
        device = next(self.native.parameters()).device
        x = history.occupancy.to(device)
        quantized, history_indices, shapes = encode_classes(self.native.vae, x)
        observed_delta, past_velocity = observed_motion(history)
        q0 = self.bootstrap(torch.cat((quantized.mean((1, 3, 4)), past_velocity.to(device, quantized.dtype)), -1))
        logits0, mode0, candidates0, delta0, _ = self._select_motion(q0)
        initial_delta = torch.cat((observed_delta.to(device, quantized.dtype), delta0[:, None]), 1)
        initial_mode = torch.nn.functional.one_hot(mode0, 3).to(quantized.dtype)[:, None].expand(-1, 5, -1)
        poses = self.native.pose_encoder(torch.cat((initial_delta, initial_mode), -1))
        work = quantized
        token_logits, token_ids, semantic_logits, trace = [], [], [], []
        mode_logits, modes, candidates, selected = [logits0], [mode0], [candidates0], [delta0]
        for step in range(1, 7):
            context = 5 + step - 1
            predicted, pose_queries = self.native.transformer.forward_autoreg_step(
                work, pose_tokens=poses, start_frame=0, mid_frame=context)
            current = predicted[:, -1]  # B,512,50,50
            if current.shape != (len(x), 512, 50, 50):
                raise ValueError('Native future-token output shape differs')
            ids = current.argmax(1)
            embedding = self.native.vae.vqvae.get_codebook_entry(ids, shape=None).permute(0, 3, 1, 2).contiguous()
            decoded = self.native.vae.forward_decoder(
                self.native.vae.vqvae.post_quant_conv(embedding), list(shapes), (len(x), 1, *GRID_SHAPE))
            token_logits.append(current)
            token_ids.append(ids)
            semantic_logits.append(decoded[:, 0])
            if step < 6:
                ml, m, c, delta, next_pose = self._select_motion(pose_queries[:, -1])
                mode_logits.append(ml); modes.append(m); candidates.append(c); selected.append(delta)
                work = torch.cat((work, embedding[:, None]), 1)
                poses = torch.cat((poses, next_pose[:, None]), 1)
            trace.append(dict(query=step, context_frames=context, future_occupancy_index=step - 1,
                offset_s=FUTURE_OFFSETS_S[step - 1], predicts_outgoing_ego=step < 6,
                ego_target_interval_s=[.5 * step, .5 * (step + 1)] if step < 6 else None))
        output = dict(
            token_logits=torch.stack(token_logits, 1), predicted_tokens=torch.stack(token_ids, 1),
            occupancy_logits=torch.stack(semantic_logits, 1),
            mode_logits=torch.stack(mode_logits, 1), predicted_modes=torch.stack(modes, 1),
            motion_candidates=torch.stack(candidates, 1), predicted_motion=torch.stack(selected, 1),
            history_token_ids=history_indices, rollout_trace=tuple(trace), future_offsets_s=FUTURE_OFFSETS_S,
            occupancy_query_indices=(1, 2, 3, 4, 5, 6), motion_query_indices=(0, 1, 2, 3, 4, 5),
            motion_target_intervals_s=((0., .5), (.5, 1.), (1., 1.5), (1.5, 2.), (2., 2.5), (2.5, 3.)))
        output['occupancy'] = output['occupancy_logits'].argmax(-1)
        if any(not torch.isfinite(value).all() for value in output.values() if isinstance(value, torch.Tensor)):
            raise FloatingPointError('Nonfinite actual C0 prediction')
        return output
