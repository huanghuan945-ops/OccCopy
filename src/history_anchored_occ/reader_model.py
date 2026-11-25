"""Feature-reader variants attached to the frozen C0 forecaster."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import torch
from history_anchored_occ.gt_forecaster import RawGTAnchorBank, GTHistory, GTStageEntry
from history_anchored_occ.quality_forecaster import QualityGTPredictor, native_head_logits
from history_anchored_occ.quality_reader import QualityGTAnchorReader
from history_anchored_occ.gt_anchor_reader import sample_qualified

CASES = ("F_pre_RO", "M_pre_RO", "M_quant_RO", "M_pre_RW")
READER_PATH = "checkpoints/reader_initialization.pt"
READER_SHA = "fa771c15c00f1d1eb65a9e759ee9472044853a387c743bfc4a70c88b1f873b70"
C0_PATH = "checkpoints/c0_final.pt"
C0_SHA = "d7087eeb5f66e8d4d9be0b5a2055fff6cfd7fda23a9d127ec82d97036e655f9f"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest_state(state):
    h = hashlib.sha256()
    for name, t in state.items():
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fresh_reader(root, case):
    if case not in CASES:
        raise ValueError("Explicit P0-2 case required")
    path = Path(root) / READER_PATH
    if sha(path) != READER_SHA:
        raise ValueError("Reviewed common initialization changed")
    initial = torch.load(path, map_location="cpu", weights_only=False)
    if initial["seed"] != 2026 or initial["reader_updates"] != 0:
        raise ValueError("Fresh seed2026 initial state required")
    r = QualityGTAnchorReader("GT-F" if case == "F_pre_RO" else "GT-Mcore", "quality", seed=2026)
    keys = r.load_state_dict(initial["reader"], strict=True)
    if keys.missing_keys or keys.unexpected_keys:
        raise AssertionError("Reader state incompatible")
    if sum(p.numel() for p in r.parameters()) != 92940:
        raise AssertionError("Reader capacity changed")
    if r.project.weight.count_nonzero() or r.project.bias.count_nonzero():
        raise AssertionError("Not the zero-output common reader")
    return r.eval()


def check_frozen(entry):
    if any(m.training for m in entry.modules()):
        raise AssertionError("All running statistics must remain eval")
    for name, p in entry.named_parameters():
        allowed = name.startswith("predictor.query_extension.")
        if p.requires_grad != allowed or (not allowed and p.grad is not None):
            raise AssertionError("Parameter boundary: " + name)
    pred = entry.predictor
    if (
        pred._active_prior is not None
        or pred._geometry_trace
        or pred._memory is not None
        or pred._pending is not None
    ):
        raise AssertionError("Forward state leaked")
    head = pred.native.transformer.conv_out
    if head._forward_pre_hooks or head._forward_hooks:
        raise AssertionError("Transient head hook leaked")


def updated_content_time_block(previous, overwrite, observation_t0, step):
    """Audit timestamps may remain on CPU while the feature mask is on CUDA."""
    mask = overwrite.to(device=previous.device, dtype=torch.bool)
    current = observation_t0.to(device=previous.device, dtype=previous.dtype) + step * 500000
    return torch.where(mask, current[:, None, None], previous)


class SlotMemory:
    """Fixed source grid, immutable original qualification/observation metadata.

    Only features and per-position content times update. No prediction becomes
    new observation support. Content time is an audit field, not an age input.
    """

    def __init__(self, bank, case, quantized_history):
        if case not in CASES or not isinstance(bank, RawGTAnchorBank):
            raise ValueError("Reviewed bank and case required")
        f = quantized_history if case == "M_quant_RO" else bank.features
        if f.shape != bank.features.shape or not torch.isfinite(f).all():
            raise ValueError("Feature adapter shape/nonfinite")
        self.case = case
        self.bank = replace(
            bank,
            features=f.detach().clone(),
            input_frame_timestamps_us=bank.input_frame_timestamps_us.detach().clone(),
            world_from_source_grid_ego=bank.world_from_source_grid_ego.detach().clone(),
            reading_qualification=bank.reading_qualification.detach().clone(),
        )
        self.content_timestamps_us = (
            self.bank.input_frame_timestamps_us[:, :, None, None].expand(-1, -1, 50, 50).clone()
        )
        self._original = (
            self.bank.input_frame_timestamps_us.clone(),
            self.bank.world_from_source_grid_ego.clone(),
            self.bank.reading_qualification.clone(),
        )
        self.completed_output_steps = 0
        self.records = []

    def assert_original_metadata(self):
        for old, now in zip(
            self._original,
            (
                self.bank.input_frame_timestamps_us,
                self.bank.world_from_source_grid_ego,
                self.bank.reading_qualification,
            ),
        ):
            if not torch.equal(old, now):
                raise AssertionError("Observation metadata changed")

    def after_output(self, step, fused_query, geometry):
        if (
            step not in range(1, 7)
            or step != self.completed_output_steps + 1
            or geometry.step != step
        ):
            raise ValueError("Exactly ordered, completed six-step outputs required")
        self.assert_original_metadata()
        if not torch.equal(
            geometry.world_from_source_ego,
            self.bank.world_from_source_grid_ego.to(geometry.world_from_source_ego),
        ) or not torch.equal(
            geometry.source_timestamps_us,
            self.bank.input_frame_timestamps_us.to(geometry.source_timestamps_us),
        ):
            raise ValueError("Write geometry is not bound to the original slots")
        if (
            fused_query.shape != (len(self.bank.features), 128, 50, 50)
            or not torch.isfinite(fused_query).all()
        ):
            raise ValueError("Native fused query required")
        if self.case != "M_pre_RW":
            self.records.append(dict(step=step, write=False, reason="read_only"))
            self.completed_output_steps = step
            return
        slot = (step - 1) % 5
        with torch.no_grad():
            query_from_source = torch.linalg.inv(
                geometry.source_from_query[:, slot].detach().double()
            )
            xx, yy = torch.meshgrid(
                torch.arange(50, device=fused_query.device, dtype=torch.float64),
                torch.arange(50, device=fused_query.device, dtype=torch.float64),
                indexing="ij",
            )
            p = torch.stack(
                (
                    -40 + (xx + 0.5) * 1.6,
                    -40 + (yy + 0.5) * 1.6,
                    torch.zeros_like(xx),
                    torch.ones_like(xx),
                ),
                -1,
            )
            xy = torch.einsum("bij,xyj->bxyi", query_from_source, p)[..., :2].to(fused_query.dtype)
            inside = ((xy >= -40) & (xy < 40)).all(-1)
            q = torch.ones(
                (len(fused_query), 1, 50, 50), device=fused_query.device, dtype=torch.bool
            )
            sampled, rho = sample_qualified(
                fused_query.detach()[:, None], q, xy[:, None, :, :, None, :]
            )
            sampled = sampled[:, 0, :, :, 0, :].permute(0, 3, 1, 2).contiguous()
            inherited = self.bank.reading_qualification[:, slot].to(inside.device)
            overwrite = inherited & inside & (rho[:, 0, :, :, 0] > 0)
            # Copy-on-write: never mutate tensors saved in an earlier CE graph.
            updated = self.bank.features.clone()
            updated[:, slot] = torch.where(overwrite[:, None], sampled, updated[:, slot])
            times = self.content_timestamps_us.clone()
            times[:, slot] = updated_content_time_block(
                times[:, slot], overwrite, self.bank.input_frame_timestamps_us[:, -1], step
            )
            self.bank = replace(self.bank, features=updated.detach())
            self.content_timestamps_us = times
            n = int(overwrite.sum())
            total = int(inherited.sum())
            self.records.append(
                dict(
                    step=step,
                    slot_0_based=slot,
                    write=True,
                    overwritten_positions=n,
                    inherited_positions=total,
                    retained_supported_positions=total - n,
                    content_detached=True,
                    observation_time_refreshed=False,
                    support_added=False,
                    future_GT_used=False,
                    next_step_reads_write=step < 6,
                )
            )
        self.completed_output_steps = step
        self.assert_original_metadata()


class P0MatchedPredictor(QualityGTPredictor):
    """Reuse the actual reviewed six-step loop, causal ego and quality reader.

    A completed step is committed immediately before the next step, after its
    semantic decode and ego prediction; the final commit occurs after q6 decode.
    """

    def __init__(self, native, case, seed=2026, query_extension=None):
        super().__init__(native, seed, query_extension, geometry_gradient_policy="detached")
        if case not in CASES:
            raise ValueError("Explicit P0-2 case required")
        self.case = case
        self._memory = None
        self._pending = None

    def _commit_pending(self):
        if self._pending is None:
            return
        step, fused, geom = self._pending
        self._memory.after_output(step, fused, geom)
        self._pending = None

    def _future_query_step(self, work, poses, bank, step):
        self._commit_pending()
        if step == 1:
            if self._memory is not None:
                raise RuntimeError("Memory not reset")
            self._memory = SlotMemory(bank, self.case, work[:, :5])
        if self._memory is None or self._active_prior is None:
            raise RuntimeError("Inactive history-only rollout")
        geom = self._active_prior.read_geometry(step)
        b, context = work.shape[:2]

        def before_head(module, args):
            if self._pending is not None:
                raise AssertionError("Native head invoked more than once per step")
            x = args[0]
            if x.shape != (b * context, 128, 50, 50):
                raise ValueError("Native context layout changed")
            views = x.reshape(b, context, 128, 50, 50)
            pre = native_head_logits(module, views[:, -1])
            fused = self.query_extension(
                views[:, -1], self._memory.bank, step, geometry=geom, pre_logits=pre
            )
            self._pending = (step, fused.detach(), geom)
            updated = views.clone()
            updated[:, -1] = fused
            return (updated.reshape_as(x), *args[1:])

        handle = self.native.transformer.conv_out.register_forward_pre_hook(before_head)
        try:
            return self.native.transformer.forward_autoreg_step(
                work, pose_tokens=poses, start_frame=0, mid_frame=context
            )
        finally:
            handle.remove()

    def forward(self, history: GTHistory):
        if not isinstance(history, GTHistory):
            raise TypeError("History-only API; no target dictionary")
        if self._memory is not None or self._pending is not None:
            raise RuntimeError("Reentrant/stale memory")
        try:
            output = super().forward(history)
            self._commit_pending()
            if self._memory is None or self._memory.completed_output_steps != 6:
                raise AssertionError("Six complete outputs required")
            output.update(
                p0_case=self.case,
                memory_update_trace=tuple(self._memory.records),
                runtime_memory_content_readonly=self.case != "M_pre_RW",
                original_observation_metadata_unchanged=True,
                original_anchor_bank_unchanged=self.case != "M_pre_RW",
                anchor_representation="codebook_embedding"
                if self.case == "M_quant_RO"
                else "prequantizer",
                content_time_is_observation_time=False,
            )
            output["feedback_logits"] = output["token_logits"][:, :5]
            return output
        finally:
            self._memory = None
            self._pending = None


def load_common_entry(root, case):
    """Strict actual CPU load. Does not read/restore optimizer or CUDA RNG."""
    from history_anchored_occ.gt_forecaster import load_gt_c0_initialization

    root = Path(root)
    path = root / C0_PATH
    if sha(path) != C0_SHA:
        raise ValueError("Fixed C0 checkpoint changed")
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    meta = json.loads(path.with_suffix(".pt.json").read_text())
    if (
        meta["sha256"] != C0_SHA
        or state["identity"]["case"] != "C0"
        or state["cursor"]["base_updates_total"] != 6214
        or state["cursor"]["reader_updates_total"] != 0
        or meta["identity"] != state["identity"]
        or meta["cursor"] != state["cursor"]
    ):
        raise ValueError("C0 lineage mismatch")
    entry, construction = load_gt_c0_initialization(root, "cpu")
    keys = entry.load_state_dict(state["model"], strict=True)
    if digest_state(entry.state_dict()) != digest_state(state["model"]):
        raise AssertionError("Actual loaded C0 differs")
    predictor = P0MatchedPredictor(entry.predictor.native, case)
    predictor.load_state_dict(entry.predictor.state_dict(), strict=True)
    predictor.new_parameters_trained = True
    entry = GTStageEntry(predictor).eval().requires_grad_(False)
    base_digest = digest_state(entry.state_dict())
    entry.predictor.query_extension = fresh_reader(root, case).requires_grad_(True)
    entry.eval()
    check_frozen(entry)
    actual = {
        k: v
        for k, v in entry.state_dict().items()
        if not k.startswith("predictor.query_extension.")
    }
    if digest_state(actual) != base_digest:
        raise AssertionError("Attaching reader altered shared base")
    info = dict(
        case=case,
        base_checkpoint=path.relative_to(root).as_posix(),
        base_checkpoint_sha256=C0_SHA,
        base_state_digest=base_digest,
        VQ_digest=state["VQ_digest"],
        reader_initialization_sha256=READER_SHA,
        reader_state_digest=digest_state(entry.predictor.query_extension.state_dict()),
        reader_trainable_parameters=sum(p.numel() for p in entry.parameters() if p.requires_grad),
        frozen_parameters=sum(p.numel() for p in entry.parameters() if not p.requires_grad),
        base_missing_keys=list(keys.missing_keys),
        base_unexpected_keys=list(keys.unexpected_keys),
        optimizer_created=False,
        optimizer_state_restored=False,
        CUDA_rng_restored=False,
        actual_network_forward_executed=False,
        construction=construction,
    )
    return entry, info
