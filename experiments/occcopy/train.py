"""Train the last-frame OccCopy head on a frozen forecaster."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from .runtime import ACCUMULATION, OUTPUT_ROOT, ROOT, UPDATES, configure_runtime, read_config, read_windows, seed_rngs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-smoke", action="store_true", help="Run one four-window update on CPU.")
    parser.add_argument("--output", type=Path, help="Write to a new directory.")
    args = parser.parse_args()
    out = args.output or OUTPUT_ROOT / ("train_smoke" if args.cpu_smoke else "train")
    train(out, cpu_smoke=args.cpu_smoke)


def train(out, *, cpu_smoke=False):
    import numpy as np
    import torch

    from history_anchored_occ.c0_model import encode_classes
    from history_anchored_occ.c0_supervision import build_future_supervision
    from history_anchored_occ.reader_checkpoint import load_endpoint
    from history_anchored_occ.reader_model import check_frozen, digest_state, sha
    from history_anchored_occ.semantic_loss import auxiliary_embedding
    from history_anchored_occ.training_utils import capture_rng, restore_rng, cpu_tree

    from .head import CopyHead, CopyRoute
    from .losses import (
        objective_sums,
        normalized_loss,
        target_counts,
        shared_feedback_ce,
        semantic_ce_sum,
    )
    from .windows import WindowData, WindowStore

    config = read_config()
    seed = config["seed"]
    rows = read_windows(config["train"], "train")
    device = configure_runtime(cpu_smoke)
    store = WindowStore(ROOT)
    seed_rngs(seed)
    entry, _ = load_endpoint(ROOT, config["reader"])
    entry = entry.to(device)
    check_frozen(entry)
    base = digest_state(entry.state_dict())
    vae = entry.predictor.native.vae
    head = CopyHead(seed=seed).to(device)
    initial = digest_state(head.state_dict())
    parameters = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(
        head.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )
    seed_rngs(seed)
    route = CopyRoute(entry, head, device)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    completed = 0

    def hashes(o):
        return {
            k: hashlib.sha256(o[k].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k in (
                "token_logits",
                "predicted_tokens",
                "occupancy",
                "predicted_motion",
                "predicted_modes",
            )
        }

    def forward(history, labels=None):
        with route.active(
            history.occupancy.to(device), history.timestamps_us, labels=labels
        ) as records:
            o = entry(history)
        if o["token_logits"].shape != (1, 6, 512, 50, 50) or len(records) != 6:
            raise AssertionError("Six copy steps required")
        return o, records

    def semantic_vjp(stored, target, shapes):
        vjp = torch.zeros_like(stored)
        total = 0.0
        for step in range(6):
            leaf = stored[:, step].to(device).clone().requires_grad_(True)
            proxy, _, _ = auxiliary_embedding(leaf, vae.vqvae.embedding.weight, temperature=1.0)
            dense = vae.forward_decoder(
                vae.vqvae.post_quant_conv(proxy), list(shapes), (1, 1, 200, 200, 16)
            )[:, 0]
            value, n = semantic_ce_sum(dense, target.future_occupancy[:, step].to(device))
            if n != 640000:
                raise AssertionError("Full-label semantic count differs")
            (vector,) = torch.autograd.grad(value / (6 * 640000), leaf)
            if not torch.isfinite(vector).all():
                raise FloatingPointError("Nonfinite decoder VJP")
            vjp[:, step] = vector.detach().cpu()
            total += float(value.detach()) / (6 * 640000)
        return vjp, total

    updates = 1 if cpu_smoke else UPDATES
    try:
        with (out / "updates.jsonl").open("x") as trace:
            for step in range(updates):
                batch = rows[ACCUMULATION * step : ACCUMULATION * step + ACCUMULATION]
                prepared = []
                tick = time.monotonic()
                for row in batch:
                    data = WindowData(store, row)
                    history = data.load_history()
                    if data.history_reads != 5:
                        raise AssertionError("History input boundary differs")
                    future = data.load_future_targets()
                    with torch.no_grad():
                        target = build_future_supervision(vae, future)
                        _, _, shapes = encode_classes(vae, future.occupancy[:, :1].to(device))
                    prepared.append((history, target, shapes))
                opt.zero_grad(set_to_none=True)
                before = digest_state(head.state_dict())
                counts = tuple(sum(target_counts(t)[k] for _, t, _ in prepared) for k in range(3))
                feedback_count = sum(t.future_token_labels[:, :5].numel() for _, t, _ in prepared)
                sums = [0.0, 0.0, 0.0]
                feedback_sum = 0.0
                semantic_mean = 0.0
                gates = []
                aux_log = []
                for history, target, shapes in prepared:
                    rng = cpu_tree(capture_rng())
                    o, records = forward(history, target.future_token_labels[0].to(device))
                    expected = hashes(o)
                    stored = o["token_logits"].detach().cpu().clone()
                    terms = objective_sums(o, target)
                    fb = shared_feedback_ce(
                        o["feedback_logits"], target.future_token_labels.to(device)
                    )
                    nfb = target.future_token_labels[:, :5].numel()
                    loss = normalized_loss(terms, counts) + 0.1 * fb * nfb / feedback_count
                    pc = sum(r["pointer_count"] for r in records)
                    gc_ = sum(r["gate_count"] for r in records)
                    pointer = sum(r["pointer_sum"] for r in records) / pc.clamp_min(1)
                    gate_bce = sum(r["gate_sum"] for r in records) / gc_.clamp_min(1)
                    loss = loss + (1.0 * pointer + 0.5 * gate_bce) / ACCUMULATION
                    aux_log.append(
                        [
                            float(pointer.detach()),
                            float(gate_bce.detach()),
                            float(sum(r["top1_correct"] for r in records) / pc.clamp_min(1)),
                            float(pc / (6 * 2500)),
                            float(sum(r["gate_target_mean"] for r in records) / gc_.clamp_min(1)),
                        ]
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite main loss")
                    loss.backward()
                    for k in range(3):
                        sums[k] += float(terms["sums"][k].detach())
                    feedback_sum += float(fb.detach()) * nfb
                    gates.append([float(r["gate"].mean()) for r in records])
                    del o, terms, fb, loss, records
                    vector, ce = semantic_vjp(stored, target, shapes)
                    restore_rng(rng)
                    replay, _ = forward(history)
                    if hashes(replay) != expected:
                        raise AssertionError("Same-weight exact replay differs")
                    torch.autograd.backward(
                        replay["token_logits"], 0.1 * vector.to(device) / ACCUMULATION
                    )
                    semantic_mean += ce / ACCUMULATION
                    del replay, stored, vector
                    gc.collect()
                check_frozen(entry)
                if digest_state(entry.state_dict()) != base:
                    raise AssertionError("Frozen base changed")
                norm = torch.nn.utils.clip_grad_norm_(
                    head.parameters(), 5.0, error_if_nonfinite=True
                )
                if not torch.isfinite(norm) or norm <= 0:
                    raise AssertionError("No copy-head learning signal")
                opt.step()
                opt.zero_grad(set_to_none=True)
                completed = step + 1
                if digest_state(head.state_dict()) == before:
                    raise AssertionError("Optimizer did not change the head")
                if counts[0] != 60000 or feedback_count != 50000:
                    raise AssertionError("Denominators differ")
                main_loss = (
                    sum(w * v / n if n else 0.0 for w, v, n in zip((1.0, 0.1, 0.1), sums, counts))
                    + 0.1 * feedback_sum / feedback_count
                )
                record = dict(
                    step=completed,
                    window_ids=[r["window_id"] for r in batch],
                    main_loss=main_loss,
                    semantic_CE=semantic_mean,
                    total_loss=main_loss + 0.1 * semantic_mean,
                    token_CE=sums[0] / counts[0],
                    gradient_norm=float(norm),
                    mean_gate_per_step=np.mean(gates, 0).tolist(),
                    seconds=time.monotonic() - tick,
                    **dict(
                        zip(
                            (
                                "pointer_NLL",
                                "gate_BCE",
                                "pointer_top1_accuracy",
                                "cells_with_correct_candidate",
                                "gate_target_rate",
                            ),
                            np.mean(aux_log, 0).tolist(),
                        )
                    ),
                )
                trace.write(json.dumps(record) + "\n")
                trace.flush()
                if completed % 10 == 0 or cpu_smoke:
                    print(
                        json.dumps(
                            dict(
                                updates=completed,
                                token_CE=record["token_CE"],
                                gate=record["mean_gate_per_step"],
                                pointer_top1=record["pointer_top1_accuracy"],
                                gate_target=record["gate_target_rate"],
                                elapsed=round(time.monotonic() - started),
                            )
                        ),
                        flush=True,
                    )

        checkpoint = out / "head.pt"
        torch.save(
            dict(head=cpu_tree(head.state_dict()), updates=completed, base_digest=base),
            checkpoint,
        )
        result = dict(
            updates=completed,
            train_windows=ACCUMULATION * completed,
            copy_head_parameters=parameters,
            seed=seed,
            cpu_smoke=cpu_smoke,
            final_head_digest=digest_state(head.state_dict()),
            initial_head_digest=initial,
            checkpoint=checkpoint.name,
            checkpoint_sha256=sha(checkpoint),
            elapsed_seconds=time.monotonic() - started,
            peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30
            if device.type == "cuda"
            else None,
        )
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    except BaseException as error:
        (out / "failed.json").write_text(
            json.dumps(dict(error=repr(error), completed_updates=completed)) + "\n"
        )
        raise


if __name__ == "__main__":
    main()
