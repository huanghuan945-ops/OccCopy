"""Evaluate last-frame OccCopy on the standard Occ3D-nuScenes validation pool."""
import argparse
import gc
import json
from pathlib import Path

from .runtime import OUTPUT_ROOT, ROOT, UPDATES, configure_runtime, read_config, read_windows, seed_rngs


HORIZONS_S = (1, 2, 3)
SCORED_STEPS = (1, 3, 5)  # Zero-based indices into six 0.5-second predictions.


def summarize(counts):
    import numpy as np

    true_positives = np.diagonal(counts, axis1=-2, axis2=-1).astype(float)
    union = counts.sum(-1) + counts.sum(-2) - true_positives
    foreground = (slice(None), slice(0, 17))
    per_class = np.divide(
        true_positives[foreground], union[foreground],
        out=np.full((len(HORIZONS_S), 17), np.nan), where=union[foreground] > 0,
    )
    miou = np.nanmean(per_class, axis=1) * 100
    intersection = counts[:, :17, :17].sum((1, 2))
    occupied_gt = counts[:, :17, :].sum((1, 2))
    occupied_pred = counts[:, :, :17].sum((1, 2))
    iou = intersection / np.maximum(occupied_gt + occupied_pred - intersection, 1) * 100
    return {
        "mIoU_mean_1_2_3s": float(miou.mean()),
        "IoU_mean_1_2_3s": float(iou.mean()),
        "mIoU_per_horizon": miou.tolist(),
        "IoU_per_horizon": iou.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-smoke", action="store_true", help="Score two windows on CPU.")
    parser.add_argument("--train-dir", type=Path, help="Read the checkpoint and result from this training directory.")
    parser.add_argument("--output", type=Path, help="Write to a new directory.")
    args = parser.parse_args()
    train_dir = args.train_dir or OUTPUT_ROOT / ("train_smoke" if args.cpu_smoke else "train")
    output = args.output or OUTPUT_ROOT / ("eval_smoke" if args.cpu_smoke else "eval")
    evaluate(train_dir, output, cpu_smoke=args.cpu_smoke)


def evaluate(train_dir, out, *, cpu_smoke=False):
    import numpy as np
    import torch

    from history_anchored_occ.reader_checkpoint import load_endpoint
    from history_anchored_occ.reader_model import check_frozen, digest_state, sha

    from .head import CopyHead, CopyRoute, bypass_reader
    from .windows import WindowData, WindowStore

    config = read_config()
    seed = config["seed"]
    rows = read_windows(config["val"], "val")
    device = configure_runtime(cpu_smoke)
    if out.exists():
        raise FileExistsError(f"Evaluation output already exists: {out}")

    training = json.loads((train_dir / "results.json").read_text())
    if training["seed"] != seed or (not cpu_smoke and training["updates"] != UPDATES):
        raise ValueError("Use the copy head trained with the main experiment configuration")
    path = train_dir / training["checkpoint"]
    if sha(path) != training["checkpoint_sha256"]:
        raise ValueError(f"Copy checkpoint checksum differs: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    store = WindowStore(ROOT)
    entry, _ = load_endpoint(ROOT, config["reader"])
    entry = entry.to(device)
    check_frozen(entry)
    base_digest = digest_state(entry.state_dict())
    if state["base_digest"] != base_digest or state["updates"] != training["updates"]:
        raise ValueError("Copy checkpoint and frozen forecaster do not match the training run")
    head = CopyHead(seed=seed).to(device).eval()
    head.load_state_dict(state["head"], strict=True)
    if digest_state(head.state_dict()) != training["final_head_digest"]:
        raise ValueError("Copy head state differs from the training result")
    route = CopyRoute(entry, head, device)

    def forward(history, gate_off=False):
        seed_rngs(seed)
        with route.active(history.occupancy.to(device), history.timestamps_us, gate_off=gate_off) as records:
            result = entry(history)
        check_frozen(entry)
        if len(records) != 6:
            raise RuntimeError("Expected six copy steps")
        return result

    scored_rows = rows[:2] if cpu_smoke else rows
    window_by_id = {row["window_id"]: row for row in rows}
    replay_ids = config["val"]["replay_ids"][:1 if cpu_smoke else 2]
    counts = np.zeros((len(HORIZONS_S), 18, 18), dtype=np.int64)
    with torch.no_grad():
        for window_id in replay_ids:
            history = WindowData(store, window_by_id[window_id]).load_history()
            closed = forward(history, gate_off=True)["predicted_tokens"]
            seed_rngs(seed)
            with bypass_reader(entry.predictor.query_extension):
                plain = entry(history)["predicted_tokens"]
            check_frozen(entry)
            if not torch.equal(closed, plain):
                raise RuntimeError(f"Closed gate does not reproduce C0 on {window_id}")
        for completed, row in enumerate(scored_rows, 1):
            window = WindowData(store, row)
            history = window.load_history()
            prediction = forward(history)["occupancy"][0, list(SCORED_STEPS)]
            if window.history_reads != 5:
                raise RuntimeError("Expected five observed frames before scoring")
            target = torch.from_numpy(window.load_scoring_labels(SCORED_STEPS)).to(device)
            if target.shape != prediction.shape:
                raise ValueError("Prediction and target grids differ")
            for step, (gt, pred) in enumerate(zip(target, prediction)):
                joined = (gt * 18 + pred).reshape(-1)
                counts[step] += torch.bincount(joined, minlength=18**2).reshape(18, 18).cpu().numpy()
            if completed % 500 == 0:
                print(f"Scored {completed}/{len(scored_rows)} windows", flush=True)
            gc.collect()
    if digest_state(entry.state_dict()) != base_digest:
        raise RuntimeError("Frozen forecaster changed during evaluation")

    result = {
        "method": "OccCopy (J=1)",
        "windows": len(scored_rows),
        "cpu_smoke": cpu_smoke,
        "horizons_s": list(HORIZONS_S),
        **summarize(counts),
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
