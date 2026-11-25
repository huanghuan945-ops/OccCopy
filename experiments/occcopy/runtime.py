"""Configuration and deterministic runtime setup for the OccCopy run."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/occcopy.json"
OUTPUT_ROOT = ROOT / "runs/occcopy"
UPDATES = 300
ACCUMULATION = 4


def read_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def read_windows(spec, split):
    path = ROOT / spec["windows"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"Window manifest checksum differs: {path}")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != spec["count"] or len({row["window_id"] for row in rows}) != len(rows):
        raise ValueError(f"Window count or IDs differ: {path}")
    if any(row["split"] != f"official_{split}" for row in rows):
        raise ValueError(f"Window split differs: {path}")
    return rows


def configure_runtime(cpu_smoke):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not cpu_smoke:
        if not torch.cuda.is_available():
            raise RuntimeError("Training and full evaluation require CUDA. Use --cpu-smoke to check the pipeline.")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    return torch.device("cpu" if cpu_smoke else "cuda")


def seed_rngs(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
