"""Checkpoint loading for the frozen OccCopy forecaster."""
import json
from pathlib import Path

import torch

from .reader_model import check_frozen, digest_state, load_common_entry, sha


def base_digest(entry):
    return digest_state(
        {
            k: v
            for k, v in entry.state_dict().items()
            if not k.startswith("predictor.query_extension.")
        }
    )


def load_endpoint(root, bound):
    """Load the frozen read-only endpoint used to construct the copy route."""
    case = "M_pre_RO"
    root = Path(root)
    path = root / bound["checkpoint"]
    meta = json.loads(path.with_suffix(".pt.json").read_text())
    if sha(path) != bound["sha256"] or meta["sha256"] != bound["sha256"]:
        raise ValueError("Endpoint hash differs")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if (
        value["format"] != "P0_matched_frozen_base_reader_300_v1"
        or value["reader_updates"] != 300
        or value["window_inputs"] != 1200
        or value["identity"]["case"] != case
        or value["identity"] != meta["identity"]
        or value["identity"]["debug_weights_inherited"]
        or not value["identity"]["optimizer_reset"]
    ):
        raise ValueError("Wrong matched endpoint")
    entry, loading = load_common_entry(root, case)
    if base_digest(entry) != value["base_digest"]:
        raise ValueError("Common base pairing differs")
    keys = entry.predictor.query_extension.load_state_dict(value["reader"], strict=True)
    if (
        keys.missing_keys
        or keys.unexpected_keys
        or digest_state(entry.predictor.query_extension.state_dict())
        != bound["final_reader_digest"]
    ):
        raise AssertionError("Reader load not exact")
    check_frozen(entry)
    # Saved pre-decoder used this exact lookup only when no codebook remapping exists.
    if entry.predictor.native.vae.vqvae.remap is not None:
        raise ValueError("Unexpected codebook remapping")
    return entry, dict(
        case=case,
        endpoint_sha256=bound["sha256"],
        reader_updates=300,
        base_unchanged=True,
        reader_digest=bound["final_reader_digest"],
        missing_keys=[],
        unexpected_keys=[],
        optimizer_objects=0,
        optimizer_state_restored=False,
        RNG_restored=False,
        codebook_remap=None,
        construction=loading,
    )
