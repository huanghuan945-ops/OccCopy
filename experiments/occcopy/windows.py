"""Load observed occupancy separately from future supervision and scoring labels."""
import io
import json

import numpy as np
from pyquaternion import Quaternion
import torch

from history_anchored_occ.archive_io import IndexedArchive
from history_anchored_occ.c0_supervision import FutureGT
from history_anchored_occ.gt_forecaster import GTHistory
from history_anchored_occ.voxel_warp import GRID_SHAPE


def read_occupancy(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as record:
        semantics = record["semantics"]
        if (semantics.shape != GRID_SHAPE or not np.issubdtype(semantics.dtype, np.integer)
                or np.any((semantics < 0) | (semantics >= 18))):
            raise ValueError("Occ3D semantics must have shape (200, 200, 16) and classes 0 through 17")
        masks = {}
        for key in ("mask_camera", "mask_lidar"):
            mask = record[key]
            if mask.shape != GRID_SHAPE or not np.isin(mask, [0, 1]).all():
                raise ValueError(f"Invalid Occ3D mask: {key}")
            masks[key] = mask.astype(bool)
        return semantics.astype(np.int64), masks


def transform(record):
    rotation = np.asarray(record["rotation"], dtype=np.float64)
    translation = np.asarray(record["translation"], dtype=np.float64)
    if (rotation.shape != (4,) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()):
        raise ValueError("Invalid calibration or pose")
    if not np.isclose(np.linalg.norm(rotation), 1.0, atol=1e-5):
        raise ValueError("Calibration or pose quaternion must have unit norm")
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = translation
    return matrix


class ValidationHistory(GTHistory):
    require_nominal_history = False


class WindowStore:
    def __init__(self, root):
        indexes = root / "data/manifests"
        metadata = json.loads((indexes / "metadata_index.json").read_text())["tables"]
        self.scenes = {row["token"]: row for row in metadata["scene"]}
        self.samples = {row["token"]: row for row in metadata["sample"]}
        self.calibrations = {row["token"]: row for row in metadata["calibrated_sensor"]}
        self.poses = {row["token"]: row for row in metadata["ego_pose"]}
        self.lidar = {}
        for row in metadata["sample_data"]:
            token = row["sample_token"]
            if token in self.lidar:
                raise ValueError(f"Duplicate LiDAR keyframe: {token}")
            self.lidar[token] = row
        self.labels = IndexedArchive(indexes / "label_archive_index.json")


class WindowData:
    def __init__(self, store, row):
        self.store = store
        self.scene_name = row["scene_name"]
        self.split = row["split"]
        self.history = tuple(json.loads(row["history_tokens"]))
        self.future = tuple(json.loads(row["future_tokens"]))
        self.sequence = self.history + self.future
        if len(self.history) != 5 or len(self.future) != 6 or len(set(self.sequence)) != 11:
            raise ValueError("Expected five observed and six distinct future frames")
        self.timestamps = tuple(json.loads(row["history_timestamps_us"])
                                + json.loads(row["future_timestamps_us"]))
        self.cutoff = int(row["t0_timestamp_us"])
        if (len(self.timestamps) != 11 or self.history[-1] != row["t0_token"]
                or self.timestamps[4] != self.cutoff):
            raise ValueError("Invalid t0 or window timestamps")
        scene = store.scenes[row["scene_token"]]
        if scene["name"] != self.scene_name or scene["log_token"] != row["log_token"]:
            raise ValueError("Window scene or log differs from nuScenes metadata")
        for token, stamp in zip(self.sequence, self.timestamps):
            sample = store.samples[token]
            if sample["scene_token"] != scene["token"] or sample["timestamp"] != stamp:
                raise ValueError(f"Window frame does not match nuScenes metadata: {token}")
            if token not in store.lidar:
                raise ValueError(f"Missing LiDAR keyframe metadata: {token}")
        self.history_reads = 0

    def _adjacent(self, first, second):
        samples = self.store.samples
        return samples[first]["next"] == second and samples[second]["prev"] == first

    def _poses(self, token, stamp):
        lidar = self.store.lidar[token]
        pose = self.store.poses[lidar["ego_pose_token"]]
        if lidar["timestamp"] != stamp or pose["timestamp"] != stamp:
            raise ValueError(f"LiDAR or pose timestamp differs for frame {token}")
        ego = transform(pose)
        return ego @ transform(self.store.calibrations[lidar["calibrated_sensor_token"]]), ego

    def _read(self, token):
        member = f"gts/{self.scene_name}/{token}/labels.npz"
        return read_occupancy(self.store.labels.read(member))

    def load_history(self):
        if any(not self._adjacent(a, b) for a, b in zip(self.history, self.history[1:])):
            raise ValueError("Observed frames must be directly adjacent")
        occupancy, lidar_poses, ego_poses = [], [], []
        self.history_reads = 0
        for token, stamp in zip(self.history, self.timestamps[:5]):
            if stamp > self.cutoff:
                raise ValueError("Observed frame after t0")
            lidar, ego = self._poses(token, stamp)
            semantics, _ = self._read(token)
            occupancy.append(semantics)
            lidar_poses.append(lidar)
            ego_poses.append(ego)
            self.history_reads += 1
        history_type = GTHistory if self.split == "official_train" else ValidationHistory
        result = history_type(
            torch.from_numpy(np.stack(occupancy)[None]),
            torch.tensor([self.timestamps[:5]], dtype=torch.int64),
            torch.from_numpy(np.stack(lidar_poses)[None]),
            torch.from_numpy(np.stack(ego_poses)[None]),
            (self.history,),
        )
        result.validate()
        return result

    def load_future_targets(self):
        sequence = self.sequence[4:]
        stamps = self.timestamps[4:]
        lidar_poses = np.full((7, 4, 4), np.nan)
        valid_pose = np.zeros(7, dtype=bool)
        for i, (token, stamp) in enumerate(zip(sequence, stamps)):
            try:
                lidar_poses[i], _ = self._poses(token, stamp)
                valid_pose[i] = True
            except (KeyError, ValueError):
                pass
        adjacency = np.array([self._adjacent(a, b) for a, b in zip(sequence, sequence[1:])])
        occupancy, masks = [], {"mask_camera": [], "mask_lidar": []}
        for token in self.future:
            semantics, raw_masks = self._read(token)
            occupancy.append(semantics)
            for key in masks:
                masks[key].append(raw_masks[key])
        return FutureGT(
            torch.from_numpy(np.stack(occupancy)[None]),
            {key: torch.from_numpy(np.stack(values)[None]) for key, values in masks.items()},
            torch.from_numpy(lidar_poses[None]),
            torch.tensor([stamps], dtype=torch.int64),
            torch.from_numpy(valid_pose[None]),
            torch.from_numpy(adjacency[None]),
            (self.future,),
        )

    def load_scoring_labels(self, steps):
        return np.stack([self._read(self.future[step])[0] for step in steps])
