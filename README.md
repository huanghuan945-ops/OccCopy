# OccCopy

OccCopy forecasts 3D semantic occupancy with a frozen OccWorld backbone. The
forecaster receives five observed occupancy grids. At each prediction step,
the copy head aligns the most recent grid to the predicted ego pose and mixes
its token distribution with the backbone prediction.

## Main result

The reported OccCopy result with one copy source (J = 1) is **18.20% mIoU and
27.48% IoU** on 4,219 Occ3D-nuScenes validation windows. The full-grid scores
are averaged over the 1, 2, and 3 second horizons. Observed occupancy is ground
truth and future ego motion is predicted.

## Setup

Use Python 3.11. Install the CUDA build of PyTorch 2.12.0 for your driver,
then install the remaining dependencies. [PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v2120)

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

## Data

- Download the nuScenes v1.0 trainval metadata archive and place it at
  `data/nuscenes_archives/v1.0-trainval_meta.tar`. The index builder also accepts
  a compressed metadata archive through `--metadata`. [nuScenes downloads](https://www.nuscenes.org/nuscenes#download)
- Download the Occ3D-nuScenes labels. Pack the extracted `gts/` directory into
  an uncompressed tar at `data/nuscenes_archives/occ3d_labels.tar`.
  [Occ3D-nuScenes downloads](https://github.com/Tsinghua-MARS-Lab/Occ3D#occ3d-nuscenes)

```bash
tar -cf data/nuscenes_archives/occ3d_labels.tar -C /path/to/occ3d gts
python tools/data_index/build_indexes.py
```

The script writes the local indexes to `data/manifests/`. The two window lists
in `experiments/occcopy/` store the sample IDs and timestamps for 1,200 training
and 4,219 validation windows.

## Train and evaluate

The frozen checkpoints are private. If you have the original artifacts, place
them under `checkpoints/` with this layout.

```text
checkpoints/
  reference_only/dynamics_fulltrain_epoch50.pth
  c0_final.pt
  c0_final.pt.json
  reader_initialization.pt
  reader_m_pre_ro.pt
  reader_m_pre_ro.pt.json
```

```bash
python -m experiments.occcopy.train
python -m experiments.occcopy.evaluate
```

Training fits the copy head over 300 AdamW updates with four windows per update
and seed 2026. The evaluator reads the trained head from `runs/occcopy/train/`
and writes the scores to `runs/occcopy/eval/results.json`. Both commands accept
`--output` for a fresh run directory. Pass `--train-dir` to evaluation when
using a custom training directory.

## License

The release uses the Apache 2.0 license. The bundled model retains the
[OccWorld license](third_party/occworld/LICENSE). [OccWorld upstream](https://github.com/wzzheng/OccWorld)
