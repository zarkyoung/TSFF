# TSFF: A Two-Stage Spatiotemporal Fusion Framework for Land Surface Temperature with Environmental Variable Correction

This repository contains the official implementation of the paper:

> **A Two-Stage Spatiotemporal Fusion Framework for Land Surface Temperature With Environmental Variable Correction**
> Yong Zhang, Yingbao Yang.
> *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2026.

---

## Overview

TSFF (Two-Stage Spatiotemporal Fusion Framework) fuses **ERA5-Land** and **MODIS** land surface temperature (LST) products to generate **hourly, MODIS-resolution** LST estimates that combine the temporal continuity of ERA5-Land with the spatial detail of MODIS. The fusion is performed in two stages:

1. **Stage 1 — BMSTF-GAN (Bidirectional Multifeature Spatiotemporal Fusion Network with GAN).** Implemented in `models/model.py` (`Recont`), this network fuses ERA5-Land LST and MODIS LST observations from a reference date with the ERA5-Land LST at the target date to produce an initial MODIS-resolution LST estimate at the target date. It uses a **bidirectional structure** to strengthen temporal constraints and a **multiscale attention mechanism** to capture complex spatial dependencies, and is trained adversarially with a generative adversarial network.
2. **Stage 2 — LST-CorrNet (LST Correction Network).** Implemented in `models/model_correction.py` (`SimpleCorrection`, wrapped by `TwoStageModel`), this lightweight network performs a **residual correction** on the Stage 1 estimate using heterogeneous environmental variables — **digital elevation model (DEM)**, **total column water vapor (TCWV)**, and **upward/downward longwave radiation** — that account for thermodynamic influences from topography, the atmosphere, and surface radiative processes. The predicted correction is bounded and added back to the base prediction.

> **Note on the paper vs. this codebase.**
> The results reported in the published paper use **only the first two stages** (BMSTF-GAN training + LST-CorrNet training). The code also contains a **Stage 3 joint fine-tuning** routine (`train_joint_stage` in `main_correction.py`), which was **not used** for the paper. It is kept in the repository for completeness and for users who wish to experiment further; running it is not required to reproduce the paper.

---

## Repository Structure

```
BMSTF-GAN_correction3/
├── main_correction.py          # Entry point: orchestrates the staged training
├── trainer.py                  # BMSTF-GAN trainer (Stage 1)
├── trainer_correction.py       # LST-CorrNet / joint trainers (Stage 2 / 3)
├── configs/
│   └── config_two_stage.yaml   # All hyperparameters and paths
├── models/
│   ├── model.py                # BMSTF-GAN base network (Recont) + losses
│   └── model_correction.py     # LST-CorrNet (SimpleCorrection) + TwoStageModel wrapper
├── utils/                      # Logger, config loader, SSIM, early stopping, ...
├── data/                       # Place .npy training/validation tensors here
├── logs/                       # Training logs and history plots
└── checkpoints/                # Saved model weights
```

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12 (CUDA recommended)
- torchvision, torchmetrics
- numpy, pyyaml, matplotlib

A minimal install:

```bash
pip install torch torchvision torchmetrics numpy pyyaml matplotlib
```

---

## Data Preparation

The training script loads pre-processed tensors directly from `.npy` files under `dataset_path` (default `./data`). The expected files are:

| File                       | Description                                  |
| -------------------------- | -------------------------------------------- |
| `train_mobj.npy` / `val_mobj.npy` | MODIS LST at the target date (ground truth) |
| `train_mref.npy` / `val_mref.npy` | MODIS LST at the reference date              |
| `train_eobj.npy` / `val_eobj.npy` | ERA5-Land LST at the target date             |
| `train_eref.npy` / `val_eref.npy` | ERA5-Land LST at the reference date          |
| `train_dem.npy`  / `val_dem.npy`  | Digital elevation model (DEM)                |
| `train_wv_obj.npy` / `val_wv_obj.npy` | Total column water vapor (TCWV) at the target date |
| `train_lw_up_obj.npy` / `val_lw_up_obj.npy` | Upward longwave radiation at the target date |
| `train_lw_down_obj.npy` / `val_lw_down_obj.npy` | Downward longwave radiation at the target date |

Each array is expected to have shape `(N, H, W)`. The script unsqueezes a channel dimension to `(N, 1, H, W)` automatically. `H = W = patch_size` (default `128`). ERA5-Land inputs should be resampled to the MODIS grid before being saved as `.npy`.

---

## Configuration

All hyperparameters live in [`configs/config_two_stage.yaml`](configs/config_two_stage.yaml). Key entries:

```yaml
dataset_path: './data'
patch_size: 128
batch_size: 32
learning_rate: 0.0001        # Stage 1
correction_learning_rate: 0.0005   # Stage 2
base_epochs: 200             # Stage 1 epochs
correction_epochs: 120       # Stage 2 epochs
joint_epochs: 50             # Stage 3 epochs (NOT used in the paper)
use_dem: true
use_water_vapor: true
use_longwave_radiation: true
```

---

## Training

To reproduce the paper, run Stages 1 and 2 only.

**Stage 1 — train BMSTF-GAN (the base fusion network):**

```bash
python main_correction.py --config configs/config_two_stage.yaml --stage base
```

**Stage 2 — train LST-CorrNet on top of the best BMSTF-GAN checkpoint:**

```bash
python main_correction.py --config configs/config_two_stage.yaml --stage correction
```

**Combined (Stages 1 + 2 + evaluation):** the `all` mode also runs Stage 3. To reproduce the paper, prefer running the two stages individually as above.

```bash
# Runs base + correction + joint + evaluation (paper uses only base + correction)
python main_correction.py --stage all
```

**Stage 3 — joint fine-tuning (optional, not used in the paper):**

```bash
python main_correction.py --stage joint
```

**Evaluation only:**

```bash
python main_correction.py --stage evaluate
```

Useful flags:

- `--gpu 0` — select GPU id
- `--skip_base` — skip Stage 1 and reuse an existing BMSTF-GAN checkpoint
- `--base_model /path/to/base.pth` — provide an explicit BMSTF-GAN checkpoint for Stage 2

Outputs:

- Checkpoints → `./checkpoints/` (`base_model_best.pth`, `best_correction_model.pth`, `best_joint_model.pth`)
- Logs and training-history plots → `./logs/`

---

## Citation

If you use this code, please cite:

```bibtex
@article{zhang2026two,
  title   = {A Two-Stage Spatiotemporal Fusion Framework for Land Surface Temperature With Environmental Variable Correction},
  author  = {Zhang, Yong and Yang, Yingbao},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2026},
  publisher = {IEEE}
}
```
