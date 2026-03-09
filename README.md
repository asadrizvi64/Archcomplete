# ArchComplete-GS — Phase 1

**Semantically-Guided Generative Completion of Architectural Scenes via 3D Gaussian Splatting and Domain-Conditioned Diffusion Priors**

MSc Thesis Codebase · Syed Muhammad Asad Rizvi · TU Dresden · Supervisor: Prof. Dr. Weigert

---

## Phase 1: Semantic-Aware 3DGS Reconstruction

This repository contains the complete Phase 1 implementation covering:

- **Semantic 3DGS** — 3D Gaussian model with per-primitive architectural labels (11 classes)
- **Depth Anything V2 integration** — monocular metric depth for augmented initialization
- **Grounded-SAM2 segmentation** — architectural vocabulary segmentation (wall, floor, ceiling, window, door, column, beam, staircase, roof, facade, balcony)
- **Planarity regularization** — novel loss enforcing flat surfaces on architectural plane classes
- **2D→3D label lifting** — ObjectGS-inspired majority-vote projection
- **Structural scene graph** — encodes wall→ceiling, window→wall, floor→wall, beam→column relationships
- **Confidence estimation** — density + semantic discontinuity + structural plausibility

---

## Installation

```bash
# 1. Clone
git clone https://github.com/your-repo/archcomplete-gs.git
cd archcomplete-gs

# 2. Create environment
conda create -n archgs python=3.10
conda activate archgs

# 3. Install PyTorch (with CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install gsplat
pip install gsplat

# 5. Install GroundingDINO
pip install git+https://github.com/IDEA-Research/GroundingDINO

# 6. Install SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2

# 7. Install remaining dependencies
pip install -r requirements.txt

# 8. Install this package
pip install -e .
```

---

## Usage

### Step 1 — Preprocess a new scene

```bash
# Full preprocessing (COLMAP + depth + segmentation)
python scripts/preprocess.py --scene data/scenes/my_building --scene-type outdoor

# Skip COLMAP if already done
python scripts/preprocess.py --scene data/scenes/my_building --skip-colmap

# Indoor scene (uses indoor metric depth model)
python scripts/preprocess.py --scene data/scenes/interior_room --scene-type indoor
```

Expected scene directory layout:
```
data/scenes/my_building/
    images/          ← Input RGB images (JPG or PNG)
    colmap/          ← Created by preprocess.py (or provide manually)
    depths/          ← Created by preprocess.py
    masks/           ← Created by preprocess.py
```

### Step 2 — Train

```bash
# Standard training (30,000 iterations)
python scripts/train.py --scene data/scenes/my_building

# Quick test run (500 iters, verifies everything works)
python scripts/train.py --scene data/scenes/my_building --test-run

# Resume from checkpoint
python scripts/train.py --scene data/scenes/my_building \
    --resume outputs/phase1_semantic_3dgs/checkpoints/gaussian_iter_010000.pt

# Override config values
python scripts/train.py --scene data/scenes/my_building \
    training.iterations=50000 training.loss.planarity=0.1
```

### Step 3 — Evaluate

```bash
python scripts/evaluate.py \
    --scene data/scenes/my_building \
    --checkpoint outputs/phase1_semantic_3dgs/checkpoints/gaussian_final.pt \
    --output outputs/phase1_semantic_3dgs/eval_results.json \
    --use-lpips
```

---

## Project Structure

```
archcomplete_gs/
├── archcomplete_gs/
│   ├── data/
│   │   ├── dataset.py          # Scene dataset, Camera dataclass
│   │   └── colmap_utils.py     # COLMAP binary parser, depth augmentation
│   ├── models/
│   │   ├── gaussian_model.py   # SemanticGaussianModel (core model)
│   │   ├── depth_estimator.py  # Depth Anything V2 wrapper + COLMAP alignment
│   │   └── segmentor.py        # Grounded-SAM2 architectural segmentor
│   ├── training/
│   │   ├── trainer.py          # Phase1Trainer — main training loop
│   │   └── losses.py           # Photometric + planarity + semantic losses
│   └── semantic/
│       ├── label_lifter.py     # 2D→3D majority-vote label lifting
│       ├── scene_graph.py      # Architectural scene graph builder
│       └── confidence.py       # Per-Gaussian confidence scoring
├── configs/
│   └── phase1.yaml             # All hyperparameters
├── scripts/
│   ├── preprocess.py           # COLMAP + depth + segmentation pipeline
│   ├── train.py                # Training entry point
│   └── evaluate.py             # PSNR/SSIM/LPIPS/mIoU/planarity evaluation
└── requirements.txt
```

---

## Key Hyperparameters (configs/phase1.yaml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.iterations` | 30,000 | Total training steps |
| `training.loss.planarity` | 0.05 | Planarity regularization weight |
| `training.loss.semantic_ce` | 0.50 | Semantic cross-entropy weight |
| `training.loss.depth_supervision` | 0.10 | Depth supervision weight |
| `label_lifting.start_iter` | 2,000 | First label lifting iteration |
| `label_lifting.lift_every` | 500 | Re-lift labels every N iters |
| `label_lifting.min_votes` | 3 | Min views for valid label |
| `scene_graph.build_after_iter` | 5,000 | When to build scene graph |
| `training.planarity.k_neighbors` | 16 | KNN for plane fitting |

---

## Architectural Vocabulary (11 classes)

| ID | Class | Planarity | Notes |
|----|-------|-----------|-------|
| 0 | background | — | Unlabeled |
| 1 | wall | ✓ | Primary planar class |
| 2 | floor | ✓ | Primary planar class |
| 3 | ceiling | ✓ | Primary planar class |
| 4 | window | — | Embedded in walls |
| 5 | door | — | Embedded in walls |
| 6 | column | — | Structural support |
| 7 | beam | — | Structural support |
| 8 | staircase | — | Vertical circulation |
| 9 | roof | — | Building top |
| 10 | facade | ✓ | Exterior face |
| 11 | balcony | — | Exterior projection |

---

## Experiment Tracking

Training is logged to Weights & Biases. Set `experiment.wandb: true` in config
(default) and ensure you're logged in: `wandb login`.

Key logged metrics:
- `loss/total`, `loss/rgb_total`, `loss/planarity`, `loss/semantic`
- `eval/psnr` (validation views, every 2000 iters)
- `train/num_gaussians`

---

## Phase 2 (Coming Next)

Phase 2 will implement:
- Confidence-guided gap region extraction
- 4-type Architectural Gap Taxonomy classifier (A: Occluded Facade, B: Inaccessible Interior, C: Material Failure, D: Access-Restricted)
- GNN-based gap type classification using the scene graph

## Citation

```bibtex
@misc{rizvi2026archcomplete,
  title={ArchComplete-GS: Semantically-Guided Generative Completion of
         Architectural Scenes via 3D Gaussian Splatting and
         Domain-Conditioned Diffusion Priors},
  author={Rizvi, Syed Muhammad Asad},
  year={2026},
  institution={Technische Universit\"at Dresden},
}
```
