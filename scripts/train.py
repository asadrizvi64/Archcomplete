"""
scripts/train.py

Main entry point for ArchComplete-GS Phase 1 training.
Loads config, initializes trainer, and runs training loop.

Usage:
    # Basic:
    python scripts/train.py --scene data/scenes/my_building

    # With custom config:
    python scripts/train.py --scene data/scenes/my_building --config configs/phase1.yaml

    # Resume from checkpoint:
    python scripts/train.py --scene data/scenes/my_building --resume outputs/phase1/checkpoints/gaussian_iter_010000.pt

    # Quick test run (500 iters):
    python scripts/train.py --scene data/scenes/my_building --test-run
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def load_config(config_path: str, overrides: list[str] = None) -> OmegaConf:
    cfg = OmegaConf.load(config_path)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="ArchComplete-GS Phase 1 Training")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene directory")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (auto-detect if None)")
    parser.add_argument("--test-run", action="store_true", help="Run 500 iters for quick testing")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("override", nargs="*", help="Config overrides: key=value")
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    cfg = load_config(args.config, args.override)
    cfg.data.scene_path = args.scene

    if args.test_run:
        cfg.training.iterations = 500
        cfg.experiment.log_every = 50
        cfg.experiment.save_every = 200
        cfg.experiment.eval_every = 200
        cfg.experiment.name = cfg.experiment.name + "_test"
        print("[Train] TEST RUN: 500 iterations only.")

    if args.no_wandb:
        cfg.experiment.wandb = False

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Train] Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = "cpu"
        print("[Train] WARNING: Running on CPU. Training will be very slow.")

    print(f"[Train] Scene: {args.scene}")
    print(f"[Train] Config: {args.config}")
    print(f"[Train] Output: {cfg.experiment.output_dir}/{cfg.experiment.name}")
    print()

    # ── Add project root to path ─────────────────────────────────────────────
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # ── Data ────────────────────────────────────────────────────────────────
    from archcomplete_gs.data.dataset import ArchitecturalSceneDataset

    print("[Train] Loading training dataset...")
    train_dataset = ArchitecturalSceneDataset(
        scene_path=cfg.data.scene_path,
        split="train",
        train_split=cfg.data.train_split,
        image_scale=cfg.data.image_scale,
        white_background=cfg.data.white_background,
        load_depth=True,
        load_masks=True,
    )

    print("[Train] Loading validation dataset...")
    val_dataset = ArchitecturalSceneDataset(
        scene_path=cfg.data.scene_path,
        split="val",
        train_split=cfg.data.train_split,
        image_scale=cfg.data.image_scale,
        white_background=cfg.data.white_background,
        load_depth=True,
        load_masks=False,   # No masks needed for eval
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    from archcomplete_gs.training.trainer import Phase1Trainer

    trainer = Phase1Trainer(
        cfg=cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
    )

    trainer.setup()

    if args.resume:
        print(f"[Train] Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # ── Train ────────────────────────────────────────────────────────────────
    trainer.train()


if __name__ == "__main__":
    main()
