"""
scripts/evaluate.py

Phase 1 evaluation script.
Computes:
  - Standard NVS metrics: PSNR, SSIM, LPIPS
  - Semantic accuracy: per-class IoU between lifted labels and GT masks
  - Planarity score: fraction of planar-class Gaussians passing planarity test
  - Coverage: voxel-level scene coverage statistics

Usage:
    python scripts/evaluate.py --scene data/scenes/my_building \
        --checkpoint outputs/phase1/checkpoints/gaussian_final.pt \
        --output outputs/phase1/eval_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).mean()
    return (-10 * torch.log10(mse + 1e-8)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    from archcomplete_gs.training.losses import ssim_loss
    return (1.0 - ssim_loss(pred, target)).item()


def compute_lpips(pred: torch.Tensor, target: torch.Tensor, lpips_fn) -> float:
    p = pred.unsqueeze(0) * 2 - 1     # [0,1] → [-1,1]
    t = target.unsqueeze(0) * 2 - 1
    return lpips_fn(p, t).item()


def compute_semantic_iou(
    pred_labels: np.ndarray,        # (H, W) predicted class ids
    gt_labels: np.ndarray,          # (H, W) ground truth class ids
    num_classes: int = 12,
) -> dict:
    """Per-class and mean IoU."""
    from archcomplete_gs.data.dataset import ArchitecturalSceneDataset
    ious = {}
    for c in range(num_classes):
        pred_c = pred_labels == c
        gt_c = gt_labels == c
        intersection = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        if union == 0:
            continue
        ious[ArchitecturalSceneDataset.ARCH_CLASSES[c]] = float(intersection) / float(union)
    mean_iou = np.mean(list(ious.values())) if ious else 0.0
    return {"per_class": ious, "mean_iou": mean_iou}


def compute_planarity_score(
    means: torch.Tensor,
    sem_labels: torch.Tensor,
    k: int = 16,
    threshold: float = 0.02,
) -> float:
    """
    Fraction of planar-class Gaussians that pass a planarity test.
    A Gaussian 'passes' if its distance to its local fitted plane < threshold.
    """
    from archcomplete_gs.data.dataset import ArchitecturalSceneDataset

    planar_ids = ArchitecturalSceneDataset.PLANAR_CLASS_IDS
    planar_mask = torch.zeros(means.shape[0], dtype=torch.bool, device=means.device)
    for c in planar_ids:
        planar_mask |= (sem_labels == c)

    if planar_mask.sum() < k + 1:
        return 0.0

    planar_means = means[planar_mask]
    N = min(planar_means.shape[0], 8192)
    if planar_means.shape[0] > N:
        idx = torch.randperm(planar_means.shape[0], device=means.device)[:N]
        planar_means = planar_means[idx]

    k_actual = min(k, planar_means.shape[0] - 1)
    dists = torch.cdist(planar_means, planar_means)
    dists.fill_diagonal_(float("inf"))
    nn_idx = dists.topk(k_actual, dim=1, largest=False).indices
    neighbors = planar_means[nn_idx]

    center = neighbors.mean(dim=1, keepdim=True)
    centered = neighbors - center
    try:
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
        normals = Vh[:, -1, :]
    except Exception:
        return 0.0

    diff = planar_means - center.squeeze(1)
    dist_to_plane = (diff * normals).sum(dim=-1).abs()
    pass_pct = (dist_to_plane < threshold).float().mean().item()
    return pass_pct


def main():
    parser = argparse.ArgumentParser(description="ArchComplete-GS Phase 1 Evaluation")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/phase1.yaml")
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--use-lpips", action="store_true", help="Compute LPIPS (requires lpips package)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from omegaconf import OmegaConf
    from archcomplete_gs.data.dataset import ArchitecturalSceneDataset
    from archcomplete_gs.models.gaussian_model import SemanticGaussianModel
    from archcomplete_gs.training.trainer import Phase1Trainer
    from archcomplete_gs.semantic.confidence import ConfidenceEstimator

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = OmegaConf.load(args.config)
    cfg.data.scene_path = args.scene

    # Load val dataset
    val_dataset = ArchitecturalSceneDataset(
        scene_path=args.scene, split="val",
        train_split=cfg.data.train_split,
        image_scale=cfg.data.image_scale,
        load_depth=True, load_masks=True,
    )

    # Load model via trainer
    trainer = Phase1Trainer(cfg, val_dataset, val_dataset, device=device)
    trainer.setup()
    trainer.load_checkpoint(args.checkpoint)
    model = trainer.model
    model.eval()

    # LPIPS model
    lpips_fn = None
    if args.use_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="vgg").to(device)
        except ImportError:
            print("[Eval] lpips not installed. Skipping LPIPS.")

    # ── NVS Metrics ─────────────────────────────────────────────────────────
    print(f"[Eval] Evaluating {len(val_dataset)} val views...")
    psnrs, ssims, lpips_vals = [], [], []

    with torch.no_grad():
        for camera in tqdm(val_dataset.get_all_cameras(), desc="NVS eval"):
            camera = camera.to(device)
            render_out = trainer._render(camera)
            pred = render_out["rgb"].clamp(0, 1)
            gt = camera.image.to(device)
            if pred.shape != gt.shape:
                gt = F.interpolate(gt.unsqueeze(0), pred.shape[-2:], mode="bilinear").squeeze(0)
            psnrs.append(compute_psnr(pred, gt))
            ssims.append(compute_ssim(pred, gt))
            if lpips_fn is not None:
                lpips_vals.append(compute_lpips(pred, gt, lpips_fn))

    # ── Semantic Accuracy ────────────────────────────────────────────────────
    print("[Eval] Computing semantic accuracy...")
    sem_ious = []
    label_lifter = trainer.label_lifter
    if label_lifter.has_labels:
        lifted_labels = label_lifter.gaussian_labels
        for camera in tqdm(val_dataset.get_all_cameras(), desc="Semantic eval"):
            if camera.seg_mask is None:
                continue
            camera = camera.to(device)
            rendered_maps = label_lifter.project_semantic_to_views(
                model.means.detach(),
                lifted_labels,
                [camera],
                camera.height, camera.width,
            )
            pred_map = rendered_maps[0].cpu().numpy()
            gt_map = camera.seg_mask.numpy()
            if pred_map.shape != gt_map.shape:
                from PIL import Image
                from PIL import Image as PILImage
                pred_pil = PILImage.fromarray(pred_map.astype(np.uint8)).resize(
                    (gt_map.shape[1], gt_map.shape[0]), PILImage.NEAREST
                )
                pred_map = np.array(pred_pil).astype(np.int64)
            iou_result = compute_semantic_iou(pred_map, gt_map)
            sem_ious.append(iou_result["mean_iou"])

    # ── Planarity Score ──────────────────────────────────────────────────────
    print("[Eval] Computing planarity score...")
    planarity_score = compute_planarity_score(
        model.means.detach(),
        model.semantic_labels.detach(),
        threshold=cfg.training.planarity.planarity_threshold,
    )

    # ── Coverage ─────────────────────────────────────────────────────────────
    print("[Eval] Computing coverage map...")
    conf_estimator = ConfidenceEstimator(
        coverage_voxel_size=cfg.confidence.coverage_voxel_size,
        device=device,
    )
    coverage = conf_estimator.compute_coverage_map(
        model.means.detach(),
        model.opacities.detach().squeeze(-1),
    )

    # ── Assemble Results ─────────────────────────────────────────────────────
    results = {
        "checkpoint": args.checkpoint,
        "scene": args.scene,
        "num_gaussians": model.num_gaussians,
        "nvs": {
            "psnr_mean": float(np.mean(psnrs)),
            "psnr_std": float(np.std(psnrs)),
            "ssim_mean": float(np.mean(ssims)),
            "ssim_std": float(np.std(ssims)),
        },
        "semantic": {
            "mean_iou": float(np.mean(sem_ious)) if sem_ious else None,
            "num_views_evaluated": len(sem_ious),
        },
        "planarity": {
            "score": planarity_score,
            "description": "Fraction of planar-class Gaussians within planarity threshold",
        },
        "coverage": {
            "coverage_pct": coverage["coverage_pct"],
            "voxel_size_m": coverage["resolution"],
            "observed_voxels": int(coverage["observed"].sum()),
            "missing_voxels": int(coverage["missing"].sum()),
            "uncertain_voxels": int(coverage["uncertain"].sum()),
        },
    }
    if lpips_vals:
        results["nvs"]["lpips_mean"] = float(np.mean(lpips_vals))

    # ── Print Summary ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATION RESULTS — ArchComplete-GS Phase 1")
    print("="*60)
    print(f"  PSNR:        {results['nvs']['psnr_mean']:.2f} ± {results['nvs']['psnr_std']:.2f} dB")
    print(f"  SSIM:        {results['nvs']['ssim_mean']:.4f}")
    if "lpips_mean" in results["nvs"]:
        print(f"  LPIPS:       {results['nvs']['lpips_mean']:.4f}")
    if results["semantic"]["mean_iou"] is not None:
        print(f"  Semantic mIoU: {results['semantic']['mean_iou']*100:.1f}%")
    print(f"  Planarity:   {results['planarity']['score']*100:.1f}% pass")
    print(f"  Coverage:    {results['coverage']['coverage_pct']:.1f}% of scene voxels observed")
    print(f"  Gaussians:   {results['num_gaussians']:,}")
    print("="*60)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Eval] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
