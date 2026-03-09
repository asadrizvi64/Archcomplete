"""
scripts/preprocess.py

Phase 1 preprocessing pipeline for ArchComplete-GS.
Runs for a new scene before training:
  1. COLMAP feature extraction + matching + reconstruction (if not done)
  2. Depth Anything V2 — estimate depth for all images
  3. Grounded-SAM2 — segment architectural elements for all images
  4. Align monocular depth to COLMAP scale

Usage:
    python scripts/preprocess.py --scene data/scenes/my_building [--skip-colmap]
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np


def run_colmap(scene_path: Path, image_scale: float = 1.0):
    """
    Run full COLMAP pipeline on a scene directory.
    Assumes images are in scene_path/images/.
    Outputs sparse reconstruction to scene_path/colmap/sparse/0/.
    """
    images_dir = scene_path / "images"
    db_path = scene_path / "colmap" / "database.db"
    sparse_dir = scene_path / "colmap" / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    (scene_path / "colmap").mkdir(exist_ok=True)

    print("[Preprocess] Running COLMAP feature extraction...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "PINHOLE",
        "--SiftExtraction.use_gpu", "1",
    ], check=True)

    print("[Preprocess] Running COLMAP feature matching...")
    subprocess.run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "1",
    ], check=True)

    print("[Preprocess] Running COLMAP sparse reconstruction...")
    subprocess.run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.init_min_tri_angle", "4",
        "--Mapper.multiple_models", "0",
    ], check=True)
    print(f"[Preprocess] COLMAP done. Sparse reconstruction in {sparse_dir}/0/")


def run_depth_estimation(scene_path: Path, image_scale: float = 1.0, scene_type: str = "outdoor"):
    """Run Depth Anything V2 on all images, align to COLMAP scale."""
    from archcomplete_gs.models.depth_estimator import (
        DepthAnythingV2Estimator,
        align_depth_to_colmap,
        extract_colmap_depths_at_pixels,
    )
    from archcomplete_gs.data.colmap_utils import load_colmap_reconstruction

    images_dir = scene_path / "images"
    depths_dir = scene_path / "depths"
    depths_dir.mkdir(exist_ok=True)

    print(f"[Preprocess] Estimating depth (scene_type={scene_type})...")
    estimator = DepthAnythingV2Estimator(use_metric=True, scene_type=scene_type)
    estimator.process_scene(images_dir, depths_dir, image_scale=image_scale)

    # Align depths to COLMAP scale
    print("[Preprocess] Aligning depth to COLMAP scale...")
    recon = load_colmap_reconstruction(scene_path / "colmap")

    aligned_count = 0
    for img in recon.images.values():
        depth_path = depths_dir / (Path(img.name).stem + ".npy")
        if not depth_path.exists():
            continue

        depth = np.load(depth_path)
        pixels, colmap_depths = extract_colmap_depths_at_pixels(
            recon, img.name, depth.shape
        )

        if len(pixels) >= 10:
            from archcomplete_gs.models.depth_estimator import align_depth_to_colmap
            aligned = align_depth_to_colmap(depth, colmap_depths, pixels, method="scale_shift")
            np.save(depth_path, aligned.astype(np.float32))
            aligned_count += 1

    print(f"[Preprocess] Aligned {aligned_count}/{len(recon.images)} depth maps to COLMAP scale.")


def run_segmentation(scene_path: Path, image_scale: float = 1.0):
    """Run Grounded-SAM2 architectural segmentation on all images."""
    from archcomplete_gs.models.segmentor import ArchitecturalSegmentor

    images_dir = scene_path / "images"
    masks_dir = scene_path / "masks"

    print("[Preprocess] Running architectural segmentation (Grounded-SAM2)...")
    segmentor = ArchitecturalSegmentor()
    segmentor.process_scene(images_dir, masks_dir, image_scale=image_scale)
    print(f"[Preprocess] Segmentation masks saved to {masks_dir}")


def validate_scene(scene_path: Path) -> bool:
    """Check that a scene has the minimum required files."""
    required = [
        scene_path / "images",
        scene_path / "colmap" / "sparse",
    ]
    all_ok = True
    for p in required:
        if not p.exists():
            print(f"[Preprocess] Missing: {p}")
            all_ok = False
    if all_ok:
        imgs = list((scene_path / "images").iterdir())
        print(f"[Preprocess] Scene OK: {len(imgs)} images found.")
    return all_ok


def print_scene_summary(scene_path: Path):
    """Print a summary of what's been preprocessed."""
    scene_path = Path(scene_path)
    print("\n" + "="*60)
    print(f"Scene: {scene_path.name}")
    print("="*60)

    imgs = list((scene_path / "images").glob("*"))
    print(f"  Images:    {len(imgs)}")

    depths = list((scene_path / "depths").glob("*.npy")) if (scene_path / "depths").exists() else []
    print(f"  Depths:    {len(depths)} / {len(imgs)}")

    masks = list((scene_path / "masks").glob("*.npy")) if (scene_path / "masks").exists() else []
    print(f"  Seg masks: {len(masks)} / {len(imgs)}")

    colmap_dir = scene_path / "colmap" / "sparse" / "0"
    if colmap_dir.exists():
        try:
            from archcomplete_gs.data.colmap_utils import load_colmap_reconstruction
            recon = load_colmap_reconstruction(scene_path / "colmap")
            print(f"  COLMAP:    {recon.num_images} images, {recon.num_points:,} points")
        except Exception as e:
            print(f"  COLMAP:    Error reading — {e}")
    else:
        print("  COLMAP:    Not found")
    print()


def main():
    parser = argparse.ArgumentParser(description="ArchComplete-GS Phase 1 Preprocessing")
    parser.add_argument("--scene", type=str, required=True, help="Path to scene directory")
    parser.add_argument("--skip-colmap", action="store_true", help="Skip COLMAP (already done)")
    parser.add_argument("--skip-depth", action="store_true", help="Skip depth estimation")
    parser.add_argument("--skip-seg", action="store_true", help="Skip segmentation")
    parser.add_argument("--scene-type", type=str, default="outdoor", choices=["indoor", "outdoor"],
                        help="Scene type for metric depth estimation")
    parser.add_argument("--image-scale", type=float, default=1.0,
                        help="Downscale images before processing (0.5 = half resolution)")
    args = parser.parse_args()

    scene_path = Path(args.scene)
    if not scene_path.exists():
        raise ValueError(f"Scene path does not exist: {scene_path}")

    print(f"\n[Preprocess] Processing scene: {scene_path}")
    print(f"  scene_type={args.scene_type}, image_scale={args.image_scale}\n")

    if not args.skip_colmap:
        run_colmap(scene_path, image_scale=args.image_scale)

    if not validate_scene(scene_path):
        print("[Preprocess] Scene validation failed. Check COLMAP output.")
        return

    if not args.skip_depth:
        run_depth_estimation(scene_path, image_scale=args.image_scale, scene_type=args.scene_type)

    if not args.skip_seg:
        run_segmentation(scene_path, image_scale=args.image_scale)

    print_scene_summary(scene_path)
    print("[Preprocess] ✓ Preprocessing complete! Ready to train.\n")
    print(f"  Run: python scripts/train.py --scene {scene_path} --config configs/phase1.yaml")


if __name__ == "__main__":
    main()
