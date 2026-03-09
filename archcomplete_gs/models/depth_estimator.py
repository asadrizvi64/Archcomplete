"""
archcomplete_gs/models/depth_estimator.py

Depth Anything V2 wrapper for monocular depth estimation.
Supports both relative and metric depth modes.
Handles batched inference and caching to disk.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DepthAnythingV2Estimator:
    """
    Wrapper around Depth Anything V2 (HuggingFace transformers API).

    Relative depth: normalized 0-1 inverse depth (faster, no calibration needed)
    Metric depth: absolute depth in metres (requires scene_type: indoor/outdoor)
    
    Usage:
        estimator = DepthAnythingV2Estimator(use_metric=True, scene_type="outdoor")
        depth = estimator.estimate(image_pil)  # returns (H, W) float32 in metres
        estimator.process_scene(images_dir, output_dir)  # batch process + cache
    """

    MODEL_MAP = {
        # (use_metric, scene_type) -> HF model ID
        (False, None):       "depth-anything/Depth-Anything-V2-Large-hf",
        (True, "indoor"):    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        (True, "outdoor"):   "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    }

    def __init__(
        self,
        use_metric: bool = True,
        scene_type: str = "outdoor",    # "indoor" or "outdoor"
        device: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.use_metric = use_metric
        self.scene_type = scene_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_name is None:
            key = (use_metric, scene_type if use_metric else None)
            model_name = self.MODEL_MAP[key]
        self.model_name = model_name

        self._pipe = None  # Lazy-loaded

    def _load(self):
        if self._pipe is not None:
            return
        print(f"[DepthEstimator] Loading {self.model_name} ...")
        from transformers import pipeline
        self._pipe = pipeline(
            task="depth-estimation",
            model=self.model_name,
            device=self.device,
        )
        print(f"[DepthEstimator] Model loaded on {self.device}")

    @torch.no_grad()
    def estimate(self, image: Image.Image) -> np.ndarray:
        """
        Estimate depth for a single PIL image.
        Returns (H, W) float32 depth map.
        - If metric: values are metres
        - If relative: values are 0-1 disparity (inverse depth)
        """
        self._load()
        result = self._pipe(image)
        depth = np.array(result["depth"]).astype(np.float32)
        return depth

    @torch.no_grad()
    def estimate_batch(self, images: list[Image.Image], batch_size: int = 4) -> list[np.ndarray]:
        """Batch depth estimation."""
        self._load()
        depths = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            results = self._pipe(batch)
            for r in results:
                depths.append(np.array(r["depth"]).astype(np.float32))
        return depths

    def process_scene(
        self,
        images_dir: Path,
        output_dir: Path,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".PNG"),
        skip_existing: bool = True,
        image_scale: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """
        Process all images in a directory. Saves .npy depth maps.
        Returns dict mapping image filename -> depth array.
        """
        from tqdm import tqdm

        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted([p for p in images_dir.iterdir() if p.suffix in extensions])
        print(f"[DepthEstimator] Processing {len(image_paths)} images → {output_dir}")

        depths = {}
        for img_path in tqdm(image_paths, desc="Depth estimation"):
            out_path = output_dir / (img_path.stem + ".npy")
            if skip_existing and out_path.exists():
                depths[img_path.name] = np.load(out_path)
                continue

            pil_img = Image.open(img_path).convert("RGB")
            if image_scale != 1.0:
                W, H = pil_img.size
                pil_img = pil_img.resize((int(W * image_scale), int(H * image_scale)), Image.LANCZOS)

            depth = self.estimate(pil_img)
            np.save(out_path, depth)
            depths[img_path.name] = depth

        print(f"[DepthEstimator] Done. Depth range: "
              f"[{min(d.min() for d in depths.values()):.2f}, "
              f"{max(d.max() for d in depths.values()):.2f}] m")
        return depths


def align_depth_to_colmap(
    depth_map: np.ndarray,
    colmap_depths: np.ndarray,   # Sparse depth values from COLMAP at known pixels
    colmap_pixels: np.ndarray,   # (N, 2) pixel coords of COLMAP points
    method: str = "scale_shift",
) -> np.ndarray:
    """
    Align monocular depth map to COLMAP sparse depths.
    
    Monocular depth is up-to-scale (relative) or has metric error.
    This function fits a scale (and optionally shift) to align it with
    the sparse but accurate COLMAP depths.
    
    Args:
        depth_map: (H, W) estimated depth
        colmap_depths: (N,) COLMAP depth values at sample points
        colmap_pixels: (N, 2) pixel coordinates of COLMAP 3D points
        method: "scale_only" or "scale_shift"
    
    Returns:
        Aligned depth map (H, W)
    """
    H, W = depth_map.shape
    px = colmap_pixels[:, 0].clip(0, W-1).astype(int)
    py = colmap_pixels[:, 1].clip(0, H-1).astype(int)
    mono_vals = depth_map[py, px]

    # Filter invalid
    valid = (mono_vals > 0) & (colmap_depths > 0)
    mono_vals = mono_vals[valid]
    colmap_vals = colmap_depths[valid]

    if len(mono_vals) < 10:
        print("[DepthAlign] Warning: too few correspondences for alignment, returning raw depth.")
        return depth_map

    if method == "scale_only":
        # Least-squares scale: min ||s * mono - colmap||^2
        scale = np.dot(mono_vals, colmap_vals) / (np.dot(mono_vals, mono_vals) + 1e-8)
        return depth_map * scale

    elif method == "scale_shift":
        # min ||s * mono + t - colmap||^2
        A = np.stack([mono_vals, np.ones_like(mono_vals)], axis=1)
        b = colmap_vals
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        scale, shift = result
        aligned = depth_map * scale + shift
        return aligned.clip(0.01)

    else:
        raise ValueError(f"Unknown alignment method: {method}")


def extract_colmap_depths_at_pixels(
    reconstruction,
    image_name: str,
    depth_map_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract COLMAP depth values at pixel locations for a given image.
    Used to align monocular depth estimates to COLMAP scale.
    
    Returns:
        pixels: (N, 2) pixel coordinates
        depths: (N,) depth values
    """
    H, W = depth_map_shape

    # Find the COLMAP image
    target_img = None
    for img in reconstruction.images.values():
        if img.name == image_name:
            target_img = img
            break
    if target_img is None:
        return np.zeros((0, 2)), np.zeros(0)

    camera = reconstruction.cameras[target_img.camera_id]
    w2c = target_img.w2c()

    pixels = []
    depths = []
    for pt_id in target_img.point3d_ids:
        if pt_id == -1 or pt_id not in reconstruction.points3d:
            continue
        pt3d = reconstruction.points3d[pt_id].xyz

        # Project to camera
        pt_cam = w2c[:3, :3] @ pt3d + w2c[:3, 3]
        z = pt_cam[2]
        if z <= 0:
            continue

        # Scale intrinsics to depth map size
        scale_x = W / camera.width
        scale_y = H / camera.height
        u = pt_cam[0] / z * camera.fx * scale_x + camera.cx * scale_x
        v = pt_cam[1] / z * camera.fy * scale_y + camera.cy * scale_y

        if 0 <= u < W and 0 <= v < H:
            pixels.append([u, v])
            depths.append(z)

    if len(pixels) == 0:
        return np.zeros((0, 2)), np.zeros(0)

    return np.array(pixels, dtype=np.float32), np.array(depths, dtype=np.float32)
