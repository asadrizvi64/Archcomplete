"""
archcomplete_gs/data/dataset.py

PyTorch Dataset for architectural scenes with precomputed depth and
segmentation masks. Supports COLMAP-organized scenes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# ─── Camera ──────────────────────────────────────────────────────────────────

@dataclass
class Camera:
    """A single calibrated camera / training view."""
    image_id: int
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: torch.Tensor           # (4, 4) camera-to-world
    image: torch.Tensor         # (3, H, W) float32 [0,1]
    depth: Optional[torch.Tensor] = None    # (1, H, W) float32 metres
    seg_mask: Optional[torch.Tensor] = None # (H, W) int64 class indices

    @property
    def device(self):
        return self.image.device

    def to(self, device):
        self.c2w = self.c2w.to(device)
        self.image = self.image.to(device)
        if self.depth is not None:
            self.depth = self.depth.to(device)
        if self.seg_mask is not None:
            self.seg_mask = self.seg_mask.to(device)
        return self

    @property
    def w2c(self) -> torch.Tensor:
        return torch.linalg.inv(self.c2w)

    def intrinsic_matrix(self) -> torch.Tensor:
        return torch.tensor([
            [self.fx,       0, self.cx],
            [      0, self.fy, self.cy],
            [      0,       0,       1],
        ], dtype=torch.float32, device=self.c2w.device)

    def project(self, xyz_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D world points to 2D pixel coords.
        Args:
            xyz_world: (N, 3)
        Returns:
            uv: (N, 2) pixel coordinates
            depth: (N,) depth values
        """
        w2c = self.w2c
        # To camera space
        ones = torch.ones(xyz_world.shape[0], 1, device=xyz_world.device, dtype=xyz_world.dtype)
        pts_h = torch.cat([xyz_world, ones], dim=-1)  # (N, 4)
        pts_cam = (w2c @ pts_h.T).T                   # (N, 4)
        x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
        u = (x / z.clamp(min=1e-6)) * self.fx + self.cx
        v = (y / z.clamp(min=1e-6)) * self.fy + self.cy
        return torch.stack([u, v], dim=-1), z


# ─── Scene Dataset ────────────────────────────────────────────────────────────

class ArchitecturalSceneDataset(Dataset):
    """
    Dataset for a single architectural scene. Loads images, depth maps,
    and segmentation masks aligned to COLMAP camera poses.

    Directory layout expected:
        scene_path/
            images/          ← RGB images
            colmap/          ← COLMAP sparse reconstruction
            depths/          ← Depth Anything V2 outputs (*.npy)
            masks/           ← Grounded-SAM label maps (*.npy, int8)
    """

    # Canonical architectural label set (11 classes + background)
    ARCH_CLASSES = [
        "background",   # 0
        "wall",         # 1
        "floor",        # 2
        "ceiling",      # 3
        "window",       # 4
        "door",         # 5
        "column",       # 6
        "beam",         # 7
        "staircase",    # 8
        "roof",         # 9
        "facade",       # 10
        "balcony",      # 11
    ]
    NUM_CLASSES = len(ARCH_CLASSES)
    CLASS_TO_IDX = {c: i for i, c in enumerate(ARCH_CLASSES)}

    # Classes that receive planarity regularization
    PLANAR_CLASSES = {"wall", "floor", "ceiling", "facade"}
    PLANAR_CLASS_IDS = [CLASS_TO_IDX[c] for c in PLANAR_CLASSES if c in CLASS_TO_IDX]

    def __init__(
        self,
        scene_path: str | Path,
        split: str = "train",
        train_split: float = 0.8,
        image_scale: float = 1.0,
        white_background: bool = False,
        load_depth: bool = True,
        load_masks: bool = True,
    ):
        self.scene_path = Path(scene_path)
        self.split = split
        self.image_scale = image_scale
        self.white_background = white_background
        self.load_depth = load_depth
        self.load_masks = load_masks

        self._cameras: list[Camera] = []
        self._load_scene(train_split)

    def _load_scene(self, train_split: float):
        """Load COLMAP reconstruction and build camera list."""
        from archcomplete_gs.data.colmap_utils import load_colmap_reconstruction

        recon = load_colmap_reconstruction(self.scene_path / "colmap")
        self.reconstruction = recon

        # Sort images by name for reproducible splits
        sorted_images = sorted(recon.images.values(), key=lambda x: x.name)
        n_total = len(sorted_images)
        n_train = int(n_total * train_split)

        if self.split == "train":
            selected = sorted_images[:n_train]
        elif self.split == "val":
            selected = sorted_images[n_train:]
        else:
            selected = sorted_images

        for colmap_img in selected:
            camera = recon.cameras[colmap_img.camera_id]
            img_path = self.scene_path / "images" / colmap_img.name

            if not img_path.exists():
                print(f"[Dataset] Warning: {img_path} not found, skipping.")
                continue

            # Load image
            pil_img = Image.open(img_path).convert("RGB")
            W_orig, H_orig = pil_img.size

            if self.image_scale != 1.0:
                W_new = int(W_orig * self.image_scale)
                H_new = int(H_orig * self.image_scale)
                pil_img = pil_img.resize((W_new, H_new), Image.LANCZOS)
            else:
                W_new, H_new = W_orig, H_orig

            img_tensor = torch.from_numpy(
                np.array(pil_img).astype(np.float32) / 255.0
            ).permute(2, 0, 1)  # (3, H, W)

            if self.white_background:
                img_tensor = img_tensor.clamp(0, 1)

            # Scale intrinsics
            scale_x = W_new / camera.width
            scale_y = H_new / camera.height
            fx = camera.fx * scale_x
            fy = camera.fy * scale_y
            cx = camera.cx * scale_x
            cy = camera.cy * scale_y

            # c2w matrix
            c2w = torch.from_numpy(colmap_img.c2w())

            # Load depth
            depth_tensor = None
            if self.load_depth:
                depth_path = self.scene_path / "depths" / (Path(colmap_img.name).stem + ".npy")
                if depth_path.exists():
                    depth_np = np.load(depth_path).astype(np.float32)
                    # Resize to match image
                    if depth_np.shape != (H_new, W_new):
                        depth_pil = Image.fromarray(depth_np).resize((W_new, H_new), Image.NEAREST)
                        depth_np = np.array(depth_pil)
                    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)  # (1, H, W)

            # Load segmentation mask
            seg_tensor = None
            if self.load_masks:
                mask_path = self.scene_path / "masks" / (Path(colmap_img.name).stem + ".npy")
                if mask_path.exists():
                    mask_np = np.load(mask_path).astype(np.int64)
                    if mask_np.shape != (H_new, W_new):
                        mask_pil = Image.fromarray(mask_np.astype(np.uint8)).resize(
                            (W_new, H_new), Image.NEAREST
                        )
                        mask_np = np.array(mask_pil).astype(np.int64)
                    seg_tensor = torch.from_numpy(mask_np)  # (H, W)

            cam = Camera(
                image_id=colmap_img.id,
                image_name=colmap_img.name,
                width=W_new,
                height=H_new,
                fx=fx, fy=fy, cx=cx, cy=cy,
                c2w=c2w,
                image=img_tensor,
                depth=depth_tensor,
                seg_mask=seg_tensor,
            )
            self._cameras.append(cam)

        print(f"[Dataset] Loaded {len(self._cameras)} cameras for split='{self.split}'")

    def __len__(self) -> int:
        return len(self._cameras)

    def __getitem__(self, idx: int) -> Camera:
        return self._cameras[idx]

    def get_point_cloud(self) -> tuple[np.ndarray, np.ndarray]:
        """Return sparse COLMAP point cloud (xyz, rgb)."""
        return self.reconstruction.point_cloud()

    def get_all_cameras(self) -> list[Camera]:
        return self._cameras

    def random_camera(self) -> Camera:
        idx = torch.randint(len(self._cameras), (1,)).item()
        return self._cameras[idx]
