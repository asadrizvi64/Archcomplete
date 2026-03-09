"""
archcomplete_gs/data/colmap_utils.py

Utilities for loading COLMAP sparse reconstructions and augmenting
point clouds with Depth Anything V2 monocular depth estimates.
"""

from __future__ import annotations
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ─── COLMAP Data Structures ───────────────────────────────────────────────────

@dataclass
class COLMAPCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray          # fx, fy, cx, cy (+ distortion for non-pinhole)

    @property
    def fx(self): return self.params[0]
    @property
    def fy(self): return self.params[1]
    @property
    def cx(self): return self.params[2]
    @property
    def cy(self): return self.params[3]

    def intrinsic_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx,       0, self.cx],
            [      0, self.fy, self.cy],
            [      0,       0,       1],
        ], dtype=np.float32)


@dataclass
class COLMAPImage:
    id: int
    qvec: np.ndarray            # (4,) quaternion wxyz
    tvec: np.ndarray            # (3,) translation
    camera_id: int
    name: str                   # image filename
    xys: np.ndarray             # (N, 2) 2D keypoints
    point3d_ids: np.ndarray     # (N,) corresponding 3D point IDs (-1 if unmatched)

    def rotation_matrix(self) -> np.ndarray:
        """Convert quaternion (wxyz) to 3x3 rotation matrix."""
        w, x, y, z = self.qvec
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
            [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
            [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ], dtype=np.float32)

    def c2w(self) -> np.ndarray:
        """Camera-to-world transform (4x4)."""
        R = self.rotation_matrix()
        t = self.tvec
        # COLMAP stores world-to-camera: [R | t]
        # c2w = inv([R | t; 0 0 0 1]) = [R^T | -R^T t; 0 0 0 1]
        Rt = R.T
        c = -Rt @ t
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = Rt
        mat[:3, 3] = c
        return mat

    def w2c(self) -> np.ndarray:
        """World-to-camera transform (4x4)."""
        R = self.rotation_matrix()
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = R
        mat[:3, 3] = self.tvec
        return mat


@dataclass
class COLMAPPoint3D:
    id: int
    xyz: np.ndarray             # (3,)
    rgb: np.ndarray             # (3,) uint8
    error: float
    image_ids: np.ndarray
    point2d_idxs: np.ndarray


@dataclass
class COLMAPReconstruction:
    cameras: dict[int, COLMAPCamera]
    images: dict[int, COLMAPImage]
    points3d: dict[int, COLMAPPoint3D]

    @property
    def num_images(self): return len(self.images)

    @property
    def num_points(self): return len(self.points3d)

    def point_cloud(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (xyz, rgb) arrays for the full sparse point cloud."""
        pts = list(self.points3d.values())
        xyz = np.stack([p.xyz for p in pts], axis=0).astype(np.float32)
        rgb = np.stack([p.rgb for p in pts], axis=0).astype(np.float32) / 255.0
        return xyz, rgb


# ─── Binary / Text Parsers ────────────────────────────────────────────────────

def _read_next_bytes(f, num_bytes, fmt):
    data = f.read(num_bytes)
    return struct.unpack(fmt, data)


def read_cameras_binary(path: Path) -> dict[int, COLMAPCamera]:
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = _read_next_bytes(f, 8, "<Q")[0]
        for _ in range(num_cameras):
            cam_id, model_id, width, height = _read_next_bytes(f, 24, "<iiQQ")
            model_names = {
                0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE",
            }
            model = model_names.get(model_id, f"UNKNOWN_{model_id}")
            num_params = {
                "SIMPLE_PINHOLE": 3, "PINHOLE": 4,
                "SIMPLE_RADIAL": 4, "RADIAL": 5, "OPENCV": 8,
            }.get(model, 4)
            params = np.array(_read_next_bytes(f, 8 * num_params, "<" + "d" * num_params))
            cameras[cam_id] = COLMAPCamera(cam_id, model, width, height, params.astype(np.float32))
    return cameras


def read_images_binary(path: Path) -> dict[int, COLMAPImage]:
    images = {}
    with open(path, "rb") as f:
        num_images = _read_next_bytes(f, 8, "<Q")[0]
        for _ in range(num_images):
            img_id, *qtvec_data = _read_next_bytes(f, 64, "<i4d3d")
            qvec = np.array(qtvec_data[:4], dtype=np.float32)
            tvec = np.array(qtvec_data[4:], dtype=np.float32)
            camera_id = _read_next_bytes(f, 4, "<i")[0]
            name = b""
            c = f.read(1)
            while c != b"\x00":
                name += c
                c = f.read(1)
            name = name.decode("utf-8")
            num_points2d = _read_next_bytes(f, 8, "<Q")[0]
            xys_ids = _read_next_bytes(f, 24 * num_points2d, "<" + "ddq" * num_points2d)
            xys = np.array(xys_ids[0::3] + xys_ids[1::3]).reshape(2, -1).T.astype(np.float32)
            # reshape properly
            flat = list(xys_ids)
            xys = np.array([(flat[i], flat[i+1]) for i in range(0, len(flat), 3)], dtype=np.float32)
            p3d_ids = np.array([flat[i+2] for i in range(0, len(flat), 3)], dtype=np.int64)
            images[img_id] = COLMAPImage(img_id, qvec, tvec, camera_id, name, xys, p3d_ids)
    return images


def read_points3d_binary(path: Path) -> dict[int, COLMAPPoint3D]:
    points3d = {}
    with open(path, "rb") as f:
        num_points = _read_next_bytes(f, 8, "<Q")[0]
        for _ in range(num_points):
            pt_id, x, y, z, r, g, b, error = _read_next_bytes(f, 43, "<Q3d3Bd")
            track_length = _read_next_bytes(f, 8, "<Q")[0]
            track_data = _read_next_bytes(f, 8 * track_length, "<" + "ii" * track_length)
            image_ids = np.array(track_data[0::2], dtype=np.int32)
            point2d_idxs = np.array(track_data[1::2], dtype=np.int32)
            points3d[pt_id] = COLMAPPoint3D(
                pt_id,
                np.array([x, y, z], dtype=np.float32),
                np.array([r, g, b], dtype=np.uint8),
                error,
                image_ids,
                point2d_idxs,
            )
    return points3d


def load_colmap_reconstruction(colmap_dir: Path) -> COLMAPReconstruction:
    """
    Load a COLMAP sparse reconstruction from a directory.
    Expects: cameras.bin, images.bin, points3D.bin
    (or cameras.txt, images.txt, points3D.txt for text format)
    """
    colmap_dir = Path(colmap_dir)
    sparse_dir = colmap_dir / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = colmap_dir

    bin_cameras = sparse_dir / "cameras.bin"
    bin_images = sparse_dir / "images.bin"
    bin_points = sparse_dir / "points3D.bin"

    if bin_cameras.exists():
        cameras = read_cameras_binary(bin_cameras)
        images = read_images_binary(bin_images)
        points3d = read_points3d_binary(bin_points)
    else:
        raise NotImplementedError("Text format COLMAP not yet implemented. Use binary format.")

    print(f"[COLMAP] Loaded {len(cameras)} cameras, {len(images)} images, {len(points3d)} 3D points")
    return COLMAPReconstruction(cameras, images, points3d)


# ─── Depth-Augmented Point Cloud ─────────────────────────────────────────────

def augment_with_depth(
    reconstruction: COLMAPReconstruction,
    depth_maps: dict[str, np.ndarray],          # image_name -> (H, W) depth array
    images_dir: Path,
    depth_scale: float = 1.0,
    subsample: int = 8,                          # Subsample depth pixels for speed
    min_depth: float = 0.1,
    max_depth: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project depth maps to 3D and merge with COLMAP sparse points.
    Returns augmented (xyz, rgb) arrays.
    
    This is the core init improvement over vanilla COLMAP sparse initialization:
    dense depth priors fill gaps that COLMAP misses (textureless walls, etc.).
    """
    from PIL import Image

    sparse_xyz, sparse_rgb = reconstruction.point_cloud()
    dense_pts_xyz = [sparse_xyz]
    dense_pts_rgb = [sparse_rgb]

    for img_id, colmap_img in reconstruction.images.items():
        if colmap_img.name not in depth_maps:
            continue

        depth = depth_maps[colmap_img.name] * depth_scale
        camera = reconstruction.cameras[colmap_img.camera_id]

        # Load corresponding color image
        img_path = images_dir / colmap_img.name
        if not img_path.exists():
            continue
        color = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        H, W = depth.shape

        # Resize color to match depth if needed
        if color.shape[:2] != (H, W):
            from PIL import Image as PILImage
            color = np.array(PILImage.fromarray((color * 255).astype(np.uint8)).resize(
                (W, H), PILImage.BILINEAR
            )).astype(np.float32) / 255.0

        # Build pixel grid (subsampled)
        ys = np.arange(0, H, subsample)
        xs = np.arange(0, W, subsample)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        d = depth[grid_y, grid_x]
        valid = (d > min_depth) & (d < max_depth)
        grid_x, grid_y, d = grid_x[valid], grid_y[valid], d[valid]

        # Unproject to camera space
        fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
        # Scale intrinsics if depth was computed at different resolution
        scale_x = W / camera.width
        scale_y = H / camera.height
        fx_s, fy_s = fx * scale_x, fy * scale_y
        cx_s, cy_s = cx * scale_x, cy * scale_y

        x_cam = (grid_x - cx_s) * d / fx_s
        y_cam = (grid_y - cy_s) * d / fy_s
        z_cam = d
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

        # Transform to world space
        c2w = colmap_img.c2w()
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        pts_world = pts_cam @ R_c2w.T + t_c2w

        # Colors
        colors = color[grid_y, grid_x]  # (N, 3)

        dense_pts_xyz.append(pts_world.astype(np.float32))
        dense_pts_rgb.append(colors.astype(np.float32))

    xyz = np.concatenate(dense_pts_xyz, axis=0)
    rgb = np.concatenate(dense_pts_rgb, axis=0)
    print(f"[Depth Augmentation] {sparse_xyz.shape[0]:,} sparse → {xyz.shape[0]:,} augmented points")
    return xyz, rgb


def compute_scene_bounds(xyz: np.ndarray, percentile: float = 99.5) -> tuple[np.ndarray, np.ndarray]:
    """Robust scene bounds excluding outliers."""
    lo = np.percentile(xyz, 100 - percentile, axis=0)
    hi = np.percentile(xyz, percentile, axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)
