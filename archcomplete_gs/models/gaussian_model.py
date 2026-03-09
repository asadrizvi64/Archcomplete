"""
archcomplete_gs/models/gaussian_model.py

Semantic Gaussian Model for ArchComplete-GS Phase 1.
Extends standard 3DGS (via gsplat) with:
  - Per-Gaussian semantic label logits (11 architectural classes)
  - Planarity regularization for architectural plane classes
  - Normal estimation from Gaussian covariance
  - Confidence scoring
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─── Gaussian Parameter Container ─────────────────────────────────────────────

class SemanticGaussianModel(nn.Module):
    """
    3D Gaussian model with architectural semantic attributes.
    
    Per-Gaussian learnable parameters:
        _means:    (N, 3) positions
        _quats:    (N, 4) quaternion rotation (w, x, y, z)
        _scales:   (N, 3) log-scales
        _opacities:(N, 1) pre-sigmoid opacity
        _sh0:      (N, 3) DC SH coefficient (base color)
        _shN:      (N, (deg+1)^2 - 1, 3) higher-order SH
        _sem_logits:(N, num_classes) semantic class logits (not activated)
    """

    NUM_CLASSES = 12   # background + 11 arch classes

    def __init__(
        self,
        sh_degree: int = 3,
        num_classes: int = 12,
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.num_classes = num_classes
        self._num_sh = (sh_degree + 1) ** 2

        # Parameters are created on first call to `initialize_from_pointcloud`
        self._means: Optional[nn.Parameter] = None
        self._quats: Optional[nn.Parameter] = None
        self._scales: Optional[nn.Parameter] = None
        self._opacities: Optional[nn.Parameter] = None
        self._sh0: Optional[nn.Parameter] = None
        self._shN: Optional[nn.Parameter] = None
        self._sem_logits: Optional[nn.Parameter] = None

        self._num_gaussians: int = 0

    # ─── Initialization ───────────────────────────────────────────────────────

    def initialize_from_pointcloud(
        self,
        xyz: np.ndarray | Tensor,        # (N, 3) positions
        rgb: np.ndarray | Tensor,         # (N, 3) colors [0,1]
        init_opacity: float = 0.1,
        init_scale_factor: float = 1.0,
    ):
        """Initialize Gaussian parameters from a point cloud (COLMAP or dense)."""
        if isinstance(xyz, np.ndarray):
            xyz = torch.from_numpy(xyz).float()
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb).float()

        N = xyz.shape[0]
        print(f"[GaussianModel] Initializing {N:,} Gaussians from point cloud.")

        # Means
        self._means = nn.Parameter(xyz.clone())

        # Quaternions (identity rotation)
        quats = torch.zeros(N, 4)
        quats[:, 0] = 1.0   # w=1, x=y=z=0
        self._quats = nn.Parameter(quats)

        # Scales: initialize from nearest-neighbor distances
        scales = self._init_scales_from_nn(xyz) * init_scale_factor
        self._scales = nn.Parameter(torch.log(scales + 1e-8))

        # Opacities
        init_op = torch.logit(torch.full((N, 1), init_opacity))
        self._opacities = nn.Parameter(init_op)

        # Spherical harmonics — DC term from RGB
        # SH DC coeff = rgb / (2 * sqrt(pi)) ... inverse of SH eval at center
        sh0_scale = 1.0 / (2.0 * np.sqrt(np.pi))
        sh0 = (rgb.clone() - 0.5) / sh0_scale
        self._sh0 = nn.Parameter(sh0.unsqueeze(1))         # (N, 1, 3)
        self._shN = nn.Parameter(torch.zeros(N, self._num_sh - 1, 3))

        # Semantic logits — uniform init (background slightly favored)
        sem = torch.zeros(N, self.num_classes)
        sem[:, 0] = 0.1
        self._sem_logits = nn.Parameter(sem)

        self._num_gaussians = N

    def _init_scales_from_nn(self, xyz: Tensor, k: int = 3) -> Tensor:
        """Estimate initial Gaussian scale from mean k-nearest neighbor distance."""
        from torch import cdist
        # For large point clouds, subsample for NN computation
        if xyz.shape[0] > 50_000:
            idx = torch.randperm(xyz.shape[0])[:50_000]
            xyz_sub = xyz[idx]
        else:
            xyz_sub = xyz

        dists = cdist(xyz_sub, xyz_sub)    # (M, M)
        dists.fill_diagonal_(float("inf"))
        knn_dists, _ = dists.topk(k, dim=1, largest=False)   # (M, k)
        mean_nn = knn_dists.mean(dim=1)                        # (M,)

        if xyz.shape[0] > 50_000:
            # Assign mean scale to all
            global_scale = mean_nn.mean()
            return torch.full((xyz.shape[0],), global_scale.item())
        return mean_nn.clamp(min=1e-4)

    # ─── Parameter Access ─────────────────────────────────────────────────────

    @property
    def means(self) -> Tensor:
        return self._means

    @property
    def quats(self) -> Tensor:
        return F.normalize(self._quats, dim=-1)

    @property
    def scales(self) -> Tensor:
        return torch.exp(self._scales)

    @property
    def opacities(self) -> Tensor:
        return torch.sigmoid(self._opacities)

    @property
    def sh_coeffs(self) -> Tensor:
        """(N, (deg+1)^2, 3) full SH coefficient tensor."""
        return torch.cat([self._sh0, self._shN], dim=1)

    @property
    def semantic_probs(self) -> Tensor:
        """(N, num_classes) softmax semantic probabilities."""
        return F.softmax(self._sem_logits, dim=-1)

    @property
    def semantic_labels(self) -> Tensor:
        """(N,) argmax semantic class labels."""
        return self._sem_logits.argmax(dim=-1)

    @property
    def num_gaussians(self) -> int:
        return self._num_gaussians

    # ─── Normal Estimation ────────────────────────────────────────────────────

    def compute_normals(self) -> Tensor:
        """
        Estimate surface normals from Gaussian covariance.
        The smallest eigenvector of the covariance matrix is the normal.
        Returns (N, 3) unit normals.
        """
        # Reconstruct covariance from quaternion + scale
        q = self.quats           # (N, 4) normalized
        s = self.scales          # (N, 3)

        # Rotation matrix from quaternion (wxyz)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=-1),
            torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=-1),
            torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=-1),
        ], dim=-2)  # (N, 3, 3)

        # Covariance = R * diag(s^2) * R^T
        S2 = torch.diag_embed(s ** 2)              # (N, 3, 3)
        cov = R @ S2 @ R.transpose(-1, -2)         # (N, 3, 3)

        # Normal = eigenvector of smallest eigenvalue
        # For flat/planar Gaussians, smallest scale = normal direction
        # Fastest: just use the column of R corresponding to min scale
        min_idx = s.argmin(dim=-1)                 # (N,) 0, 1, or 2
        normals = R[torch.arange(R.shape[0]), :, min_idx]  # (N, 3)
        return F.normalize(normals, dim=-1)

    # ─── Densification / Pruning ──────────────────────────────────────────────

    def get_param_groups(self, lr_config: dict) -> list[dict]:
        """Return parameter groups for the optimizer with per-param learning rates."""
        return [
            {"params": [self._means],      "lr": lr_config["means"],      "name": "means"},
            {"params": [self._quats],      "lr": lr_config["quats"],      "name": "quats"},
            {"params": [self._scales],     "lr": lr_config["scales"],     "name": "scales"},
            {"params": [self._opacities],  "lr": lr_config["opacities"],  "name": "opacities"},
            {"params": [self._sh0],        "lr": lr_config["sh0"],        "name": "sh0"},
            {"params": [self._shN],        "lr": lr_config["shN"],        "name": "shN"},
            {"params": [self._sem_logits], "lr": lr_config.get("semantic_features", 1e-3), "name": "sem_logits"},
        ]

    @torch.no_grad()
    def densify_and_prune(
        self,
        grads: Tensor,                   # (N,) per-Gaussian grad magnitude
        grad_threshold: float,
        min_opacity: float,
        extent: float,                   # scene scale
        max_gaussians: int,
    ):
        """
        Standard 3DGS adaptive density control:
          - Clone Gaussians with large gradient and small scale (under-reconstruction)
          - Split Gaussians with large gradient and large scale (over-reconstruction)
          - Prune Gaussians with low opacity
        """
        N = self.num_gaussians
        if N == 0:
            return

        selected = grads > grad_threshold  # (N,)
        scales_max = self.scales.max(dim=-1).values  # (N,)

        # Clone: small Gaussians with high gradient
        clone_mask = selected & (scales_max < 0.01 * extent)
        # Split: large Gaussians with high gradient
        split_mask = selected & (scales_max >= 0.01 * extent)
        # Prune: low opacity
        prune_mask = (self.opacities.squeeze(-1) < min_opacity)

        if clone_mask.any():
            self._clone_gaussians(clone_mask)
        if split_mask.any():
            self._split_gaussians(split_mask)
        if prune_mask.any():
            self._prune_gaussians(prune_mask)

        # Cap total
        if self.num_gaussians > max_gaussians:
            keep_idx = torch.randperm(self.num_gaussians)[:max_gaussians]
            self._select_gaussians(keep_idx)

    @torch.no_grad()
    def _clone_gaussians(self, mask: Tensor):
        new_means = self._means[mask].detach().clone()
        new_quats = self._quats[mask].detach().clone()
        new_scales = self._scales[mask].detach().clone()
        new_ops = self._opacities[mask].detach().clone()
        new_sh0 = self._sh0[mask].detach().clone()
        new_shN = self._shN[mask].detach().clone()
        new_sem = self._sem_logits[mask].detach().clone()
        self._concat_params(new_means, new_quats, new_scales, new_ops, new_sh0, new_shN, new_sem)

    @torch.no_grad()
    def _split_gaussians(self, mask: Tensor, n_splits: int = 2):
        N_split = mask.sum().item()
        scales_orig = self._scales[mask]
        # New scale = original / 1.6 (empirical from 3DGS paper)
        new_scales = scales_orig - torch.log(torch.tensor(1.6))
        # Sample new means by perturbing in scale direction
        normals = self.compute_normals()[mask]
        offset = self.scales[mask].max(dim=-1, keepdim=True).values * normals
        new_means_a = self._means[mask] + offset * 0.5
        new_means_b = self._means[mask] - offset * 0.5
        for nm in [new_means_a, new_means_b]:
            self._concat_params(
                nm.detach(), self._quats[mask].detach(), new_scales.detach(),
                self._opacities[mask].detach(), self._sh0[mask].detach(),
                self._shN[mask].detach(), self._sem_logits[mask].detach()
            )
        # Remove original split Gaussians
        keep = ~mask
        self._select_gaussians(keep.nonzero(as_tuple=True)[0])

    @torch.no_grad()
    def _prune_gaussians(self, mask: Tensor):
        keep = (~mask).nonzero(as_tuple=True)[0]
        self._select_gaussians(keep)

    @torch.no_grad()
    def _concat_params(self, means, quats, scales, ops, sh0, shN, sem):
        def _cat(param_attr, new_val):
            old = getattr(self, param_attr).data
            combined = torch.cat([old, new_val], dim=0)
            setattr(self, param_attr, nn.Parameter(combined))
        _cat("_means", means)
        _cat("_quats", quats)
        _cat("_scales", scales)
        _cat("_opacities", ops)
        _cat("_sh0", sh0)
        _cat("_shN", shN)
        _cat("_sem_logits", sem)
        self._num_gaussians = self._means.shape[0]

    @torch.no_grad()
    def _select_gaussians(self, keep_indices: Tensor):
        self._means = nn.Parameter(self._means.data[keep_indices])
        self._quats = nn.Parameter(self._quats.data[keep_indices])
        self._scales = nn.Parameter(self._scales.data[keep_indices])
        self._opacities = nn.Parameter(self._opacities.data[keep_indices])
        self._sh0 = nn.Parameter(self._sh0.data[keep_indices])
        self._shN = nn.Parameter(self._shN.data[keep_indices])
        self._sem_logits = nn.Parameter(self._sem_logits.data[keep_indices])
        self._num_gaussians = self._means.shape[0]

    @torch.no_grad()
    def reset_opacities(self, reset_value: float = 0.01):
        """Reset opacities periodically (prevents Gaussian bloat)."""
        self._opacities.data = torch.logit(
            torch.clamp(self.opacities, max=reset_value)
        )

    # ─── Export ───────────────────────────────────────────────────────────────

    def save_ply(self, path: str):
        """Export Gaussian model as PLY file (compatible with standard viewers)."""
        from plyfile import PlyData, PlyElement
        import numpy as np

        means = self._means.detach().cpu().numpy()
        opacs = self.opacities.detach().cpu().numpy().squeeze(-1)
        scales = self.scales.detach().cpu().numpy()
        quats = self.quats.detach().cpu().numpy()
        sh = self.sh_coeffs[:, 0].detach().cpu().numpy()  # DC only for PLY
        sem_labels = self.semantic_labels.detach().cpu().numpy()

        vertex_dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            ("semantic_label", "i4"),
        ]
        N = means.shape[0]
        verts = np.zeros(N, dtype=vertex_dtype)
        verts["x"] = means[:, 0]
        verts["y"] = means[:, 1]
        verts["z"] = means[:, 2]
        verts["opacity"] = opacs
        verts["scale_0"] = scales[:, 0]
        verts["scale_1"] = scales[:, 1]
        verts["scale_2"] = scales[:, 2]
        verts["rot_0"] = quats[:, 0]
        verts["rot_1"] = quats[:, 1]
        verts["rot_2"] = quats[:, 2]
        verts["rot_3"] = quats[:, 3]
        verts["f_dc_0"] = sh[:, 0]
        verts["f_dc_1"] = sh[:, 1]
        verts["f_dc_2"] = sh[:, 2]
        verts["semantic_label"] = sem_labels

        el = PlyElement.describe(verts, "vertex")
        PlyData([el]).write(path)
        print(f"[GaussianModel] Saved {N:,} Gaussians to {path}")

    def stats(self) -> dict:
        """Return model statistics for logging."""
        return {
            "num_gaussians": self.num_gaussians,
            "mean_opacity": self.opacities.mean().item(),
            "mean_scale": self.scales.mean().item(),
            "class_distribution": {
                i: (self.semantic_labels == i).sum().item()
                for i in range(self.num_classes)
            },
        }
