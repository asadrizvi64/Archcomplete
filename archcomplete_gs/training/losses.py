"""
archcomplete_gs/training/losses.py

Loss functions for ArchComplete-GS Phase 1:
  - RGB photometric loss (L1 + SSIM)
  - Depth supervision loss (L1 / SILog)
  - Planarity regularization (novel: enforces flat surfaces on arch plane classes)
  - Normal consistency loss
  - Semantic cross-entropy loss
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from archcomplete_gs.data.dataset import ArchitecturalSceneDataset


# ─── Photometric ─────────────────────────────────────────────────────────────

def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    return torch.abs(pred - target).mean()


def ssim_loss(pred: Tensor, target: Tensor, window_size: int = 11) -> Tensor:
    """Differentiable SSIM. pred/target: (C, H, W) float [0,1]."""
    C, H, W = pred.shape
    if H < window_size or W < window_size:
        return torch.tensor(0.0, device=pred.device)

    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, device=pred.device, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)  # (1,1,W,W)
    window = window.expand(C, -1, -1, -1)

    pred_b = pred.unsqueeze(0)        # (1, C, H, W)
    target_b = target.unsqueeze(0)

    mu_p = F.conv2d(pred_b, window, padding=window_size//2, groups=C)
    mu_t = F.conv2d(target_b, window, padding=window_size//2, groups=C)
    mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p * mu_t

    sigma_p2 = F.conv2d(pred_b**2, window, padding=window_size//2, groups=C) - mu_p2
    sigma_t2 = F.conv2d(target_b**2, window, padding=window_size//2, groups=C) - mu_t2
    sigma_pt = F.conv2d(pred_b*target_b, window, padding=window_size//2, groups=C) - mu_pt

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_pt + C1) * (2*sigma_pt + C2)) / \
               ((mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2))
    return 1.0 - ssim_map.mean()


def photometric_loss(
    pred: Tensor,           # (3, H, W)
    target: Tensor,         # (3, H, W)
    lambda_ssim: float = 0.2,
) -> tuple[Tensor, dict]:
    l1 = l1_loss(pred, target)
    ssim = ssim_loss(pred, target)
    total = (1 - lambda_ssim) * l1 + lambda_ssim * ssim
    return total, {"loss/rgb_l1": l1.item(), "loss/rgb_ssim": ssim.item()}


# ─── Depth Supervision ────────────────────────────────────────────────────────

def depth_l1_loss(pred_depth: Tensor, gt_depth: Tensor) -> Tensor:
    """
    L1 depth loss on valid pixels.
    pred_depth, gt_depth: (H, W) or (1, H, W)
    """
    pred = pred_depth.squeeze()
    gt = gt_depth.squeeze()
    valid = (gt > 0) & torch.isfinite(gt)
    if not valid.any():
        return torch.tensor(0.0, device=pred.device)
    return torch.abs(pred[valid] - gt[valid]).mean()


def silog_loss(pred_depth: Tensor, gt_depth: Tensor, var_lambda: float = 0.15) -> Tensor:
    """
    Scale-invariant log depth loss (SILog). Good when absolute scale is uncertain.
    """
    pred = pred_depth.squeeze().clamp(min=1e-4)
    gt = gt_depth.squeeze()
    valid = (gt > 0) & torch.isfinite(gt)
    if not valid.any():
        return torch.tensor(0.0, device=pred.device)

    log_diff = torch.log(pred[valid]) - torch.log(gt[valid])
    n = valid.sum().float()
    silog = (log_diff ** 2).mean() - var_lambda * (log_diff.mean() ** 2)
    return silog


# ─── Planarity Regularization ─────────────────────────────────────────────────

class PlanarityLoss(nn.Module):
    """
    Planarity regularization for architectural Gaussian primitives.

    Key insight from the proposal: walls, floors, ceilings, and facades
    should lie on flat planes. Standard 3DGS does not enforce this.
    We fit a plane to each Gaussian's local neighborhood and penalize
    deviation from that plane.

    Algorithm:
      1. For each architectural Gaussian, find its k nearest neighbors.
      2. Fit a plane via PCA (smallest eigenvector = normal).
      3. Penalize: mean squared distance of Gaussians from the fitted plane.
    """

    # Classes that receive planarity regularization
    PLANAR_CLASS_IDS = ArchitecturalSceneDataset.PLANAR_CLASS_IDS  # wall, floor, ceiling, facade

    def __init__(
        self,
        k_neighbors: int = 16,
        min_gaussians_for_loss: int = 100,
    ):
        super().__init__()
        self.k = k_neighbors
        self.min_gaussians = min_gaussians_for_loss

    def forward(
        self,
        means: Tensor,               # (N, 3) all Gaussian positions
        sem_labels: Tensor,          # (N,) argmax class IDs
    ) -> Tensor:
        """
        Compute planarity loss for planar-class Gaussians.
        Returns scalar loss value.
        """
        # Select planar Gaussians
        planar_mask = torch.zeros(means.shape[0], dtype=torch.bool, device=means.device)
        for cls_id in self.PLANAR_CLASS_IDS:
            planar_mask |= (sem_labels == cls_id)

        planar_means = means[planar_mask]
        N_planar = planar_means.shape[0]

        if N_planar < self.min_gaussians:
            return torch.tensor(0.0, device=means.device, requires_grad=True)

        # For efficiency, subsample if too many planar Gaussians
        max_pts = 8192
        if N_planar > max_pts:
            sample_idx = torch.randperm(N_planar, device=means.device)[:max_pts]
            planar_means = planar_means[sample_idx]
            N_planar = max_pts

        # KNN in 3D space
        k = min(self.k, N_planar - 1)
        # Pairwise distances (N, N)
        dists = torch.cdist(planar_means, planar_means)   # (N, N)
        dists.fill_diagonal_(float("inf"))
        nn_idx = dists.topk(k, dim=1, largest=False).indices   # (N, k)

        # For each point, gather its k neighbors
        neighbors = planar_means[nn_idx]              # (N, k, 3)

        # PCA-based plane fitting per local neighborhood
        # Center the neighborhood
        center = neighbors.mean(dim=1, keepdim=True)  # (N, 1, 3)
        centered = neighbors - center                  # (N, k, 3)

        # SVD: smallest singular vector = normal direction
        try:
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)  # Vh: (N, 3, 3)
            normals = Vh[:, -1, :]  # (N, 3) — row of Vh = right singular vector
        except Exception:
            return torch.tensor(0.0, device=means.device)

        # Distance of each point to its local plane
        diff = planar_means - center.squeeze(1)       # (N, 3)
        dist_to_plane = (diff * normals).sum(dim=-1)  # (N,) dot product = signed distance
        planarity_loss = (dist_to_plane ** 2).mean()

        return planarity_loss


# ─── Normal Consistency ───────────────────────────────────────────────────────

def normal_consistency_loss(
    normals: Tensor,            # (N, 3) Gaussian normals
    means: Tensor,              # (N, 3) Gaussian positions
    sem_labels: Tensor,         # (N,) class labels
    k: int = 8,
    planar_class_ids: Optional[list] = None,
) -> Tensor:
    """
    Encourage normals of neighboring Gaussians on the same plane class to be consistent.
    """
    if planar_class_ids is None:
        planar_class_ids = ArchitecturalSceneDataset.PLANAR_CLASS_IDS

    planar_mask = torch.zeros(means.shape[0], dtype=torch.bool, device=means.device)
    for cls_id in planar_class_ids:
        planar_mask |= (sem_labels == cls_id)

    if planar_mask.sum() < k + 1:
        return torch.tensor(0.0, device=means.device)

    p_means = means[planar_mask]
    p_normals = normals[planar_mask]

    N = min(p_means.shape[0], 4096)
    if p_means.shape[0] > N:
        idx = torch.randperm(p_means.shape[0], device=means.device)[:N]
        p_means = p_means[idx]
        p_normals = p_normals[idx]

    dists = torch.cdist(p_means, p_means)
    dists.fill_diagonal_(float("inf"))
    k_actual = min(k, N - 1)
    nn_idx = dists.topk(k_actual, dim=1, largest=False).indices   # (N, k)

    # Cosine similarity between normal and its k neighbors' normals
    neighbor_normals = p_normals[nn_idx]                           # (N, k, 3)
    dot = (p_normals.unsqueeze(1) * neighbor_normals).sum(dim=-1)  # (N, k)
    # Want dot = 1 (normals aligned) → loss = 1 - |dot|
    loss = (1.0 - dot.abs()).mean()
    return loss


# ─── Semantic Cross-Entropy ────────────────────────────────────────────────────

def semantic_loss(
    sem_logits: Tensor,         # (N, C) Gaussian semantic logits
    projected_labels: Tensor,   # (N,) pseudo-labels from projected masks
    valid_mask: Optional[Tensor] = None,   # (N,) bool mask for valid pseudo-labels
    label_smoothing: float = 0.05,
) -> Tensor:
    """
    Cross-entropy loss between Gaussian semantic logits and projected 2D labels.
    Only applied to Gaussians that have valid pseudo-labels.
    """
    if valid_mask is not None:
        sem_logits = sem_logits[valid_mask]
        projected_labels = projected_labels[valid_mask]

    if sem_logits.shape[0] == 0:
        return torch.tensor(0.0, device=sem_logits.device)

    return F.cross_entropy(sem_logits, projected_labels, label_smoothing=label_smoothing)


# ─── Combined Loss ────────────────────────────────────────────────────────────

@dataclass_style_dict_result = dict  # just for readability

class ArchCompleteLoss(nn.Module):
    """
    Combined loss for ArchComplete-GS Phase 1 training.
    Aggregates all loss terms with configurable weights.
    """

    def __init__(
        self,
        lambda_rgb_l1: float = 0.8,
        lambda_rgb_ssim: float = 0.2,
        lambda_depth: float = 0.1,
        lambda_planarity: float = 0.05,
        lambda_normal: float = 0.01,
        lambda_semantic: float = 0.5,
        depth_loss_type: str = "l1",        # "l1" or "silog"
        k_neighbors_planarity: int = 16,
    ):
        super().__init__()
        self.lambda_rgb_l1 = lambda_rgb_l1
        self.lambda_rgb_ssim = lambda_rgb_ssim
        self.lambda_depth = lambda_depth
        self.lambda_planarity = lambda_planarity
        self.lambda_normal = lambda_normal
        self.lambda_semantic = lambda_semantic
        self.depth_loss_type = depth_loss_type

        self.planarity = PlanarityLoss(k_neighbors=k_neighbors_planarity)

    def forward(
        self,
        # Rendering outputs
        rendered_rgb: Tensor,               # (3, H, W)
        target_rgb: Tensor,                 # (3, H, W)
        rendered_depth: Optional[Tensor],   # (H, W)
        target_depth: Optional[Tensor],     # (1, H, W)
        # Gaussian state
        means: Tensor,                      # (N, 3)
        normals: Tensor,                    # (N, 3)
        sem_logits: Tensor,                 # (N, C)
        sem_labels: Tensor,                 # (N,) argmax
        # Pseudo-labels from label lifting
        projected_labels: Optional[Tensor] = None,  # (N,) int
        projected_valid: Optional[Tensor] = None,   # (N,) bool
        # Control flags
        use_depth: bool = True,
        use_planarity: bool = True,
        use_semantic: bool = True,
    ) -> tuple[Tensor, dict]:

        logs = {}
        total = torch.tensor(0.0, device=rendered_rgb.device)

        # RGB photometric loss
        rgb_total, rgb_logs = photometric_loss(
            rendered_rgb, target_rgb,
            lambda_ssim=self.lambda_rgb_ssim / (self.lambda_rgb_l1 + self.lambda_rgb_ssim)
        )
        total = total + rgb_total
        logs.update(rgb_logs)
        logs["loss/rgb_total"] = rgb_total.item()

        # Depth supervision
        if use_depth and rendered_depth is not None and target_depth is not None:
            if self.depth_loss_type == "silog":
                d_loss = silog_loss(rendered_depth, target_depth)
            else:
                d_loss = depth_l1_loss(rendered_depth, target_depth)
            total = total + self.lambda_depth * d_loss
            logs["loss/depth"] = d_loss.item()

        # Planarity regularization
        if use_planarity and means.shape[0] > 0:
            plan_loss = self.planarity(means, sem_labels)
            total = total + self.lambda_planarity * plan_loss
            logs["loss/planarity"] = plan_loss.item()

            # Normal consistency
            norm_loss = normal_consistency_loss(normals, means, sem_labels)
            total = total + self.lambda_normal * norm_loss
            logs["loss/normal_consistency"] = norm_loss.item()

        # Semantic cross-entropy
        if use_semantic and projected_labels is not None:
            sem_loss_val = semantic_loss(sem_logits, projected_labels, projected_valid)
            total = total + self.lambda_semantic * sem_loss_val
            logs["loss/semantic"] = sem_loss_val.item()

        logs["loss/total"] = total.item()
        return total, logs
