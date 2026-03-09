"""
archcomplete_gs/semantic/label_lifter.py

Lifts 2D architectural segmentation masks into 3D Gaussian space.
Implements the ObjectGS-inspired majority-vote projection:
  For each Gaussian, project it to all training views, sample the
  segmentation label at that pixel, and assign the most frequent label.

This is the core semantic 3DGS extension in the ArchComplete-GS proposal.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from archcomplete_gs.data.dataset import ArchitecturalSceneDataset, Camera


class SemanticLabelLifter:
    """
    Lifts 2D segmentation masks to 3D Gaussian semantic labels via
    majority-vote projection across training views.

    Algorithm (ObjectGS adaptation for architectural scenes):
      1. For each Gaussian g_i with mean μ_i:
         a. Project μ_i into each training view j that observes it.
         b. Sample the 2D label map at the projected pixel.
         c. Record the vote.
      2. Assign the majority vote label to g_i.
      3. Optionally smooth labels using KNN on Gaussian positions.

    The lifter stores:
      - gaussian_labels: (N,) int64 — majority vote labels
      - gaussian_labels_valid: (N,) bool — True if Gaussian had enough votes
      - label_confidence: (N,) float — fraction of votes for winning label
    """

    def __init__(
        self,
        train_dataset: ArchitecturalSceneDataset,
        min_votes: int = 3,
        k_smoothing: int = 8,
        min_visibility_fraction: float = 0.3,   # Fraction of scene extent for frustum check
        device: str = "cuda",
    ):
        self.train_dataset = train_dataset
        self.cameras = train_dataset.get_all_cameras()
        self.num_classes = ArchitecturalSceneDataset.NUM_CLASSES
        self.min_votes = min_votes
        self.k_smoothing = k_smoothing
        self.min_visibility_fraction = min_visibility_fraction
        self.device = device

        # Check which cameras have masks
        self._cameras_with_masks = [
            cam for cam in self.cameras if cam.seg_mask is not None
        ]
        print(f"[LabelLifter] {len(self._cameras_with_masks)}/{len(self.cameras)} cameras have seg masks.")

        self.gaussian_labels: Optional[torch.Tensor] = None         # (N,) int64
        self.gaussian_labels_valid: Optional[torch.Tensor] = None   # (N,) bool
        self.label_confidence: Optional[torch.Tensor] = None        # (N,) float

    @property
    def has_labels(self) -> bool:
        return self.gaussian_labels is not None

    @torch.no_grad()
    def lift(self, means: torch.Tensor) -> torch.Tensor:
        """
        Run majority-vote label lifting on current Gaussian positions.
        
        Args:
            means: (N, 3) Gaussian positions (detached from graph)
        
        Returns:
            labels: (N,) int64 tensor of semantic labels
        """
        if len(self._cameras_with_masks) == 0:
            print("[LabelLifter] No cameras with masks. Skipping label lifting.")
            self.gaussian_labels = torch.zeros(means.shape[0], dtype=torch.int64, device=self.device)
            self.gaussian_labels_valid = torch.zeros(means.shape[0], dtype=torch.bool, device=self.device)
            return self.gaussian_labels

        N = means.shape[0]
        # Accumulate votes: vote_counts[i, c] = number of views that labeled Gaussian i as class c
        vote_counts = torch.zeros(N, self.num_classes, dtype=torch.float32, device=self.device)

        # Process each camera
        for camera in self._cameras_with_masks:
            camera = camera.to(self.device)
            seg_mask = camera.seg_mask.to(self.device)  # (H, W) int64
            H, W = seg_mask.shape

            # Project Gaussian means into this camera
            uv, depths = camera.project(means)          # (N, 2), (N,)

            # Only count Gaussians in front of camera and within image bounds
            in_front = depths > 0.01
            in_bounds = (
                (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) &
                (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
            )
            visible = in_front & in_bounds              # (N,)

            if not visible.any():
                continue

            # Sample segmentation labels at projected pixel locations
            vis_uv = uv[visible]                        # (M, 2)
            # Normalize to [-1, 1] for grid_sample
            norm_u = (vis_uv[:, 0] / (W - 1)) * 2 - 1
            norm_v = (vis_uv[:, 1] / (H - 1)) * 2 - 1
            grid = torch.stack([norm_u, norm_v], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, M, 2)

            # Sample: use nearest neighbor for discrete labels
            mask_float = seg_mask.float().unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
            sampled = F.grid_sample(
                mask_float, grid,
                mode="nearest", align_corners=True
            ).squeeze()                                 # (M,)
            sampled_labels = sampled.long().clamp(0, self.num_classes - 1)

            # Accumulate votes
            vis_indices = visible.nonzero(as_tuple=True)[0]   # (M,)
            vote_counts.scatter_add_(
                0,
                vis_indices.unsqueeze(1).expand(-1, self.num_classes),
                F.one_hot(sampled_labels, self.num_classes).float()
            )

        # Majority vote
        total_votes = vote_counts.sum(dim=1)    # (N,)
        labels = vote_counts.argmax(dim=1)      # (N,) — class with most votes
        max_votes = vote_counts.max(dim=1).values

        # Valid if Gaussian received enough votes
        valid = total_votes >= self.min_votes
        confidence = torch.where(
            total_votes > 0,
            max_votes / total_votes.clamp(min=1),
            torch.zeros_like(total_votes)
        )

        # KNN smoothing: smooth labels using spatial neighborhood
        if self.k_smoothing > 0:
            labels = self._smooth_labels(means, labels, valid, vote_counts)

        self.gaussian_labels = labels
        self.gaussian_labels_valid = valid
        self.label_confidence = confidence

        # Print label distribution
        total_valid = valid.sum().item()
        print(f"[LabelLifter] Lifted labels: {total_valid}/{N} valid Gaussians")
        self._print_label_stats(labels, valid)

        return labels

    def _smooth_labels(
        self,
        means: torch.Tensor,        # (N, 3)
        labels: torch.Tensor,       # (N,) int64
        valid: torch.Tensor,        # (N,) bool
        vote_counts: torch.Tensor,  # (N, C)
    ) -> torch.Tensor:
        """
        Smooth labels by aggregating votes from KNN neighbors in Gaussian space.
        Helps when a Gaussian is only visible in few views.
        """
        N = means.shape[0]
        k = min(self.k_smoothing, N - 1)
        if k <= 0:
            return labels

        # Subsample for efficiency
        max_sample = 50_000
        if N > max_sample:
            return labels   # Skip smoothing for very large models

        # Pairwise distances
        dists = torch.cdist(means, means)
        dists.fill_diagonal_(float("inf"))
        nn_idx = dists.topk(k, dim=1, largest=False).indices    # (N, k)

        # Aggregate neighbor vote counts
        neighbor_votes = vote_counts[nn_idx]                    # (N, k, C)
        aggregated = vote_counts + neighbor_votes.sum(dim=1)    # (N, C)

        smoothed_labels = aggregated.argmax(dim=1)
        return smoothed_labels

    def _print_label_stats(self, labels: torch.Tensor, valid: torch.Tensor):
        classes = ArchitecturalSceneDataset.ARCH_CLASSES
        valid_labels = labels[valid]
        total = max(valid.sum().item(), 1)
        for i, name in enumerate(classes):
            count = (valid_labels == i).sum().item()
            if count > 0:
                pct = 100.0 * count / total
                bar = "█" * int(pct / 2)
                print(f"  {name:12s} {pct:5.1f}% {bar}")

    def project_semantic_to_views(
        self,
        means: torch.Tensor,
        labels: torch.Tensor,
        cameras: list[Camera],
        H: int,
        W: int,
    ) -> list[torch.Tensor]:
        """
        Project Gaussian semantic labels back to 2D views for visualization.
        Returns list of (H, W) label maps.
        """
        rendered_label_maps = []
        with torch.no_grad():
            for camera in cameras:
                camera = camera.to(self.device)
                label_map = torch.zeros(H, W, dtype=torch.int64, device=self.device)

                uv, depths = camera.project(means)
                in_front = depths > 0.01
                in_bounds = (
                    (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) &
                    (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
                )
                visible = in_front & in_bounds

                if visible.any():
                    vis_uv = uv[visible].long()
                    vis_labels = labels[visible]
                    # Z-buffer: only paint if this is front-most Gaussian
                    vis_depths = depths[visible]
                    depth_buf = torch.full((H, W), float("inf"), device=self.device)
                    for j in range(vis_uv.shape[0]):
                        px, py = vis_uv[j, 0].item(), vis_uv[j, 1].item()
                        d = vis_depths[j].item()
                        if d < depth_buf[py, px]:
                            depth_buf[py, px] = d
                            label_map[py, px] = vis_labels[j]

                rendered_label_maps.append(label_map)
        return rendered_label_maps
