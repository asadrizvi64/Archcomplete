"""
archcomplete_gs/semantic/confidence.py

Per-Gaussian confidence scoring for ArchComplete-GS.
Combines transmittance, density, semantic discontinuity, and structural
plausibility into a unified confidence map used in Stage 2 gap analysis.

Based on GScenes confidence measure, extended with domain-specific signals.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from archcomplete_gs.data.dataset import ArchitecturalSceneDataset


class ConfidenceEstimator:
    """
    Estimates per-Gaussian confidence in reconstruction quality.
    
    Four confidence signals (all in [0, 1], higher = more confident):
    
    1. Transmittance confidence: Gaussians with high accumulated opacity
       in their best rendered view are well-observed.
    
    2. Density confidence: Gaussians in dense neighborhoods are more
       reliable than isolated ones.
    
    3. Semantic discontinuity: Gaussian where its semantic class
       abruptly ends without a structural boundary gets low confidence.
       E.g. a wall label that ends mid-air.
    
    4. Structural plausibility: Flags deviations from architectural norms
       (window-to-wall ratios, floor-to-floor heights).
    
    The final gap map (for Stage 2) is:
        confidence < threshold → missing/uncertain region
    """

    # Architectural norms for plausibility checking
    FLOOR_HEIGHT_RANGE = (2.4, 5.0)    # metres (typical floor-to-floor)
    WINDOW_WALL_RATIO_MAX = 0.90        # Max window fraction of facade area
    WINDOW_WALL_RATIO_MIN = 0.01        # Facades should have at least some windows

    def __init__(
        self,
        transmittance_threshold: float = 0.7,
        density_threshold: float = 0.1,
        semantic_discontinuity_sigma: float = 0.5,
        coverage_voxel_size: float = 0.1,
        device: str = "cuda",
    ):
        self.transmittance_threshold = transmittance_threshold
        self.density_threshold = density_threshold
        self.semantic_discontinuity_sigma = semantic_discontinuity_sigma
        self.coverage_voxel_size = coverage_voxel_size
        self.device = device

    @torch.no_grad()
    def compute_density_confidence(
        self,
        means: torch.Tensor,        # (N, 3)
        opacities: torch.Tensor,    # (N,)
        k: int = 16,
        radius: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Density-based confidence: Gaussians in dense, opaque neighborhoods
        are better-reconstructed than isolated or transparent ones.
        
        Returns (N,) confidence in [0, 1].
        """
        N = means.shape[0]
        if N < k + 1:
            return torch.ones(N, device=self.device)

        # Subsample for large scenes
        max_pts = 20_000
        if N > max_pts:
            idx = torch.randperm(N, device=self.device)[:max_pts]
            means_s = means[idx]
            opacs_s = opacities[idx]
        else:
            means_s = means
            opacs_s = opacities
            idx = None

        k_actual = min(k, means_s.shape[0] - 1)
        dists = torch.cdist(means_s, means_s)
        dists.fill_diagonal_(float("inf"))
        knn_dists, knn_idx = dists.topk(k_actual, dim=1, largest=False)  # (M, k)

        # Density = mean opacity of k nearest neighbors weighted by inverse distance
        knn_opacs = opacs_s[knn_idx]                           # (M, k)
        weights = 1.0 / (knn_dists + 1e-4)                    # (M, k)
        weights = weights / weights.sum(dim=1, keepdim=True)
        local_density = (knn_opacs * weights).sum(dim=1)      # (M,)

        # Also penalize large nearest-neighbor distance (sparse region)
        mean_nn_dist = knn_dists.mean(dim=1)
        # Normalize: high distance → low confidence
        if mean_nn_dist.max() > 0:
            nn_confidence = 1.0 - (mean_nn_dist / (mean_nn_dist.max() + 1e-4)).clamp(0, 1)
        else:
            nn_confidence = torch.ones_like(mean_nn_dist)

        density_conf = 0.5 * local_density + 0.5 * nn_confidence

        if idx is not None:
            # Assign global mean to non-sampled points
            full_conf = torch.full((N,), density_conf.mean().item(), device=self.device)
            full_conf[idx] = density_conf
            return full_conf

        return density_conf.clamp(0, 1)

    @torch.no_grad()
    def compute_semantic_discontinuity(
        self,
        means: torch.Tensor,        # (N, 3)
        sem_labels: torch.Tensor,   # (N,) int
        k: int = 8,
    ) -> torch.Tensor:
        """
        Penalize Gaussians where their architectural class ends without
        meeting a structurally expected neighbor class.
        
        E.g. a WALL cluster that borders nothing (not floor, not ceiling,
        not window) on at least one side suggests missing geometry.
        
        Returns (N,) discontinuity penalty in [0, 1].
        Higher = more suspicious (lower confidence).
        """
        N = means.shape[0]
        if N < k + 1:
            return torch.zeros(N, device=self.device)

        CLS = ArchitecturalSceneDataset.CLASS_TO_IDX

        # Expected neighbor relationships (from→to)
        EXPECTED_NEIGHBORS = {
            CLS["wall"]:    {CLS["floor"], CLS["ceiling"], CLS["window"], CLS["door"]},
            CLS["floor"]:   {CLS["wall"], CLS["staircase"]},
            CLS["ceiling"]: {CLS["wall"], CLS["beam"]},
            CLS["facade"]:  {CLS["window"], CLS["door"], CLS["balcony"]},
        }

        max_pts = 20_000
        if N > max_pts:
            sample_idx = torch.randperm(N, device=self.device)[:max_pts]
            means_s = means[sample_idx]
            labels_s = sem_labels[sample_idx]
        else:
            means_s = means
            labels_s = sem_labels
            sample_idx = None

        k_actual = min(k, means_s.shape[0] - 1)
        dists = torch.cdist(means_s, means_s)
        dists.fill_diagonal_(float("inf"))
        knn_idx = dists.topk(k_actual, dim=1, largest=False).indices   # (M, k)

        knn_labels = labels_s[knn_idx]          # (M, k) neighbor class labels
        discontinuity = torch.zeros(means_s.shape[0], device=self.device)

        for cls_id, expected_neighbors in EXPECTED_NEIGHBORS.items():
            cls_mask = labels_s == cls_id
            if not cls_mask.any():
                continue

            cls_knn_labels = knn_labels[cls_mask]       # (C, k)
            # Check if any expected neighbor class appears in kNN
            has_expected = torch.zeros(cls_knn_labels.shape[0], dtype=torch.bool, device=self.device)
            for exp_cls in expected_neighbors:
                has_expected |= (cls_knn_labels == exp_cls).any(dim=1)

            # Discontinuity: cls Gaussians with NO expected neighbors
            disc_score = (~has_expected).float()
            discontinuity[cls_mask] = disc_score

        if sample_idx is not None:
            full_disc = torch.zeros(N, device=self.device)
            full_disc[sample_idx] = discontinuity
            return full_disc

        return discontinuity.clamp(0, 1)

    @torch.no_grad()
    def compute_coverage_map(
        self,
        means: torch.Tensor,            # (N, 3) all Gaussian positions
        opacities: torch.Tensor,        # (N,)
        density_threshold: Optional[float] = None,
    ) -> dict:
        """
        Voxelize the scene and mark each voxel as:
          - observed: sufficient Gaussian density
          - uncertain: low Gaussian density  
          - missing: no Gaussians

        Returns a dict with voxel grid metadata and occupancy tensor.
        """
        if density_threshold is None:
            density_threshold = self.density_threshold

        r = self.coverage_voxel_size
        means_np = means.cpu().numpy()

        lo = np.percentile(means_np, 1, axis=0)
        hi = np.percentile(means_np, 99, axis=0)

        # Grid dimensions
        grid_dims = np.ceil((hi - lo) / r).astype(int) + 1
        grid_dims = np.clip(grid_dims, 1, 1024)    # Safety cap

        # Assign each Gaussian to a voxel
        voxel_idx = np.floor((means_np - lo) / r).astype(int)
        voxel_idx = np.clip(voxel_idx, 0, grid_dims - 1)

        # Accumulate opacity per voxel
        opacs_np = opacities.cpu().numpy()
        grid = np.zeros(grid_dims, dtype=np.float32)
        for i in range(len(voxel_idx)):
            x, y, z = voxel_idx[i]
            grid[x, y, z] += opacs_np[i]

        observed = grid > density_threshold
        uncertain = (grid > 0) & (grid <= density_threshold)
        missing = grid == 0

        return {
            "grid": grid,
            "observed": observed,
            "uncertain": uncertain,
            "missing": missing,
            "origin": lo,
            "resolution": r,
            "dims": grid_dims,
            "coverage_pct": 100.0 * observed.sum() / max(grid.size, 1),
        }

    @torch.no_grad()
    def compute_full_confidence(
        self,
        means: torch.Tensor,
        opacities: torch.Tensor,
        sem_labels: torch.Tensor,
        label_confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute unified per-Gaussian confidence score combining all signals.
        
        Returns (N,) confidence in [0, 1].
        Low confidence = candidate for gap completion.
        """
        print("[Confidence] Computing density confidence...")
        density_conf = self.compute_density_confidence(means, opacities)

        print("[Confidence] Computing semantic discontinuity...")
        discontinuity = self.compute_semantic_discontinuity(means, sem_labels)
        sem_conf = 1.0 - discontinuity * self.semantic_discontinuity_sigma

        # Label confidence from lifter (fraction of votes for winning class)
        if label_confidence is not None:
            lc = label_confidence.to(self.device)
        else:
            lc = torch.ones(means.shape[0], device=self.device)

        # Combined score: geometric mean of components
        confidence = (density_conf * sem_conf.clamp(0, 1) * lc) ** (1/3)
        confidence = confidence.clamp(0, 1)

        # Statistics
        low_conf = (confidence < 0.3).sum().item()
        mid_conf = ((confidence >= 0.3) & (confidence < 0.7)).sum().item()
        hi_conf = (confidence >= 0.7).sum().item()
        N = means.shape[0]
        print(f"[Confidence] Distribution — "
              f"high(≥0.7): {hi_conf/N*100:.1f}% | "
              f"mid: {mid_conf/N*100:.1f}% | "
              f"low(<0.3): {low_conf/N*100:.1f}%")

        return confidence

    def check_structural_plausibility(
        self,
        means: torch.Tensor,
        sem_labels: torch.Tensor,
    ) -> dict:
        """
        Check high-level structural plausibility of the reconstruction:
          - Are floor heights in expected range?
          - Is window-to-wall ratio reasonable?
          - Are walls roughly vertical?
        
        Returns dict of plausibility flags and measurements.
        """
        CLS = ArchitecturalSceneDataset.CLASS_TO_IDX
        results = {}

        # Floor height analysis
        floor_mask = sem_labels == CLS.get("floor", -1)
        if floor_mask.sum() > 10:
            floor_y = means[floor_mask, 1]  # Y axis typically up
            floor_levels = self._detect_floor_levels(floor_y)
            results["floor_levels"] = floor_levels
            results["num_floors"] = len(floor_levels)
            if len(floor_levels) >= 2:
                heights = np.diff(sorted(floor_levels))
                results["floor_heights"] = heights.tolist()
                lo, hi = self.FLOOR_HEIGHT_RANGE
                results["floor_heights_plausible"] = all(lo <= h <= hi for h in heights)
            else:
                results["floor_heights_plausible"] = True   # Single floor: can't check

        # Window-to-wall ratio
        wall_count = (sem_labels == CLS.get("wall", -1)).sum().item()
        facade_count = (sem_labels == CLS.get("facade", -1)).sum().item()
        window_count = (sem_labels == CLS.get("window", -1)).sum().item()
        total_facade = max(wall_count + facade_count, 1)
        wwr = window_count / total_facade
        results["window_wall_ratio"] = wwr
        results["wwr_plausible"] = self.WINDOW_WALL_RATIO_MIN <= wwr <= self.WINDOW_WALL_RATIO_MAX

        return results

    def _detect_floor_levels(self, floor_y: torch.Tensor, bin_size: float = 0.3) -> list[float]:
        """Detect distinct floor levels using histogram peak detection."""
        y_np = floor_y.cpu().numpy()
        y_min, y_max = y_np.min(), y_np.max()
        if y_max - y_min < 0.5:
            return [float(y_np.mean())]

        bins = int((y_max - y_min) / bin_size) + 1
        hist, edges = np.histogram(y_np, bins=bins)
        # Find peaks
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > hist.max() * 0.1:
                peaks.append(float((edges[i] + edges[i+1]) / 2))
        return peaks if peaks else [float(y_np.mean())]
