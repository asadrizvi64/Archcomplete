"""
archcomplete_gs/semantic/scene_graph.py

Lightweight architectural scene graph built over Gaussian primitives.
Encodes structural relationships between architectural elements:
  - wall_supports_ceiling
  - window_in_wall
  - door_in_wall
  - floor_below_wall
  - beam_connects_column

Used downstream in Stage 2 (gap classification) and Stage 3 (completion conditioning).
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from archcomplete_gs.data.dataset import ArchitecturalSceneDataset


# ─── Relationship Types ───────────────────────────────────────────────────────

RELATIONSHIP_TYPES = [
    "wall_supports_ceiling",
    "window_in_wall",
    "door_in_wall",
    "floor_below_wall",
    "beam_connects_column",
    "spatial_proximity",         # Generic spatial neighbor edge
]
REL_TO_IDX = {r: i for i, r in enumerate(RELATIONSHIP_TYPES)}


# ─── Scene Graph ─────────────────────────────────────────────────────────────

class ArchitecturalSceneGraph:
    """
    Spatial scene graph over Gaussian primitives.

    Nodes: Gaussian clusters (obtained by spatially grouping Gaussians per class)
    Edges: structural or spatial relationships between nodes

    This lightweight representation captures:
      - Which walls border which floors/ceilings
      - Where windows are relative to walls
      - Spatial adjacency between architectural elements

    The graph is used as:
      1. Feature in the gap classifier (Stage 2)
      2. Conditioning signal for the diffusion model (Stage 3)
      3. Structural plausibility check in confidence estimation
    """

    CLASS_IDX = ArchitecturalSceneDataset.CLASS_TO_IDX

    def __init__(
        self,
        k_spatial: int = 8,
        relationship_types: Optional[list[str]] = None,
        cluster_resolution: float = 0.5,   # Grid resolution for clustering (metres)
    ):
        self.k_spatial = k_spatial
        self.relationship_types = relationship_types or RELATIONSHIP_TYPES
        self.cluster_resolution = cluster_resolution

        # Graph data
        self.node_positions: Optional[torch.Tensor] = None   # (M, 3) cluster centers
        self.node_classes: Optional[torch.Tensor] = None     # (M,) class IDs
        self.node_sizes: Optional[torch.Tensor] = None       # (M,) num Gaussians
        self.edge_index: Optional[torch.Tensor] = None       # (2, E) source/target
        self.edge_types: Optional[torch.Tensor] = None       # (E,) relationship type IDs
        self.edge_weights: Optional[torch.Tensor] = None     # (E,) proximity weights

    @property
    def num_nodes(self) -> int:
        return self.node_positions.shape[0] if self.node_positions is not None else 0

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1] if self.edge_index is not None else 0

    def build(
        self,
        means: torch.Tensor,        # (N, 3) Gaussian positions
        sem_labels: torch.Tensor,   # (N,) class labels
    ):
        """
        Build the scene graph from Gaussian positions and semantic labels.

        Step 1: Cluster Gaussians spatially per class → nodes
        Step 2: Add spatial proximity edges (k-NN between cluster centers)
        Step 3: Add semantic relationship edges (wall→ceiling, window→wall, etc.)
        """
        print(f"[SceneGraph] Building from {means.shape[0]:,} Gaussians...")

        # Step 1: Cluster
        node_positions, node_classes, node_sizes = self._cluster_gaussians(means, sem_labels)
        self.node_positions = node_positions
        self.node_classes = node_classes
        self.node_sizes = node_sizes

        if self.num_nodes < 2:
            print("[SceneGraph] Too few nodes to build edges.")
            return

        # Step 2: Spatial proximity edges
        src_spatial, dst_spatial, w_spatial = self._spatial_edges(node_positions)

        # Step 3: Semantic structural edges
        src_sem, dst_sem, type_sem = self._semantic_edges(node_positions, node_classes)

        # Combine all edges
        all_src = torch.cat([src_spatial, src_sem], dim=0)
        all_dst = torch.cat([dst_spatial, dst_sem], dim=0)

        # Edge types: spatial proximity = last index
        spatial_types = torch.full((src_spatial.shape[0],), REL_TO_IDX["spatial_proximity"])
        all_types = torch.cat([spatial_types, type_sem], dim=0)

        # Deduplicate edges
        edge_pairs = torch.stack([all_src, all_dst], dim=1)
        unique_pairs, inv_idx = torch.unique(edge_pairs, dim=0, return_inverse=True)
        # Keep edge type for each unique edge (use first occurrence)
        unique_types = all_types[torch.unique(inv_idx)]

        self.edge_index = unique_pairs.T    # (2, E)
        self.edge_types = unique_types      # (E,)

        # Edge weights from distances
        diffs = node_positions[self.edge_index[0]] - node_positions[self.edge_index[1]]
        dists = diffs.norm(dim=-1)
        self.edge_weights = 1.0 / (dists + 1e-4)

        print(f"[SceneGraph] Built: {self.num_nodes} nodes, {self.num_edges} edges")
        self._print_stats()

    def _cluster_gaussians(
        self,
        means: torch.Tensor,
        sem_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Spatially cluster Gaussians using a grid with resolution `cluster_resolution`.
        Returns per-cluster center, class, and size.
        """
        r = self.cluster_resolution
        voxel_coords = (means / r).long()     # (N, 3) integer grid coords
        # Combine voxel coords and class into a unique key
        cls = sem_labels.unsqueeze(1)
        keys = torch.cat([voxel_coords, cls], dim=1)  # (N, 4)

        unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)

        num_clusters = unique_keys.shape[0]
        cluster_positions = torch.zeros(num_clusters, 3, device=means.device)
        cluster_sizes = torch.zeros(num_clusters, dtype=torch.long, device=means.device)

        # Aggregate means per cluster
        cluster_positions.scatter_add_(0, inverse.unsqueeze(1).expand(-1, 3), means)
        cluster_sizes.scatter_add_(0, inverse, torch.ones(means.shape[0], dtype=torch.long, device=means.device))

        # Average positions
        cluster_positions = cluster_positions / cluster_sizes.float().unsqueeze(1).clamp(min=1)
        cluster_classes = unique_keys[:, 3]

        return cluster_positions, cluster_classes, cluster_sizes

    def _spatial_edges(
        self,
        positions: torch.Tensor,    # (M, 3)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build k-NN spatial proximity edges."""
        M = positions.shape[0]
        k = min(self.k_spatial, M - 1)
        if k <= 0:
            empty = torch.zeros(0, dtype=torch.long, device=positions.device)
            return empty, empty, empty

        dists = torch.cdist(positions, positions)
        dists.fill_diagonal_(float("inf"))
        knn_idx = dists.topk(k, dim=1, largest=False).indices   # (M, k)

        src = torch.arange(M, device=positions.device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = knn_idx.reshape(-1)
        weights = dists[src, dst]

        return src, dst, weights

    def _semantic_edges(
        self,
        positions: torch.Tensor,
        classes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add edges based on architectural rules:
          - wall clusters → ceiling clusters above them (vertical proximity)
          - window clusters → nearest wall cluster
          - door clusters → nearest wall cluster
          - floor clusters → wall clusters above them
          - beam clusters → column clusters
        """
        CLS = self.CLASS_IDX
        srcs, dsts, types_ = [], [], []

        def get_clusters(cls_name: str):
            idx = CLS.get(cls_name, -1)
            if idx < 0:
                return torch.zeros(0, dtype=torch.long, device=positions.device)
            return (classes == idx).nonzero(as_tuple=True)[0]

        def nearest_pair(src_ids, dst_ids, rel_type: str, max_dist: float = 5.0):
            if src_ids.numel() == 0 or dst_ids.numel() == 0:
                return
            src_pos = positions[src_ids]
            dst_pos = positions[dst_ids]
            dists = torch.cdist(src_pos, dst_pos)   # (|src|, |dst|)
            nearest_dst_local = dists.argmin(dim=1)  # (|src|,)
            nearest_dist = dists.min(dim=1).values
            valid = nearest_dist < max_dist
            for s_local, d_local in zip(
                src_ids[valid].tolist(),
                dst_ids[nearest_dst_local[valid]].tolist()
            ):
                srcs.append(s_local)
                dsts.append(d_local)
                types_.append(REL_TO_IDX.get(rel_type, 0))

        walls = get_clusters("wall")
        ceilings = get_clusters("ceiling")
        floors = get_clusters("floor")
        windows = get_clusters("window")
        doors = get_clusters("door")
        beams = get_clusters("beam")
        columns = get_clusters("column")

        nearest_pair(walls, ceilings, "wall_supports_ceiling", max_dist=8.0)
        nearest_pair(windows, walls, "window_in_wall", max_dist=3.0)
        nearest_pair(doors, walls, "door_in_wall", max_dist=3.0)
        nearest_pair(floors, walls, "floor_below_wall", max_dist=8.0)
        nearest_pair(beams, columns, "beam_connects_column", max_dist=10.0)

        if not srcs:
            empty = torch.zeros(0, dtype=torch.long, device=positions.device)
            return empty, empty, empty

        src_t = torch.tensor(srcs, dtype=torch.long, device=positions.device)
        dst_t = torch.tensor(dsts, dtype=torch.long, device=positions.device)
        type_t = torch.tensor(types_, dtype=torch.long, device=positions.device)
        return src_t, dst_t, type_t

    def _print_stats(self):
        classes = ArchitecturalSceneDataset.ARCH_CLASSES
        print("[SceneGraph] Node class distribution:")
        for i, name in enumerate(classes):
            count = (self.node_classes == i).sum().item()
            if count > 0:
                print(f"  {name:12s}: {count} clusters")
        print("[SceneGraph] Edge type distribution:")
        for rel_type, rel_idx in REL_TO_IDX.items():
            count = (self.edge_types == rel_idx).sum().item()
            if count > 0:
                print(f"  {rel_type:30s}: {count} edges")

    def save(self, path: str | Path):
        data = {
            "node_positions": self.node_positions,
            "node_classes": self.node_classes,
            "node_sizes": self.node_sizes,
            "edge_index": self.edge_index,
            "edge_types": self.edge_types,
            "edge_weights": self.edge_weights,
        }
        torch.save(data, path)

    @classmethod
    def load(cls, path: str | Path) -> "ArchitecturalSceneGraph":
        graph = cls()
        data = torch.load(path, map_location="cpu")
        graph.node_positions = data["node_positions"]
        graph.node_classes = data["node_classes"]
        graph.node_sizes = data["node_sizes"]
        graph.edge_index = data["edge_index"]
        graph.edge_types = data["edge_types"]
        graph.edge_weights = data["edge_weights"]
        return graph

    def to_pyg(self):
        """
        Convert to PyTorch Geometric Data object for GNN processing.
        Requires torch_geometric.
        """
        from torch_geometric.data import Data
        return Data(
            x=torch.cat([
                self.node_positions,
                F.one_hot(self.node_classes, ArchitecturalSceneDataset.NUM_CLASSES).float(),
                self.node_sizes.float().unsqueeze(1),
            ], dim=1),
            edge_index=self.edge_index,
            edge_attr=torch.stack([
                self.edge_types.float(),
                self.edge_weights,
            ], dim=1),
        )
