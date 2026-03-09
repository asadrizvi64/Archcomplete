"""
archcomplete_gs/training/trainer.py

Main training loop for ArchComplete-GS Phase 1.
Orchestrates:
  - 3DGS optimization via gsplat
  - Periodic semantic label lifting
  - Densification and pruning
  - Depth supervision
  - Evaluation and checkpointing
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from archcomplete_gs.data.dataset import ArchitecturalSceneDataset, Camera
from archcomplete_gs.models.gaussian_model import SemanticGaussianModel
from archcomplete_gs.training.losses import ArchCompleteLoss
from archcomplete_gs.semantic.label_lifter import SemanticLabelLifter
from archcomplete_gs.semantic.scene_graph import ArchitecturalSceneGraph
from archcomplete_gs.semantic.confidence import ConfidenceEstimator


class Phase1Trainer:
    """
    Orchestrates Phase 1 training: semantic-aware 3DGS reconstruction.

    Training flow per iteration:
      1. Sample a random camera
      2. Render RGB + depth via gsplat
      3. Compute losses (photometric + depth + planarity + semantic)
      4. Backward + optimizer step
      5. Periodic: densify/prune, lift labels, reset opacities
      6. Periodic: evaluate + save checkpoint

    Usage:
        trainer = Phase1Trainer(cfg, train_dataset, val_dataset)
        trainer.setup()
        trainer.train()
    """

    def __init__(
        self,
        cfg,                         # OmegaConf config
        train_dataset: ArchitecturalSceneDataset,
        val_dataset: ArchitecturalSceneDataset,
        device: str = "cuda",
    ):
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.output_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[SemanticGaussianModel] = None
        self.optimizer: Optional[Adam] = None
        self.loss_fn: Optional[ArchCompleteLoss] = None
        self.label_lifter: Optional[SemanticLabelLifter] = None

        self.step: int = 0
        self.grad_accum: Optional[torch.Tensor] = None  # Per-Gaussian gradient accumulator

        # Experiment tracking
        self._wandb = None
        if cfg.experiment.wandb:
            self._init_wandb()

    def _init_wandb(self):
        try:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=self.cfg.experiment.wandb_project,
                name=self.cfg.experiment.name,
                config=dict(self.cfg),
            )
        except ImportError:
            print("[Trainer] wandb not installed, skipping.")

    def setup(self):
        """Initialize model, optimizer, and auxiliary components."""
        torch.manual_seed(self.cfg.experiment.seed)
        np.random.seed(self.cfg.experiment.seed)

        # Initialize Gaussian model
        self.model = SemanticGaussianModel(
            sh_degree=self.cfg.gaussian.sh_degree,
            num_classes=self.cfg.gaussian.num_semantic_classes + 1,
        ).to(self.device)

        # Get initial point cloud
        xyz, rgb = self.train_dataset.get_point_cloud()
        # Optionally augment with depth (from cached depth maps)
        xyz, rgb = self._augment_with_depth(xyz, rgb)

        xyz_t = torch.from_numpy(xyz).to(self.device)
        rgb_t = torch.from_numpy(rgb).to(self.device)
        self.model.initialize_from_pointcloud(xyz_t, rgb_t)

        # Optimizer
        lr = self.cfg.training.lr
        param_groups = self.model.get_param_groups({
            "means":      lr.means,
            "quats":      lr.quats,
            "scales":     lr.scales,
            "opacities":  lr.opacities,
            "sh0":        lr.sh0,
            "shN":        lr.shN,
            "semantic_features": lr.semantic_features,
        })
        self.optimizer = Adam(param_groups, lr=0.0, eps=1e-15)

        # Gradient accumulator for densification
        self.grad_accum = torch.zeros(
            self.model.num_gaussians, device=self.device
        )

        # Loss
        loss_cfg = self.cfg.training.loss
        self.loss_fn = ArchCompleteLoss(
            lambda_rgb_l1=loss_cfg.rgb_l1,
            lambda_rgb_ssim=loss_cfg.rgb_ssim,
            lambda_depth=loss_cfg.depth_supervision,
            lambda_planarity=loss_cfg.planarity,
            lambda_normal=loss_cfg.normal_consistency,
            lambda_semantic=loss_cfg.semantic_ce,
            depth_loss_type=self.cfg.depth.depth_loss_type,
            k_neighbors_planarity=self.cfg.training.planarity.k_neighbors,
        )

        # Label lifter
        self.label_lifter = SemanticLabelLifter(
            train_dataset=self.train_dataset,
            min_votes=self.cfg.label_lifting.min_votes,
            k_smoothing=self.cfg.label_lifting.smoothing_k,
            device=self.device,
        )

        print(f"[Trainer] Setup complete. {self.model.num_gaussians:,} Gaussians.")
        print(f"[Trainer] Output dir: {self.output_dir}")

    def _augment_with_depth(
        self, xyz: np.ndarray, rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Augment COLMAP sparse points with back-projected depth map points."""
        depths_dir = Path(self.cfg.data.scene_path) / self.cfg.data.depth_dir
        if not depths_dir.exists():
            print("[Trainer] No depth dir found, using COLMAP sparse only.")
            return xyz, rgb

        from archcomplete_gs.data.colmap_utils import augment_with_depth
        recon = self.train_dataset.reconstruction
        image_names = [img.name for img in recon.images.values()]
        depth_maps = {}
        for name in image_names:
            dp = depths_dir / (Path(name).stem + ".npy")
            if dp.exists():
                depth_maps[name] = np.load(dp)

        if not depth_maps:
            return xyz, rgb

        images_dir = Path(self.cfg.data.scene_path) / self.cfg.data.images_dir
        return augment_with_depth(recon, depth_maps, images_dir)

    # ─── Rendering ────────────────────────────────────────────────────────────

    def _render(self, camera: Camera) -> dict:
        """
        Render a view using gsplat's rasterization pipeline.
        Returns dict with rendered_rgb, rendered_depth, rendered_alpha.
        """
        try:
            from gsplat import rasterization
        except ImportError:
            raise ImportError("gsplat not installed. Run: pip install gsplat")

        means = self.model.means
        quats = self.model.quats
        scales = self.model.scales
        opacities = self.model.opacities.squeeze(-1)
        sh_coeffs = self.model.sh_coeffs

        # Camera parameters
        c2w = camera.c2w.to(self.device)
        K = camera.intrinsic_matrix().to(self.device)
        H, W = camera.height, camera.width

        # Viewmat: world-to-camera (4x4)
        viewmat = torch.linalg.inv(c2w).unsqueeze(0)                # (1, 4, 4)

        # Camera direction for SH evaluation
        cam_pos = c2w[:3, 3]
        dirs = means - cam_pos.unsqueeze(0)
        dirs = F.normalize(dirs, dim=-1)

        # Evaluate SH to get colors
        colors = self._eval_sh(sh_coeffs, dirs)                     # (N, 3)

        renders, alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K.unsqueeze(0),
            width=W,
            height=H,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB+D",   # returns depth in 4th channel
            packed=False,
        )

        # renders: (1, H, W, 4) — RGB + depth
        rendered = renders[0]                   # (H, W, 4)
        rendered_rgb = rendered[..., :3].permute(2, 0, 1)   # (3, H, W)
        rendered_depth = rendered[..., 3]                    # (H, W)
        rendered_alpha = alphas[0, ..., 0]                   # (H, W)

        return {
            "rgb": rendered_rgb,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "meta": meta,
        }

    def _eval_sh(self, sh_coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate spherical harmonics to get per-Gaussian RGB colors.
        sh_coeffs: (N, (deg+1)^2, 3)
        dirs: (N, 3) normalized view directions
        Returns: (N, 3) colors in [0, 1]
        """
        # DC term only for simplicity; full SH eval for high-quality rendering
        C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))
        colors = sh_coeffs[:, 0] * C0 + 0.5

        if self.model.sh_degree > 0 and sh_coeffs.shape[1] > 1:
            # SH degree 1 basis functions
            x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
            C1 = 0.4886025119029199
            colors = colors + C1 * (
                -sh_coeffs[:, 1] * y +
                 sh_coeffs[:, 2] * z +
                -sh_coeffs[:, 3] * x
            )
        # Higher degrees omitted for brevity; gsplat's built-in SH eval is preferred
        return colors.clamp(0, 1)

    # ─── Training Step ────────────────────────────────────────────────────────

    def _train_step(self, camera: Camera) -> dict:
        camera = camera.to(self.device)
        self.optimizer.zero_grad()

        # Render
        render_out = self._render(camera)
        rendered_rgb = render_out["rgb"]
        rendered_depth = render_out["depth"]
        meta = render_out["meta"]

        # Semantic state
        sem_labels = self.model.semantic_labels    # (N,)
        sem_logits = self.model._sem_logits        # (N, C)
        normals = self.model.compute_normals()

        # Pseudo-labels from lifter (if available)
        projected_labels = None
        projected_valid = None
        if (
            self.label_lifter.has_labels
            and self.step >= self.cfg.label_lifting.start_iter
        ):
            projected_labels = self.label_lifter.gaussian_labels.to(self.device)
            projected_valid = self.label_lifter.gaussian_labels_valid.to(self.device)

        # Resize if needed
        target_rgb = camera.image.to(self.device)
        if target_rgb.shape[-2:] != rendered_rgb.shape[-2:]:
            target_rgb = F.interpolate(
                target_rgb.unsqueeze(0), rendered_rgb.shape[-2:], mode="bilinear"
            ).squeeze(0)

        # Compute losses
        loss, logs = self.loss_fn(
            rendered_rgb=rendered_rgb,
            target_rgb=target_rgb,
            rendered_depth=rendered_depth,
            target_depth=camera.depth,
            means=self.model.means,
            normals=normals,
            sem_logits=sem_logits,
            sem_labels=sem_labels,
            projected_labels=projected_labels,
            projected_valid=projected_valid,
            use_depth=(camera.depth is not None),
            use_planarity=(self.step >= 1000),    # Warmup before planarity
            use_semantic=(projected_labels is not None),
        )

        loss.backward()

        # Accumulate gradients for densification
        if hasattr(meta, "radii") or "radii" in (meta or {}):
            with torch.no_grad():
                grad_norms = self.model.means.grad.norm(dim=-1) if self.model.means.grad is not None else torch.zeros_like(self.grad_accum)
                if self.grad_accum.shape[0] != grad_norms.shape[0]:
                    self.grad_accum = torch.zeros_like(grad_norms)
                self.grad_accum += grad_norms

        self.optimizer.step()
        return logs

    # ─── Main Training Loop ───────────────────────────────────────────────────

    def train(self):
        """Main training loop for Phase 1 (default: 30,000 iterations)."""
        cfg_t = self.cfg.training
        total_iters = cfg_t.iterations

        print(f"\n[Trainer] Starting training for {total_iters:,} iterations...")
        t0 = time.time()

        for self.step in range(total_iters):
            # Sample random training camera
            camera = self.train_dataset.random_camera()

            # Training step
            logs = self._train_step(camera)

            # ── Densification ──────────────────────────────────────────────
            if (
                cfg_t.densify_from_iter <= self.step <= cfg_t.densify_until_iter
                and self.step % 100 == 0
            ):
                extent = self._compute_scene_extent()
                self.model.densify_and_prune(
                    grads=self.grad_accum / 100,   # average over 100 steps
                    grad_threshold=cfg_t.densify_grad_threshold,
                    min_opacity=cfg_t.min_opacity,
                    extent=extent,
                    max_gaussians=self.cfg.gaussian.max_num_gaussians,
                )
                self.grad_accum = torch.zeros(
                    self.model.num_gaussians, device=self.device
                )

            # ── Opacity Reset ──────────────────────────────────────────────
            if self.step > 0 and self.step % cfg_t.opacity_reset_interval == 0:
                self.model.reset_opacities()

            # ── Semantic Label Lifting ─────────────────────────────────────
            lift_cfg = self.cfg.label_lifting
            if (
                self.step >= lift_cfg.start_iter
                and self.step % lift_cfg.lift_every == 0
            ):
                self.label_lifter.lift(self.model.means.detach())

            # ── Scene Graph (build once after iter 5000) ───────────────────
            if self.step == self.cfg.scene_graph.build_after_iter:
                self._build_scene_graph()

            # ── Logging ────────────────────────────────────────────────────
            if self.step % self.cfg.experiment.log_every == 0:
                elapsed = time.time() - t0
                logs["train/num_gaussians"] = self.model.num_gaussians
                logs["train/step"] = self.step
                logs["train/elapsed_min"] = elapsed / 60
                self._log(logs)

                # Console
                print(
                    f"[{self.step:6d}/{total_iters}] "
                    f"loss={logs['loss/total']:.4f} | "
                    f"rgb={logs['loss/rgb_total']:.4f} | "
                    f"plan={logs.get('loss/planarity', 0):.4f} | "
                    f"N={self.model.num_gaussians:,} | "
                    f"t={elapsed/60:.1f}min"
                )

            # ── Checkpoint ────────────────────────────────────────────────
            if self.step % self.cfg.experiment.save_every == 0 and self.step > 0:
                self._save_checkpoint()

            # ── Evaluation ────────────────────────────────────────────────
            if self.step % self.cfg.experiment.eval_every == 0 and self.step > 0:
                self._evaluate()

        # Final save
        self._save_checkpoint(final=True)
        print(f"\n[Trainer] Training complete! Total time: {(time.time()-t0)/60:.1f} min")

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _compute_scene_extent(self) -> float:
        means = self.model.means.detach()
        lo = torch.quantile(means, 0.01, dim=0)
        hi = torch.quantile(means, 0.99, dim=0)
        return (hi - lo).max().item()

    def _build_scene_graph(self):
        print(f"[Trainer] Building architectural scene graph at iter {self.step}...")
        graph = ArchitecturalSceneGraph(
            k_spatial=self.cfg.scene_graph.k_spatial,
            relationship_types=self.cfg.scene_graph.relationship_types,
        )
        graph.build(
            means=self.model.means.detach(),
            sem_labels=self.model.semantic_labels.detach(),
        )
        self.scene_graph = graph
        graph.save(self.output_dir / "scene_graph.pt")
        print(f"[Trainer] Scene graph built: {graph.num_nodes} nodes, {graph.num_edges} edges")

    def _save_checkpoint(self, final: bool = False):
        suffix = "final" if final else f"iter_{self.step:06d}"
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        # Save model parameters
        state = {
            "step": self.step,
            "means": self.model._means.data,
            "quats": self.model._quats.data,
            "scales": self.model._scales.data,
            "opacities": self.model._opacities.data,
            "sh0": self.model._sh0.data,
            "shN": self.model._shN.data,
            "sem_logits": self.model._sem_logits.data,
            "sh_degree": self.model.sh_degree,
            "num_classes": self.model.num_classes,
        }
        torch.save(state, ckpt_dir / f"gaussian_{suffix}.pt")

        # Also save PLY for visualization
        ply_dir = self.output_dir / "ply"
        ply_dir.mkdir(exist_ok=True)
        self.model.save_ply(str(ply_dir / f"gaussian_{suffix}.ply"))

        print(f"[Trainer] Checkpoint saved: {suffix}")

    @torch.no_grad()
    def _evaluate(self):
        """Evaluate on validation cameras. Computes PSNR, SSIM."""
        if len(self.val_dataset) == 0:
            return

        psnrs, ssims = [], []
        for camera in self.val_dataset.get_all_cameras()[:10]:   # Max 10 for speed
            camera = camera.to(self.device)
            render_out = self._render(camera)
            pred = render_out["rgb"].clamp(0, 1)
            gt = camera.image.to(self.device)

            if pred.shape != gt.shape:
                gt = F.interpolate(gt.unsqueeze(0), pred.shape[-2:], mode="bilinear").squeeze(0)

            mse = ((pred - gt) ** 2).mean()
            psnr = -10 * torch.log10(mse + 1e-8)
            psnrs.append(psnr.item())

        mean_psnr = np.mean(psnrs)
        logs = {"eval/psnr": mean_psnr, "eval/step": self.step}
        self._log(logs)
        print(f"[Eval] iter={self.step} PSNR={mean_psnr:.2f} dB")

    def _log(self, logs: dict):
        if self._wandb is not None:
            self._wandb.log(logs, step=self.step)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.step = state["step"]
        self.model._means = torch.nn.Parameter(state["means"])
        self.model._quats = torch.nn.Parameter(state["quats"])
        self.model._scales = torch.nn.Parameter(state["scales"])
        self.model._opacities = torch.nn.Parameter(state["opacities"])
        self.model._sh0 = torch.nn.Parameter(state["sh0"])
        self.model._shN = torch.nn.Parameter(state["shN"])
        self.model._sem_logits = torch.nn.Parameter(state["sem_logits"])
        self.model._num_gaussians = state["means"].shape[0]
        print(f"[Trainer] Loaded checkpoint from {path} (step {self.step})")
