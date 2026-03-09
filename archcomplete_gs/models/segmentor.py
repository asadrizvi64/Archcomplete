"""
archcomplete_gs/models/segmentor.py

Grounded-SAM2 wrapper for architectural semantic segmentation.
Queries GroundingDINO with architectural vocabulary then segments with SAM2.
Produces per-pixel class label maps cached to disk.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from archcomplete_gs.data.dataset import ArchitecturalSceneDataset


# ─── Vocabulary ───────────────────────────────────────────────────────────────

ARCH_VOCABULARY: list[str] = [
    "wall",
    "floor",
    "ceiling",
    "window",
    "door",
    "column",
    "beam",
    "staircase",
    "roof",
    "facade",
    "balcony",
]

# Grounding DINO prompt: all classes as dot-separated string
GDINO_PROMPT = " . ".join(ARCH_VOCABULARY) + " ."


# ─── Segmentor ────────────────────────────────────────────────────────────────

class ArchitecturalSegmentor:
    """
    Two-stage architectural segmentor:
      1. GroundingDINO: open-vocabulary detection → bounding boxes per class
      2. SAM2: segment each detected box → per-class masks
      3. Merge masks into a single (H, W) int8 label map

    Lazy-loaded: models are only instantiated on first call.
    All outputs are cached to disk as .npy files.
    """

    def __init__(
        self,
        grounding_dino_model: str = "IDEA-Research/grounding-dino-tiny",
        sam2_checkpoint: str = "checkpoints/sam2_hiera_large.pt",
        sam2_config: str = "sam2_hiera_l.yaml",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        device: Optional[str] = None,
        vocabulary: list[str] = ARCH_VOCABULARY,
    ):
        self.gdino_model_id = grounding_dino_model
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocabulary = vocabulary
        self.prompt = " . ".join(vocabulary) + " ."
        self.class_to_idx = ArchitecturalSceneDataset.CLASS_TO_IDX

        self._gdino = None
        self._sam2 = None

    def _load_models(self):
        if self._gdino is not None:
            return
        print(f"[Segmentor] Loading GroundingDINO: {self.gdino_model_id}")
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self._gdino_processor = AutoProcessor.from_pretrained(self.gdino_model_id)
        self._gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.gdino_model_id
        ).to(self.device).eval()
        print("[Segmentor] GroundingDINO loaded.")

        print(f"[Segmentor] Loading SAM2: {self.sam2_checkpoint}")
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2_model = build_sam2(self.sam2_config, self.sam2_checkpoint, device=self.device)
            self._sam2 = SAM2ImagePredictor(sam2_model)
        except ImportError:
            print("[Segmentor] SAM2 not available, falling back to SAM1.")
            from segment_anything import sam_model_registry, SamPredictor
            # Use a SAM1 fallback (user needs to provide checkpoint)
            sam = sam_model_registry["vit_h"](checkpoint=self.sam2_checkpoint)
            sam.to(self.device)
            self._sam2 = SamPredictor(sam)
        print("[Segmentor] SAM2 loaded.")

    @torch.no_grad()
    def _detect_boxes(
        self,
        image: Image.Image,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """
        Run GroundingDINO to get bounding boxes.
        Returns:
            boxes: (N, 4) in (x1, y1, x2, y2) normalized [0,1]
            labels: list of class strings (length N)
            scores: (N,) confidence scores
        """
        inputs = self._gdino_processor(
            images=image,
            text=self.prompt,
            return_tensors="pt",
        ).to(self.device)

        outputs = self._gdino(**inputs)
        results = self._gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes = results["boxes"].cpu().numpy()      # (N, 4) abs pixel coords
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]                  # list of strings

        # Normalize labels: match to vocabulary
        clean_labels = []
        for lbl in labels:
            lbl = lbl.strip().lower()
            # Match to nearest vocab word
            matched = "background"
            for vocab_word in self.vocabulary:
                if vocab_word in lbl or lbl in vocab_word:
                    matched = vocab_word
                    break
            clean_labels.append(matched)

        return boxes, clean_labels, scores

    @torch.no_grad()
    def _segment_boxes(
        self,
        image: Image.Image,
        boxes: np.ndarray,
    ) -> np.ndarray:
        """
        Run SAM2 to get per-box binary masks.
        Returns (N, H, W) bool array.
        """
        img_np = np.array(image.convert("RGB"))
        H, W = img_np.shape[:2]

        if len(boxes) == 0:
            return np.zeros((0, H, W), dtype=bool)

        self._sam2.set_image(img_np)
        # SAM2 accepts batched boxes
        masks, _, _ = self._sam2.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        # masks: (N, 1, H, W) or (N, H, W)
        if masks.ndim == 4:
            masks = masks[:, 0]
        return masks.astype(bool)

    def segment_image(self, image: Image.Image) -> np.ndarray:
        """
        Segment a single image.
        Returns (H, W) int8 label map with class indices.
        Background (unlabeled) = 0.
        """
        self._load_models()
        W, H = image.size
        label_map = np.zeros((H, W), dtype=np.int8)
        confidence_map = np.zeros((H, W), dtype=np.float32)

        boxes, labels, scores = self._detect_boxes(image)
        if len(boxes) == 0:
            return label_map

        masks = self._segment_boxes(image, boxes)

        # Merge masks: higher confidence overwrites lower confidence (conflict resolution)
        for mask, label, score in sorted(
            zip(masks, labels, scores),
            key=lambda x: x[2],   # sort by score ascending so highest wins
        ):
            class_id = self.class_to_idx.get(label, 0)
            if class_id == 0:
                continue
            # Only write where this mask has higher confidence than existing
            region = mask & (confidence_map < score)
            label_map[region] = class_id
            confidence_map[region] = score

        return label_map

    def process_scene(
        self,
        images_dir: Path,
        output_dir: Path,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".PNG"),
        skip_existing: bool = True,
        image_scale: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """
        Process all images in a directory. Saves .npy label maps.
        Returns dict mapping image filename -> label map.
        """
        from tqdm import tqdm

        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted([p for p in images_dir.iterdir() if p.suffix in extensions])
        print(f"[Segmentor] Processing {len(image_paths)} images → {output_dir}")

        label_maps = {}
        class_pixel_counts = np.zeros(ArchitecturalSceneDataset.NUM_CLASSES, dtype=np.int64)

        for img_path in tqdm(image_paths, desc="Architectural segmentation"):
            out_path = output_dir / (img_path.stem + ".npy")
            if skip_existing and out_path.exists():
                lm = np.load(out_path)
                label_maps[img_path.name] = lm
                continue

            pil_img = Image.open(img_path).convert("RGB")
            if image_scale != 1.0:
                W, H = pil_img.size
                pil_img = pil_img.resize((int(W * image_scale), int(H * image_scale)), Image.LANCZOS)

            lm = self.segment_image(pil_img)
            np.save(out_path, lm)
            label_maps[img_path.name] = lm

            # Track class distribution
            for c in range(ArchitecturalSceneDataset.NUM_CLASSES):
                class_pixel_counts[c] += (lm == c).sum()

        # Print coverage summary
        total_px = class_pixel_counts.sum()
        if total_px > 0:
            print("\n[Segmentor] Label distribution across scene:")
            for i, cls_name in enumerate(ArchitecturalSceneDataset.ARCH_CLASSES):
                pct = 100.0 * class_pixel_counts[i] / total_px
                bar = "█" * int(pct / 2)
                print(f"  {cls_name:12s} {pct:5.1f}% {bar}")

        return label_maps


# ─── Visualization ────────────────────────────────────────────────────────────

# Color palette for visualization (RGB)
ARCH_CLASS_COLORS = np.array([
    [0,   0,   0  ],  # 0  background
    [255, 200, 150],  # 1  wall       (warm beige)
    [180, 220, 180],  # 2  floor      (light green)
    [200, 200, 255],  # 3  ceiling    (light blue)
    [100, 180, 255],  # 4  window     (sky blue)
    [200, 120,  60],  # 5  door       (brown)
    [160, 160, 160],  # 6  column     (grey)
    [120, 100,  80],  # 7  beam       (dark brown)
    [255, 240, 100],  # 8  staircase  (yellow)
    [220, 150,  80],  # 9  roof       (terracotta)
    [100, 200, 100],  # 10 facade     (green)
    [255, 150, 200],  # 11 balcony    (pink)
], dtype=np.uint8)


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    """
    Convert (H, W) int label map to (H, W, 3) RGB visualization.
    """
    H, W = label_map.shape
    rgb = ARCH_CLASS_COLORS[label_map.clip(0, len(ARCH_CLASS_COLORS) - 1)]
    return rgb


def overlay_labels_on_image(
    image: np.ndarray,              # (H, W, 3) uint8
    label_map: np.ndarray,          # (H, W) int
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend label colorization over original image."""
    color = colorize_label_map(label_map).astype(np.float32)
    base = image.astype(np.float32)
    overlay = (1 - alpha) * base + alpha * color
    return overlay.clip(0, 255).astype(np.uint8)
