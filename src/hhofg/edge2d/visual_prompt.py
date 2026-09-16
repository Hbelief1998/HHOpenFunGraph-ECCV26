from __future__ import annotations

import hashlib
from typing import Any

import cv2
import numpy as np
from PIL import Image

from hhofg.core.types import Detection2D

from .types import Edge2DCandidate


def image_sha_rgb(image_rgb: np.ndarray) -> str:
    arr = np.asarray(image_rgb, dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def build_edge_prompt(candidate: Edge2DCandidate) -> str:
    direction = {
        "O-C": "Does the CHILD functional carrier belong to, or form a functional part of, the PARENT object?",
        "O-U": (
            "Is the CHILD interactive unit functionally associated with, part of, or used to "
            "operate the PARENT object, possibly through an intermediate functional carrier?"
        ),
    }.get(candidate.edge_type, "Does the CHILD functionally belong to the PARENT?")
    return f"""You are scoring one functional relation in an RGB image.

The image is annotated:
- PARENT is marked in RED and labeled PARENT.
- CHILD is marked in BLUE and labeled CHILD.

PARENT label: {candidate.parent_label}
CHILD label: {candidate.child_label}
Relation type: {candidate.edge_type}
Question: {direction}

Return JSON only:
{{
  "s2d_score": <a number from 0.01 to 0.99>,
  "reason": "<one short visual reason>"
}}

Use a high score only when the annotated child is visually on, attached to, contained by, or clearly operating the annotated parent. Use a low score when they are separate, ambiguous, or only overlap because of perspective."""


def _clip_box(box: list[float], h: int, w: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)


def build_annotated_edge_image(
    *,
    image_rgb: np.ndarray,
    parent_det: Detection2D,
    child_det: Detection2D,
    parent_mask: np.ndarray,
    child_mask: np.ndarray,
    max_side: int = 768,
    crop_margin_ratio: float = 0.25,
) -> Image.Image:
    img = np.asarray(image_rgb, dtype=np.uint8).copy()
    h, w = img.shape[:2]
    px1, py1, px2, py2 = _clip_box(parent_det.box_xyxy, h, w)
    cx1, cy1, cx2, cy2 = _clip_box(child_det.box_xyxy, h, w)
    ux1, uy1 = min(px1, cx1), min(py1, cy1)
    ux2, uy2 = max(px2, cx2), max(py2, cy2)
    margin = int(round(max(ux2 - ux1 + 1, uy2 - uy1 + 1) * float(crop_margin_ratio)))
    x1, y1 = max(0, ux1 - margin), max(0, uy1 - margin)
    x2, y2 = min(w - 1, ux2 + margin), min(h - 1, uy2 + margin)
    crop = img[y1 : y2 + 1, x1 : x2 + 1].copy()
    pm = np.asarray(parent_mask, dtype=bool)[y1 : y2 + 1, x1 : x2 + 1]
    cm = np.asarray(child_mask, dtype=bool)[y1 : y2 + 1, x1 : x2 + 1]
    overlay = crop.copy()
    overlay[pm] = (0.55 * overlay[pm] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    overlay[cm] = (0.55 * overlay[cm] + 0.45 * np.array([0, 80, 255])).astype(np.uint8)
    crop = overlay

    pbox = (px1 - x1, py1 - y1, px2 - x1, py2 - y1)
    cbox = (cx1 - x1, cy1 - y1, cx2 - x1, cy2 - y1)
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (pbox[0], pbox[1]), (pbox[2], pbox[3]), (0, 0, 255), 4)
    cv2.rectangle(bgr, (cbox[0], cbox[1]), (cbox[2], cbox[3]), (255, 80, 0), 4)
    cv2.putText(bgr, f"PARENT: {parent_det.label}", (max(0, pbox[0]), max(28, pbox[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(bgr, f"CHILD: {child_det.label}", (max(0, cbox[0]), min(bgr.shape[0] - 8, cbox[3] + 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 0), 2, cv2.LINE_AA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    if max(pil.size) > int(max_side):
        scale = float(max_side) / float(max(pil.size))
        pil = pil.resize((max(1, int(round(pil.width * scale))), max(1, int(round(pil.height * scale)))), Image.BICUBIC)
    return pil
