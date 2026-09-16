from __future__ import annotations

import numpy as np

from .types import Frame2DResult


def validate_frame2d_arrays(
    result: Frame2DResult,
    boxes: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    n = len(result.detections)
    if boxes.shape != (n, 4):
        raise ValueError(f"boxes must have shape ({n}, 4), got {boxes.shape}")
    if scores.shape != (n,):
        raise ValueError(f"scores must have shape ({n},), got {scores.shape}")
    if masks.ndim != 3 or masks.shape[0] != n:
        raise ValueError(f"masks must have shape ({n}, H, W), got {masks.shape}")
    if n == 0 and masks.shape[1:] != (result.image_height, result.image_width):
        raise ValueError("empty masks must preserve image height/width")
    if n > 0 and masks.shape[1:] != (result.image_height, result.image_width):
        raise ValueError("mask height/width must match raw result image dimensions")
    if not np.all(np.isfinite(boxes)):
        raise ValueError("boxes contain NaN or Inf")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain NaN or Inf")
    expected_ids = list(range(n))
    got_ids = [d.det_id for d in result.detections]
    if got_ids != expected_ids:
        raise ValueError(f"det_id must be consecutive 0..N-1, got {got_ids}")
    return boxes, scores, masks
