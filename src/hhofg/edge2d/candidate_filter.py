from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from hhofg.core.types import Detection2D, LocalRelation2D

from .types import Edge2DCandidate


def _sha_mask(mask: np.ndarray) -> str:
    arr = np.asarray(mask, dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _bbox_mask(shape: tuple[int, int], box_xyxy: list[float]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    left = int(np.clip(np.floor(x1), 0, width))
    top = int(np.clip(np.floor(y1), 0, height))
    right = int(np.clip(np.ceil(x2), 0, width))
    bottom = int(np.clip(np.ceil(y2), 0, height))
    mask = np.zeros(shape, dtype=bool)
    if right > left and bottom > top:
        mask[top:bottom, left:right] = True
    return mask


def build_parent_support_mask(
    parent_mask: np.ndarray,
    parent_box: list[float],
    *,
    closing_radius_px: int = 2,
) -> np.ndarray:
    """Complete a parent part-whole support region, strictly inside its bbox."""

    raw = np.asarray(parent_mask, dtype=bool)
    bbox = _bbox_mask(raw.shape, parent_box)
    clipped = (raw & bbox).astype(np.uint8)
    if not np.any(clipped):
        return np.zeros_like(raw)
    try:
        import cv2

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            clipped, connectivity=8
        )
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            clipped = (labels == largest).astype(np.uint8)

        contours, _ = cv2.findContours(
            clipped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled = np.zeros_like(clipped)
        if contours:
            cv2.drawContours(filled, contours, -1, color=1, thickness=cv2.FILLED)

        radius = max(0, int(closing_radius_px))
        if radius:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)

        points = cv2.findNonZero(filled)
        support = np.zeros_like(filled)
        if points is not None and len(points) >= 3:
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(support, hull, color=1)
        else:
            support = filled
        return support.astype(bool) & bbox
    except Exception:
        return raw & bbox


def edge_candidate_stats(
    *,
    frame_key: str,
    relation: LocalRelation2D,
    detections_by_id: dict[int, Detection2D],
    masks_raw: np.ndarray,
    sdet_threshold: float = 0.25,
    gcamc_threshold: float = 0.90,
    dilation_radius_px: int | None = None,
    dilation_alpha: float = 0.0,
    dilation_radius_min_px: int = 2,
    dilation_radius_max_px: int = 12,
    image_sha: str = "",
    candidate_hash: str = "",
    candidate_sources: list[str] | None = None,
    semantic_sources: list[str] | None = None,
    semantic_compatible: bool | None = None,
    support_closing_radius_px: int = 2,
) -> Edge2DCandidate:
    parent = detections_by_id[int(relation.parent_det_id)]
    child = detections_by_id[int(relation.child_det_id)]
    sdet = min(float(parent.score), float(child.score))
    parent_mask = np.asarray(masks_raw[parent.det_id]).astype(bool)
    child_mask = np.asarray(masks_raw[child.det_id]).astype(bool)
    semantic_compatible = bool(
        semantic_sources if semantic_compatible is None else semantic_compatible
    )
    if dilation_radius_px is None:
        if float(dilation_alpha) > 0.0:
            box = np.asarray(child.box_xyxy, dtype=np.float32)
            diag = float(np.linalg.norm(box[2:] - box[:2]))
            radius = int(round(float(dilation_alpha) * diag))
            dilation_radius_px = int(np.clip(radius, int(dilation_radius_min_px), int(dilation_radius_max_px)))
        else:
            dilation_radius_px = 2
    try:
        import cv2

        k = max(1, int(dilation_radius_px) * 2 + 1)
        kernel = np.ones((k, k), dtype=np.uint8)
        dilated_parent = cv2.dilate(parent_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        dilated_parent = parent_mask
    child_pixels = int(child_mask.sum())
    gcamc_raw = (
        0.0
        if child_pixels == 0
        else float((child_mask & dilated_parent).sum() / child_pixels)
    )
    support_mask = build_parent_support_mask(
        parent_mask,
        parent.box_xyxy,
        closing_radius_px=support_closing_radius_px,
    )
    gcamc_support = (
        0.0
        if child_pixels == 0
        else float((child_mask & support_mask).sum() / child_pixels)
    )
    child_box = np.asarray(child.box_xyxy, dtype=np.float64)
    parent_box = np.asarray(parent.box_xyxy, dtype=np.float64)
    child_center = 0.5 * (child_box[:2] + child_box[2:])
    center_inside = bool(
        parent_box[0] <= child_center[0] <= parent_box[2]
        and parent_box[1] <= child_center[1] <= parent_box[3]
    )
    support_mask_used = bool(
        semantic_compatible and gcamc_support > gcamc_raw
    )
    gcamc = (
        max(gcamc_raw, gcamc_support)
        if semantic_compatible
        else gcamc_raw
    )
    expected_roles = {
        "O-C": ("O", "C"),
        "O-U": ("O", "U"),
        "C-U": ("C", "U"),
    }.get(relation.edge_type)
    fail = ""
    if parent.det_id == child.det_id:
        fail = "self_loop"
    elif expected_roles is None or (parent.role, child.role) != expected_roles:
        fail = "illegal_role_pair"
    elif semantic_compatible and not center_inside:
        fail = "child_center_outside_parent_box"
    elif sdet <= float(sdet_threshold):
        fail = "sdet_below_threshold"
    elif gcamc <= float(gcamc_threshold):
        fail = "gcamc_below_threshold"
    return Edge2DCandidate(
        frame_key=frame_key,
        parent_det_id=parent.det_id,
        child_det_id=child.det_id,
        parent_label=parent.label,
        child_label=child.label,
        parent_score=float(parent.score),
        child_score=float(child.score),
        edge_type=relation.edge_type,
        sdet=sdet,
        gcamc=gcamc,
        gcamc_raw=gcamc_raw,
        gcamc_support=gcamc_support,
        gcamc_used=gcamc,
        support_mask_used=support_mask_used,
        child_center_inside_parent_box=center_inside,
        pass_prefilter=not fail,
        fail_reason=fail,
        image_sha=image_sha,
        mask_sha=hashlib.sha256((_sha_mask(parent_mask) + _sha_mask(child_mask)).encode("ascii")).hexdigest(),
        candidate_hash=candidate_hash,
        candidate_sources=list(candidate_sources or []),
        semantic_sources=list(semantic_sources or []),
    )
