from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


def robust_mad(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    med = float(np.median(vals))
    return float(np.median(np.abs(vals - med)))


def robust_iqr(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return 0.0
    q1, q3 = np.quantile(vals, [0.25, 0.75])
    return float(max(0.0, q3 - q1))


def keep_primary_connected_component(mask: np.ndarray, box_xyxy: list[float]) -> tuple[np.ndarray, dict[str, Any]]:
    m = np.asarray(mask).astype(np.uint8)
    if int(m.sum()) == 0:
        return m.astype(bool), {"component_count": 0, "selected_component": -1, "reason": "empty"}
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m.astype(bool), {"component_count": 1, "selected_component": 1, "reason": "single"}
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    cx = int(round((x1 + x2) * 0.5))
    cy = int(round((y1 + y2) * 0.5))
    chosen = -1
    if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1] and labels[cy, cx] > 0:
        chosen = int(labels[cy, cx])
        reason = "box_center_component"
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        chosen = int(np.argmax(areas) + 1)
        reason = "largest_component"
    out = labels == chosen
    return out, {
        "component_count": int(n - 1),
        "selected_component": int(chosen),
        "reason": reason,
        "selected_area": int(out.sum()),
    }


def erosion_seed_mask(mask: np.ndarray, *, min_pixels: int, iterations: int = 1) -> np.ndarray:
    m = np.asarray(mask).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(m, kernel, iterations=int(iterations)).astype(bool)
    if int(eroded.sum()) < int(min_pixels):
        return m.astype(bool)
    return eroded


@dataclass
class DepthMode:
    index: int
    count: int
    median: float
    mad: float
    min_depth: float
    max_depth: float
    distance_to_seed: float | None
    distance_to_parent: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "count": self.count,
            "median": self.median,
            "mad": self.mad,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "distance_to_seed": self.distance_to_seed,
            "distance_to_parent": self.distance_to_parent,
        }


def split_depth_modes(depth_values: np.ndarray, *, gap_m: float, min_count: int, seed_depth: float | None = None, parent_depth: float | None = None) -> list[DepthMode]:
    vals = np.sort(np.asarray(depth_values, dtype=np.float32).reshape(-1))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return []
    if vals.size < max(2, int(min_count)):
        return [
            DepthMode(
                index=0,
                count=int(vals.size),
                median=float(np.median(vals)),
                mad=robust_mad(vals),
                min_depth=float(vals.min()),
                max_depth=float(vals.max()),
                distance_to_seed=None if seed_depth is None else abs(float(np.median(vals)) - float(seed_depth)),
                distance_to_parent=None if parent_depth is None else abs(float(np.median(vals)) - float(parent_depth)),
            )
        ]
    gaps = np.diff(vals)
    cuts = np.where(gaps > float(gap_m))[0] + 1
    chunks = np.split(vals, cuts)
    modes: list[DepthMode] = []
    for idx, chunk in enumerate(chunks):
        if chunk.size < int(min_count):
            continue
        med = float(np.median(chunk))
        modes.append(
            DepthMode(
                index=len(modes),
                count=int(chunk.size),
                median=med,
                mad=robust_mad(chunk),
                min_depth=float(chunk.min()),
                max_depth=float(chunk.max()),
                distance_to_seed=None if seed_depth is None else abs(med - float(seed_depth)),
                distance_to_parent=None if parent_depth is None else abs(med - float(parent_depth)),
            )
        )
    if not modes:
        med = float(np.median(vals))
        modes.append(
            DepthMode(
                index=0,
                count=int(vals.size),
                median=med,
                mad=robust_mad(vals),
                min_depth=float(vals.min()),
                max_depth=float(vals.max()),
                distance_to_seed=None if seed_depth is None else abs(med - float(seed_depth)),
                distance_to_parent=None if parent_depth is None else abs(med - float(parent_depth)),
            )
        )
    return modes


def choose_depth_mode(modes: list[DepthMode], *, seed_depth: float | None, parent_depth: float | None, parent_mad: float | None, cfg: dict[str, Any]) -> DepthMode | None:
    if not modes:
        return None
    parent_band = float(cfg.get("parent_band_min_m", 0.025))
    if parent_depth is not None:
        parent_band = max(parent_band, float(cfg.get("parent_band_mad_scale", 3.0)) * max(float(parent_mad or 0.0), 0.002))

    def score(mode: DepthMode) -> tuple[int, float, float, int]:
        parent_ok = 0
        parent_dist = mode.distance_to_parent if mode.distance_to_parent is not None else 999.0
        if parent_depth is not None and parent_dist <= parent_band:
            parent_ok = 1
        seed_dist = mode.distance_to_seed if mode.distance_to_seed is not None else 999.0
        compact = float(mode.mad)
        return (parent_ok, -seed_dist, -compact, mode.count)

    if parent_depth is not None:
        compatible = [m for m in modes if (m.distance_to_parent is not None and m.distance_to_parent <= parent_band)]
        if compatible:
            return sorted(compatible, key=score, reverse=True)[0]
    if seed_depth is not None:
        seed_band = float(cfg.get("seed_band_m", 0.05))
        near_seed = [m for m in modes if m.distance_to_seed is not None and m.distance_to_seed <= seed_band]
        if near_seed:
            return sorted(near_seed, key=score, reverse=True)[0]
    return sorted(modes, key=lambda m: (m.mad, -m.count))[0]


def score_observation_quality(
    *,
    detection_score: float,
    mask_quality: float,
    box_touches_border: bool,
    num_points: int,
    min_points: int,
    depth_mad: float,
    depth_iqr: float,
    cluster_ratio: float,
    parent_residual: float | None,
    mask_area_ratio: float,
    tiny_far: bool,
    cfg: dict[str, Any],
) -> tuple[float, str]:
    weights = cfg.get("quality_weights", {})
    score = 0.0
    score += float(weights.get("detection", 0.20)) * float(np.clip(detection_score, 0.0, 1.0))
    score += float(weights.get("mask", 0.15)) * float(np.clip(mask_quality, 0.0, 1.0))
    point_score = min(1.0, float(num_points) / max(1.0, float(min_points) * float(cfg.get("quality_point_full_scale", 3.0))))
    score += float(weights.get("points", 0.20)) * point_score
    compact = np.exp(-max(float(depth_mad), float(depth_iqr) * 0.5) / max(float(cfg.get("quality_depth_scale_m", 0.05)), 1e-6))
    score += float(weights.get("depth_compactness", 0.20)) * float(compact)
    score += float(weights.get("cluster_purity", 0.15)) * float(np.clip(cluster_ratio, 0.0, 1.0))
    parent_score = 1.0
    if parent_residual is not None:
        parent_score = float(np.exp(-abs(float(parent_residual)) / max(float(cfg.get("quality_parent_scale_m", 0.04)), 1e-6)))
    score += float(weights.get("parent", 0.10)) * parent_score
    if box_touches_border:
        score -= float(cfg.get("border_penalty", 0.15))
    if tiny_far:
        score -= float(cfg.get("tiny_far_penalty", 0.05))
    if mask_area_ratio < float(cfg.get("min_mask_area_ratio_for_reliable", 1e-5)):
        score -= float(cfg.get("tiny_mask_penalty", 0.10))
    score = float(np.clip(score, 0.0, 1.0))
    if score >= float(cfg.get("reliable_quality", 0.62)):
        status = "reliable"
    elif score >= float(cfg.get("provisional_quality", 0.38)):
        status = "provisional"
    elif score >= float(cfg.get("low_quality", 0.20)):
        status = "low_quality"
    else:
        status = "rejected"
    return score, status
