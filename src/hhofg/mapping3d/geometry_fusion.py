from __future__ import annotations

from typing import Any

import numpy as np

from .types import MapNode3D, Observation3D


def robust_bounds(points: np.ndarray, *, q_low: float = 2.0, q_high: float = 98.0) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        z = np.zeros(3, dtype=np.float32)
        return z, z
    lo = np.percentile(pts, float(q_low), axis=0).astype(np.float32)
    hi = np.percentile(pts, float(q_high), axis=0).astype(np.float32)
    return lo, hi


def spatially_uniform_sample(points: np.ndarray, colors: np.ndarray, *, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors, dtype=np.float32)
    if pts.shape[0] <= int(max_points):
        return pts, cols
    # Deterministic farthest-ish proxy: sort by coarse voxel hash then stride.
    q = np.floor(pts / max(float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) / 128.0, 1e-4)).astype(np.int64)
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    idx = order[np.linspace(0, order.size - 1, int(max_points)).astype(np.int64)]
    return pts[idx], cols[idx]


def should_accept_geometry_fusion(node: MapNode3D, obs: Observation3D, cfg: dict[str, Any]) -> tuple[bool, str]:
    if obs.geometry_status not in {"reliable", "provisional"}:
        return False, f"obs_{obs.geometry_status}"
    if obs.geometry_is_imputed and node.state == "stable":
        return False, "imputed_geometry_not_fused_into_stable"
    if float(obs.geometry_quality) < float(cfg.get("fusion_min_observation_quality", 0.35)):
        return False, "quality_gate"
    centroid_residual = float(np.linalg.norm(obs.centroid_world - node.centroid_world))
    max_residual = float(cfg.get("fusion_max_centroid_residual_m", 0.12 if node.state == "stable" else 0.18))
    if centroid_residual > max_residual:
        return False, "centroid_residual_gate"
    old_diag = max(float(node.bbox_diag_world), 1e-6)
    new_min = np.minimum(node.bbox_min_world, obs.bbox_min_world)
    new_max = np.maximum(node.bbox_max_world, obs.bbox_max_world)
    expansion = float(np.linalg.norm(np.maximum(new_max - new_min, 0.0)) / old_diag)
    if node.obs_count >= int(cfg.get("fusion_bbox_expansion_after_obs", 2)) and expansion > float(cfg.get("fusion_max_bbox_expansion_ratio", 2.5)):
        return False, "bbox_expansion_gate"
    return True, "accepted"


def refresh_geometry(node: MapNode3D, *, q_low: float = 2.0, q_high: float = 98.0) -> None:
    points = node.core_points_world if node.core_points_world is not None else node.points_world
    node.centroid_world = np.median(points, axis=0).astype(np.float32)
    node.bbox_min_world, node.bbox_max_world = robust_bounds(points, q_low=q_low, q_high=q_high)
    node.bbox_diag_world = float(np.linalg.norm(np.maximum(node.bbox_max_world - node.bbox_min_world, 0.0)))
