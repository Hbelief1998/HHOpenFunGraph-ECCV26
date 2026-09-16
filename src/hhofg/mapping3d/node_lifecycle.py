from __future__ import annotations

from typing import Any

import numpy as np

from .types import MapNode3D, Observation3D


def required_support_frames(role: str, cfg: dict[str, Any]) -> int:
    return int(cfg.get("stable_min_support_frames", {}).get(role, 2))


def can_promote_to_stable(node: MapNode3D, cfg: dict[str, Any]) -> tuple[bool, str]:
    if node.state == "rejected":
        return False, "rejected"
    min_frames = required_support_frames(node.role, cfg)
    if len(set(node.support_frames)) < min_frames:
        return False, "insufficient_support_frames"
    min_real = int(cfg.get("stable_min_real_depth_frames", {}).get(node.role, 2))
    if len(set(node.real_depth_support_frames)) < min_real:
        return False, "insufficient_real_depth_support"
    if float(node.geometry_quality_ema) < float(cfg.get("stable_min_quality_ema", 0.55)):
        return False, "low_quality_ema"
    if len(node.recent_centroids) >= 2:
        pts = np.asarray(node.recent_centroids[-min(5, len(node.recent_centroids)) :], dtype=np.float32)
        spread = float(np.max(np.linalg.norm(pts - np.median(pts, axis=0)[None, :], axis=1)))
        if spread > float(cfg.get("stable_centroid_spread_m", 0.08)):
            return False, "centroid_unstable"
    return True, "stable"


def update_node_lifecycle(node: MapNode3D, obs: Observation3D, cfg: dict[str, Any]) -> None:
    if obs.frame_idx not in node.support_frames:
        node.support_frames.append(obs.frame_idx)
    if obs.geometry_is_imputed:
        if obs.frame_idx not in node.imputed_support_frames:
            node.imputed_support_frames.append(obs.frame_idx)
    else:
        if obs.frame_idx not in node.real_depth_support_frames:
            node.real_depth_support_frames.append(obs.frame_idx)
    node.recent_centroids.append([float(v) for v in obs.centroid_world.tolist()])
    node.recent_centroids = node.recent_centroids[-int(cfg.get("recent_history", 12)) :]
    node.recent_bboxes.append(
        {
            "min": [float(v) for v in obs.bbox_min_world.tolist()],
            "max": [float(v) for v in obs.bbox_max_world.tolist()],
        }
    )
    node.recent_bboxes = node.recent_bboxes[-int(cfg.get("recent_history", 12)) :]
    node.recent_quality.append(float(obs.geometry_quality))
    node.recent_quality = node.recent_quality[-int(cfg.get("recent_history", 12)) :]
    alpha = float(cfg.get("geometry_quality_ema_alpha", 0.25))
    node.geometry_quality_ema = float((1.0 - alpha) * node.geometry_quality_ema + alpha * obs.geometry_quality)
    ok, _reason = can_promote_to_stable(node, cfg)
    if ok:
        node.state = "stable"
