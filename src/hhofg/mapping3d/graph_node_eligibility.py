from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .types import FrameEdgeCandidate, MapNode3D


def _centroid_spread_m(node: MapNode3D) -> float:
    points = np.asarray(node.recent_centroids, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        return 0.0
    center = np.median(points, axis=0)
    return float(np.max(np.linalg.norm(points - center, axis=1)))


def evaluate_graph_node_eligibility(
    nodes: dict[str, MapNode3D],
    candidates: list[FrameEdgeCandidate] | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, MapNode3D], dict[str, Any]]:
    """Evaluate nodes from node-local evidence only.

    ``candidates`` remains as a compatibility argument for staged callers, but
    is deliberately ignored: an edge score must never rescue a weak node.
    """

    del candidates
    cfg = cfg or {}
    policy = str(
        cfg.get(
            "node_policy",
            "stable_only"
            if bool(cfg.get("stable_only", True))
            else "stable_or_confirmed_provisional",
        )
    )
    valid_policies = {"stable_only", "stable_or_confirmed_provisional"}
    if policy not in valid_policies:
        raise ValueError(
            f"edge_optimizer.node_policy must be one of {sorted(valid_policies)}"
        )

    min_total_support = int(
        cfg.get("provisional_min_total_support_frames", 2)
    )
    min_real_depth = int(cfg.get("provisional_min_real_depth_frames", 1))
    min_anchor = float(cfg.get("provisional_min_anchor_quality", 0.55))
    min_geometry = float(
        cfg.get("provisional_min_geometry_quality_ema", 0.50)
    )
    require_non_border = bool(
        cfg.get("provisional_require_non_border_anchor", True)
    )
    max_spread = float(
        cfg.get("provisional_max_recent_centroid_spread_m", 0.12)
    )

    reasons: Counter[str] = Counter()
    confirmed = 0
    single_frame_provisional = 0
    for node in nodes.values():
        eligible = False
        confidence = 0.0
        reason = "unknown"
        support_frames = {
            int(frame)
            for frame in (
                list(node.support_frames) + list(node.identity_support_frames)
            )
        }
        real_depth_frames = {int(frame) for frame in node.real_depth_support_frames}
        centroid_spread = _centroid_spread_m(node)
        if node.state == "rejected":
            reason = "rejected"
        elif node.state == "stable":
            eligible = True
            confidence = 1.0
            reason = "stable"
        elif policy == "stable_only":
            reason = "provisional_disallowed_by_stable_only"
        elif len(support_frames) < min_total_support:
            reason = "insufficient_total_support_frames"
            if len(support_frames) <= 1:
                single_frame_provisional += 1
        elif len(real_depth_frames) < min_real_depth:
            reason = "insufficient_real_depth_frames"
        elif float(node.anchor_quality) < min_anchor:
            reason = "anchor_quality_below_threshold"
        elif float(node.geometry_quality_ema) < min_geometry:
            reason = "geometry_quality_below_threshold"
        elif require_non_border and bool(node.anchor_touches_border):
            reason = "extreme_boundary_anchor"
        elif centroid_spread > max_spread:
            reason = "centroid_spread_above_threshold"
        else:
            eligible = True
            confidence = float(
                (
                    float(node.anchor_quality)
                    + float(node.geometry_quality_ema)
                    + min(1.0, len(support_frames) / max(min_total_support, 1))
                )
                / 3.0
            )
            reason = "confirmed_provisional"
            confirmed += 1

        node.graph_eligible = eligible
        node.graph_node_confidence = confidence
        node.graph_eligibility_reason = reason
        reasons[reason] += 1

    eligible_nodes = {
        node_id: node for node_id, node in nodes.items() if node.graph_eligible
    }
    metadata = {
        "node_policy": policy,
        "num_stable_graph_nodes": int(
            sum(node.state == "stable" for node in eligible_nodes.values())
        ),
        "num_confirmed_provisional_graph_nodes": int(confirmed),
        # Kept as a summary compatibility alias, with node-only semantics.
        "num_rescued_provisional_graph_nodes": int(confirmed),
        "num_single_frame_provisional_nodes": int(single_frame_provisional),
        "num_ineligible_nodes": int(len(nodes) - len(eligible_nodes)),
        "eligibility_reasons": dict(sorted(reasons.items())),
        "eligibility_uses_edge_evidence": False,
    }
    return eligible_nodes, metadata
