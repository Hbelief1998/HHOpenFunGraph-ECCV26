from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .features import cosine01, l2_normalize, voxel_downsample_points_colors
from .geometry_fusion import refresh_geometry, spatially_uniform_sample
from .types import MapNode3D


@dataclass
class ConsolidationEvent:
    event_type: str
    source_node_id: str
    target_node_id: str
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)


def update_covisibility(
    covisibility: set[tuple[str, str]], node_ids: list[str]
) -> None:
    ids = sorted(set(node_ids))
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            covisibility.add((a, b))


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _normal_label(label: str) -> str:
    return " ".join(str(label).lower().replace("_", " ").replace("-", " ").split())


def _sample(points: np.ndarray, limit: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] <= limit:
        return pts
    indices = np.linspace(0, pts.shape[0] - 1, int(limit)).astype(np.int64)
    return pts[indices]


def _nearest_overlap(source: np.ndarray, target: np.ndarray, threshold: float) -> float:
    if source.shape[0] == 0 or target.shape[0] == 0:
        return 0.0
    try:
        from scipy.spatial import cKDTree

        distances = cKDTree(target).query(source, k=1)[0]
        return float(np.mean(distances <= float(threshold)))
    except Exception:
        threshold2 = float(threshold) ** 2
        hits = 0
        for start in range(0, source.shape[0], 128):
            block = source[start : start + 128]
            dist2 = np.sum(
                (block[:, None, :] - target[None, :, :]) ** 2, axis=2
            )
            hits += int(np.sum(np.min(dist2, axis=1) <= threshold2))
        return float(hits / max(1, source.shape[0]))


def _directional_point_overlaps(
    a: MapNode3D,
    b: MapNode3D,
    *,
    threshold: float,
    max_points: int,
) -> tuple[float, float]:
    a_points = _sample(
        np.asarray(a.core_points_world if a.core_points_world is not None else a.points_world),
        max_points,
    )
    b_points = _sample(
        np.asarray(b.core_points_world if b.core_points_world is not None else b.points_world),
        max_points,
    )
    return (
        _nearest_overlap(a_points, b_points, threshold),
        _nearest_overlap(b_points, a_points, threshold),
    )


def _bbox_iou(a: MapNode3D, b: MapNode3D, *, padding: float) -> float:
    # Map points sample visible surfaces, so an AABB can be effectively planar
    # and have zero volume.  Symmetric metric padding makes the overlap measure
    # well-defined without changing either node's stored geometry.
    pad = float(max(0.0, padding))
    a_min = np.asarray(a.bbox_min_world, dtype=np.float32) - pad
    a_max = np.asarray(a.bbox_max_world, dtype=np.float32) + pad
    b_min = np.asarray(b.bbox_min_world, dtype=np.float32) - pad
    b_max = np.asarray(b.bbox_max_world, dtype=np.float32) + pad
    intersection = float(
        np.prod(np.maximum(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0.0))
    )
    a_volume = float(np.prod(np.maximum(a_max - a_min, 0.0)))
    b_volume = float(np.prod(np.maximum(b_max - b_min, 0.0)))
    return float(intersection / max(a_volume + b_volume - intersection, 1e-12))


def _association_extent(node: MapNode3D) -> float:
    value = getattr(node, "association_bbox_diag_world", None)
    return float(value if value is not None else node.bbox_diag_world)


def _bbox_surface_gap(a: MapNode3D, b: MapNode3D) -> float:
    a_min = np.asarray(
        getattr(a, "association_bbox_min_world", None)
        if getattr(a, "association_bbox_min_world", None) is not None
        else a.bbox_min_world,
        dtype=np.float32,
    )
    a_max = np.asarray(
        getattr(a, "association_bbox_max_world", None)
        if getattr(a, "association_bbox_max_world", None) is not None
        else a.bbox_max_world,
        dtype=np.float32,
    )
    b_min = np.asarray(
        getattr(b, "association_bbox_min_world", None)
        if getattr(b, "association_bbox_min_world", None) is not None
        else b.bbox_min_world,
        dtype=np.float32,
    )
    b_max = np.asarray(
        getattr(b, "association_bbox_max_world", None)
        if getattr(b, "association_bbox_max_world", None) is not None
        else b.bbox_max_world,
        dtype=np.float32,
    )
    axis_gaps = np.maximum(np.maximum(a_min - b_max, b_min - a_max), 0.0)
    return float(np.linalg.norm(axis_gaps))


def _observed_together(a: MapNode3D, b: MapNode3D) -> bool:
    return bool(set(a.observed_frames) & set(b.observed_frames))


def _canonical(node_id: str, aliases: dict[str, str]) -> str:
    current = str(node_id)
    visited: set[str] = set()
    while current in aliases and current not in visited:
        visited.add(current)
        current = aliases[current]
    return current


def _identity_anchor_index(
    nodes: dict[str, MapNode3D],
    aliases: dict[str, str],
    cfg: dict[str, Any],
) -> dict[str, set[str]]:
    """Reverse unique per-frame parent evidence into parent -> child anchors.

    A lower-level track is an identity anchor only when a frame gives it one
    unambiguous parent.  The mechanism is role based: U can anchor C and C/U
    can anchor O.  No semantic label is enumerated here.
    """

    require_stable = bool(cfg.get("anchor_require_stable_child", True))
    min_parent_frames = max(1, int(cfg.get("anchor_min_parent_support_frames", 1)))
    parent_to_children: dict[str, set[str]] = {}
    for child in nodes.values():
        if require_stable and child.state != "stable":
            continue
        history_specs = []
        if child.role in {"C", "U"}:
            history_specs.append((child.object_parent_context_history, "O"))
        if child.role == "U":
            history_specs.append((child.carrier_parent_context_history, "C"))
        for history, expected_parent_role in history_specs:
            parents_by_frame: dict[str, set[str]] = {}
            for record in history:
                parent_id = str(record.get("parent_node_id", ""))
                frame_key = str(record.get("frame_key", ""))
                if not parent_id or not frame_key:
                    continue
                parent_id = _canonical(parent_id, aliases)
                parent = nodes.get(parent_id)
                if parent is None or parent.role != expected_parent_role:
                    continue
                parents_by_frame.setdefault(frame_key, set()).add(parent_id)
            support: dict[str, set[str]] = {}
            for frame_key, parent_ids in parents_by_frame.items():
                if len(parent_ids) != 1:
                    continue
                parent_id = next(iter(parent_ids))
                support.setdefault(parent_id, set()).add(frame_key)
            for parent_id, frame_keys in support.items():
                if len(frame_keys) >= min_parent_frames:
                    parent_to_children.setdefault(parent_id, set()).add(child.node_id)
    return parent_to_children


def _anchor_relation(
    a: MapNode3D,
    b: MapNode3D,
    *,
    parent_to_children: dict[str, set[str]],
    nodes: dict[str, MapNode3D],
    covisibility: set[tuple[str, str]],
) -> tuple[int, bool]:
    a_children = parent_to_children.get(a.node_id, set())
    b_children = parent_to_children.get(b.node_id, set())
    shared = a_children & b_children
    for a_child_id in a_children - shared:
        for b_child_id in b_children - shared:
            a_child = nodes.get(a_child_id)
            b_child = nodes.get(b_child_id)
            if a_child is None or b_child is None:
                continue
            if (
                _pair_key(a_child_id, b_child_id) in covisibility
                or _observed_together(a_child, b_child)
            ):
                return len(shared), True
    return len(shared), False


def _merge_parent_id(a: str | None, b: str | None) -> str | None:
    if not a:
        return b
    if not b:
        return a
    return a if a == b else None


def _merge_nodes(target: MapNode3D, source: MapNode3D, cfg: dict[str, Any]) -> None:
    max_points = int(cfg.get("merge_max_points_per_node", 20000))
    voxel_size = float(cfg.get("merge_voxel_size_m", 0.004))
    points = np.concatenate([target.points_world, source.points_world], axis=0)
    colors = np.concatenate([target.colors_rgb, source.colors_rgb], axis=0)
    points, colors = voxel_downsample_points_colors(
        points, colors, voxel_size, max_points=max_points
    )
    points, colors = spatially_uniform_sample(
        points, colors, max_points=max_points
    )
    target.points_world = points
    target.colors_rgb = colors
    target.core_points_world = points.copy()
    refresh_geometry(target)

    target_assoc = np.asarray(
        getattr(target, "association_points_world", None)
        if getattr(target, "association_points_world", None) is not None
        else target.points_world,
        dtype=np.float32,
    )
    source_assoc = np.asarray(
        getattr(source, "association_points_world", None)
        if getattr(source, "association_points_world", None) is not None
        else source.points_world,
        dtype=np.float32,
    )
    association_points = np.concatenate([target_assoc, source_assoc], axis=0)
    association_points, _ = voxel_downsample_points_colors(
        association_points,
        np.zeros_like(association_points),
        float(cfg.get("association_voxel_size_m", voxel_size)),
        max_points=int(cfg.get("association_max_points_per_node", 4096)),
    )
    target.association_points_world = association_points
    target.association_bbox_min_world = np.percentile(
        association_points, 2.0, axis=0
    ).astype(np.float32)
    target.association_bbox_max_world = np.percentile(
        association_points, 98.0, axis=0
    ).astype(np.float32)
    target.association_bbox_diag_world = float(
        np.linalg.norm(
            np.maximum(
                target.association_bbox_max_world
                - target.association_bbox_min_world,
                0.0,
            )
        )
    )

    target_weight = max(float(target.semantic_weight_sum), 1e-6)
    source_weight = max(float(source.semantic_weight_sum), 1e-6)
    total_weight = target_weight + source_weight
    target.semantic_feature = l2_normalize(
        target.semantic_feature * target_weight
        + source.semantic_feature * source_weight
    )
    target.appearance_hist = l2_normalize(
        target.appearance_hist * target_weight
        + source.appearance_hist * source_weight
    )
    target.semantic_weight_sum = total_weight
    for label, vote in source.label_votes.items():
        target.label_votes[label] = target.label_votes.get(label, 0.0) + float(vote)
    target.top_label = sorted(
        target.label_votes.items(), key=lambda item: (-item[1], item[0])
    )[0][0]

    target.obs_count += int(source.obs_count)
    target.first_seen_frame = min(target.first_seen_frame, source.first_seen_frame)
    if source.last_seen_frame > target.last_seen_frame:
        target.last_box_xyxy = source.last_box_xyxy
        target.last_frame_key = source.last_frame_key
    target.last_seen_frame = max(target.last_seen_frame, source.last_seen_frame)
    target.observed_frames = sorted(set(target.observed_frames + source.observed_frames))
    target.observation_ids = list(
        dict.fromkeys(target.observation_ids + source.observation_ids)
    )
    target.support_frames = sorted(set(target.support_frames + source.support_frames))
    target.real_depth_support_frames = sorted(
        set(target.real_depth_support_frames + source.real_depth_support_frames)
    )
    target.imputed_support_frames = sorted(
        set(target.imputed_support_frames + source.imputed_support_frames)
    )
    target.identity_support_frames = sorted(
        set(target.identity_support_frames + source.identity_support_frames)
    )
    target.cross_role_duplicate_witness_frames = sorted(
        set(
            getattr(target, "cross_role_duplicate_witness_frames", [])
            + getattr(source, "cross_role_duplicate_witness_frames", [])
        )
    )
    target.recent_centroids = (target.recent_centroids + source.recent_centroids)[-20:]
    target.recent_bboxes = (target.recent_bboxes + source.recent_bboxes)[-20:]
    target.recent_quality = (target.recent_quality + source.recent_quality)[-20:]
    target.object_parent_context_history.extend(source.object_parent_context_history)
    target.carrier_parent_context_history.extend(source.carrier_parent_context_history)
    target.parent_context_history.extend(source.parent_context_history)
    target.stable_object_parent_id = _merge_parent_id(
        target.stable_object_parent_id, source.stable_object_parent_id
    )
    target.stable_carrier_parent_id = _merge_parent_id(
        target.stable_carrier_parent_id, source.stable_carrier_parent_id
    )
    target.stable_parent_id = _merge_parent_id(
        target.stable_parent_id, source.stable_parent_id
    )
    target.num_geometry_accepts += int(source.num_geometry_accepts)
    target.num_geometry_rejects += int(source.num_geometry_rejects)
    target.geometry_quality_ema = float(
        (
            target.geometry_quality_ema * target_weight
            + source.geometry_quality_ema * source_weight
        )
        / total_weight
    )
    if source.anchor_quality > target.anchor_quality:
        target.anchor_observation_id = source.anchor_observation_id
        target.anchor_quality = source.anchor_quality
        target.anchor_points_world = source.anchor_points_world.copy()
        target.anchor_touches_border = source.anchor_touches_border
    if target.state != "stable" and source.state == "stable":
        target.state = "stable"
    target.graph_eligible = False
    target.graph_node_confidence = 0.0
    target.graph_eligibility_reason = "not_evaluated_after_consolidation"


def _target_source(a: MapNode3D, b: MapNode3D) -> tuple[MapNode3D, MapNode3D]:
    def priority(node: MapNode3D) -> tuple[int, int, float, int]:
        return (
            int(node.state == "stable"),
            int(node.obs_count),
            float(node.anchor_quality),
            -int(node.node_id[1:]),
        )

    return (a, b) if priority(a) >= priority(b) else (b, a)


def retire_cross_role_duplicates(
    nodes: dict[str, MapNode3D],
    covisibility: set[tuple[str, str]],
    cfg: dict[str, Any],
) -> list[ConsolidationEvent]:
    """Retire old C-only tracks proven to duplicate a U interpretation.

    A U track becomes eligible only after an actual same-frame C/U box
    collision was observed during lifting.  Final retirement then requires
    strong 3D identity, so a merely nearby, legitimate C-U hierarchy is kept.
    The rule is role based and does not name cap, lid, or any other label.
    """

    if not bool(cfg.get("cross_role_duplicate_retirement_enabled", True)):
        return []
    witnesses = [
        node
        for node in nodes.values()
        if node.role == "U"
        and getattr(node, "cross_role_duplicate_witness_frames", [])
    ]
    if not witnesses:
        return []

    centroid_floor = float(
        cfg.get("cross_role_duplicate_max_centroid_distance_m", 0.03)
    )
    centroid_ratio = float(
        cfg.get("cross_role_duplicate_max_centroid_diag_ratio", 0.35)
    )
    point_floor = float(cfg.get("cross_role_duplicate_point_distance_m", 0.01))
    point_ratio = float(
        cfg.get("cross_role_duplicate_point_distance_diag_ratio", 0.0)
    )
    min_overlap_required = float(
        cfg.get("cross_role_duplicate_min_directional_point_overlap", 0.45)
    )
    max_overlap_required = float(
        cfg.get("cross_role_duplicate_min_one_way_point_overlap", 0.80)
    )
    max_points = int(cfg.get("merge_overlap_max_points", 1024))
    events: list[ConsolidationEvent] = []

    for carrier in sorted(
        (node for node in nodes.values() if node.role == "C"),
        key=lambda node: node.node_id,
    ):
        matches: list[tuple[float, float, float, MapNode3D, dict[str, float]]] = []
        for unit in witnesses:
            max_extent = max(
                _association_extent(carrier), _association_extent(unit)
            )
            min_extent = min(
                _association_extent(carrier), _association_extent(unit)
            )
            centroid_distance = float(
                np.linalg.norm(carrier.centroid_world - unit.centroid_world)
            )
            centroid_limit = max(centroid_floor, centroid_ratio * max_extent)
            if centroid_distance > centroid_limit:
                continue
            point_distance = max(point_floor, point_ratio * min_extent)
            overlap_cu, overlap_uc = _directional_point_overlaps(
                carrier,
                unit,
                threshold=point_distance,
                max_points=max_points,
            )
            min_overlap = min(overlap_cu, overlap_uc)
            max_overlap = max(overlap_cu, overlap_uc)
            if (
                min_overlap < min_overlap_required
                or max_overlap < max_overlap_required
            ):
                continue
            metrics = {
                "centroid_distance_m": centroid_distance,
                "centroid_distance_limit_m": centroid_limit,
                "point_distance_m": point_distance,
                "point_overlap_carrier_to_unit": overlap_cu,
                "point_overlap_unit_to_carrier": overlap_uc,
                "min_directional_point_overlap": min_overlap,
                "max_directional_point_overlap": max_overlap,
                "unit_witness_frames": float(
                    len(getattr(unit, "cross_role_duplicate_witness_frames", []))
                ),
            }
            matches.append(
                (-min_overlap, -max_overlap, centroid_distance, unit, metrics)
            )
        if not matches:
            continue
        _neg_min, _neg_max, _distance, unit, metrics = sorted(
            matches, key=lambda item: (item[0], item[1], item[2], item[3].node_id)
        )[0]
        nodes.pop(carrier.node_id, None)
        events.append(
            ConsolidationEvent(
                event_type="retire",
                source_node_id=carrier.node_id,
                target_node_id=unit.node_id,
                reason="cross_role_duplicate_witness_and_geometry",
                metrics=metrics,
            )
        )

    if not events:
        return []
    retired_ids = {event.source_node_id for event in events}
    retired_pairs = {
        pair
        for pair in covisibility
        if pair[0] in retired_ids or pair[1] in retired_ids
    }
    covisibility.difference_update(retired_pairs)
    for node in nodes.values():
        if node.stable_object_parent_id in retired_ids:
            node.stable_object_parent_id = None
        if node.stable_carrier_parent_id in retired_ids:
            node.stable_carrier_parent_id = None
        if node.stable_parent_id in retired_ids:
            node.stable_parent_id = None
    return events


def conservative_consolidate(
    nodes: dict[str, MapNode3D],
    covisibility: set[tuple[str, str]],
    cfg: dict[str, Any],
    aliases: dict[str, str] | None = None,
) -> list[ConsolidationEvent]:
    """Merge strong, non-covisible track fragments without label special cases."""

    if not bool(cfg.get("enabled", True)):
        return []
    aliases = aliases if aliases is not None else {}
    events: list[ConsolidationEvent] = []
    min_semantic = float(cfg.get("merge_min_semantic_similarity", 0.98))
    min_appearance = float(cfg.get("merge_min_appearance_similarity", 0.0))
    max_centroid = float(cfg.get("merge_max_centroid_distance_m", 0.05))
    max_centroid_diag_ratio = float(cfg.get("merge_max_centroid_diag_ratio", 0.75))
    anchored_centroid_diag_ratio = float(
        cfg.get("merge_anchored_max_centroid_diag_ratio", 1.0)
    )
    anchored_max_surface_gap = float(
        cfg.get("merge_anchored_max_surface_gap_m", 0.03)
    )
    anchored_max_surface_gap_ratio = float(
        cfg.get("merge_anchored_max_surface_gap_diag_ratio", 0.15)
    )
    min_bbox_iou = float(cfg.get("merge_min_bbox_iou", 0.30))
    bbox_padding = float(cfg.get("merge_bbox_padding_m", 0.005))
    bbox_padding_diag_ratio = float(cfg.get("merge_bbox_padding_diag_ratio", 0.01))
    point_threshold = float(cfg.get("merge_point_distance_m", 0.01))
    point_threshold_diag_ratio = float(
        cfg.get("merge_point_distance_diag_ratio", 0.0)
    )
    min_directional_overlap = float(
        cfg.get("merge_min_directional_point_overlap", 0.35)
    )
    min_one_way_overlap = float(cfg.get("merge_min_one_way_point_overlap", 0.65))
    overlap_max_points = int(cfg.get("merge_overlap_max_points", 1024))

    while True:
        parent_to_children = _identity_anchor_index(nodes, aliases, cfg)
        candidates: list[
            tuple[int, float, float, float, float, str, str, dict[str, float]]
        ] = []
        ids = sorted(nodes)
        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1 :]:
                a, b = nodes[a_id], nodes[b_id]
                if a.role != b.role:
                    continue
                if _normal_label(a.top_label) != _normal_label(b.top_label):
                    continue
                # The runtime cache is the primary source; observed_frames is
                # an independent hard fallback for old/resumed checkpoints.
                if (
                    _pair_key(a_id, b_id) in covisibility
                    or _observed_together(a, b)
                ):
                    continue
                semantic = cosine01(a.semantic_feature, b.semantic_feature)
                if semantic < min_semantic:
                    continue
                appearance = cosine01(a.appearance_hist, b.appearance_hist)
                if appearance < min_appearance:
                    continue
                shared_anchor_count, anchor_conflict = _anchor_relation(
                    a,
                    b,
                    parent_to_children=parent_to_children,
                    nodes=nodes,
                    covisibility=covisibility,
                )
                if anchor_conflict:
                    continue
                distance = float(np.linalg.norm(a.centroid_world - b.centroid_world))
                max_extent = max(_association_extent(a), _association_extent(b))
                min_extent = min(_association_extent(a), _association_extent(b))
                centroid_ratio = (
                    anchored_centroid_diag_ratio
                    if shared_anchor_count > 0
                    else max_centroid_diag_ratio
                )
                adaptive_centroid_limit = max(
                    max_centroid,
                    centroid_ratio * max_extent,
                )
                if distance > adaptive_centroid_limit:
                    continue
                adaptive_bbox_padding = max(
                    bbox_padding, bbox_padding_diag_ratio * min_extent
                )
                bbox_iou = _bbox_iou(a, b, padding=adaptive_bbox_padding)
                adaptive_point_threshold = max(
                    point_threshold, point_threshold_diag_ratio * min_extent
                )
                overlap_ab, overlap_ba = _directional_point_overlaps(
                    a,
                    b,
                    threshold=adaptive_point_threshold,
                    max_points=overlap_max_points,
                )
                min_overlap = min(overlap_ab, overlap_ba)
                max_overlap = max(overlap_ab, overlap_ba)
                strict_geometric_duplicate = bool(
                    bbox_iou >= min_bbox_iou
                    and min_overlap >= min_directional_overlap
                    and max_overlap >= min_one_way_overlap
                )
                surface_gap = _bbox_surface_gap(a, b)
                anchored_gap_limit = max(
                    anchored_max_surface_gap,
                    anchored_max_surface_gap_ratio * max_extent,
                )
                anchored_complementary_fragment = bool(
                    shared_anchor_count > 0 and surface_gap <= anchored_gap_limit
                )
                if not (
                    strict_geometric_duplicate or anchored_complementary_fragment
                ):
                    continue
                metrics = {
                    "semantic_similarity": semantic,
                    "appearance_similarity": appearance,
                    "centroid_distance_m": distance,
                    "centroid_distance_limit_m": adaptive_centroid_limit,
                    "bbox_iou": bbox_iou,
                    "bbox_padding_m": adaptive_bbox_padding,
                    "point_distance_m": adaptive_point_threshold,
                    "point_overlap_a_to_b": overlap_ab,
                    "point_overlap_b_to_a": overlap_ba,
                    "min_directional_point_overlap": min_overlap,
                    "max_directional_point_overlap": max_overlap,
                    "surface_gap_m": surface_gap,
                    "surface_gap_limit_m": anchored_gap_limit,
                    "shared_identity_anchors": float(shared_anchor_count),
                    "strict_geometric_duplicate": float(strict_geometric_duplicate),
                }
                candidates.append(
                    (
                        -int(shared_anchor_count > 0),
                        surface_gap / max(max_extent, 1e-6),
                        -min_overlap,
                        -max_overlap,
                        -bbox_iou,
                        a_id,
                        b_id,
                        metrics,
                    )
                )
        if not candidates:
            break
        (
            _neg_has_anchor,
            _normalized_gap,
            _neg_min_overlap,
            _neg_max_overlap,
            _neg_iou,
            a_id,
            b_id,
            metrics,
        ) = sorted(candidates)[0]
        target, source = _target_source(nodes[a_id], nodes[b_id])
        _merge_nodes(target, source, cfg)
        nodes.pop(source.node_id)
        aliases[source.node_id] = target.node_id
        for key, value in list(aliases.items()):
            if value == source.node_id:
                aliases[key] = target.node_id
        events.append(
            ConsolidationEvent(
                event_type="merge",
                source_node_id=source.node_id,
                target_node_id=target.node_id,
                reason=(
                    "shared_child_anchor_complementary_fragment"
                    if metrics["shared_identity_anchors"] > 0
                    and not metrics["strict_geometric_duplicate"]
                    else "non_covisible_strong_geometric_duplicate"
                ),
                metrics=metrics,
            )
        )

        remapped: set[tuple[str, str]] = set()
        for left, right in covisibility:
            left = target.node_id if left == source.node_id else left
            right = target.node_id if right == source.node_id else right
            if left != right:
                remapped.add(_pair_key(left, right))
        covisibility.clear()
        covisibility.update(remapped)
    return events
