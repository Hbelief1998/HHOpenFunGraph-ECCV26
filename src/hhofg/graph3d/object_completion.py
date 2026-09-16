from __future__ import annotations

import re
from typing import Any

import numpy as np

from hhofg.mapping3d.semantic_candidate_builder import (
    SemanticRelationIndex,
    labels_compatible,
    normalize_semantic_label,
)
from hhofg.mapping3d.types import MapNode3D


def _association_bounds(node: MapNode3D) -> tuple[np.ndarray, np.ndarray]:
    lower = (
        node.association_bbox_min_world
        if node.association_bbox_min_world is not None
        else node.bbox_min_world
    )
    upper = (
        node.association_bbox_max_world
        if node.association_bbox_max_world is not None
        else node.bbox_max_world
    )
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _aabb_surface_gap(first: MapNode3D, second: MapNode3D) -> float:
    first_min, first_max = _association_bounds(first)
    second_min, second_max = _association_bounds(second)
    gap = np.maximum(np.maximum(first_min - second_max, second_min - first_max), 0.0)
    return float(np.linalg.norm(gap))


def _box_iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(
        intersection / max(first_area + second_area - intersection, 1e-12)
    )


def _surface_normal(node: MapNode3D, max_points: int) -> np.ndarray | None:
    points = np.asarray(
        node.association_points_world
        if node.association_points_world is not None
        else node.points_world,
        dtype=np.float64,
    )
    if len(points) < 3:
        return None
    points = points[_sample_indices(len(points), max_points)]
    centered = points - np.median(points, axis=0)
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues[-1]) <= 1e-12:
        return None
    normal = eigenvectors[:, 0]
    return normal / max(float(np.linalg.norm(normal)), 1e-12)


def _semantic_contexts(evidence: dict[str, Any]) -> set[str]:
    contexts: set[str] = set()
    direct = evidence.get("object_contexts", [])
    if isinstance(direct, dict):
        contexts.update(
            normalize_semantic_label(label)
            for label, count in direct.items()
            if int(count) > 0
        )
    else:
        contexts.update(normalize_semantic_label(label) for label in direct or [])
    for chain in evidence.get("semantic_chains", []) or []:
        if not isinstance(chain, dict):
            continue
        contexts.update(
            normalize_semantic_label(label)
            for label in chain.get("object_contexts", []) or []
        )
    return {context for context in contexts if context}


def _context_matches(owner_label: str, contexts: set[str]) -> bool:
    return bool(contexts) and any(
        labels_compatible(owner_label, context)
        or labels_compatible(context, owner_label)
        for context in contexts
    )


def _component_normal_is_consistent(
    members: list[MapNode3D],
    normals: dict[str, np.ndarray],
    min_abs_dot: float,
) -> bool:
    available = [normals[node.node_id] for node in members if node.node_id in normals]
    if len(available) != len(members):
        return False
    reference = available[0]
    aligned = [normal if float(normal @ reference) >= 0.0 else -normal for normal in available]
    consensus = np.sum(aligned, axis=0)
    length = float(np.linalg.norm(consensus))
    if length <= 1e-12:
        return False
    consensus /= length
    return all(abs(float(normal @ consensus)) >= min_abs_dot for normal in available)


def _evidence_link_groups(
    *,
    owner_label: str,
    nodes: list[MapNode3D],
    observation_boxes: dict[str, dict[str, list[float]]],
    owner_hints: dict[str, set[tuple[str, str, int]]],
    cfg: dict[str, Any],
) -> tuple[list[list[MapNode3D]], list[dict[str, Any]]]:
    """Build owner membership components from shared 2D hints and 3D geometry."""

    max_gap = float(cfg.get("max_pair_surface_gap_m", 0.12))
    min_normal_dot = float(cfg.get("min_pca_normal_abs_dot", 0.90))
    min_iou = float(cfg.get("min_covis_box_iou", 1e-4))
    min_frames = max(int(cfg.get("min_covis_overlap_frames", 1)), 1)
    min_hint_frames = max(int(cfg.get("min_shared_owner_hint_frames", 1)), 1)
    normal_points = max(int(cfg.get("pca_max_points", 2048)), 3)
    normals = {
        node.node_id: normal
        for node in nodes
        if (normal := _surface_normal(node, normal_points)) is not None
    }
    links: list[dict[str, Any]] = []
    ordered_nodes = sorted(nodes, key=lambda item: item.node_id)
    for index, first in enumerate(ordered_nodes):
        for second in ordered_nodes[index + 1 :]:
            first_hints = {
                hint
                for hint in owner_hints.get(first.node_id, set())
                if labels_compatible(hint[0], owner_label)
                or labels_compatible(owner_label, hint[0])
            }
            second_hints = {
                hint
                for hint in owner_hints.get(second.node_id, set())
                if labels_compatible(hint[0], owner_label)
                or labels_compatible(owner_label, hint[0])
            }
            shared_hints = first_hints & second_hints
            shared_frames = {hint[1] for hint in shared_hints}
            if len(shared_frames) < min_hint_frames:
                continue
            overlap_ious = [
                _box_iou(
                    observation_boxes[first.node_id][frame_key],
                    observation_boxes[second.node_id][frame_key],
                )
                for frame_key in sorted(shared_frames)
                if frame_key in observation_boxes.get(first.node_id, {})
                and frame_key in observation_boxes.get(second.node_id, {})
            ]
            positive_ious = [value for value in overlap_ious if value > min_iou]
            if len(positive_ious) < min_frames:
                continue
            first_normal = normals.get(first.node_id)
            second_normal = normals.get(second.node_id)
            if first_normal is None or second_normal is None:
                continue
            normal_dot = abs(float(first_normal @ second_normal))
            if normal_dot < min_normal_dot:
                continue
            surface_gap = _aabb_surface_gap(first, second)
            if surface_gap > max_gap:
                continue
            links.append(
                {
                    "first": first.node_id,
                    "second": second.node_id,
                    "shared_owner_hint_frames": len(shared_frames),
                    "covis_overlap_frames": len(positive_ious),
                    "max_covis_box_iou": max(positive_ious),
                    "pca_normal_abs_dot": normal_dot,
                    "surface_gap_m": surface_gap,
                }
            )

    components: dict[str, list[MapNode3D]] = {
        node.node_id: [node] for node in nodes
    }
    parent = {node.node_id: node.node_id for node in nodes}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    for link in sorted(
        links,
        key=lambda item: (
            -int(item["shared_owner_hint_frames"]),
            -int(item["covis_overlap_frames"]),
            -float(item["max_covis_box_iou"]),
            float(item["surface_gap_m"]),
            str(item["first"]),
            str(item["second"]),
        ),
    ):
        first_root = find(str(link["first"]))
        second_root = find(str(link["second"]))
        if first_root == second_root:
            continue
        merged = components[first_root] + components[second_root]
        if not _component_normal_is_consistent(merged, normals, min_normal_dot):
            continue
        keep_root, drop_root = sorted((first_root, second_root))
        parent[drop_root] = keep_root
        components[keep_root] = merged
        components.pop(drop_root)

    groups = [
        sorted(group, key=lambda node: node.node_id)
        for group in components.values()
        if len(group) >= 2
    ]
    groups.sort(key=lambda group: [node.node_id for node in group])
    return groups, links


def _sample_indices(size: int, limit: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, num=limit, dtype=np.int64)


def _next_object_ids(existing_ids: set[str], count: int) -> list[str]:
    numeric = [
        int(match.group(1))
        for node_id in existing_ids
        if (match := re.fullmatch(r"O(\d+)", node_id)) is not None
    ]
    next_value = max(numeric, default=0) + 1
    result = []
    while len(result) < count:
        candidate = f"O{next_value:06d}"
        next_value += 1
        if candidate not in existing_ids:
            result.append(candidate)
    return result


def _weighted_mean(
    members: list[MapNode3D], attribute: str, weights: np.ndarray
) -> np.ndarray:
    arrays = [np.asarray(getattr(node, attribute), dtype=np.float32) for node in members]
    if not arrays or any(array.shape != arrays[0].shape for array in arrays):
        return np.zeros((0,), dtype=np.float32)
    return np.average(np.stack(arrays), axis=0, weights=weights).astype(np.float32)


def _materialize_object(
    *,
    node_id: str,
    owner_label: str,
    members: list[MapNode3D],
    max_points: int,
    max_association_points: int,
) -> MapNode3D:
    points = np.concatenate([node.points_world for node in members], axis=0)
    colors = np.concatenate([node.colors_rgb for node in members], axis=0)
    keep = _sample_indices(len(points), max_points)
    points = points[keep].astype(np.float32)
    colors = colors[keep].astype(np.float32)

    association_parts = [
        node.association_points_world
        if node.association_points_world is not None
        else node.points_world
        for node in members
    ]
    association_points = np.concatenate(association_parts, axis=0).astype(np.float32)
    association_points = association_points[
        _sample_indices(len(association_points), max_association_points)
    ]

    frames = sorted({frame for node in members for frame in node.observed_frames})
    support_frames = sorted({frame for node in members for frame in node.support_frames})
    real_depth_frames = sorted(
        {frame for node in members for frame in node.real_depth_support_frames}
    )
    weights = np.asarray(
        [max(node.graph_node_confidence, node.geometry_quality_ema, 1e-3) for node in members],
        dtype=np.float64,
    )
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    association_min = association_points.min(axis=0)
    association_max = association_points.max(axis=0)
    anchor = max(members, key=lambda node: (node.anchor_quality, node.node_id))
    recent_centroids = [
        list(map(float, node.centroid_world)) for node in members
    ]

    return MapNode3D(
        node_id=node_id,
        role="O",
        label_votes={owner_label: float(weights.sum())},
        top_label=owner_label,
        points_world=points,
        colors_rgb=colors,
        centroid_world=np.median(points, axis=0).astype(np.float32),
        bbox_min_world=bbox_min,
        bbox_max_world=bbox_max,
        bbox_diag_world=float(np.linalg.norm(bbox_max - bbox_min)),
        appearance_hist=_weighted_mean(members, "appearance_hist", weights),
        semantic_feature=_weighted_mean(members, "semantic_feature", weights),
        semantic_weight_sum=float(weights.sum()),
        obs_count=sum(node.obs_count for node in members),
        first_seen_frame=min(node.first_seen_frame for node in members),
        last_seen_frame=max(node.last_seen_frame for node in members),
        observed_frames=frames,
        observation_ids=[
            observation_id
            for node in members
            for observation_id in node.observation_ids
        ],
        last_box_xyxy=anchor.last_box_xyxy,
        last_frame_key=anchor.last_frame_key,
        association_points_world=association_points,
        association_bbox_min_world=association_min,
        association_bbox_max_world=association_max,
        association_bbox_diag_world=float(
            np.linalg.norm(association_max - association_min)
        ),
        state="stable",
        anchor_observation_id=anchor.anchor_observation_id,
        anchor_quality=float(np.mean([node.anchor_quality for node in members])),
        anchor_points_world=points.copy(),
        core_points_world=points.copy(),
        geometry_quality_ema=float(
            np.mean([node.geometry_quality_ema for node in members])
        ),
        num_geometry_accepts=sum(node.num_geometry_accepts for node in members),
        num_geometry_rejects=sum(node.num_geometry_rejects for node in members),
        support_frames=support_frames,
        real_depth_support_frames=real_depth_frames,
        identity_support_frames=sorted(
            {frame for node in members for frame in node.identity_support_frames}
        ),
        recent_centroids=recent_centroids,
        graph_eligible=True,
        graph_node_confidence=float(min(node.graph_node_confidence for node in members)),
        graph_eligibility_reason="derived_from_stable_carrier_group",
    )


def complete_deferred_objects(
    *,
    nodes: dict[str, MapNode3D],
    object_edges: list[dict[str, Any]],
    semantic_index: SemanticRelationIndex,
    deferred_object_labels: set[str],
    local_edges: list[dict[str, Any]] | None = None,
    observation_boxes: dict[str, dict[str, list[float]]] | None = None,
    owner_hints: dict[str, set[tuple[str, str, int]]] | None = None,
    cfg: dict[str, Any] | None = None,
    reserved_node_ids: set[str] | None = None,
) -> tuple[dict[str, MapNode3D], list[dict[str, Any]], dict[str, Any]]:
    """Materialize missing objects from stable, semantically scoped carriers.

    The policy supplies only which object labels are intentionally deferred.
    Atlas O-C priors define legal owners.  A carrier participates only after
    it wins a typed, observed C-U relation.  Shared deferred-owner hints,
    raw SAM3 boxes and orientation-independent 3D geometry then define a
    transitive membership graph.  No carrier or object label is hard-coded.
    """

    cfg = cfg or {}
    local_edges = local_edges or []
    observation_boxes = observation_boxes or {}
    owner_hints = owner_hints or {}
    enabled = bool(cfg.get("enabled", True))
    deferred = {
        normalize_semantic_label(label) for label in deferred_object_labels if label
    }
    empty_meta = {
        "enabled": enabled,
        "deferred_object_labels": sorted(deferred),
        "num_unowned_carriers_considered": 0,
        "num_ambiguous_owner_carriers": 0,
        "num_carriers_without_qualified_unit": 0,
        "num_carriers_without_owner_hint": 0,
        "num_membership_links": 0,
        "num_singleton_objects_from_repeated_hint": 0,
        "num_derived_objects": 0,
        "num_derived_object_edges": 0,
        "qualified_unit_winners": [],
        "membership_links": [],
        "derived_objects": [],
    }
    if not enabled or not deferred:
        return {}, [], empty_meta

    owned_carriers = {
        str(edge.get("child_node_id"))
        for edge in object_edges
        if str(edge.get("edge_type")) == "O-C"
        and not edge.get("provisional", False)
        and edge.get("status") != "provisional"
    }
    owner_members: dict[str, list[MapNode3D]] = {}
    ambiguous = 0
    considered = 0
    for carrier in nodes.values():
        if carrier.role != "C" or carrier.node_id in owned_carriers:
            continue
        if not carrier.graph_eligible or carrier.state != "stable":
            continue
        considered += 1
        possible = semantic_index.parent_labels_for(
            "O-C",
            carrier.top_label,
            allowed_parent_labels=deferred,
        )
        possible = {
            owner
            for owner in possible
            if any(
                labels_compatible(owner, label) or labels_compatible(label, owner)
                for label in deferred
            )
        }
        if len(possible) != 1:
            ambiguous += int(len(possible) > 1)
            continue
        owner_members.setdefault(next(iter(possible)), []).append(carrier)

    min_members = max(int(cfg.get("min_member_carriers", 2)), 2)
    groups: list[tuple[str, list[MapNode3D]]] = []
    rejected_small_groups = 0
    without_unit = 0
    without_hint = 0
    all_unit_winners: list[dict[str, Any]] = []
    all_membership_links: list[dict[str, Any]] = []
    singleton_objects = 0
    for owner_label, carriers in sorted(owner_members.items()):
        carrier_ids = {c.node_id for c in carriers}
        unit_winners = [
            {"carrier_id": e["parent_node_id"], "unit_id": e["child_node_id"],
             "owner_label": owner_label, "strong_2d_count": e.get("strong_2d_count", 0)}
            for e in local_edges
            if e["edge_type"] == "C-U" and e["parent_node_id"] in carrier_ids
            and int(e.get("strong_2d_count", 0)) > 0
            and _context_matches(owner_label, _semantic_contexts(e))
        ]
        active_ids = {e["carrier_id"] for e in unit_winners}
        all_unit_winners.extend(unit_winners)
        without_unit += sum(carrier.node_id not in active_ids for carrier in carriers)
        active = [carrier for carrier in carriers if carrier.node_id in active_ids]
        hinted = [
            carrier
            for carrier in active
            if any(
                labels_compatible(hint[0], owner_label)
                or labels_compatible(owner_label, hint[0])
                for hint in owner_hints.get(carrier.node_id, set())
            )
        ]
        without_hint += len(active) - len(hinted)
        owner_groups, membership_links = _evidence_link_groups(
            owner_label=owner_label,
            nodes=hinted,
            observation_boxes=observation_boxes,
            owner_hints=owner_hints,
            cfg=cfg,
        )
        for link in membership_links:
            all_membership_links.append({"owner_label": owner_label, **link})
        grouped_ids = {
            member.node_id for group in owner_groups for member in group
        }
        min_singleton_hint_frames = max(
            int(cfg.get("min_singleton_owner_hint_frames", 2)), 2
        )
        for carrier in hinted:
            if carrier.node_id in grouped_ids:
                continue
            hint_frames = {
                hint[1]
                for hint in owner_hints.get(carrier.node_id, set())
                if labels_compatible(hint[0], owner_label)
                or labels_compatible(owner_label, hint[0])
            }
            if len(hint_frames) >= min_singleton_hint_frames:
                owner_groups.append([carrier])
                grouped_ids.add(carrier.node_id)
                singleton_objects += 1
        rejected_small_groups += len(hinted) - len(grouped_ids)
        groups.extend(
            (owner_label, group)
            for group in owner_groups
            if len(group) >= min_members or len(group) == 1
        )

    # Graph eligibility is a subset of the map. Never reuse an ineligible
    # node's ID: stored observations/covisibility still refer to that ID.
    object_ids = _next_object_ids(set(nodes) | set(reserved_node_ids or ()), len(groups))
    derived_nodes: dict[str, MapNode3D] = {}
    derived_edges: list[dict[str, Any]] = []
    provenance = []
    for object_id, (owner_label, members) in zip(object_ids, groups):
        derived = _materialize_object(
            node_id=object_id,
            owner_label=owner_label,
            members=members,
            max_points=max(int(cfg.get("max_points_per_object", 20000)), 1),
            max_association_points=max(
                int(cfg.get("max_association_points_per_object", 4096)), 1
            ),
        )
        derived_nodes[object_id] = derived
        member_ids = sorted(member.node_id for member in members)
        provenance.append(
            {
                "node_id": object_id,
                "label": owner_label,
                "member_carrier_ids": member_ids,
            }
        )
        for member in sorted(members, key=lambda node: node.node_id):
            metadata = semantic_index.metadata_for(
                "O-C", owner_label, member.top_label, object_context=owner_label
            )
            derived_edges.append(
                {
                    "parent_node_id": object_id,
                    "child_node_id": member.node_id,
                    "edge_type": "O-C",
                    "parent_role": "O",
                    "child_role": "C",
                    "parent_label": owner_label,
                    "child_label": member.top_label,
                    "source": "derived_object_completion",
                    "object_completion_method": (
                        "typed_unit_owner_hint_covis_pca_chain"
                    ),
                    "member_carrier_ids": member_ids,
                    "semantic_sources": metadata.get("semantic_sources", []),
                    "semantic_chains": [metadata] if metadata else [],
                    "relation_text": str(metadata.get("relation_text", "")),
                }
            )

    meta = {
        **empty_meta,
        "num_unowned_carriers_considered": considered,
        "num_ambiguous_owner_carriers": ambiguous,
        "num_carriers_without_qualified_unit": without_unit,
        "num_carriers_without_owner_hint": without_hint,
        "num_rejected_small_carrier_groups": rejected_small_groups,
        "num_membership_links": len(all_membership_links),
        "num_singleton_objects_from_repeated_hint": singleton_objects,
        "num_derived_objects": len(derived_nodes),
        "num_derived_object_edges": len(derived_edges),
        "qualified_unit_winners": all_unit_winners,
        "membership_links": all_membership_links,
        "derived_objects": provenance,
    }
    return derived_nodes, derived_edges, meta
