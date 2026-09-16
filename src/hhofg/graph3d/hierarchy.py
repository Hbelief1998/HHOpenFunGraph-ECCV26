from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - exercised only in minimal installs
    cKDTree = None

from hhofg.mapping3d.semantic_candidate_builder import labels_compatible
from hhofg.mapping3d.types import MapNode3D


def carrier_unit_surface_distance(
    carrier: MapNode3D,
    unit: MapNode3D,
    *,
    max_points: int = 1024,
    near_fraction: float = 0.20,
) -> float:
    """Robust point-to-surface distance for hierarchy ranking.

    Association geometry preserves the detected instance extent and is more
    appropriate here than the conservative fused core.  The median of the
    nearest 20% unit-to-carrier distances captures physical contact while
    resisting a few noisy depth points.
    """

    carrier_points = np.asarray(
        carrier.association_points_world
        if carrier.association_points_world is not None
        else carrier.points_world,
        dtype=np.float64,
    )
    unit_points = np.asarray(
        unit.association_points_world
        if unit.association_points_world is not None
        else unit.points_world,
        dtype=np.float64,
    )
    if len(carrier_points) > max_points:
        carrier_points = carrier_points[
            np.linspace(0, len(carrier_points) - 1, max_points, dtype=np.int64)
        ]
    if len(unit_points) > max_points:
        unit_points = unit_points[
            np.linspace(0, len(unit_points) - 1, max_points, dtype=np.int64)
        ]
    if cKDTree is not None:
        distances = cKDTree(carrier_points).query(unit_points, k=1, workers=1)[0]
    else:
        distances = np.empty(len(unit_points), dtype=np.float64)
        for start in range(0, len(unit_points), 128):
            block = unit_points[start : start + 128]
            squared = np.sum(
                (block[:, None, :] - carrier_points[None, :, :]) ** 2, axis=-1
            )
            distances[start : start + len(block)] = np.sqrt(np.min(squared, axis=1))
    count = max(1, int(np.ceil(len(distances) * np.clip(near_fraction, 0.01, 1.0))))
    nearest = np.partition(distances, count - 1)[:count]
    return float(np.median(nearest))


def _association_aabb_distance(carrier: MapNode3D, unit: MapNode3D) -> float:
    carrier_min = np.asarray(
        carrier.association_bbox_min_world
        if carrier.association_bbox_min_world is not None
        else carrier.bbox_min_world,
        dtype=np.float64,
    )
    carrier_max = np.asarray(
        carrier.association_bbox_max_world
        if carrier.association_bbox_max_world is not None
        else carrier.bbox_max_world,
        dtype=np.float64,
    )
    unit_min = np.asarray(
        unit.association_bbox_min_world
        if unit.association_bbox_min_world is not None
        else unit.bbox_min_world,
        dtype=np.float64,
    )
    unit_max = np.asarray(
        unit.association_bbox_max_world
        if unit.association_bbox_max_world is not None
        else unit.bbox_max_world,
        dtype=np.float64,
    )
    gap = np.maximum(np.maximum(carrier_min - unit_max, unit_min - carrier_max), 0.0)
    return float(np.linalg.norm(gap))


def _semantic_sources(evidence: dict[str, Any]) -> set[str]:
    value = evidence.get("semantic_sources", [])
    if isinstance(value, dict):
        return {str(key) for key, count in value.items() if int(count) > 0}
    return {str(item) for item in value or []}


def semantic_compatibility(
    evidence: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[float | None, str, int, int]:
    sources = _semantic_sources(evidence)
    support_frames = int(evidence.get("semantic_chain_support_frames", 0))
    allowed_support = int(evidence.get("allowed_parent_support", 0))
    if "atlas" in sources:
        return (
            float(cfg.get("cprior_atlas", 0.90)),
            "atlas",
            support_frames,
            allowed_support,
        )
    if sources & {"frame_result", "graph_frame_result"}:
        return (
            float(cfg.get("cprior_frame", 0.85)),
            "frame_result",
            support_frames,
            allowed_support,
        )
    if "allowed_parent_map" in sources:
        return (
            float(cfg.get("cprior_allowed_parent", 0.75)),
            "allowed_parent_map",
            support_frames,
            max(1, allowed_support),
        )
    return None, "no_semantic_support", support_frames, allowed_support


def _semantic_context_values(evidence: dict[str, Any], key: str) -> set[str]:
    direct = evidence.get(key, [])
    values: set[str] = (
        {str(name) for name, count in direct.items() if int(count) > 0}
        if isinstance(direct, dict)
        else {str(name) for name in direct or [] if str(name)}
    )
    edge_type = str(evidence.get("edge_type") or "")
    for context in evidence.get("semantic_chains", []) or []:
        if not isinstance(context, dict):
            continue
        if str(context.get("edge_type") or edge_type) != edge_type:
            continue
        values.update(str(value) for value in context.get(key, []) or [] if str(value))
    return values


def _has_strong_direct_object_context(
    *,
    object_edge: dict[str, Any],
    object_evidence: dict[str, Any] | None,
    nodes: dict[str, MapNode3D],
) -> bool:
    if object_edge.get("provisional") or object_edge.get("status") == "provisional":
        return False
    if object_evidence is None:
        return False
    if int(object_evidence.get("strong_2d_count", 0)) <= 0:
        return False
    if "direct" not in _semantic_context_values(object_evidence, "owner_kinds"):
        return False
    parent = nodes.get(str(object_edge.get("parent_node_id") or ""))
    if parent is None:
        return False
    object_contexts = _semantic_context_values(
        object_evidence, "object_contexts"
    )
    return bool(object_contexts) and any(
        labels_compatible(parent.top_label, context)
        for context in object_contexts
    )


def resolve_immediate_parent_edges(
    *,
    nodes: dict[str, MapNode3D],
    optimized_edges: list[dict[str, Any]],
    candidate_edges: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve local ownership independently of a carrier's object proposal.

    Unknown object ownership is not negative local evidence. Observed C-U may
    correct a weak O-U fallback; confirmed direct O-U remains protected.
    """

    cfg = cfg or {}
    require_semantic = bool(cfg.get("require_semantic_cu_support", True))
    max_surface_distance = float(cfg.get("max_surface_distance_m", 0.08))
    min_score = float(cfg.get("min_hierarchy_score", 0.80))
    min_margin = max(float(cfg.get("min_carrier_margin", 0.0)), 0.0)
    distance_scale = max(float(cfg.get("gnear_scale_m", 0.03)), 1e-6)
    max_points = max(int(cfg.get("surface_distance_max_points", 1024)), 1)
    near_fraction = float(cfg.get("surface_distance_near_fraction", 0.20))
    semantic_weight = max(float(cfg.get("semantic_weight", 0.30)), 0.0)
    geometry_weight = max(float(cfg.get("surface_weight", 0.30)), 0.0)
    observation_weight = max(float(cfg.get("observation_weight", 0.40)), 0.0)
    weight_sum = max(
        semantic_weight + geometry_weight + observation_weight, 1e-9
    )

    object_parent: dict[str, str] = {}
    direct_ou: dict[str, dict[str, Any]] = {}
    oc_edges: dict[str, dict[str, Any]] = {}
    for edge in optimized_edges:
        edge_type = str(edge.get("edge_type"))
        parent_id = str(edge.get("parent_node_id"))
        child_id = str(edge.get("child_node_id"))
        if edge_type == "O-C":
            object_parent[child_id] = parent_id
            oc_edges[child_id] = dict(edge)
        elif edge_type == "O-U":
            object_parent[child_id] = parent_id
            direct_ou[child_id] = dict(edge)

    object_evidence = {
        (str(item.get("parent_node_id")), str(item.get("child_node_id"))): item
        for item in candidate_edges
        if str(item.get("edge_type")) == "O-U"
    }
    candidates_by_unit: dict[str, list[dict[str, Any]]] = {}
    rejected_reasons: Counter[str] = Counter()
    surface_distances: list[float] = []
    hierarchy_scores: list[float] = []

    for evidence in candidate_edges:
        if str(evidence.get("edge_type")) != "C-U":
            continue
        carrier_id = str(evidence.get("parent_node_id"))
        unit_id = str(evidence.get("child_node_id"))
        carrier = nodes.get(carrier_id)
        unit = nodes.get(unit_id)
        if carrier is None or unit is None:
            rejected_reasons["node_missing"] += 1
            continue
        if carrier.role != "C" or unit.role != "U":
            rejected_reasons["illegal_role_pair"] += 1
            continue

        object_for_carrier = object_parent.get(carrier_id)
        object_for_unit = object_parent.get(unit_id)
        strong_2d_count = int(evidence.get("strong_2d_count", 0))
        object_edge = direct_ou.get(unit_id)
        # Class-context disagreement is not proof of an instance conflict.
        # Protect a confirmed direct O-U, but never lock an unknown/weak O.
        if object_edge and _has_strong_direct_object_context(
            object_edge=object_edge,
            object_evidence=object_evidence.get((object_for_unit, unit_id)),
            nodes=nodes,
        ):
            rejected_reasons["direct_object_context_preferred"] += 1
            continue
        if (object_for_unit is not None
                and object_for_unit != object_for_carrier and strong_2d_count <= 0
                and (object_for_carrier is None or (
                    not object_edge.get("provisional", False)
                    and object_edge.get("status") != "provisional"))):
            rejected_reasons[
                "unobserved_carrier_cannot_replace_measured_object"
                if object_for_carrier is None else "cross_object_candidate"
            ] += 1
            continue
        owner_node = nodes.get(object_for_carrier)
        contexts = _semantic_context_values(evidence, "object_contexts")
        context_matches = bool(owner_node and contexts) and any(
            labels_compatible(owner_node.top_label, context)
            or labels_compatible(context, owner_node.top_label)
            for context in contexts
        )
        # Explicit local observations can disagree with upstream class naming.
        # Retain that disagreement for auditing; do not invent class aliases.
        if owner_node and contexts and not context_matches and strong_2d_count <= 0:
            rejected_reasons["semantic_object_context_mismatch"] += 1
            continue
        cprior, cprior_source, chain_support, allowed_support = semantic_compatibility(
            evidence, cfg
        )
        if cprior is None and require_semantic:
            rejected_reasons["no_semantic_cu_support"] += 1
            continue
        if cprior is None:
            cprior = float(cfg.get("cprior_default", 0.5))

        aabb_distance = _association_aabb_distance(carrier, unit)
        if aabb_distance > max_surface_distance:
            rejected_reasons["aabb_gate_above_threshold"] += 1
            continue
        distance = carrier_unit_surface_distance(
            carrier,
            unit,
            max_points=max_points,
            near_fraction=near_fraction,
        )
        surface_distances.append(distance)
        if distance > max_surface_distance:
            rejected_reasons["surface_distance_above_threshold"] += 1
            continue
        gnear = float(np.exp(-distance / distance_scale))
        strong_2d_count = int(evidence.get("strong_2d_count", 0))
        observed_support = float(1.0 - np.exp(-max(strong_2d_count, 0)))
        ownership_score = float(
            np.clip(
                float(evidence.get("ownership_geometry_score_max", 0.0)) / 3.0,
                0.0,
                1.0,
            )
        )
        # A selected 2D relation is the strongest instance-level ownership
        # signal.  Semantic-only candidates still receive a small geometry
        # preference, but never accumulate fake multi-frame support.
        observation_score = float(
            0.75 * observed_support + 0.25 * ownership_score
            if strong_2d_count > 0
            else 0.15 * ownership_score
        )
        hierarchy_score = float(
            (
                semantic_weight * cprior
                + geometry_weight * gnear
                + observation_weight * observation_score
            )
            / weight_sum
        )
        hierarchy_scores.append(hierarchy_score)

        record = dict(evidence)
        record.update(
            {
                "hierarchy_score": hierarchy_score,
                "surface_distance_m": distance,
                "aabb_gate_distance_m": aabb_distance,
                "gnear": gnear,
                "observation_score": observation_score,
                "observed_support": observed_support,
                "ownership_score": ownership_score,
                "cprior": cprior,
                "cprior_source": cprior_source,
                "semantic_chain_support_frames": chain_support,
                "allowed_parent_support": allowed_support,
                "carrier_object_parent_id": object_for_carrier,
                # The selected carrier supplies O context only when known;
                # selecting C-U never invents an upper-level owner.
                "unit_object_parent_id": object_for_carrier,
                "unit_object_parent_source": (
                    "unknown" if object_for_carrier is None else
                    "optimized_O-U" if object_for_unit == object_for_carrier
                    else "carrier_corrected" if object_for_unit else "carrier_induced"
                ),
                "previous_unit_object_parent_id": object_for_unit,
                "object_parent_consistent": True if object_for_carrier else None,
                "object_parent_status": (
                    "unknown" if object_for_carrier is None else
                    "provisional" if oc_edges[carrier_id].get("provisional") else "confirmed"
                ),
                "semantic_object_context_match": context_matches if owner_node and contexts else None,
            }
        )
        candidates_by_unit.setdefault(unit_id, []).append(record)

    qualified_cu: dict[str, dict[str, Any]] = {}
    candidate_margins: dict[str, float | None] = {}
    for unit_id, records in sorted(candidates_by_unit.items()):
        ranked = sorted(
            records,
            key=lambda record: (
                -float(record["hierarchy_score"]),
                float(record["surface_distance_m"]),
                str(record["parent_node_id"]),
            ),
        )
        margin = (
            None
            if len(ranked) < 2
            else float(ranked[0]["hierarchy_score"] - ranked[1]["hierarchy_score"])
        )
        candidate_margins[unit_id] = margin
        selected = ranked[0]
        provisional_reasons = []
        if float(selected["hierarchy_score"]) <= min_score:
            provisional_reasons.append("hierarchy_score_below_threshold")
            rejected_reasons["provisional_hierarchy_score_below_threshold"] += 1
        if margin is not None and margin < min_margin:
            provisional_reasons.append("carrier_margin_below_threshold")
            rejected_reasons["provisional_carrier_margin_below_threshold"] += 1
        selected["status"] = (
            "provisional" if provisional_reasons else "confirmed"
        )
        selected["provisional"] = bool(provisional_reasons)
        selected["provisional_reasons"] = provisional_reasons
        qualified_cu[unit_id] = selected

    resolved_edges: list[dict[str, Any]] = [
        dict(edge) for _, edge in sorted(oc_edges.items())
    ]
    no_carrier_units = 0
    direct_context_units = 0
    cu_rejected_by_direct_context = rejected_reasons["direct_object_context_preferred"]
    selected_cu: dict[str, dict[str, Any]] = {}
    parent_decisions: dict[str, dict[str, Any]] = {}
    for unit_id in sorted(set(direct_ou).union(qualified_cu)):
        object_edge = direct_ou.get(unit_id)
        carrier_edge = qualified_cu.get(unit_id)
        strong_direct_context = bool(
            object_edge is not None
            and _has_strong_direct_object_context(
                object_edge=object_edge,
                object_evidence=object_evidence.get(
                    (str(object_edge["parent_node_id"]), unit_id)
                ),
                nodes=nodes,
            )
        )
        if object_edge is not None and strong_direct_context:
            direct_context_units += 1
            record = dict(object_edge)
            record["source"] = "optimized_direct_object_context"
            record["immediate_parent_decision"] = "direct_object_context"
            resolved_edges.append(record)
            parent_decisions[unit_id] = {
                "selected_parent_id": str(object_edge["parent_node_id"]),
                "selected_edge_type": "O-U",
                "reason": "strong_direct_object_context",
            }
        elif carrier_edge is not None:
            record = dict(carrier_edge)
            record["immediate_parent_decision"] = "qualified_carrier"
            resolved_edges.append(record)
            selected_cu[unit_id] = record
            parent_decisions[unit_id] = {
                "selected_parent_id": str(carrier_edge["parent_node_id"]),
                "selected_edge_type": "C-U",
                "reason": "local_semantic_surface_best",
                "candidate_margin": candidate_margins.get(unit_id),
                "status": carrier_edge.get("status", "confirmed"),
                "previous_object_parent_id": (
                    str(object_edge["parent_node_id"]) if object_edge else None
                ),
                "selected_object_parent_id": carrier_edge["carrier_object_parent_id"],
            }
        elif object_edge is not None:
            no_carrier_units += 1
            record = dict(object_edge)
            record["source"] = "optimized_object_parent_no_carrier"
            record["immediate_parent_decision"] = "object_fallback"
            resolved_edges.append(record)
            parent_decisions[unit_id] = {
                "selected_parent_id": str(object_edge["parent_node_id"]),
                "selected_edge_type": "O-U",
                "reason": "no_qualified_local_carrier",
            }

    meta = {
        "graph_stage": "immediate_parent_resolution",
        "require_semantic_cu_support": require_semantic,
        "requires_object_anchor": False,
        "max_surface_distance_m": max_surface_distance,
        "surface_distance_near_fraction": near_fraction,
        "min_hierarchy_score": min_score,
        "min_carrier_margin": min_margin,
        "hierarchy_score_weights": {
            "semantic": semantic_weight / weight_sum,
            "surface": geometry_weight / weight_sum,
            "observation": observation_weight / weight_sum,
        },
        "selection_policy": "optimistic_current_best_with_later_replacement",
        "num_units_with_selected_carrier": len(selected_cu),
        "num_units_without_selected_carrier": no_carrier_units,
        "num_units_with_direct_object_context": direct_context_units,
        "num_cu_rejected_by_direct_object_context": cu_rejected_by_direct_context,
        "immediate_parent_decisions": parent_decisions,
        "hierarchy_reject_reasons": dict(sorted(rejected_reasons.items())),
        "surface_distance_m": {
            "count": len(surface_distances),
            "min": min(surface_distances) if surface_distances else None,
            "median": float(np.median(surface_distances)) if surface_distances else None,
            "max": max(surface_distances) if surface_distances else None,
        },
        "hierarchy_score": {
            "count": len(hierarchy_scores),
            "min": min(hierarchy_scores) if hierarchy_scores else None,
            "median": float(np.median(hierarchy_scores)) if hierarchy_scores else None,
            "max": max(hierarchy_scores) if hierarchy_scores else None,
        },
    }
    return resolved_edges, meta


def format_final_hierarchy(
    *,
    selected_edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Format already-resolved immediate-parent edges and audit invariants."""

    object_parents = {
        str(e["child_node_id"]): e for e in selected_edges if e["edge_type"] == "O-C"
    }
    final_edges = []
    for edge in selected_edges:
        record = dict(edge)
        record["graph_stage"] = "final_hierarchical_graph"
        record["functional_status"] = "complete" if str(record.get("relation_text") or "").strip() else "pending"
        if record["edge_type"] == "C-U":
            owner = object_parents.get(str(record["parent_node_id"]))
            owner_id = owner["parent_node_id"] if owner else None
            record["carrier_object_parent_id"] = owner_id
            record["unit_object_parent_id"] = owner_id
            record["object_parent_consistent"] = True if owner else None
            record["object_parent_status"] = (
                "unknown" if owner is None else
                "provisional" if owner.get("provisional") else "confirmed"
            )
            if owner and owner.get("source") == "derived_object_completion":
                record["unit_object_parent_source"] = "derived_object_completion"
        final_edges.append(record)

    cross_object_cu_edges = sum(
        1
        for edge in final_edges
        if str(edge.get("edge_type")) == "C-U"
        if edge.get("carrier_object_parent_id") is not None
        and edge.get("unit_object_parent_id") is not None
        and edge.get("carrier_object_parent_id")
        != edge.get("unit_object_parent_id")
    )
    unanchored_cu_edges = sum(
        1
        for edge in final_edges
        if str(edge.get("edge_type")) == "C-U"
        if edge.get("carrier_object_parent_id") is None
        or edge.get("unit_object_parent_id") is None
    )

    child_parent_count: Counter[str] = Counter(
        str(edge["child_node_id"]) for edge in final_edges
    )
    duplicate_children = sorted(
        child_id
        for child_id, count in child_parent_count.items()
        if count > 1
    )
    if duplicate_children:
        raise ValueError(
            "immediate parents must be resolved before hierarchy formatting: "
            + ", ".join(duplicate_children)
        )
    cprior_sources = Counter(
        str(edge.get("cprior_source", "not_applicable")) for edge in final_edges
    )
    meta = {
        "graph_stage": "final_hierarchical_graph",
        "is_final_graph": True,
        "allows_multiple_parents": False,
        "num_edges": len(final_edges),
        "final_multi_immediate_parent_children": int(
            len(duplicate_children)
        ),
        "cross_object_cu_edges": cross_object_cu_edges,
        "unanchored_cu_edges": unanchored_cu_edges,
        "hierarchy_cprior_sources": dict(sorted(cprior_sources.items())),
    }
    return final_edges, meta


def shape_final_hierarchy(
    *,
    nodes: dict[str, MapNode3D],
    optimized_edges: list[dict[str, Any]],
    candidate_edges: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compatibility entry point composed from resolution then formatting."""

    selected_edges, resolution_meta = resolve_immediate_parent_edges(
        nodes=nodes,
        optimized_edges=optimized_edges,
        candidate_edges=candidate_edges,
        cfg=cfg,
    )
    final_edges, format_meta = format_final_hierarchy(
        selected_edges=selected_edges
    )
    return final_edges, {**resolution_meta, **format_meta}
