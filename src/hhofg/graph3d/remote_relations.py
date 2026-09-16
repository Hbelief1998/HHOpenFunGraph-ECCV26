from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from hhofg.mapping3d.semantic_candidate_builder import (
    labels_compatible,
    normalize_semantic_label,
)
from hhofg.mapping3d.types import MapNode3D


def _labels_match(instance_label: str, prior_label: str) -> bool:
    return labels_compatible(instance_label, prior_label) or labels_compatible(
        prior_label, instance_label
    )


def _pair_key(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first < second else (second, first)


def _remote_templates(atlas: dict[str, Any] | None) -> list[dict[str, Any]]:
    templates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in (atlas or {}).get("remote_relation_candidates", []) or []:
        if not isinstance(raw, dict):
            continue
        source_label = normalize_semantic_label(raw.get("from_object"))
        target_label = normalize_semantic_label(raw.get("to_object"))
        relation_text = str(raw.get("relation") or "").strip()
        if not source_label or not target_label or not relation_text:
            continue
        key = (source_label, normalize_semantic_label(relation_text), target_label)
        templates.setdefault(
            key,
            {
                "source_label": source_label,
                "target_label": target_label,
                "relation_text": relation_text,
                "directional": bool(raw.get("directional", True)),
            },
        )
    return [templates[key] for key in sorted(templates)]


def instantiate_remote_relations(
    *,
    nodes: dict[str, MapNode3D],
    atlas: dict[str, Any] | None,
    covisibility: set[tuple[str, str]] | None = None,
    observed_evidence: dict[tuple[str, str], dict[str, Any]] | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Instantiate LLM remote templates on the current 3D map.

    Semantic compatibility is the only required evidence.  Geometry,
    co-visibility, endpoint maturity and observed 2D remote evidence rank
    competing source instances; they never suppress the sole legal pair.  For
    each template and target instance, exactly one current-best source is
    emitted.  Re-running this function after the map grows naturally replaces
    that provisional source when a better candidate becomes available.
    """

    cfg = cfg or {}
    if not bool(cfg.get("enabled", True)):
        return [], {"enabled": False, "num_remote_edges": 0}
    covisibility = covisibility or set()
    observed_evidence = observed_evidence or {}
    templates = _remote_templates(atlas)

    weights = {
        "prior": max(float(cfg.get("prior_weight", 0.40)), 0.0),
        "observed": max(float(cfg.get("observed_weight", 0.25)), 0.0),
        "distance": max(float(cfg.get("distance_weight", 0.15)), 0.0),
        "covisibility": max(float(cfg.get("covisibility_weight", 0.15)), 0.0),
        "maturity": max(float(cfg.get("maturity_weight", 0.05)), 0.0),
    }
    weight_sum = max(sum(weights.values()), 1e-9)

    centroids = np.asarray(
        [node.centroid_world for node in nodes.values()], dtype=np.float64
    )
    scene_diag = (
        float(np.linalg.norm(centroids.max(axis=0) - centroids.min(axis=0)))
        if len(centroids) > 1
        else 1.0
    )
    distance_scale = max(
        float(cfg.get("distance_scale_m", 0.0)),
        scene_diag * float(cfg.get("distance_scale_scene_fraction", 0.35)),
        0.25,
    )
    covis_saturation = max(int(cfg.get("covisibility_saturation_frames", 3)), 1)

    label_index: dict[str, list[MapNode3D]] = {}
    for node in nodes.values():
        label_index.setdefault(normalize_semantic_label(node.top_label), []).append(node)

    def matching(label: str) -> list[MapNode3D]:
        exact = label_index.get(label, [])
        return exact or [
            node for node in nodes.values() if _labels_match(node.top_label, label)
        ]

    edges: list[dict[str, Any]] = []
    candidate_count = 0
    selection_switchable = 0
    source_counts: Counter[str] = Counter()
    for template_idx, template in enumerate(templates):
        source_nodes = matching(template["source_label"])
        target_nodes = matching(template["target_label"])
        for target in sorted(target_nodes, key=lambda node: node.node_id):
            ranked: list[tuple[float, str, dict[str, Any]]] = []
            for source in source_nodes:
                if source.node_id == target.node_id:
                    continue
                candidate_count += 1
                distance = float(
                    np.linalg.norm(source.centroid_world - target.centroid_world)
                )
                distance_score = float(np.exp(-distance / distance_scale))
                common_frames = len(
                    set(source.observed_frames) & set(target.observed_frames)
                )
                if _pair_key(source.node_id, target.node_id) in covisibility:
                    common_frames = max(common_frames, 1)
                covis_score = min(1.0, common_frames / covis_saturation)
                observation = observed_evidence.get(
                    (source.node_id, target.node_id), {}
                )
                observation_count = int(observation.get("count", 0))
                observation_score = float(
                    1.0 - np.exp(-max(observation_count, 0))
                )
                maturity_score = float(
                    np.clip(
                        0.5
                        * (
                            float(getattr(source, "graph_node_confidence", 0.0))
                            + float(getattr(target, "graph_node_confidence", 0.0))
                        ),
                        0.0,
                        1.0,
                    )
                )
                score = float(
                    (
                        weights["prior"]
                        + weights["observed"] * observation_score
                        + weights["distance"] * distance_score
                        + weights["covisibility"] * covis_score
                        + weights["maturity"] * maturity_score
                    )
                    / weight_sum
                )
                record = {
                    "parent_node_id": source.node_id,
                    "child_node_id": target.node_id,
                    "edge_type": "remote",
                    "parent_role": source.role,
                    "child_role": target.role,
                    "parent_label": source.top_label,
                    "child_label": target.top_label,
                    "relation_text": template["relation_text"],
                    "source": "llm_remote_prior_3d_current_best",
                    "status": "confirmed" if observation_count > 0 else "provisional",
                    "provisional": observation_count == 0,
                    "remote_score": score,
                    "prior_score": 1.0,
                    "observed_remote_count": observation_count,
                    "observed_remote_evidence_sum": float(
                        observation.get("evidence_sum", 0.0)
                    ),
                    "covisibility_frames": common_frames,
                    "distance_m": distance,
                    "distance_score": distance_score,
                    "maturity_score": maturity_score,
                    "template_index": template_idx,
                    "template_source_label": template["source_label"],
                    "template_target_label": template["target_label"],
                    "selection_policy": "optimistic_current_best_with_later_replacement",
                }
                ranked.append((-score, source.node_id, record))
            if not ranked:
                continue
            ranked.sort()
            selected = ranked[0][2]
            selected["candidate_count"] = len(ranked)
            selected["selection_margin"] = (
                None if len(ranked) < 2 else float(-ranked[0][0] + ranked[1][0])
            )
            selection_switchable += int(len(ranked) > 1)
            source_counts[selected["status"]] += 1
            edges.append(selected)

    # Identical atlas templates or label aliases can resolve to the same
    # triplet.  Keep the strongest single record without collapsing distinct
    # relation predicates.
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in edges:
        key = (
            str(edge["parent_node_id"]),
            str(edge["child_node_id"]),
            normalize_semantic_label(edge["relation_text"]),
        )
        if key not in dedup or float(edge["remote_score"]) > float(
            dedup[key]["remote_score"]
        ):
            dedup[key] = edge
    final = sorted(
        dedup.values(),
        key=lambda edge: (
            edge["parent_node_id"],
            edge["child_node_id"],
            edge["relation_text"],
        ),
    )
    return final, {
        "enabled": True,
        "num_templates": len(templates),
        "num_instance_candidates": candidate_count,
        "num_remote_edges": len(final),
        "num_targets_with_multiple_source_candidates": selection_switchable,
        "edge_status_counts": dict(sorted(source_counts.items())),
        "distance_scale_m": distance_scale,
        "score_weights": {
            key: value / weight_sum for key, value in weights.items()
        },
        "selection_policy": "optimistic_current_best_with_later_replacement",
        "semantic_prior_is_sufficient": True,
    }
