from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from hhofg.core.serialization import write_json_atomic
from hhofg.graph3d.edge_optimizer import optimize_object_edges
from hhofg.graph3d.hierarchy import (
    format_final_hierarchy,
    resolve_immediate_parent_edges,
)
from hhofg.graph3d.object_completion import complete_deferred_objects
from hhofg.graph3d.remote_relations import instantiate_remote_relations
from hhofg.mapping3d.checkpoint import load_final_map_state
from hhofg.mapping3d.edge_candidates import (
    annotate_candidate_semantics, candidate_to_mapped_evidence, recover_semantic_carrier_candidates,
)
from hhofg.mapping3d.graph_node_eligibility import evaluate_graph_node_eligibility
from hhofg.mapping3d.serialization import append_jsonl, save_map
from hhofg.mapping3d.semantic_candidate_builder import (
    build_global_semantic_relation_index, labels_compatible, select_relation_text,
)
from hhofg.mapping3d.types import FrameEdgeCandidate, MapNode3D
from hhofg.mapping3d.visualization import (
    infer_scene_base_ply_path,
    read_ply_points_rgb,
    write_scene_graph_overlay_ply,
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    records = []
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _candidate_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(record.get("frame_key", "")),
        str(record.get("edge_type", "")),
        int(record.get("parent_det_id", -1)),
        int(record.get("child_det_id", -1)),
    )


def _load_aliases(mapping_run: Path) -> dict[str, str]:
    path = mapping_run / "map" / "node_aliases.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    aliases = payload.get("aliases", payload)
    return {str(key): str(value) for key, value in aliases.items()}


def _load_hierarchy_semantics(
    mapping_run: Path,
) -> tuple[dict[str, Any], set[str]]:
    """Load the atlas and generic deferred-object policy used by Stage C."""

    run_config_path = mapping_run / "run_config.json"
    if not run_config_path.is_file():
        return {}, set()
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    frontend_value = run_config.get("frontend_run")
    if not frontend_value:
        return {}, set()
    frontend_run = Path(str(frontend_value))
    atlas_path = frontend_run / "global_atlas.json"
    frontend_config_path = frontend_run / "run_config.json"
    atlas = (
        json.loads(atlas_path.read_text(encoding="utf-8"))
        if atlas_path.is_file()
        else {}
    )
    frontend_config = (
        json.loads(frontend_config_path.read_text(encoding="utf-8"))
        if frontend_config_path.is_file()
        else {}
    )
    deferred = {
        str(label)
        for label in (frontend_config.get("policy", {}).get("hint_only_objects", []) or [])
        if str(label).strip()
    }
    return atlas, deferred


def _load_deferred_object_evidence(
    mapping_run: Path,
    association_maps: dict[str, dict[int, str]],
    aliases: dict[str, str],
) -> tuple[
    dict[str, dict[str, list[float]]],
    dict[str, set[tuple[str, str, int]]],
    dict[str, int],
]:
    """Map cached raw SAM3 boxes and deferred-owner hints onto 3D tracks."""

    run_config_path = mapping_run / "run_config.json"
    if not run_config_path.is_file():
        return {}, {}, {"frames": 0, "boxes": 0, "passing_owner_hints": 0}
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    frontend_value = run_config.get("frontend_run")
    frontend_frames = Path(str(frontend_value)) / "frames" if frontend_value else None
    if frontend_frames is None or not frontend_frames.is_dir():
        return {}, {}, {"frames": 0, "boxes": 0, "passing_owner_hints": 0}

    observation_boxes: dict[str, dict[str, list[float]]] = {}
    owner_hints: dict[str, set[tuple[str, str, int]]] = {}
    frame_count = 0
    box_count = 0
    hint_count = 0
    for path in sorted(frontend_frames.glob("frame_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame_key = str(payload.get("frame_key") or path.stem)
        det_to_node = association_maps.get(frame_key)
        if not det_to_node:
            continue
        frame_count += 1
        detections = payload.get("detections", []) or []
        by_index: dict[int, dict[str, Any]] = {}
        for index, detection in enumerate(detections):
            det_id = int(detection.get("det_id", index))
            by_index[index] = detection
            node_id = _canonical(det_to_node.get(det_id), aliases)
            box = detection.get("box_xyxy")
            if node_id is None or not isinstance(box, list) or len(box) != 4:
                continue
            observation_boxes.setdefault(node_id, {})[frame_key] = [
                float(value) for value in box
            ]
            box_count += 1

        local_relation = (
            ((payload.get("local_rel_debug") or {}).get("local_rel") or {})
            .get("after_cov", {})
        )
        # New caches may expose a generic name.  ``cabinet_hints`` is the
        # legacy schema name; the owner label in each hit remains authoritative.
        hint_payload = local_relation.get("deferred_object_hints") or local_relation.get(
            "cabinet_hints"
        ) or {}
        for hit in hint_payload.get("carrier_hits", []) or []:
            if not bool(hit.get("pass_thr", False)):
                continue
            child_index = int(hit.get("child_idx", -1))
            detection = by_index.get(child_index)
            if detection is None:
                continue
            det_id = int(detection.get("det_id", child_index))
            node_id = _canonical(det_to_node.get(det_id), aliases)
            owner_label = str(
                hit.get("owner_label") or hit.get("cabinet_label") or ""
            ).strip()
            parent_index = int(
                hit.get(
                    "owner_parent_idx_pre_drop",
                    hit.get("cabinet_parent_idx_pre_drop", -1),
                )
            )
            if node_id is None or not owner_label or parent_index < 0:
                continue
            owner_hints.setdefault(node_id, set()).add(
                (owner_label, frame_key, parent_index)
            )
            hint_count += 1
    return observation_boxes, owner_hints, {
        "frames": frame_count,
        "boxes": box_count,
        "passing_owner_hints": hint_count,
    }


def _load_remote_observation_evidence(
    mapping_run: Path,
    association_maps: dict[str, dict[int, str]],
    aliases: dict[str, str],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, int]]:
    """Join cached 2D remote observations to their canonical 3D endpoints."""

    run_config_path = mapping_run / "run_config.json"
    if not run_config_path.is_file():
        return {}, {"frames": 0, "observations": 0, "mapped_observations": 0}
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    frontend_value = run_config.get("frontend_run")
    frames_dir = Path(str(frontend_value)) / "frames" if frontend_value else None
    if frames_dir is None or not frames_dir.is_dir():
        return {}, {"frames": 0, "observations": 0, "mapped_observations": 0}

    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    frame_count = observation_count = mapped_count = 0
    for path in sorted(frames_dir.glob("frame_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame_key = str(payload.get("frame_key") or path.stem)
        det_to_node = association_maps.get(frame_key, {})
        remote = payload.get("remote_rel_2d") or {}
        observed = remote.get("observed_candidates") or []
        if observed:
            frame_count += 1
        for item in observed:
            if not isinstance(item, dict):
                continue
            observation_count += 1
            try:
                source_id = _canonical(
                    det_to_node.get(int(item.get("from_det_idx"))), aliases
                )
                target_id = _canonical(
                    det_to_node.get(int(item.get("to_det_idx"))), aliases
                )
            except (TypeError, ValueError):
                continue
            if not source_id or not target_id or source_id == target_id:
                continue
            mapped_count += 1
            record = evidence.setdefault(
                (source_id, target_id),
                {"count": 0, "evidence_sum": 0.0, "frames": []},
            )
            record["count"] += 1
            record["evidence_sum"] += float(item.get("evidence", 0.0) or 0.0)
            if frame_key not in record["frames"]:
                record["frames"].append(frame_key)
    return evidence, {
        "frames": frame_count,
        "observations": observation_count,
        "mapped_observations": mapped_count,
    }


def _canonical(node_id: str | None, aliases: dict[str, str]) -> str | None:
    if not node_id:
        return None
    current = str(node_id)
    visited = set()
    while current in aliases and current not in visited:
        visited.add(current)
        current = aliases[current]
    return current


def _association_maps(mapping_run: Path) -> dict[str, dict[int, str]]:
    out = {}
    for path in sorted((mapping_run / "associations").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame_key = str(payload.get("frame_key", path.stem))
        out[frame_key] = {
            int(det_id): str(node_id)
            for det_id, node_id in payload.get("det_to_node", {}).items()
        }
    return out


def join_scores_to_candidates(
    candidate_records: list[dict[str, Any]],
    score_records: list[dict[str, Any]],
) -> tuple[list[FrameEdgeCandidate], dict[str, Any]]:
    score_by_hash = {
        str(record.get("candidate_hash")): record
        for record in score_records
        if record.get("candidate_hash")
    }
    score_by_key = {_candidate_key(record): record for record in score_records}
    candidates = []
    joined = 0
    invalid_self_loops = 0
    for record in candidate_records:
        candidate = FrameEdgeCandidate(**record)
        if candidate.parent_det_id == candidate.child_det_id:
            invalid_self_loops += 1
            continue
        score = score_by_hash.get(candidate.candidate_hash) or score_by_key.get(
            (
                candidate.frame_key,
                candidate.edge_type,
                candidate.parent_det_id,
                candidate.child_det_id,
            )
        )
        if score is not None:
            joined += 1
            stats = score.get("candidate", {}) or {}
            candidate.sdet = stats.get("sdet", candidate.sdet)
            candidate.gcamc = stats.get("gcamc", candidate.gcamc)
            candidate.gcamc_raw = stats.get("gcamc_raw", candidate.gcamc_raw)
            candidate.gcamc_support = stats.get(
                "gcamc_support", candidate.gcamc_support
            )
            candidate.gcamc_used = stats.get(
                "gcamc_used", candidate.gcamc_used
            )
            candidate.support_mask_used = bool(
                stats.get("support_mask_used", candidate.support_mask_used)
            )
            candidate.child_center_inside_parent_box = bool(
                stats.get(
                    "child_center_inside_parent_box",
                    candidate.child_center_inside_parent_box,
                )
            )
            candidate.pass_prefilter = stats.get("pass_prefilter", candidate.pass_prefilter)
            candidate.prefilter_fail_reason = str(
                stats.get("fail_reason", candidate.prefilter_fail_reason)
            )
            candidate.mask_sha = str(stats.get("mask_sha", candidate.mask_sha))
            candidate.s2d_score = score.get("s2d_score")
            candidate.s2d_available = bool(score.get("available", False))
            candidate.s2d_error = str(score.get("error", ""))
            candidate.s2d_cache_key = str(score.get("cache_key", ""))
            candidate.s2d_cache_hit = bool(
                (score.get("metadata", {}) or {}).get("cache_hit", False)
            )
            candidate.s2d_model_id = str(score.get("model_id", ""))
            candidate.s2d_raw_response = str(score.get("raw_response", ""))
        candidates.append(candidate)
    return candidates, {
        "num_candidates": len(candidates),
        "num_score_records": len(score_records),
        "num_scores_joined": joined,
        "num_invalid_detection_self_loops": invalid_self_loops,
        "score_join_coverage": float(joined / max(1, len([c for c in candidates if c.edge_type in {"O-C", "O-U"}]))),
    }


def remap_candidate_endpoints(
    candidates: list[FrameEdgeCandidate],
    association_maps: dict[str, dict[int, str]],
    nodes: dict[str, MapNode3D],
    aliases: dict[str, str],
) -> list[dict[str, Any]]:
    unresolved = []
    for candidate in candidates:
        det_map = association_maps.get(candidate.frame_key, {})
        candidate.parent_node_id = _canonical(det_map.get(candidate.parent_det_id), aliases)
        candidate.child_node_id = _canonical(det_map.get(candidate.child_det_id), aliases)
        parent = nodes.get(candidate.parent_node_id or "")
        child = nodes.get(candidate.child_node_id or "")
        candidate.parent_node_state = parent.state if parent is not None else None
        candidate.child_node_state = child.state if child is not None else None
        candidate.parent_lifted = candidate.parent_node_id is not None
        candidate.child_lifted = candidate.child_node_id is not None
        reason = ""
        if candidate.frame_key not in association_maps:
            reason = "association_frame_missing"
        elif candidate.parent_node_id is None and candidate.child_node_id is None:
            reason = "both_detection_endpoints_unmapped"
        elif candidate.parent_node_id is None:
            reason = "parent_detection_endpoint_unmapped"
        elif candidate.child_node_id is None:
            reason = "child_detection_endpoint_unmapped"
        elif parent is None or child is None:
            reason = "canonical_node_missing"
        if reason:
            candidate.unresolved_endpoint_reason = reason
            unresolved.append(
                {
                    "frame_key": candidate.frame_key,
                    "edge_type": candidate.edge_type,
                    "parent_det_id": candidate.parent_det_id,
                    "child_det_id": candidate.child_det_id,
                    "candidate_hash": candidate.candidate_hash,
                    "unresolved_endpoint_reason": reason,
                }
            )
    return unresolved


def aggregate_candidates(
    candidates: Iterable[FrameEdgeCandidate],
    *,
    node_ids: set[str] | None = None,
    predicate=None,
) -> list[dict[str, Any]]:
    stats: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        if not candidate.parent_node_id or not candidate.child_node_id:
            continue
        if node_ids is not None and (
            candidate.parent_node_id not in node_ids
            or candidate.child_node_id not in node_ids
        ):
            continue
        if predicate is not None and not predicate(candidate):
            continue
        key = (candidate.parent_node_id, candidate.child_node_id, candidate.edge_type)
        record = stats.setdefault(
            key,
            {
                "parent_node_id": candidate.parent_node_id,
                "child_node_id": candidate.child_node_id,
                "edge_type": candidate.edge_type,
                "parent_role": candidate.parent_role,
                "child_role": candidate.child_role,
                "parent_label": candidate.parent_label,
                "child_label": candidate.child_label,
                "evidence_count": 0,
                "selected_2d_count": 0,
                "pass_threshold_count": 0,
                "observed_instance_count": 0,
                "strong_2d_count": 0,
                "semantic_only_count": 0,
                "prefilter_pass_count": 0,
                "s2d_available_count": 0,
                "candidate_sources": {},
                "semantic_sources": {},
                "semantic_chains": [],
                "semantic_chain_support_frames": 0,
                "allowed_parent_support": 0,
                "ownership_geometry_score_max": 0.0,
                "ownership_child_center_inside_count": 0,
                "ownership_child_box_coverage_max": 0.0,
                "_semantic_frames": set(),
                "first_frame_idx": candidate.frame_idx,
                "last_frame_idx": candidate.frame_idx,
                "relation_text": candidate.relation_text,
                "_relation_counts": Counter(),
            },
        )
        text = str(candidate.relation_text or "").strip()
        if text:
            record["_relation_counts"][text] += 1
        record["evidence_count"] += 1
        record["selected_2d_count"] += int(candidate.selected_2d)
        record["pass_threshold_count"] += int(candidate.pass_threshold)
        record["observed_instance_count"] += int(
            candidate.from_raw_relation or candidate.from_final_relation
        )
        record["strong_2d_count"] += int(
            candidate.selected_2d or candidate.pass_threshold
        )
        record["semantic_only_count"] += int(candidate.semantic_only)
        record["prefilter_pass_count"] += int(bool(candidate.pass_prefilter))
        record["s2d_available_count"] += int(candidate.s2d_available)
        geometry = candidate.ownership_geometry or {}
        record["ownership_geometry_score_max"] = max(
            float(record["ownership_geometry_score_max"]),
            float(geometry.get("score", 0.0) or 0.0),
        )
        record["ownership_child_center_inside_count"] += int(
            bool(geometry.get("child_center_inside_parent_box", False))
        )
        record["ownership_child_box_coverage_max"] = max(
            float(record["ownership_child_box_coverage_max"]),
            float(geometry.get("child_box_coverage", 0.0) or 0.0),
        )
        for source in candidate.candidate_sources:
            record["candidate_sources"][source] = record["candidate_sources"].get(source, 0) + 1
        for source in candidate.semantic_sources:
            record["semantic_sources"][source] = record["semantic_sources"].get(source, 0) + 1
            if source in {"atlas", "frame_result", "graph_frame_result"}:
                record["_semantic_frames"].add(candidate.frame_key)
            if source == "allowed_parent_map":
                record["allowed_parent_support"] += 1
        for chain in candidate.semantic_chains:
            if chain not in record["semantic_chains"]:
                record["semantic_chains"].append(chain)
        record["first_frame_idx"] = min(record["first_frame_idx"], candidate.frame_idx)
        record["last_frame_idx"] = max(record["last_frame_idx"], candidate.frame_idx)
    out = []
    for record in stats.values():
        counts = record.pop("_relation_counts")
        record["relation_text"] = select_relation_text(counts, labels=(record["parent_label"], record["child_label"]))
        record["semantic_chain_support_frames"] = len(record.pop("_semantic_frames"))
        out.append(record)
    return sorted(
        out,
        key=lambda record: (
            -int(record["evidence_count"]),
            record["edge_type"],
            record["parent_node_id"],
            record["child_node_id"],
        ),
    )


def _scene(mapping_run: Path, nodes: dict[str, MapNode3D]) -> tuple[np.ndarray, np.ndarray, str]:
    run_config_path = mapping_run / "run_config.json"
    run_config = (
        json.loads(run_config_path.read_text(encoding="utf-8"))
        if run_config_path.is_file()
        else {}
    )
    data = run_config.get("effective_data", run_config.get("mapping3d_config", {}).get("data", {}))
    base = infer_scene_base_ply_path(data.get("dataset_root", ""), data.get("sequence", ""))
    if base is not None:
        points, colors = read_ply_points_rgb(base)
        return points, colors, str(base)
    ordered = [nodes[key] for key in sorted(nodes)]
    points = np.concatenate([node.points_world for node in ordered], axis=0) if ordered else np.zeros((0, 3), np.float32)
    colors = np.concatenate([node.colors_rgb for node in ordered], axis=0).astype(np.uint8) if ordered else np.zeros((0, 3), np.uint8)
    return points, colors, "mapping3d_fused_map_points"


def build_functional_graph_artifacts(
    *,
    mapping_run: str | Path,
    candidate_jsonl: str | Path,
    score_jsonl: str | Path,
    cfg: dict[str, Any],
    output_dir: str | Path,
    write_ply: bool = True,
) -> dict[str, Any]:
    mapping_run = Path(mapping_run)
    output_dir = Path(output_dir)
    map_dir = output_dir / "map"
    mapped_dir = output_dir / "mapped_edges"
    map_dir.mkdir(parents=True, exist_ok=True)
    mapped_dir.mkdir(parents=True, exist_ok=True)

    state = load_final_map_state(mapping_run)
    nodes = state.nodes
    candidate_records = _read_jsonl(candidate_jsonl)
    score_records = _read_jsonl(score_jsonl)
    candidates, join_meta = join_scores_to_candidates(candidate_records, score_records)
    aliases = _load_aliases(mapping_run)
    association_maps = _association_maps(mapping_run)
    unresolved = remap_candidate_endpoints(
        candidates,
        association_maps,
        nodes,
        aliases,
    )
    global_atlas, deferred_object_labels = _load_hierarchy_semantics(mapping_run)
    semantic_index = build_global_semantic_relation_index(global_atlas)
    # Contextual priors may be reused; instance predicates must stay on their
    # observed endpoint pair, never be promoted to a category-wide truth.
    for candidate in candidates:
        for chain in candidate.semantic_chains:
            pair = chain.get("semantic_pair")
            if isinstance(pair, list) and len(pair) == 2:
                for evidence in chain.get("relation_evidence", []):
                    semantic_index.add(candidate.edge_type, *pair,
                        source=evidence["source"], relation_text=evidence["relation_text"],
                        object_context=evidence.get("object_context"))
    predicates_missing_before = sum(
        bool(c.parent_node_id and c.child_node_id) and not str(c.relation_text or "").strip()
        for c in candidates
    )
    annotate_candidate_semantics(candidates, semantic_index)
    predicate_missing_candidates = [
        candidate
        for candidate in candidates
        if candidate.parent_node_id
        and candidate.child_node_id
        and not str(candidate.relation_text or "").strip()
    ]
    eligible_nodes, eligibility_meta = evaluate_graph_node_eligibility(
        nodes,
        candidates,
        cfg.get("edge_optimizer", {}),
    )
    eligible_ids = set(eligible_nodes)

    evidence = []
    for candidate in candidates:
        mapped = candidate_to_mapped_evidence(candidate)
        if mapped is not None:
            evidence.append(mapped)
    evidence_path = mapped_dir / "frame_edges.jsonl"
    unresolved_path = mapped_dir / "unresolved_endpoints.jsonl"
    for path in (evidence_path, unresolved_path):
        if path.exists():
            path.unlink()
    append_jsonl(evidence_path, evidence)
    append_jsonl(unresolved_path, unresolved)

    optimizer_input = [
        edge
        for edge in evidence
        if edge.edge_type in {"O-C", "O-U"}
        and edge.parent_node_id in eligible_ids
        and edge.child_node_id in eligible_ids
    ]
    optimizer_cfg = dict(cfg.get("edge_optimizer", {}))
    optimizer_cfg["stable_only"] = False
    optimized_edges, optimizer_meta, optimizer_state = optimize_object_edges(
        optimizer_input,
        eligible_nodes,
        optimizer_cfg,
    )
    posterior_path = mapped_dir / "edge_posteriors_by_frame.jsonl"
    if posterior_path.exists():
        posterior_path.unlink()
    append_jsonl(posterior_path, optimizer_state.posterior_history)

    all_mapped = aggregate_candidates(candidates)
    graph_eligible_mapped = aggregate_candidates(candidates, node_ids=eligible_ids)
    final_local = aggregate_candidates(
        candidates, predicate=lambda candidate: candidate.from_final_relation
    )
    raw_local = aggregate_candidates(
        candidates, predicate=lambda candidate: candidate.from_raw_relation
    )
    semantic = aggregate_candidates(
        candidates, predicate=lambda candidate: bool(candidate.semantic_sources)
    )
    prefilter = aggregate_candidates(
        candidates, predicate=lambda candidate: bool(candidate.pass_prefilter)
    )
    scored = aggregate_candidates(
        candidates, predicate=lambda candidate: bool(candidate.s2d_available)
    )
    immediate_parent_candidates = aggregate_candidates(
        candidates,
        node_ids=eligible_ids,
        predicate=lambda candidate: candidate.edge_type in {"O-U", "C-U"},
    )
    selected_immediate_edges, resolution_meta = resolve_immediate_parent_edges(
        nodes=eligible_nodes, optimized_edges=optimized_edges,
        candidate_edges=immediate_parent_candidates, cfg=cfg.get("hierarchy", {}),
    )
    linked_units = {str(e["child_node_id"]) for e in selected_immediate_edges}
    unlinked_units = {key for key, n in eligible_nodes.items() if n.role == "U" and key not in linked_units}
    run_config_path = mapping_run / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.is_file() else {}
    recovered = []
    if cfg.get("candidate_generation", {}).get("recover_orphan_carriers", True):
        recovered = recover_semantic_carrier_candidates(
            frontend_run=run_config.get("frontend_run"),
            association_maps={frame: {det: _canonical(nid, aliases) for det, nid in dets.items()}
                              for frame, dets in association_maps.items()},
            nodes=eligible_nodes, index=semantic_index, unlinked_units=unlinked_units,
            existing_pairs={(e["parent_node_id"], e["child_node_id"]) for e in immediate_parent_candidates},
            cfg=cfg.get("candidate_generation", {}),
            max_distance=float(cfg.get("hierarchy", {}).get("max_surface_distance_m", 0.08)),
            mask_threshold=float(cfg.get("edge2d", {}).get("gcamc_threshold", 0.90)),
        )
    if recovered:
        candidates.extend(recovered)
        immediate_parent_candidates = aggregate_candidates(
            candidates, node_ids=eligible_ids,
            predicate=lambda c: c.edge_type in {"O-U", "C-U"},
        )
        selected_immediate_edges, resolution_meta = resolve_immediate_parent_edges(
            nodes=eligible_nodes, optimized_edges=optimized_edges,
            candidate_edges=immediate_parent_candidates, cfg=cfg.get("hierarchy", {}),
        )
    write_json_atomic(mapped_dir / "recovered_carrier_candidates.json", aggregate_candidates(recovered))
    observation_boxes, owner_hints, completion_evidence_meta = (
        _load_deferred_object_evidence(mapping_run, association_maps, aliases)
    )
    derived_nodes, derived_object_edges, completion_meta = complete_deferred_objects(
        nodes=eligible_nodes,
        object_edges=optimized_edges,
        local_edges=selected_immediate_edges,
        semantic_index=semantic_index,
        deferred_object_labels=deferred_object_labels,
        observation_boxes=observation_boxes,
        owner_hints=owner_hints,
        cfg=cfg.get("hierarchy", {}).get("derived_object_completion", {}),
        reserved_node_ids=set(nodes) | set(aliases),
    )
    completion_meta["input_evidence"] = completion_evidence_meta
    nodes.update(derived_nodes)
    eligible_nodes.update(derived_nodes)
    eligible_ids.update(derived_nodes)
    completed_children = {str(edge["child_node_id"]) for edge in derived_object_edges}
    completed_object_edges = [
        edge for edge in optimized_edges
        if str(edge["child_node_id"]) not in completed_children
    ] + derived_object_edges
    # Completion changes only upper-level ownership; it cannot erase or
    # rerank the independently selected local C-U links.
    selected_immediate_edges = [
        e for e in selected_immediate_edges if str(e["child_node_id"]) not in completed_children
    ] + derived_object_edges
    by_pair: dict[tuple[str, str, str], list[FrameEdgeCandidate]] = {}
    for candidate in candidates:
        by_pair.setdefault((candidate.parent_node_id, candidate.child_node_id, candidate.edge_type), []).append(candidate)
    owners = {e["child_node_id"]: e["parent_node_id"] for e in selected_immediate_edges if e["edge_type"] == "O-C"}
    for edge in selected_immediate_edges:
        parent = eligible_nodes[edge["parent_node_id"]]
        child = eligible_nodes[edge["child_node_id"]]
        owner = eligible_nodes.get(owners.get(parent.node_id)) if parent.role == "C" else parent
        context = owner.top_label if owner is not None else None
        evidence = by_pair.get((parent.node_id, child.node_id, edge["edge_type"]), [])
        instance_texts = [t for c in evidence for t in c.instance_relation_texts]
        if instance_texts:
            text = select_relation_text(instance_texts, labels=(parent.top_label, child.top_label, context))
            source = "instance" if text else "instance_conflict"
        else:
            historical = []
            for candidate in evidence:
                if not candidate.relation_text:
                    continue
                records = [r for chain in candidate.semantic_chains
                           for r in chain.get("relation_evidence", [])
                           if r.get("relation_text") == candidate.relation_text]
                if context and records and not any(
                    not r.get("object_context") or
                    labels_compatible(context, r["object_context"]) or
                    labels_compatible(r["object_context"], context) for r in records
                ):
                    continue
                historical.append(candidate.relation_text)
            if historical:
                text = select_relation_text(historical, labels=(parent.top_label, child.top_label, context))
                source = "pair_history" if text else "pair_history_conflict"
            else:
                text = semantic_index.relation_for(edge["edge_type"], parent.top_label, child.top_label,
                                                  object_context=context)
                source = "contextual_semantics" if text else "unresolved"
        edge["relation_text"] = text
        edge["predicate_source"] = source
        edge["predicate_object_context"] = context
    local_final_edges, format_meta = format_final_hierarchy(
        selected_edges=selected_immediate_edges
    )
    remote_observations, remote_observation_meta = (
        _load_remote_observation_evidence(mapping_run, association_maps, aliases)
    )
    canonical_covisibility = set()
    for first, second in getattr(state, "covisibility", set()):
        first_id = _canonical(first, aliases)
        second_id = _canonical(second, aliases)
        if (
            first_id
            and second_id
            and first_id != second_id
            and first_id in eligible_ids
            and second_id in eligible_ids
        ):
            canonical_covisibility.add(tuple(sorted((first_id, second_id))))
    remote_edges, remote_meta = instantiate_remote_relations(
        nodes=eligible_nodes,
        atlas=global_atlas,
        covisibility=canonical_covisibility,
        observed_evidence=remote_observations,
        cfg=cfg.get("remote_relations", {}),
    )
    remote_meta["observed_2d"] = remote_observation_meta
    final_edges = [*local_final_edges, *remote_edges]
    format_meta["num_local_edges"] = len(local_final_edges)
    format_meta["num_remote_edges"] = len(remote_edges)
    format_meta["num_edges"] = len(final_edges)
    hierarchy_meta = {
        **resolution_meta,
        **format_meta,
        "derived_object_completion": completion_meta,
        "remote_relations": remote_meta,
    }

    prefilter_object = [
        candidate
        for candidate in candidates
        if candidate.edge_type in {"O-C", "O-U"} and candidate.pass_prefilter
    ]
    all_prefilter_scored = all(candidate.s2d_available for candidate in prefilter_object)
    mapped_or_reported = all(
        candidate.parent_node_id and candidate.child_node_id
        or bool(candidate.unresolved_endpoint_reason)
        for candidate in candidates
        if candidate.s2d_available
    )
    children_with_object_candidate = {
        candidate.child_node_id
        for candidate in candidates
        if candidate.edge_type in {"O-C", "O-U"}
        and candidate.child_node_id in eligible_ids
        and bool(str(candidate.relation_text or "").strip())
    }
    children_with_object_candidate.update(
        str(edge["child_node_id"]) for edge in derived_object_edges
    )
    children_with_object_candidate.update(
        str(edge["child_node_id"])
        for edge in local_final_edges
        if str(edge.get("edge_type")) == "C-U"
        and edge.get("carrier_object_parent_id") is not None
    )
    no_candidate = [
        {
            "node_id": node_id,
            "role": node.role,
            "reason": "no_object_candidate",
        }
        for node_id, node in eligible_nodes.items()
        if node.role in {"C", "U"} and node_id not in children_with_object_candidate
    ]
    write_json_atomic(map_dir / "graph_eligible_children_without_object_candidate.json", no_candidate)
    child_parent_counts = Counter(
        str(edge["child_node_id"]) for edge in local_final_edges
    )
    # A candidate in an overlay is not a final link. Account for missing
    # parents explicitly so PLY gaps cannot be hidden by candidate coverage.
    incident_ids = {
        str(edge[key]) for edge in final_edges
        for key in ("parent_node_id", "child_node_id")
    }
    candidates_by_child: dict[str, list[FrameEdgeCandidate]] = {}
    for candidate in candidates:
        if candidate.child_node_id:
            candidates_by_child.setdefault(candidate.child_node_id, []).append(candidate)
    unlinked_children = []
    for node_id, node in sorted(eligible_nodes.items()):
        if node.role not in {"C", "U"} or child_parent_counts[node_id]:
            continue
        child_candidates = candidates_by_child.get(node_id, [])
        mapped = [c for c in child_candidates if c.parent_node_id in eligible_ids]
        object_candidates = [c for c in mapped if c.edge_type in {"O-C", "O-U"}]
        measured = [c for c in object_candidates if c.s2d_score is not None]
        carriers = [c for c in mapped if c.edge_type == "C-U"]
        if not child_candidates:
            reason = "no_parent_candidate"
        elif not mapped:
            reason = "parent_unmapped_or_ineligible"
        elif carriers:
            reason = "carrier_candidate_rejected_by_local_evidence_or_conflict"
        elif object_candidates and not measured:
            reason = (
                "object_s2d_unavailable_after_prefilter"
                if any(c.pass_prefilter for c in object_candidates)
                else "object_candidate_not_instance_validated"
            )
        else:
            reason = "parent_resolution_unavailable"
        unlinked_children.append({
            "node_id": node_id, "label": node.top_label, "role": node.role,
            "reason": reason, "isolated": node_id not in incident_ids,
            "candidate_count": len(child_candidates),
            "mapped_eligible_candidate_count": len(mapped),
            "measured_object_candidate_count": len(measured),
            "carrier_candidate_count": len(carriers),
        })
    link_diagnostics = {
        "num_children_without_local_parent": len(unlinked_children),
        "num_isolated_nodes": len(eligible_ids - incident_ids),
        "isolated_node_ids": sorted(eligible_ids - incident_ids),
        "missing_parent_reasons": dict(Counter(r["reason"] for r in unlinked_children)),
        "children_without_local_parent": unlinked_children,
    }
    write_json_atomic(map_dir / "local_link_diagnostics.json", link_diagnostics)
    invalid_ids = [
        edge
        for edge in final_edges
        if edge.get("parent_node_id") not in eligible_ids
        or edge.get("child_node_id") not in eligible_ids
    ]
    predicate_missing_final_edges = [
        edge
        for edge in final_edges
        if not str(edge.get("relation_text") or "").strip()
    ]
    has_cap = int(cfg.get("edge2d", {}).get("max_edges_per_frame", 0) or 0) > 0 or int(
        cfg.get("edge2d", {}).get("max_edges_per_run", 0) or 0
    ) > 0
    smoke_mode = not bool(cfg.get("output", {}).get("graph_complete", True))
    completeness = {
        "edge_funnel_complete": bool(candidate_records),
        "node_endpoint_mapping_complete": bool(mapped_or_reported),
        "optimizer_complete": True,
        "hierarchy_complete": True,
        "object_prefilter_pass_candidates": len(prefilter_object),
        "object_s2d_available_candidates": sum(
            1 for candidate in prefilter_object if candidate.s2d_available
        ),
        "unresolved_endpoint_count": len(unresolved),
        "graph_eligible_children_without_object_candidate": len(no_candidate),
    }
    graph_complete = bool(
        completeness["edge_funnel_complete"]
        and all_prefilter_scored
        and completeness["node_endpoint_mapping_complete"]
        and not unresolved
        and not no_candidate
        and not unlinked_children
        and not invalid_ids
        and not predicate_missing_final_edges
        and not any(count > 1 for count in child_parent_counts.values())
        and int(hierarchy_meta.get("cross_object_cu_edges", 0)) == 0
        and not has_cap
        and not smoke_mode
    )
    completeness["graph_complete"] = graph_complete
    completeness["incomplete_reasons"] = [
        reason
        for condition, reason in (
            (not all_prefilter_scored, "s2d_coverage_incomplete"),
            (not mapped_or_reported, "endpoint_mapping_unaccounted"),
            (bool(unresolved), "endpoint_mapping_sparse"),
            (bool(no_candidate), "graph_eligible_child_has_no_object_candidate"),
            (bool(unlinked_children), "graph_eligible_child_has_no_local_parent"),
            (bool(invalid_ids), "illegal_node_id"),
            (bool(predicate_missing_final_edges), "functional_predicate_missing"),
            (any(count > 1 for count in child_parent_counts.values()), "multiple_immediate_parents"),
            (
                int(hierarchy_meta.get("cross_object_cu_edges", 0)) > 0,
                "cross_object_carrier_unit_edge",
            ),
            (has_cap, "artificial_edge_cap"),
            (smoke_mode, "smoke_mode"),
        )
        if condition
    ]

    save_map(nodes, map_dir)
    stage_edges = {
        "raw_local_candidate_overlay": (raw_local, nodes),
        "final_local_relation_overlay": (final_local, nodes),
        "semantic_candidate_overlay": (semantic, nodes),
        "prefilter_pass_edge_overlay": (prefilter, nodes),
        "scored_edge_overlay": (scored, nodes),
        "mapped_graph_eligible_edge_overlay": (graph_eligible_mapped, eligible_nodes),
        "optimized_object_graph": (optimized_edges, eligible_nodes),
        "completed_object_graph": (completed_object_edges, eligible_nodes),
        "remote_relation_graph": (remote_edges, eligible_nodes),
        "final_hierarchical_graph": (final_edges, eligible_nodes),
    }
    scene_source = ""
    if write_ply:
        scene_points, scene_colors, scene_source = _scene(mapping_run, nodes)
        overlay_cfg = cfg.get("output", {}).get("scene_graph_overlay", {})
        for stage, (edges, stage_nodes) in stage_edges.items():
            extra = {
                "graph_stage": stage,
                "is_final_graph": bool(stage == "final_hierarchical_graph" and graph_complete),
                **completeness,
            }
            if stage == "optimized_object_graph":
                extra.update(optimizer_meta)
            elif stage == "final_hierarchical_graph":
                extra.update(hierarchy_meta)
            write_scene_graph_overlay_ply(
                scene_points=scene_points,
                scene_colors=scene_colors,
                nodes=stage_nodes,
                edges=edges,
                output_ply_path=map_dir / f"{stage}.ply",
                metadata_path=map_dir / f"{stage}.json",
                scene_keep_ratio=float(overlay_cfg.get("scene_keep_ratio", 1.0)),
                random_seed=int(overlay_cfg.get("random_seed", 0)),
                clear_scene_near_graph=bool(overlay_cfg.get("clear_scene_near_graph", True)),
                metadata_extra=extra,
            )
    else:
        for stage, (edges, stage_nodes) in stage_edges.items():
            stage_meta = (
                optimizer_meta if stage == "optimized_object_graph"
                else hierarchy_meta if stage == "final_hierarchical_graph" else {}
            )
            write_json_atomic(
                map_dir / f"{stage}.json",
                {
                    "graph_stage": stage,
                    "nodes": sorted(stage_nodes),
                    "edges": edges,
                    **completeness,
                    **stage_meta,
                },
            )
    write_json_atomic(map_dir / "graph_overlay_source.json", {"scene_source": scene_source})
    summary = {
        **join_meta,
        **eligibility_meta,
        **completeness,
        "local_link_diagnostics": link_diagnostics,
        "num_mapped_evidence": len(evidence),
        "num_recovered_carrier_pairs": len(aggregate_candidates(recovered)),
        "num_predicates_recovered": predicates_missing_before - len(predicate_missing_candidates),
        "num_predicate_pending_candidates": len(predicate_missing_candidates),
        "num_predicate_missing_final_edges": len(predicate_missing_final_edges),
        "num_optimizer_input": len(optimizer_input),
        "num_optimized_edges": len(optimized_edges),
        "num_derived_objects": len(derived_nodes),
        "num_derived_object_edges": len(derived_object_edges),
        "num_children_rejected_by_null_parent": int(
            optimizer_meta.get("num_children_rejected_by_null_parent", 0)
        ),
        "num_children_rejected_by_parent_margin": int(
            optimizer_meta.get("num_children_rejected_by_parent_margin", 0)
        ),
        "num_children_provisional_by_null_parent": int(
            optimizer_meta.get("num_children_provisional_by_null_parent", 0)
        ),
        "num_children_provisional_by_parent_margin": int(
            optimizer_meta.get("num_children_provisional_by_parent_margin", 0)
        ),
        "num_hierarchy_candidates": sum(
            str(candidate.get("edge_type")) == "C-U"
            for candidate in immediate_parent_candidates
        ),
        "num_immediate_parent_candidates": len(immediate_parent_candidates),
        "num_final_edges": len(final_edges),
        "num_final_local_edges": len(local_final_edges),
        "num_remote_edges": len(remote_edges),
        "remote_relations": remote_meta,
        "num_units_with_selected_carrier": int(
            hierarchy_meta.get("num_units_with_selected_carrier", 0)
        ),
        "num_units_without_selected_carrier": int(
            hierarchy_meta.get("num_units_without_selected_carrier", 0)
        ),
        "num_units_with_direct_object_context": int(
            hierarchy_meta.get("num_units_with_direct_object_context", 0)
        ),
        "num_cu_rejected_by_direct_object_context": int(
            hierarchy_meta.get(
                "num_cu_rejected_by_direct_object_context", 0
            )
        ),
        "hierarchy_surface_distance_m": hierarchy_meta.get(
            "surface_distance_m", {}
        ),
        "hierarchy_reject_reasons": hierarchy_meta.get(
            "hierarchy_reject_reasons", {}
        ),
        "final_multi_immediate_parent_children": int(
            sum(1 for count in child_parent_counts.values() if count > 1)
        ),
        "cross_object_cu_edges": int(hierarchy_meta.get("cross_object_cu_edges", 0)),
        "unanchored_cu_edges": int(hierarchy_meta.get("unanchored_cu_edges", 0)),
        "derived_object_completion": completion_meta,
        "stage_edge_counts": {stage: len(edges) for stage, (edges, _nodes) in stage_edges.items()},
    }
    write_json_atomic(output_dir / "functional_graph_summary.json", summary)
    return summary
