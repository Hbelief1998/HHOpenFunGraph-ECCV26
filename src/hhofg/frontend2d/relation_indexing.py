from __future__ import annotations

from typing import Any, Mapping, Sequence


RELATION_INDEX_SCHEMA_VERSION = 2


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _relation_stages(local_rel_debug: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(local_rel_debug, dict):
        return {}
    stages = local_rel_debug.get("local_rel", local_rel_debug)
    return stages if isinstance(stages, dict) else {}


def _rebuild_edge_lookups(payload: dict[str, Any]) -> None:
    by_child: dict[int, list[int]] = {}
    by_parent: dict[int, list[int]] = {}
    for eid, edge in enumerate(payload.get("edges", []) or []):
        child_idx = _as_int(edge.get("child_idx"))
        parent_idx = _as_int(edge.get("parent_idx"))
        if child_idx is not None:
            by_child.setdefault(child_idx, []).append(eid)
        if parent_idx is not None:
            by_parent.setdefault(parent_idx, []).append(eid)
    payload["by_child"] = by_child
    payload["by_parent"] = by_parent


def _remap_assignments(
    assignments: Any,
    index_map: Mapping[int, int],
    edge_id_map: Mapping[int, int],
) -> dict[int, Any]:
    if not isinstance(assignments, dict):
        return {}
    remapped: dict[int, Any] = {}
    for old_child_key, edge_types in assignments.items():
        old_child_idx = _as_int(old_child_key)
        if old_child_idx is None or old_child_idx not in index_map or not isinstance(edge_types, dict):
            continue
        new_edge_types: dict[str, Any] = {}
        for edge_type, assignment in edge_types.items():
            if not isinstance(assignment, dict):
                continue
            item = dict(assignment)
            item["child_idx"] = index_map[old_child_idx]

            candidate_eids = []
            for old_eid in item.get("candidate_eids", []) or []:
                old_eid_int = _as_int(old_eid)
                if old_eid_int is not None and old_eid_int in edge_id_map:
                    candidate_eids.append(edge_id_map[old_eid_int])
            item["candidate_eids"] = candidate_eids

            candidates = []
            for candidate in item.get("candidates", []) or []:
                if not isinstance(candidate, dict):
                    continue
                old_parent_idx = _as_int(candidate.get("parent_idx"))
                old_eid = _as_int(candidate.get("eid"))
                if (
                    old_parent_idx is None
                    or old_parent_idx not in index_map
                    or old_eid is None
                    or old_eid not in edge_id_map
                ):
                    continue
                candidate = dict(candidate)
                candidate["parent_idx"] = index_map[old_parent_idx]
                candidate["eid"] = edge_id_map[old_eid]
                candidates.append(candidate)
            item["candidates"] = candidates

            old_chosen_eid = _as_int(item.get("chosen_eid"))
            old_chosen_parent = _as_int(item.get("chosen_parent_idx"))
            if (
                old_chosen_eid is not None
                and old_chosen_eid in edge_id_map
                and old_chosen_parent is not None
                and old_chosen_parent in index_map
            ):
                item["chosen_eid"] = edge_id_map[old_chosen_eid]
                item["chosen_parent_idx"] = index_map[old_chosen_parent]
            else:
                item["chosen_eid"] = None
                item["chosen_parent_idx"] = None
                if str(item.get("status") or "").startswith("confirmed"):
                    item["status"] = "endpoint_filtered"
            new_edge_types[str(edge_type)] = item
        remapped[index_map[old_child_idx]] = new_edge_types
    return remapped


def _remap_stage_payload(
    payload: dict[str, Any],
    *,
    index_map: Mapping[int, int],
    new_source_det_ids: Sequence[int],
    final_labels: Sequence[str] | None = None,
    remap_cabinet_hints: bool = True,
) -> None:
    old_edges = list(payload.get("edges", []) or [])
    new_edges: list[dict[str, Any]] = []
    edge_id_map: dict[int, int] = {}
    discarded_by_reason: dict[str, int] = {}

    for old_eid, edge in enumerate(old_edges):
        if not isinstance(edge, dict):
            discarded_by_reason["invalid_edge"] = discarded_by_reason.get("invalid_edge", 0) + 1
            continue
        old_parent_idx = _as_int(edge.get("parent_idx"))
        old_child_idx = _as_int(edge.get("child_idx"))
        if (
            old_parent_idx is None
            or old_child_idx is None
            or old_parent_idx not in index_map
            or old_child_idx not in index_map
        ):
            discarded_by_reason["endpoint_filtered"] = discarded_by_reason.get("endpoint_filtered", 0) + 1
            continue

        new_parent_idx = index_map[old_parent_idx]
        new_child_idx = index_map[old_child_idx]
        if final_labels is not None:
            if not (0 <= new_parent_idx < len(final_labels) and 0 <= new_child_idx < len(final_labels)):
                discarded_by_reason["endpoint_out_of_range"] = discarded_by_reason.get("endpoint_out_of_range", 0) + 1
                continue
            parent_label = edge.get("parent_label")
            child_label = edge.get("child_label")
            if parent_label is not None and str(parent_label) != str(final_labels[new_parent_idx]):
                discarded_by_reason["parent_label_mismatch"] = discarded_by_reason.get("parent_label_mismatch", 0) + 1
                continue
            if child_label is not None and str(child_label) != str(final_labels[new_child_idx]):
                discarded_by_reason["child_label_mismatch"] = discarded_by_reason.get("child_label_mismatch", 0) + 1
                continue

        edge = dict(edge)
        edge["parent_idx"] = new_parent_idx
        edge["child_idx"] = new_child_idx
        edge["parent_source_det_id"] = int(new_source_det_ids[new_parent_idx])
        edge["child_source_det_id"] = int(new_source_det_ids[new_child_idx])
        edge_id_map[old_eid] = len(new_edges)
        new_edges.append(edge)

    old_assignments = payload.get("assignments")
    payload["edges"] = new_edges
    payload["assignments"] = _remap_assignments(old_assignments, index_map, edge_id_map)

    new_chains = []
    for chain in payload.get("ocu_chains", []) or []:
        if not isinstance(chain, dict):
            continue
        old_o = _as_int(chain.get("o_idx"))
        old_c = _as_int(chain.get("c_idx"))
        old_u = _as_int(chain.get("u_idx"))
        if (
            old_o is None
            or old_c is None
            or old_u is None
            or old_o not in index_map
            or old_c not in index_map
            or old_u not in index_map
        ):
            continue
        chain = dict(chain)
        chain["o_idx"] = index_map[old_o]
        chain["c_idx"] = index_map[old_c]
        chain["u_idx"] = index_map[old_u]
        new_chains.append(chain)
    payload["ocu_chains"] = new_chains

    new_rechecks = []
    for recheck in payload.get("carrier_child_recheck", []) or []:
        if not isinstance(recheck, dict):
            continue
        old_parent = _as_int(recheck.get("parent_idx"))
        if old_parent is None or old_parent not in index_map:
            continue
        recheck = dict(recheck)
        recheck["parent_idx"] = index_map[old_parent]
        for key in ("cu_selected_eids", "removed_eids"):
            recheck[key] = [
                edge_id_map[eid_int]
                for eid in recheck.get(key, []) or []
                if (eid_int := _as_int(eid)) is not None and eid_int in edge_id_map
            ]
        new_scores = []
        for score in recheck.get("scores", []) or []:
            if not isinstance(score, dict):
                continue
            old_eid = _as_int(score.get("eid"))
            if old_eid is None or old_eid not in edge_id_map:
                continue
            score = dict(score)
            score["eid"] = edge_id_map[old_eid]
            new_scores.append(score)
        recheck["scores"] = new_scores
        new_rechecks.append(recheck)
    if "carrier_child_recheck" in payload:
        payload["carrier_child_recheck"] = new_rechecks

    cabinet_hints = payload.get("cabinet_hints")
    if remap_cabinet_hints and isinstance(cabinet_hints, dict):
        remapped_hits = []
        for hit in cabinet_hints.get("carrier_hits", []) or []:
            if not isinstance(hit, dict) or "child_idx" not in hit:
                continue
            old_child = _as_int(hit.get("child_idx"))
            if old_child is None or old_child not in index_map:
                continue
            hit = dict(hit)
            hit["child_idx"] = index_map[old_child]
            old_parent = _as_int(hit.get("cabinet_parent_idx"))
            hit["cabinet_parent_idx"] = index_map.get(old_parent, -1) if old_parent is not None else -1
            remapped_hits.append(hit)
        cabinet_hints["carrier_hits"] = remapped_hits

    _rebuild_edge_lookups(payload)
    payload["source_det_ids"] = [int(v) for v in new_source_det_ids]
    payload["index_space"] = "final_dense"
    payload["relation_index_schema_version"] = RELATION_INDEX_SCHEMA_VERSION
    if discarded_by_reason:
        payload["discarded_edge_count"] = int(sum(discarded_by_reason.values()))
        payload["discarded_edges_by_reason"] = discarded_by_reason


def canonicalize_local_relation_debug(
    local_rel_debug: dict[str, Any] | None,
    *,
    final_source_det_ids: Sequence[int],
    final_labels: Sequence[str],
) -> None:
    """Map every relation snapshot into the current final-detection index space.

    Each snapshot must carry ``snapshot_source_det_ids``: a mapping from that
    snapshot's dense indices to immutable IDs assigned at the raw snapshot.
    """
    if not isinstance(local_rel_debug, dict):
        return
    final_source_ids = [int(v) for v in final_source_det_ids]
    source_to_final = {source_id: idx for idx, source_id in enumerate(final_source_ids)}

    for payload in _relation_stages(local_rel_debug).values():
        if not isinstance(payload, dict) or "edges" not in payload:
            continue
        snapshot_ids = payload.get("snapshot_source_det_ids")
        if not isinstance(snapshot_ids, list):
            # Legacy debug data has no trustworthy cross-stage lineage.
            payload["index_space"] = "legacy_unverified"
            continue
        index_map = {
            snapshot_idx: source_to_final[source_id]
            for snapshot_idx, source_id in enumerate(snapshot_ids)
            if source_id in source_to_final
        }
        _remap_stage_payload(
            payload,
            index_map=index_map,
            new_source_det_ids=final_source_ids,
            final_labels=final_labels,
            # Runtime canonicalizes cabinet hints separately because their
            # ``*_pre_drop`` fields belong to a dedicated hint snapshot.
            remap_cabinet_hints=False,
        )

    local_rel_debug["final_source_det_ids"] = final_source_ids
    local_rel_debug["relation_index_schema_version"] = RELATION_INDEX_SCHEMA_VERSION


def reindex_local_relation_debug(
    local_rel_debug: dict[str, Any] | None,
    *,
    keep: Sequence[int],
) -> None:
    """Atomically reindex all canonical relation structures after detection filtering."""
    if not isinstance(local_rel_debug, dict):
        return
    old_source_ids = local_rel_debug.get("final_source_det_ids")
    if not isinstance(old_source_ids, list):
        # Legacy payloads are deliberately left untouched; validated extraction
        # will fail closed instead of guessing their index lineage.
        return
    keep_indices = [int(i) for i in keep]
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
    new_source_ids = [int(old_source_ids[i]) for i in keep_indices]
    for payload in _relation_stages(local_rel_debug).values():
        if not isinstance(payload, dict) or payload.get("index_space") != "final_dense":
            continue
        _remap_stage_payload(
            payload,
            index_map=index_map,
            new_source_det_ids=new_source_ids,
        )
    local_rel_debug["final_source_det_ids"] = new_source_ids
    local_rel_debug["relation_index_schema_version"] = RELATION_INDEX_SCHEMA_VERSION
