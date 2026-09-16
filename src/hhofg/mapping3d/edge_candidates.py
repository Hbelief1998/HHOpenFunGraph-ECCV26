from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from hhofg.core.types import Detection2D, Frame2DResult, LocalRelation2D

from .semantic_candidate_builder import (
    SemanticRelationIndex, select_relation_text,
    build_semantic_relation_index,
)
from .types import FrameAssociationResult, FrameEdgeCandidate, MapNode3D, MappedFrameEdgeEvidence


def _det_by_id(result: Frame2DResult) -> dict[int, Detection2D]:
    return {int(d.det_id): d for d in result.detections}


def _candidate_from_relation(
    *,
    result: Frame2DResult,
    relation: LocalRelation2D,
    detections_by_id: dict[int, Detection2D],
    association: FrameAssociationResult,
    lifted_det_ids: set[int],
    nodes: dict[str, MapNode3D],
    image_sha: str,
    source: str,
    semantic_sources: list[str] | None = None,
    semantic_chain: dict[str, Any] | None = None,
    from_raw: bool = False,
    from_final: bool = False,
    semantic_only: bool = False,
    ownership_geometry: dict[str, Any] | None = None,
) -> FrameEdgeCandidate | None:
    p = int(relation.parent_det_id)
    c = int(relation.child_det_id)
    if p == c:
        return None
    parent = detections_by_id.get(p)
    child = detections_by_id.get(c)
    if parent is None or child is None:
        return None
    parent_node_id = association.det_to_node.get(p)
    child_node_id = association.det_to_node.get(c)
    parent_node = nodes.get(parent_node_id or "")
    child_node = nodes.get(child_node_id or "")
    return FrameEdgeCandidate(
        frame_idx=result.frame_idx,
        frame_key=result.frame_key,
        edge_type=relation.edge_type,
        parent_det_id=p,
        child_det_id=c,
        parent_label=parent.label,
        child_label=child.label,
        parent_score=parent.score,
        child_score=child.score,
        parent_role=relation.parent_role,
        child_role=relation.child_role,
        selected_2d=relation.selected,
        pass_threshold=relation.pass_threshold,
        contain=relation.contain,
        mask_contain=relation.mask_contain,
        relation_text=relation.relation_text,
        instance_relation_texts=[relation.relation_text] if relation.relation_text and (from_raw or from_final) else [],
        parent_node_id=parent_node_id,
        child_node_id=child_node_id,
        parent_node_state=parent_node.state if parent_node is not None else None,
        child_node_state=child_node.state if child_node is not None else None,
        parent_lifted=p in lifted_det_ids,
        child_lifted=c in lifted_det_ids,
        image_sha=image_sha,
        candidate_source=source,
        candidate_sources=[source],
        semantic_sources=list(semantic_sources or []),
        semantic_chains=[semantic_chain] if semantic_chain else [],
        from_raw_relation=from_raw,
        from_final_relation=from_final,
        semantic_only=semantic_only,
        ownership_geometry=dict(ownership_geometry or {}),
    )


def _ownership_geometry(parent: Detection2D, child: Detection2D) -> dict[str, Any]:
    parent_box = np.asarray(parent.box_xyxy, dtype=np.float64)
    child_box = np.asarray(child.box_xyxy, dtype=np.float64)
    parent_size = np.maximum(parent_box[2:] - parent_box[:2], 1.0)
    child_size = np.maximum(child_box[2:] - child_box[:2], 0.0)
    child_center = 0.5 * (child_box[:2] + child_box[2:])
    parent_center = 0.5 * (parent_box[:2] + parent_box[2:])
    center_inside = bool(
        parent_box[0] <= child_center[0] <= parent_box[2]
        and parent_box[1] <= child_center[1] <= parent_box[3]
    )
    intersection_size = np.maximum(
        np.minimum(parent_box[2:], child_box[2:])
        - np.maximum(parent_box[:2], child_box[:2]),
        0.0,
    )
    child_area = float(np.prod(child_size))
    child_box_coverage = (
        0.0 if child_area <= 0.0 else float(np.prod(intersection_size) / child_area)
    )
    normalized_center_distance = float(
        np.linalg.norm((child_center - parent_center) / parent_size)
    )
    score = (
        float(center_inside)
        + child_box_coverage
        + float(np.exp(-normalized_center_distance))
    )
    return {
        "child_center_inside_parent_box": center_inside,
        "child_box_coverage": child_box_coverage,
        "normalized_center_distance": normalized_center_distance,
        "score": score,
    }


def _relation_ownership_geometry(
    relation: LocalRelation2D,
    detections_by_id: dict[int, Detection2D],
) -> dict[str, Any]:
    parent = detections_by_id.get(int(relation.parent_det_id))
    child = detections_by_id.get(int(relation.child_det_id))
    if parent is None or child is None:
        return {}
    return _ownership_geometry(parent, child)


def _merge_candidate(dst: FrameEdgeCandidate, src: FrameEdgeCandidate) -> None:
    dst.candidate_sources = sorted(set(dst.candidate_sources) | set(src.candidate_sources))
    dst.semantic_sources = sorted(set(dst.semantic_sources) | set(src.semantic_sources))
    seen_chains = {repr(sorted(chain.items())) for chain in dst.semantic_chains}
    for chain in src.semantic_chains:
        marker = repr(sorted(chain.items()))
        if marker not in seen_chains:
            dst.semantic_chains.append(chain)
            seen_chains.add(marker)
    dst.selected_2d = bool(dst.selected_2d or src.selected_2d)
    dst.pass_threshold = bool(dst.pass_threshold or src.pass_threshold)
    dst.from_raw_relation = bool(dst.from_raw_relation or src.from_raw_relation)
    dst.from_final_relation = bool(dst.from_final_relation or src.from_final_relation)
    dst.semantic_only = bool(
        not dst.from_raw_relation and not dst.from_final_relation
    )
    if float(src.ownership_geometry.get("score", -1.0)) > float(
        dst.ownership_geometry.get("score", -1.0)
    ):
        dst.ownership_geometry = dict(src.ownership_geometry)
    if dst.contain is None:
        dst.contain = src.contain
    if dst.mask_contain is None:
        dst.mask_contain = src.mask_contain
    dst.instance_relation_texts = sorted(set(dst.instance_relation_texts + src.instance_relation_texts))
    if dst.instance_relation_texts:
        dst.relation_text = select_relation_text(dst.instance_relation_texts,
                                                labels=(dst.parent_label, dst.child_label))
    elif not dst.relation_text:
        dst.relation_text = src.relation_text


def _add_candidate(
    dedup: dict[tuple[str, str, int, int], FrameEdgeCandidate],
    candidate: FrameEdgeCandidate | None,
) -> None:
    if candidate is None:
        return
    key = (
        candidate.frame_key,
        candidate.edge_type,
        candidate.parent_det_id,
        candidate.child_det_id,
    )
    if key in dedup:
        _merge_candidate(dedup[key], candidate)
    else:
        dedup[key] = candidate


def annotate_candidate_semantics(
    candidates: Iterable[FrameEdgeCandidate],
    index: SemanticRelationIndex,
) -> None:
    candidates = list(candidates)
    contexts: dict[tuple[str, int], set[str]] = {}
    for c in candidates:
        if c.edge_type == "O-C" and not c.semantic_only and (c.selected_2d or c.pass_threshold):
            contexts.setdefault((c.frame_key, c.child_det_id), set()).add(c.parent_label)
    for candidate in candidates:
        owners = contexts.get((candidate.frame_key, candidate.parent_det_id), set())
        context = candidate.parent_label if candidate.parent_role == "O" else (
            next(iter(owners)) if len(owners) == 1 else None)
        if candidate.instance_relation_texts:
            candidate.relation_text = select_relation_text(candidate.instance_relation_texts,
                labels=(candidate.parent_label, candidate.child_label, context))
        sources = index.sources_for(
            candidate.edge_type, candidate.parent_label, candidate.child_label
        )
        if not sources:
            continue
        candidate.semantic_sources = sorted(
            set(candidate.semantic_sources) | sources
        )
        metadata = index.metadata_for(
            candidate.edge_type, candidate.parent_label, candidate.child_label,
            object_context=context,
        )
        if metadata not in candidate.semantic_chains:
            candidate.semantic_chains.append(metadata)
        if not candidate.relation_text and not candidate.instance_relation_texts:
            candidate.relation_text = index.relation_for(
                candidate.edge_type,
                candidate.parent_label,
                candidate.child_label,
                object_context=context,
            )


def _semantic_expansion(
    *,
    result: Frame2DResult,
    index: SemanticRelationIndex,
    observed: dict[tuple[str, str, int, int], FrameEdgeCandidate],
    detections_by_id: dict[int, Detection2D],
    association: FrameAssociationResult,
    lifted_det_ids: set[int],
    nodes: dict[str, MapNode3D],
    image_sha: str,
    cfg: dict[str, Any],
) -> list[FrameEdgeCandidate]:
    topk = max(0, int(cfg.get("semantic_topk_per_child", 2)))
    expand_only_missing = bool(
        cfg.get("expand_only_when_observed_missing", True)
    )
    if topk == 0:
        return []
    role_by_edge = {
        "O-C": ("O", "C"),
        "O-U": ("O", "U"),
        "C-U": ("C", "U"),
    }
    observed_parents: dict[tuple[str, int], set[int]] = {}
    for candidate in observed.values():
        observed_parents.setdefault(
            (candidate.edge_type, candidate.child_det_id), set()
        ).add(candidate.parent_det_id)

    expanded: list[FrameEdgeCandidate] = []
    for edge_type, (parent_role, child_role) in role_by_edge.items():
        parents = [det for det in result.detections if det.role == parent_role]
        children = [det for det in result.detections if det.role == child_role]
        for child in children:
            existing = observed_parents.get((edge_type, child.det_id), set())
            if existing and expand_only_missing:
                continue
            slots = max(0, topk - len(existing))
            if slots == 0:
                continue
            ranked: list[tuple[float, int, Detection2D, dict[str, Any], set[str]]] = []
            for parent in parents:
                if parent.det_id == child.det_id or parent.det_id in existing:
                    continue
                sources = index.sources_for(edge_type, parent.label, child.label)
                if not sources:
                    continue
                geometry = _ownership_geometry(parent, child)
                ranked.append(
                    (
                        -float(geometry["score"]),
                        int(parent.det_id),
                        parent,
                        geometry,
                        sources,
                    )
                )
            for _, _, parent, geometry, sources in sorted(ranked)[:slots]:
                relation = LocalRelation2D(
                    edge_type=edge_type,
                    parent_det_id=parent.det_id,
                    child_det_id=child.det_id,
                    parent_role=parent_role,
                    child_role=child_role,
                    contain=None,
                    mask_contain=None,
                    pass_threshold=False,
                    selected=False,
                    relation_text=index.relation_for(
                        edge_type, parent.label, child.label
                    ),
                    source_stage="semantic_prior",
                )
                candidate = _candidate_from_relation(
                    result=result,
                    relation=relation,
                    detections_by_id=detections_by_id,
                    association=association,
                    lifted_det_ids=lifted_det_ids,
                    nodes=nodes,
                    image_sha=image_sha,
                    source="semantic_prior",
                    semantic_sources=sorted(sources),
                    semantic_chain=index.metadata_for(
                        edge_type, parent.label, child.label
                    ),
                    semantic_only=True,
                    ownership_geometry=geometry,
                )
                if candidate is not None:
                    expanded.append(candidate)
    return expanded


def collect_frame_edge_candidates(
    result: Frame2DResult,
    association: FrameAssociationResult,
    lifted_det_ids: set[int],
    nodes: dict[str, MapNode3D],
    *,
    image_sha: str = "",
    mask_sha: str = "",
    global_atlas: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[FrameEdgeCandidate]:
    del mask_sha  # Per-pair hashes are computed during prefiltering.
    cfg = cfg or {}
    detections_by_id = _det_by_id(result)
    dedup: dict[tuple[str, str, int, int], FrameEdgeCandidate] = {}

    for rel in result.local_relation_candidates_raw or []:
        _add_candidate(
            dedup,
            _candidate_from_relation(
                result=result,
                relation=rel,
                detections_by_id=detections_by_id,
                association=association,
                lifted_det_ids=lifted_det_ids,
                nodes=nodes,
                image_sha=image_sha,
                source="raw_local_relation",
                from_raw=True,
                ownership_geometry=_relation_ownership_geometry(
                    rel, detections_by_id
                ),
            ),
        )
    for rel in result.local_relations_final or []:
        _add_candidate(
            dedup,
            _candidate_from_relation(
                result=result,
                relation=rel,
                detections_by_id=detections_by_id,
                association=association,
                lifted_det_ids=lifted_det_ids,
                nodes=nodes,
                image_sha=image_sha,
                source="final_local_relation",
                from_final=True,
                ownership_geometry=_relation_ownership_geometry(
                    rel, detections_by_id
                ),
            ),
        )
    index = build_semantic_relation_index(result, global_atlas)
    annotate_candidate_semantics(dedup.values(), index)
    for candidate in _semantic_expansion(
        result=result,
        index=index,
        observed=dedup,
        detections_by_id=detections_by_id,
        association=association,
        lifted_det_ids=lifted_det_ids,
        nodes=nodes,
        image_sha=image_sha,
        cfg=cfg,
    ):
        _add_candidate(dedup, candidate)

    priority = {
        "raw_local_relation": 0,
        "final_local_relation": 1,
        "semantic_prior": 2,
    }
    out = []
    for candidate in dedup.values():
        candidate.candidate_sources.sort(key=lambda value: (priority.get(value, 99), value))
        candidate.candidate_source = candidate.candidate_sources[0]
        candidate.semantic_only = bool(
            not candidate.from_raw_relation and not candidate.from_final_relation
        )
        out.append(candidate)
    return sorted(out, key=lambda c: (c.frame_idx, c.edge_type, c.parent_det_id, c.child_det_id))


def candidate_to_relation(candidate: FrameEdgeCandidate) -> LocalRelation2D:
    return LocalRelation2D(
        edge_type=candidate.edge_type,
        parent_det_id=candidate.parent_det_id,
        child_det_id=candidate.child_det_id,
        parent_role=candidate.parent_role,
        child_role=candidate.child_role,
        contain=candidate.contain,
        mask_contain=candidate.mask_contain,
        pass_threshold=candidate.pass_threshold,
        selected=candidate.selected_2d,
        relation_text=candidate.relation_text,
        source_stage=candidate.candidate_source,
    )


def candidate_to_mapped_evidence(candidate: FrameEdgeCandidate) -> MappedFrameEdgeEvidence | None:
    if not candidate.parent_node_id or not candidate.child_node_id:
        return None
    # Structural ownership evidence survives missing functional semantics.
    # Final export marks pending predicates; functional evaluation excludes them.
    return MappedFrameEdgeEvidence(
        frame_idx=candidate.frame_idx,
        frame_key=candidate.frame_key,
        edge_type=candidate.edge_type,
        parent_det_id=candidate.parent_det_id,
        child_det_id=candidate.child_det_id,
        parent_node_id=candidate.parent_node_id,
        child_node_id=candidate.child_node_id,
        parent_role=candidate.parent_role,
        child_role=candidate.child_role,
        pass_threshold=candidate.pass_threshold,
        selected_2d=candidate.selected_2d,
        contain=candidate.contain,
        mask_contain=candidate.mask_contain,
        relation_text=candidate.relation_text,
        s2d_score=candidate.s2d_score,
        candidate_hash=candidate.candidate_hash,
        candidate_sources=list(candidate.candidate_sources),
        semantic_sources=list(candidate.semantic_sources),
        semantic_chains=list(candidate.semantic_chains),
        parent_label=candidate.parent_label,
        child_label=candidate.child_label,
        semantic_only=candidate.semantic_only,
        evidence_score_source=(
            "vlm_s2d"
            if candidate.s2d_score is not None
            else "observed_2d_prior"
            if candidate.selected_2d or candidate.pass_threshold
            else "semantic_prior"
        ),
    )


def mapped_candidate_drop_reason(
    candidate: FrameEdgeCandidate,
    reject_reasons: dict[int, str] | None = None,
) -> dict[str, Any] | None:
    if candidate.parent_node_id and candidate.child_node_id:
        return None
    reject_reasons = reject_reasons or {}
    if not candidate.parent_lifted or not candidate.child_lifted:
        reason = "endpoint_not_lifted"
    else:
        reason = "endpoint_not_associated"
    return {
        "frame_idx": candidate.frame_idx,
        "frame_key": candidate.frame_key,
        "edge_type": candidate.edge_type,
        "parent_det_id": candidate.parent_det_id,
        "child_det_id": candidate.child_det_id,
        "candidate_hash": candidate.candidate_hash,
        "reason": reason,
        "unresolved_endpoint_reason": reason,
        "parent_reject_reason": reject_reasons.get(candidate.parent_det_id),
        "child_reject_reason": reject_reasons.get(candidate.child_det_id),
        "candidate_sources": candidate.candidate_sources,
    }


def recover_semantic_carrier_candidates(
    *, frontend_run: str | Path | None, association_maps: dict[str, dict[int, str]],
    nodes: dict[str, MapNode3D], index: SemanticRelationIndex,
    unlinked_units: set[str], existing_pairs: set[tuple[str, str]],
    cfg: dict[str, Any], max_distance: float, mask_threshold: float,
) -> list[FrameEdgeCandidate]:
    """Revisit historical masks for still-unlinked units using the same prior.

    Only real co-observed, graph-eligible endpoints are used. Missing masks or
    detections cannot be replaced by 3D proximity. No selected flags or S2D
    scores are synthesized, and only top-K distinct parents survive per unit.
    """
    from hhofg.core.serialization import load_frame2d_result
    from hhofg.edge2d.candidate_filter import build_parent_support_mask
    from hhofg.graph3d.hierarchy import _association_aabb_distance, carrier_unit_surface_distance

    topk = max(0, int(cfg.get("semantic_topk_per_child", 2)))
    if not frontend_run or not unlinked_units or not topk:
        return []
    recovered = []
    distances = {}
    for frame_key, det_map in sorted(association_maps.items()):
        if not unlinked_units.intersection(det_map.values()):
            continue
        path = Path(frontend_run) / "frames" / f"{frame_key}.json"
        if not path.is_file() or not path.with_suffix('.npz').is_file():
            continue
        result, _, _, masks = load_frame2d_result(path, path.with_suffix('.npz'))
        by_id = _det_by_id(result)
        mask_by_id = {d.det_id: masks[i] for i, d in enumerate(result.detections)}
        assoc = FrameAssociationResult(result.frame_idx, frame_key, [], [], [], [], det_map, {})
        # Restrict expansion before ranking: a stale/unmapped parent must not
        # occupy a slot ahead of a usable historical instance.
        subset = replace(result, detections=[d for d in result.detections
            if det_map.get(d.det_id) in nodes
            and (d.role == 'C' or det_map.get(d.det_id) in unlinked_units)])
        proposals = _semantic_expansion(
            result=subset, index=index, observed={}, detections_by_id=by_id,
            association=assoc, lifted_det_ids=set(det_map), nodes=nodes,
            image_sha='', cfg=cfg,
        )
        for c in proposals:
            if c.edge_type != 'C-U':
                continue
            pair = (c.parent_node_id, c.child_node_id)
            if pair in existing_pairs or not c.ownership_geometry['child_center_inside_parent_box']:
                continue
            if pair not in distances:
                parent, child = (nodes[k] for k in pair)
                distances[pair] = (carrier_unit_surface_distance(parent, child)
                    if _association_aabb_distance(parent, child) <= max_distance else float('inf'))
            if distances[pair] > max_distance:
                continue
            support = build_parent_support_mask(mask_by_id[c.parent_det_id], by_id[c.parent_det_id].box_xyxy)
            child_mask = np.asarray(mask_by_id[c.child_det_id], dtype=bool)
            coverage = float(np.count_nonzero(support & child_mask) / max(1, np.count_nonzero(child_mask)))
            if coverage <= mask_threshold:
                continue
            c.ownership_geometry['historical_mask_coverage'] = coverage
            recovered.append(c)
    kept = set()
    for unit in sorted(unlinked_units):
        pairs = {(c.parent_node_id, c.child_node_id) for c in recovered if c.child_node_id == unit}
        kept.update(sorted(pairs, key=lambda pair: (distances[pair], pair[0]))[:topk])
    return [c for c in recovered if (c.parent_node_id, c.child_node_id) in kept]
