from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from hhofg.core.types import Frame2DResult, FrameRecord

from .features import cosine01
from .hungarian import solve_non_exhaustive_assignment
from .projection import box_iou_xyxy, project_map_node_box
from .types import AssociationPairScore, FrameAssociationResult, MappedFrameEdgeEvidence, MapNode3D, Observation3D


def score_observation_node_pair(
    obs: Observation3D,
    node: MapNode3D,
    frame: FrameRecord,
    cfg: dict[str, Any],
) -> AssociationPairScore:
    node_extent = float(
        getattr(node, "association_bbox_diag_world", None)
        or node.bbox_diag_world
    )
    observation_extent = float(
        getattr(obs, "association_bbox_diag_world", None)
        or obs.bbox_diag_world
    )
    # A conservative fusion cluster must not collapse the identity gate.  Use
    # the larger observation/map association extent and a single generic floor
    # shared by every semantic label.
    tau = min(
        float(cfg.get("max_distance_m", 0.15)),
        max(
            float(cfg.get("min_distance_m", 0.03)),
            float(cfg.get("bbox_diag_scale", 0.5))
            * max(node_extent, observation_extent),
        ),
    )
    distance = float(np.linalg.norm(obs.centroid_world - node.centroid_world))
    projected = None
    score_iou = 0.0
    score_geo = float(np.exp(-(distance * distance) / (2.0 * float(cfg.get("sigma_m", 0.08)) ** 2)))
    score_app = cosine01(obs.appearance_hist, node.appearance_hist)
    score_sem = cosine01(obs.semantic_feature, node.semantic_feature)
    reject = ""
    valid = True
    if obs.role != node.role:
        valid = False
        reject = "role_mismatch"
    elif _stable_parent_conflict(obs, node, cfg):
        valid = False
        reject = "stable_parent_conflict"
    elif frame.camera is None or frame.T_c2w is None:
        valid = False
        reject = "missing_camera_or_pose"
    else:
        projected = project_map_node_box(
            np.asarray(
                getattr(node, "association_points_world", None)
                if getattr(node, "association_points_world", None) is not None
                else node.points_world,
                dtype=np.float32,
            ),
            frame.camera.K,
            frame.T_c2w,
            frame.camera.height,
            frame.camera.width,
            min_projected_points=int(cfg.get("min_projected_points", 3)),
            depth_m=frame.depth_m if bool(cfg.get("depth_aware_projection", True)) else None,
            depth_tolerance_m=float(cfg.get("projection_depth_tolerance_m", 0.08)),
            percentile_low=float(cfg.get("projection_percentile_low", 2.0)),
            percentile_high=float(cfg.get("projection_percentile_high", 98.0)),
        )
        if projected is None:
            valid = False
            reject = "not_projectable"
        else:
            score_iou = box_iou_xyxy(obs.box_xyxy, projected)
    total = (
        float(cfg.get("weight_iou", 0.5)) * score_iou
        + float(cfg.get("weight_geo", 1.0 / 6.0)) * score_geo
        + float(cfg.get("weight_app", 1.0 / 6.0)) * score_app
        + float(cfg.get("weight_sem", 1.0 / 6.0)) * score_sem
    )
    if valid and not (distance < tau):
        valid = False
        reject = "distance_gate"
    if valid and bool(cfg.get("require_positive_iou", True)) and not (score_iou > 0.0):
        valid = False
        reject = "iou_gate"
    if valid and total < float(cfg.get("min_score", 0.45)):
        valid = False
        reject = "score_gate"
    if valid:
        reject = ""
    return AssociationPairScore(
        observation_id=obs.observation_id,
        node_id=node.node_id,
        role=obs.role,
        distance_m=distance,
        tau_distance_m=float(tau),
        projected_box_xyxy=projected,
        score_iou=float(score_iou),
        score_geo=float(np.clip(score_geo, 0.0, 1.0)),
        score_app=float(score_app),
        score_sem=float(score_sem),
        score_total=float(total),
        valid=bool(valid),
        reject_reason=reject,
    )


def score_identity_only_pair(
    obs: Observation3D,
    node: MapNode3D,
    frame: FrameRecord,
    cfg: dict[str, Any],
) -> AssociationPairScore:
    # Reuse the paper's four-clue score without changing its weights, while
    # applying an identity-specific distance gate suitable for small nodes.
    relaxed = dict(cfg)
    relaxed["max_distance_m"] = 1.0e9
    relaxed["bbox_diag_scale"] = 1.0e9
    pair = score_observation_node_pair(obs, node, frame, relaxed)
    role_floors = cfg.get(
        "low_quality_identity_role_min_distance_m",
        {"O": 0.08, "C": 0.05, "U": 0.03},
    )
    role_floor = float(role_floors.get(obs.role, 0.0))
    max_distance = float(cfg.get("low_quality_identity_max_distance_m", 0.25))
    diag_scale = float(cfg.get("identity_bbox_diag_scale", 0.5))
    tau_identity = max(role_floor, min(max_distance, diag_scale * float(node.bbox_diag_world)))
    pair.tau_distance_m = float(tau_identity)
    pair.association_mode = "identity_only"
    pair.geometry_fusion = "not_attempted_identity_only"

    if node.state != "stable" and not bool(getattr(node, "graph_eligible", False)):
        pair.valid = False
        pair.reject_reason = "node_not_identity_eligible"
    elif pair.valid and pair.score_sem < float(cfg.get("low_quality_identity_min_semantic_similarity", 0.5)):
        pair.valid = False
        pair.reject_reason = "semantic_gate"
    elif pair.valid and not (pair.distance_m < tau_identity):
        pair.valid = False
        pair.reject_reason = "identity_distance_gate"
    if pair.valid:
        pair.reject_reason = ""
    return pair


def _stable_parent_conflict(obs: Observation3D, node: MapNode3D, cfg: dict[str, Any]) -> bool:
    if not bool(cfg.get("stable_parent_conflict_gate", False)):
        return False
    if obs.role not in {"C", "U"}:
        return False
    det_to_node = cfg.get("_current_det_to_node", {})
    nodes = cfg.get("_current_nodes", {})
    current_parents = [det_to_node.get(int(det_id)) for det_id in obs.local_parent_det_ids]
    current_parents = [p for p in current_parents if p in nodes and getattr(nodes[p], "state", "") == "stable"]
    if not current_parents:
        return False
    by_role: dict[str, list[str]] = {"O": [], "C": []}
    for parent_id in current_parents:
        parent_role = str(getattr(nodes[parent_id], "role", ""))
        if parent_role in by_role:
            by_role[parent_role].append(str(parent_id))
    object_parent = getattr(node, "stable_object_parent_id", None)
    carrier_parent = getattr(node, "stable_carrier_parent_id", None)
    # Backward compatibility for v2 C checkpoints only. A legacy U context is
    # intentionally not reused because its object/carrier semantics are mixed.
    if obs.role == "C" and not object_parent:
        object_parent = getattr(node, "stable_parent_id", None)
    if object_parent and by_role["O"] and all(p != str(object_parent) for p in by_role["O"]):
        return True
    if obs.role == "U" and carrier_parent and by_role["C"] and all(
        p != str(carrier_parent) for p in by_role["C"]
    ):
        return True
    return False


def associate_observations_to_nodes(
    observations: list[Observation3D],
    nodes: dict[str, MapNode3D],
    frame: FrameRecord,
    cfg: dict[str, Any],
    *,
    identity_only: bool = False,
) -> tuple[list[tuple[Observation3D, MapNode3D, AssociationPairScore]], list[Observation3D], list[str], list[AssociationPairScore], dict[str, list[int]]]:
    matches: list[tuple[Observation3D, MapNode3D, AssociationPairScore]] = []
    pair_scores: list[AssociationPairScore] = []
    matched_obs: set[str] = set()
    matched_nodes: set[str] = set()
    role_shapes: dict[str, list[int]] = {}
    for role in ("O", "C", "U"):
        obs_role = [o for o in observations if o.role == role]
        node_role = [n for n in nodes.values() if n.role == role]
        role_shapes[role] = [len(obs_role), len(node_role)]
        if not obs_role or not node_role:
            continue
        scores = np.zeros((len(obs_role), len(node_role)), dtype=np.float32)
        valid = np.zeros_like(scores, dtype=bool)
        pair_lookup: dict[tuple[int, int], AssociationPairScore] = {}
        for i, obs in enumerate(obs_role):
            for j, node in enumerate(node_role):
                pair = (
                    score_identity_only_pair(obs, node, frame, cfg)
                    if identity_only
                    else score_observation_node_pair(obs, node, frame, cfg)
                )
                scores[i, j] = pair.score_total
                valid[i, j] = pair.valid
                pair_scores.append(pair)
                pair_lookup[(i, j)] = pair
        for i, j in solve_non_exhaustive_assignment(scores, valid):
            pair = pair_lookup[(i, j)]
            pair.selected = True
            obs = obs_role[i]
            node = node_role[j]
            matches.append((obs, node, pair))
            matched_obs.add(obs.observation_id)
            matched_nodes.add(node.node_id)
    births = [o for o in observations if o.observation_id not in matched_obs]
    unmatched_existing = [node_id for node_id in sorted(nodes) if node_id not in matched_nodes]
    return matches, births, unmatched_existing, pair_scores, role_shapes


def build_frame_association_result(
    frame: FrameRecord,
    matches: list[tuple[Observation3D, MapNode3D, AssociationPairScore]],
    births: list[Observation3D],
    birth_node_ids: dict[str, str],
    unmatched_existing_nodes: list[str],
    pair_scores: list[AssociationPairScore],
    timing_ms: dict[str, float],
    role_shapes: dict[str, list[int]],
) -> FrameAssociationResult:
    det_to_node: dict[int, str] = {}
    match_records = []
    for obs, node, pair in matches:
        det_to_node[int(obs.det_id)] = node.node_id
        match_records.append(
            {
                "observation_id": obs.observation_id,
                "det_id": obs.det_id,
                "node_id": node.node_id,
                "role": obs.role,
                "score_total": pair.score_total,
                "association_mode": pair.association_mode,
                "geometry_fusion": pair.geometry_fusion,
            }
        )
    birth_records = []
    for obs in births:
        node_id = birth_node_ids[obs.observation_id]
        det_to_node[int(obs.det_id)] = node_id
        birth_records.append({"observation_id": obs.observation_id, "det_id": obs.det_id, "node_id": node_id, "role": obs.role})
    return FrameAssociationResult(
        frame_idx=frame.frame_idx,
        frame_key=frame.frame_key,
        matches=match_records,
        births=birth_records,
        unmatched_existing_nodes=unmatched_existing_nodes,
        pair_scores=pair_scores,
        det_to_node=det_to_node,
        timing_ms=timing_ms,
        role_matrix_shapes=role_shapes,
        reject_reason_counts=dict(Counter(p.reject_reason or "valid" for p in pair_scores)),
    )


def map_frame_edge_evidence(
    result2d: Frame2DResult,
    association: FrameAssociationResult,
    lifted_det_ids: set[int],
    reject_reasons: dict[int, str] | None = None,
) -> tuple[list[MappedFrameEdgeEvidence], list[dict[str, Any]]]:
    reject_reasons = reject_reasons or {}
    evidence: list[MappedFrameEdgeEvidence] = []
    dropped: list[dict[str, Any]] = []
    for rel in result2d.local_relations:
        p = int(rel.parent_det_id)
        c = int(rel.child_det_id)
        if p in association.det_to_node and c in association.det_to_node and p in lifted_det_ids and c in lifted_det_ids:
            evidence.append(
                MappedFrameEdgeEvidence(
                    frame_idx=result2d.frame_idx,
                    frame_key=result2d.frame_key,
                    edge_type=rel.edge_type,
                    parent_det_id=p,
                    child_det_id=c,
                    parent_node_id=association.det_to_node[p],
                    child_node_id=association.det_to_node[c],
                    parent_role=rel.parent_role,
                    child_role=rel.child_role,
                    pass_threshold=rel.pass_threshold,
                    selected_2d=rel.selected,
                    contain=rel.contain,
                    mask_contain=rel.mask_contain,
                    relation_text=rel.relation_text,
                    s2d_score=None,
                )
            )
        else:
            dropped.append(
                {
                    "frame_idx": result2d.frame_idx,
                    "frame_key": result2d.frame_key,
                    "edge_type": rel.edge_type,
                    "parent_det_id": p,
                    "child_det_id": c,
                    "reason": "endpoint_not_lifted_or_unmapped",
                    "parent_reject_reason": reject_reasons.get(p),
                    "child_reject_reason": reject_reasons.get(c),
                }
            )
    return evidence, dropped
