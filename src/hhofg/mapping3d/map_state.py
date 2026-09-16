from __future__ import annotations

import time
from collections import Counter
from typing import Any

from hhofg.core.types import FrameRecord

from .association import associate_observations_to_nodes, build_frame_association_result
from .map_node import create_map_node, update_map_node
from .node_consolidation import (
    conservative_consolidate,
    retire_cross_role_duplicates,
    update_covisibility,
)
from .types import FrameAssociationResult, MapNode3D, Observation3D


class MapState3D:
    def __init__(self, association_cfg: dict[str, Any], update_cfg: dict[str, Any], lifting_cfg: dict[str, Any] | None = None) -> None:
        self.cfg = association_cfg
        self.update_cfg = update_cfg
        self.lifting_cfg = lifting_cfg or {}
        self.nodes: dict[str, MapNode3D] = {}
        self.next_role_id = {"O": 1, "C": 1, "U": 1}
        self.peak_map_points = 0
        self.skipped_low_quality_births = 0
        self.covisibility: set[tuple[str, str]] = set()
        self.node_aliases: dict[str, str] = {}
        self.consolidation_events: list[dict[str, Any]] = []
        self.identity_attempts = 0
        self.identity_matches = 0
        self.identity_reject_reasons: Counter[str] = Counter()
        self.identity_by_role: Counter[str] = Counter()

    def _next_id(self, role: str) -> str:
        idx = self.next_role_id[role]
        self.next_role_id[role] += 1
        return f"{role}{idx:06d}"

    def create_node(self, obs: Observation3D) -> MapNode3D:
        node_id = self._next_id(obs.role)
        node = create_map_node(node_id, obs)
        self.nodes[node_id] = node
        obs.matched_node_id = node_id
        return node

    def update_node(self, node: MapNode3D, obs: Observation3D) -> None:
        update_map_node(
            node,
            obs,
            voxel_size=float(self.lifting_cfg.get("voxel_size_m", {}).get(obs.role, 0.004)),
            appearance_ema_alpha=float(self.update_cfg.get("appearance_ema_alpha", 0.3)),
            max_points=int(self.update_cfg.get("max_points_per_node", 20000)),
            update_cfg=self.update_cfg,
        )
        obs.matched_node_id = node.node_id

    def process_frame(self, observations: list[Observation3D], frame: FrameRecord) -> FrameAssociationResult:
        t0 = time.perf_counter()
        matches = []
        births: list[Observation3D] = []
        pair_scores = []
        role_shapes: dict[str, list[int]] = {}
        matched_nodes: set[str] = set()
        current_det_to_node: dict[int, str] = {}
        for role in ("O", "C", "U"):
            obs_role = [o for o in observations if o.role == role and o.geometry_status in {"reliable", "provisional"}]
            low_quality = [o for o in observations if o.role == role and o.geometry_status == "low_quality"]
            rejected = [o for o in observations if o.role == role and o.geometry_status not in {"reliable", "provisional", "low_quality"}]
            self.skipped_low_quality_births += len(low_quality) + len(rejected)
            assoc_cfg = dict(self.cfg)
            assoc_cfg["_current_det_to_node"] = current_det_to_node
            assoc_cfg["_current_nodes"] = self.nodes
            role_matches, role_births, _unmatched, role_scores, role_shape = associate_observations_to_nodes(obs_role, self.nodes, frame, assoc_cfg)
            role_shapes.update(role_shape)
            pair_scores.extend(role_scores)
            for obs, node, pair in role_matches:
                pair.association_mode = "full_geometry"
                pair.geometry_fusion = "accepted"
                self.update_node(self.nodes[node.node_id], obs)
                matches.append((obs, node, pair))
                matched_nodes.add(node.node_id)
                current_det_to_node[int(obs.det_id)] = node.node_id
            if bool(self.cfg.get("low_quality_identity_enable", False)) and low_quality:
                stable_role_nodes = {
                    nid: n
                    for nid, n in self.nodes.items()
                    if n.role == role and (n.state == "stable" or bool(getattr(n, "graph_eligible", False)))
                }
                identity_cfg = dict(assoc_cfg)
                identity_cfg["min_score"] = float(self.cfg.get("low_quality_identity_min_score", self.cfg.get("min_score", 0.45)))
                identity_cfg["require_positive_iou"] = bool(self.cfg.get("low_quality_identity_require_positive_iou", True))
                identity_matches, _identity_births, _identity_unmatched, identity_scores, identity_shape = associate_observations_to_nodes(
                    low_quality,
                    stable_role_nodes,
                    frame,
                    identity_cfg,
                    identity_only=True,
                )
                self.identity_attempts += len(identity_scores)
                self.identity_reject_reasons.update(
                    pair.reject_reason or "valid" for pair in identity_scores
                )
                role_shapes[role] = [
                    int(role_shapes.get(role, [0, 0])[0]) + int(identity_shape.get(role, [0, 0])[0]),
                    int(role_shapes.get(role, [0, 0])[1]),
                ]
                pair_scores.extend(identity_scores)
                for obs, node, pair in identity_matches:
                    if obs.observation_id in {m[0].observation_id for m in matches}:
                        continue
                    pair.association_mode = "identity_only"
                    pair.geometry_fusion = "not_attempted_identity_only"
                    matches.append((obs, node, pair))
                    matched_nodes.add(node.node_id)
                    current_det_to_node[int(obs.det_id)] = node.node_id
                    self.identity_matches += 1
                    self.identity_by_role[role] += 1
                    if obs.frame_idx not in node.identity_support_frames:
                        node.identity_support_frames.append(obs.frame_idx)
            for obs in sorted(role_births, key=lambda o: o.det_id):
                node = self.create_node(obs)
                births.append(obs)
                current_det_to_node[int(obs.det_id)] = node.node_id
                matched_nodes.add(node.node_id)
            role_shapes.setdefault(role, [len(obs_role), sum(1 for n in self.nodes.values() if n.role == role)])
        t_assoc = (time.perf_counter() - t0) * 1000.0
        birth_node_ids: dict[str, str] = {}
        for obs in births:
            if obs.matched_node_id:
                birth_node_ids[obs.observation_id] = obs.matched_node_id
        unmatched_existing = [node_id for node_id in sorted(self.nodes) if node_id not in matched_nodes]
        total_points = sum(n.points_world.shape[0] for n in self.nodes.values())
        update_covisibility(self.covisibility, list(current_det_to_node.values()))
        self.peak_map_points = max(self.peak_map_points, int(total_points))
        return build_frame_association_result(
            frame,
            matches,
            births,
            birth_node_ids,
            unmatched_existing,
            pair_scores,
            {"association": t_assoc},
            role_shapes,
        )

    def nodes_by_role(self) -> dict[str, int]:
        return {role: int(sum(1 for n in self.nodes.values() if n.role == role)) for role in ("O", "C", "U")}

    def consolidate_non_covisible_tracks(
        self, cfg: dict[str, Any]
    ) -> list[dict[str, Any]]:
        # Old checkpoints predate these audit fields; initialize them lazily so
        # a resumed run remains loadable.
        if not hasattr(self, "node_aliases"):
            self.node_aliases = {}
        if not hasattr(self, "consolidation_events"):
            self.consolidation_events = []
        events = conservative_consolidate(
            self.nodes,
            self.covisibility,
            cfg,
            aliases=self.node_aliases,
        )
        events.extend(
            retire_cross_role_duplicates(self.nodes, self.covisibility, cfg)
        )
        records = [
            {
                "event_type": event.event_type,
                "source_node_id": event.source_node_id,
                "target_node_id": event.target_node_id,
                "reason": event.reason,
                "metrics": dict(event.metrics),
            }
            for event in events
        ]
        def canonical(node_id: str | None) -> str | None:
            current = node_id
            visited: set[str] = set()
            while current and current in self.node_aliases and current not in visited:
                visited.add(current)
                current = self.node_aliases[current]
            return current

        for node in self.nodes.values():
            node.stable_object_parent_id = canonical(node.stable_object_parent_id)
            node.stable_carrier_parent_id = canonical(node.stable_carrier_parent_id)
            node.stable_parent_id = canonical(node.stable_parent_id)
        self.consolidation_events.extend(records)
        return records
