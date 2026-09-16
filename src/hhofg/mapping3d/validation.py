from __future__ import annotations

import numpy as np

from .types import FrameAssociationResult, MapNode3D, Observation3D


def validate_observations(observations: list[Observation3D]) -> None:
    for obs in observations:
        for name in ("points_camera", "points_world", "colors_rgb", "centroid_world", "bbox_min_world", "bbox_max_world"):
            arr = getattr(obs, name)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{obs.observation_id}: {name} contains NaN/Inf")
        if obs.points_camera.shape[0] != obs.points_world.shape[0]:
            raise ValueError(f"{obs.observation_id}: camera/world point count mismatch")
        if obs.points_camera.shape[0] == 0:
            raise ValueError(f"{obs.observation_id}: empty point cloud")
        if not np.all(obs.points_camera[:, 2] > 0):
            raise ValueError(f"{obs.observation_id}: nonpositive camera z")


def validate_nodes(nodes: dict[str, MapNode3D]) -> None:
    for node in nodes.values():
        for name in ("points_world", "colors_rgb", "centroid_world", "bbox_min_world", "bbox_max_world"):
            if not np.all(np.isfinite(getattr(node, name))):
                raise ValueError(f"{node.node_id}: {name} contains NaN/Inf")
        if node.points_world.shape[0] == 0:
            raise ValueError(f"{node.node_id}: empty point cloud")


def validate_association(result: FrameAssociationResult, observations: list[Observation3D]) -> None:
    by_id = {o.observation_id: o for o in observations}
    eligible = {o.observation_id for o in observations if o.geometry_status in {"reliable", "provisional"}}
    obs_ids = [m["observation_id"] for m in result.matches] + [b["observation_id"] for b in result.births]
    if len(obs_ids) != len(set(obs_ids)):
        raise ValueError(f"{result.frame_key}: duplicate observation assignment")
    assigned = set(obs_ids)
    identity_low_quality = {
        str(match["observation_id"])
        for match in result.matches
        if str(match.get("association_mode")) == "identity_only"
        and by_id.get(str(match["observation_id"])) is not None
        and by_id[str(match["observation_id"])].geometry_status == "low_quality"
    }
    low_quality_births = [
        birth["observation_id"]
        for birth in result.births
        if by_id.get(str(birth["observation_id"])) is not None
        and by_id[str(birth["observation_id"])].geometry_status == "low_quality"
    ]
    if low_quality_births:
        raise ValueError(f"{result.frame_key}: low-quality observations must not birth nodes: {low_quality_births}")
    if assigned != eligible | identity_low_quality:
        missing = sorted(eligible - assigned)
        extra = sorted(assigned - eligible - identity_low_quality)
        raise ValueError(f"{result.frame_key}: eligible observation assignment mismatch missing={missing} extra={extra}")
    node_ids = [m["node_id"] for m in result.matches]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError(f"{result.frame_key}: same node matched more than once in frame")
    for p in result.pair_scores:
        if p.selected:
            if not p.valid or p.distance_m >= p.tau_distance_m:
                raise ValueError(f"{result.frame_key}: invalid selected pair {p.observation_id}->{p.node_id}")
