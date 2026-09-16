from __future__ import annotations

import numpy as np

from .features import l2_normalize, voxel_downsample_points_colors
from .geometry_fusion import refresh_geometry, should_accept_geometry_fusion, spatially_uniform_sample
from .node_lifecycle import update_node_lifecycle
from .types import MapNode3D, Observation3D


def create_map_node(node_id: str, obs: Observation3D) -> MapNode3D:
    initial_state = "provisional" if obs.geometry_status in {"reliable", "provisional"} else "rejected"
    return MapNode3D(
        node_id=node_id,
        role=obs.role,
        label_votes={obs.label: float(obs.score)},
        top_label=obs.label,
        points_world=obs.points_world.copy(),
        colors_rgb=obs.colors_rgb.copy(),
        centroid_world=obs.centroid_world.copy(),
        bbox_min_world=obs.bbox_min_world.copy(),
        bbox_max_world=obs.bbox_max_world.copy(),
        bbox_diag_world=float(obs.bbox_diag_world),
        appearance_hist=obs.appearance_hist.copy(),
        semantic_feature=obs.semantic_feature.copy(),
        semantic_weight_sum=float(obs.score),
        obs_count=1,
        first_seen_frame=obs.frame_idx,
        last_seen_frame=obs.frame_idx,
        observed_frames=[obs.frame_idx],
        observation_ids=[obs.observation_id],
        last_box_xyxy=list(obs.box_xyxy),
        last_frame_key=obs.frame_key,
        association_points_world=obs.association_points_world.copy(),
        association_bbox_min_world=obs.association_bbox_min_world.copy(),
        association_bbox_max_world=obs.association_bbox_max_world.copy(),
        association_bbox_diag_world=float(obs.association_bbox_diag_world),
        state=initial_state,
        anchor_observation_id=obs.observation_id,
        anchor_quality=float(obs.geometry_quality),
        anchor_points_world=obs.points_world.copy(),
        core_points_world=obs.points_world.copy(),
        geometry_quality_ema=float(obs.geometry_quality),
        num_geometry_accepts=1 if initial_state != "rejected" else 0,
        num_geometry_rejects=0,
        support_frames=[obs.frame_idx],
        real_depth_support_frames=[] if obs.geometry_is_imputed else [obs.frame_idx],
        imputed_support_frames=[obs.frame_idx] if obs.geometry_is_imputed else [],
        recent_centroids=[[float(v) for v in obs.centroid_world.tolist()]],
        recent_bboxes=[{"min": [float(v) for v in obs.bbox_min_world.tolist()], "max": [float(v) for v in obs.bbox_max_world.tolist()]}],
        recent_quality=[float(obs.geometry_quality)],
        anchor_touches_border=bool(obs.box_touches_border),
        cross_role_duplicate_witness_frames=(
            [obs.frame_idx] if obs.cross_role_duplicate_det_ids else []
        ),
    )


def update_map_node(
    node: MapNode3D,
    obs: Observation3D,
    *,
    voxel_size: float,
    appearance_ema_alpha: float,
    max_points: int,
    update_cfg: dict | None = None,
) -> None:
    fusion_cfg = {
        "fusion_min_observation_quality": 0.35,
        "fusion_max_centroid_residual_m": 0.12,
        "fusion_max_bbox_expansion_ratio": 2.5,
        "fusion_bbox_expansion_after_obs": 2,
    }
    if update_cfg:
        fusion_cfg.update(update_cfg)
    accept_geometry, reject_reason = should_accept_geometry_fusion(node, obs, fusion_cfg)
    if accept_geometry:
        points = np.concatenate([node.points_world, obs.points_world], axis=0)
        colors = np.concatenate([node.colors_rgb, obs.colors_rgb], axis=0)
        points, colors = voxel_downsample_points_colors(points, colors, float(voxel_size), max_points=int(max_points))
        points, colors = spatially_uniform_sample(points, colors, max_points=int(max_points))
        node.points_world = points
        node.colors_rgb = colors
        node.core_points_world = points.copy()
        refresh_geometry(node)
        association_points = np.concatenate(
            [
                np.asarray(
                    getattr(node, "association_points_world", node.points_world),
                    dtype=np.float32,
                ),
                np.asarray(obs.association_points_world, dtype=np.float32),
            ],
            axis=0,
        )
        association_points, _ = voxel_downsample_points_colors(
            association_points,
            np.zeros_like(association_points, dtype=np.float32),
            float(update_cfg.get("association_voxel_size_m", voxel_size) if update_cfg else voxel_size),
            max_points=int(
                (update_cfg or {}).get("association_max_points_per_node", 4096)
            ),
        )
        association_points, _ = spatially_uniform_sample(
            association_points,
            np.zeros_like(association_points, dtype=np.float32),
            max_points=int(
                (update_cfg or {}).get("association_max_points_per_node", 4096)
            ),
        )
        node.association_points_world = association_points
        node.association_bbox_min_world = np.percentile(
            association_points, 2.0, axis=0
        ).astype(np.float32)
        node.association_bbox_max_world = np.percentile(
            association_points, 98.0, axis=0
        ).astype(np.float32)
        node.association_bbox_diag_world = float(
            np.linalg.norm(
                np.maximum(
                    node.association_bbox_max_world
                    - node.association_bbox_min_world,
                    0.0,
                )
            )
        )
        node.num_geometry_accepts += 1
        node.last_geometry_reject_reason = None
    else:
        node.num_geometry_rejects += 1
        node.last_geometry_reject_reason = reject_reason
    alpha = float(appearance_ema_alpha)
    node.appearance_hist = l2_normalize((1.0 - alpha) * node.appearance_hist + alpha * obs.appearance_hist)
    sem_sum = node.semantic_feature * float(node.semantic_weight_sum) + obs.semantic_feature * float(obs.score)
    node.semantic_weight_sum = float(node.semantic_weight_sum) + float(obs.score)
    node.semantic_feature = l2_normalize(sem_sum)
    node.label_votes[obs.label] = node.label_votes.get(obs.label, 0.0) + float(obs.score)
    node.top_label = sorted(node.label_votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    node.obs_count += 1
    node.last_seen_frame = obs.frame_idx
    if obs.frame_idx not in node.observed_frames:
        node.observed_frames.append(obs.frame_idx)
    node.observation_ids.append(obs.observation_id)
    node.last_box_xyxy = list(obs.box_xyxy)
    node.last_frame_key = obs.frame_key
    if not hasattr(node, "cross_role_duplicate_witness_frames"):
        node.cross_role_duplicate_witness_frames = []
    if (
        obs.cross_role_duplicate_det_ids
        and obs.frame_idx not in node.cross_role_duplicate_witness_frames
    ):
        node.cross_role_duplicate_witness_frames.append(obs.frame_idx)
    if float(obs.geometry_quality) > float(node.anchor_quality) and not obs.geometry_is_imputed and obs.geometry_status in {"reliable", "provisional"}:
        node.anchor_observation_id = obs.observation_id
        node.anchor_quality = float(obs.geometry_quality)
        node.anchor_points_world = obs.points_world.copy()
        node.anchor_touches_border = bool(obs.box_touches_border)
    update_node_lifecycle(node, obs, update_cfg or {})
