from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

VALID_ROLES_3D = {"O", "C", "U"}
VALID_GEOMETRY_STATUS = {"reliable", "provisional", "low_quality", "rejected"}
VALID_NODE_STATES = {"provisional", "stable", "rejected"}


def _array3(name: str, value: np.ndarray, *, allow_empty: bool = False) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}")
    if not allow_empty and arr.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def _vec(name: str, value: np.ndarray, dim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if dim is not None and arr.shape[0] != dim:
        raise ValueError(f"{name} must have length {dim}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


@dataclass
class Observation3D:
    observation_id: str
    frame_idx: int
    source_index: int
    frame_key: str
    det_id: int
    label: str
    role: str
    score: float
    box_xyxy: list[float]
    mask_area: int
    box_touches_border: bool
    points_camera: np.ndarray
    points_world: np.ndarray
    colors_rgb: np.ndarray
    centroid_camera: np.ndarray
    centroid_world: np.ndarray
    bbox_min_camera: np.ndarray
    bbox_max_camera: np.ndarray
    bbox_min_world: np.ndarray
    bbox_max_world: np.ndarray
    bbox_diag_world: float
    appearance_hist: np.ndarray
    semantic_feature: np.ndarray
    num_raw_depth_points: int
    num_depth_filtered_points: int
    num_cluster_points: int
    dbscan_applied: bool
    depth_median: float
    depth_min: float
    depth_max: float
    allowed_parent_labels: set[str]
    local_parent_det_ids: list[int]
    # Geometry used only for identity association.  It retains the spatial
    # extent of the selected depth mode before conservative DBSCAN filtering;
    # points_world remains the geometry that is safe to fuse into the map.
    association_points_world: np.ndarray | None = None
    association_bbox_min_world: np.ndarray | None = None
    association_bbox_max_world: np.ndarray | None = None
    association_bbox_diag_world: float | None = None
    matched_node_id: str | None = None
    lifting_warnings: list[str] = field(default_factory=list)
    depth_seed_m: float | None = None
    depth_mad_m: float | None = None
    depth_iqr_m: float | None = None
    depth_cluster_count: int = 0
    chosen_cluster_size: int = 0
    chosen_cluster_ratio: float = 0.0
    chosen_cluster_depth_median: float | None = None
    parent_depth_used: bool = False
    parent_depth_median: float | None = None
    parent_depth_mad: float | None = None
    parent_depth_residual: float | None = None
    extraction_method: str = "legacy_iqr_dbscan"
    geometry_quality: float = 1.0
    geometry_status: str = "reliable"
    geometry_is_imputed: bool = False
    mask_quality: float = 1.0
    visible_pixel_count: int = 0
    effective_voxel_size_m: float = 0.0
    mask_area_ratio: float = 0.0
    tiny_far: bool = False
    depth_modes: list[dict[str, Any]] = field(default_factory=list)
    # A same-frame C detection whose 2D extent is almost identical to this U
    # detection is a cross-role duplicate witness.  Keep the witness on the U
    # observation so the map can retire an older C-only track at the final
    # consolidation stage without naming any particular semantic class.
    cross_role_duplicate_det_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.frame_idx = int(self.frame_idx)
        self.source_index = int(self.source_index)
        self.det_id = int(self.det_id)
        self.role = str(self.role)
        if self.role not in VALID_ROLES_3D:
            raise ValueError("role must be O/C/U")
        expected_id = f"{self.frame_key}:{self.det_id}"
        if self.observation_id != expected_id:
            raise ValueError(f"observation_id must be {expected_id}")
        self.score = float(self.score)
        self.box_xyxy = [float(v) for v in self.box_xyxy]
        self.mask_area = int(self.mask_area)
        self.box_touches_border = bool(self.box_touches_border)
        self.points_camera = _array3("points_camera", self.points_camera)
        self.points_world = _array3("points_world", self.points_world)
        self.colors_rgb = _array3("colors_rgb", self.colors_rgb)
        if self.colors_rgb.shape[0] != self.points_world.shape[0]:
            raise ValueError("colors_rgb must align with points")
        self.centroid_camera = _vec("centroid_camera", self.centroid_camera, 3)
        self.centroid_world = _vec("centroid_world", self.centroid_world, 3)
        self.bbox_min_camera = _vec("bbox_min_camera", self.bbox_min_camera, 3)
        self.bbox_max_camera = _vec("bbox_max_camera", self.bbox_max_camera, 3)
        self.bbox_min_world = _vec("bbox_min_world", self.bbox_min_world, 3)
        self.bbox_max_world = _vec("bbox_max_world", self.bbox_max_world, 3)
        self.bbox_diag_world = float(self.bbox_diag_world)
        self.appearance_hist = _vec("appearance_hist", self.appearance_hist)
        self.semantic_feature = _vec("semantic_feature", self.semantic_feature)
        self.allowed_parent_labels = {str(v) for v in self.allowed_parent_labels}
        self.local_parent_det_ids = [int(v) for v in self.local_parent_det_ids]
        self.cross_role_duplicate_det_ids = [
            int(v) for v in self.cross_role_duplicate_det_ids
        ]
        if self.association_points_world is None:
            self.association_points_world = self.points_world.copy()
        else:
            self.association_points_world = _array3(
                "association_points_world", self.association_points_world
            )
        if self.association_bbox_min_world is None:
            self.association_bbox_min_world = self.bbox_min_world.copy()
        else:
            self.association_bbox_min_world = _vec(
                "association_bbox_min_world", self.association_bbox_min_world, 3
            )
        if self.association_bbox_max_world is None:
            self.association_bbox_max_world = self.bbox_max_world.copy()
        else:
            self.association_bbox_max_world = _vec(
                "association_bbox_max_world", self.association_bbox_max_world, 3
            )
        if self.association_bbox_diag_world is None:
            self.association_bbox_diag_world = float(
                np.linalg.norm(
                    np.maximum(
                        self.association_bbox_max_world
                        - self.association_bbox_min_world,
                        0.0,
                    )
                )
            )
        else:
            self.association_bbox_diag_world = float(
                self.association_bbox_diag_world
            )
        self.geometry_quality = float(np.clip(float(self.geometry_quality), 0.0, 1.0))
        self.mask_quality = float(np.clip(float(self.mask_quality), 0.0, 1.0))
        self.geometry_status = str(self.geometry_status)
        if self.geometry_status not in VALID_GEOMETRY_STATUS:
            raise ValueError(f"geometry_status must be one of {sorted(VALID_GEOMETRY_STATUS)}")
        self.visible_pixel_count = int(self.visible_pixel_count)
        self.chosen_cluster_size = int(self.chosen_cluster_size)
        self.depth_cluster_count = int(self.depth_cluster_count)
        self.chosen_cluster_ratio = float(self.chosen_cluster_ratio)
        self.effective_voxel_size_m = float(self.effective_voxel_size_m)
        self.mask_area_ratio = float(self.mask_area_ratio)
        self.tiny_far = bool(self.tiny_far)
        self.parent_depth_used = bool(self.parent_depth_used)
        self.geometry_is_imputed = bool(self.geometry_is_imputed)


@dataclass
class MapNode3D:
    node_id: str
    role: str
    label_votes: dict[str, float]
    top_label: str
    points_world: np.ndarray
    colors_rgb: np.ndarray
    centroid_world: np.ndarray
    bbox_min_world: np.ndarray
    bbox_max_world: np.ndarray
    bbox_diag_world: float
    appearance_hist: np.ndarray
    semantic_feature: np.ndarray
    semantic_weight_sum: float
    obs_count: int
    first_seen_frame: int
    last_seen_frame: int
    observed_frames: list[int]
    observation_ids: list[str]
    last_box_xyxy: list[float] | None
    last_frame_key: str | None
    association_points_world: np.ndarray | None = None
    association_bbox_min_world: np.ndarray | None = None
    association_bbox_max_world: np.ndarray | None = None
    association_bbox_diag_world: float | None = None
    state: str = "stable"
    anchor_observation_id: str | None = None
    anchor_quality: float = 1.0
    anchor_points_world: np.ndarray | None = None
    core_points_world: np.ndarray | None = None
    geometry_quality_ema: float = 1.0
    num_geometry_accepts: int = 1
    num_geometry_rejects: int = 0
    last_geometry_reject_reason: str | None = None
    support_frames: list[int] = field(default_factory=list)
    real_depth_support_frames: list[int] = field(default_factory=list)
    imputed_support_frames: list[int] = field(default_factory=list)
    recent_centroids: list[list[float]] = field(default_factory=list)
    recent_bboxes: list[dict[str, list[float]]] = field(default_factory=list)
    recent_quality: list[float] = field(default_factory=list)
    object_parent_context_history: list[dict[str, Any]] = field(default_factory=list)
    carrier_parent_context_history: list[dict[str, Any]] = field(default_factory=list)
    stable_object_parent_id: str | None = None
    stable_carrier_parent_id: str | None = None
    identity_support_frames: list[int] = field(default_factory=list)
    anchor_touches_border: bool = False
    graph_eligible: bool = False
    graph_node_confidence: float = 0.0
    graph_eligibility_reason: str = "not_evaluated"
    # Deprecated v2 fields kept for loading old checkpoints and JSON.
    parent_context_history: list[dict[str, Any]] = field(default_factory=list)
    stable_parent_id: str | None = None
    cross_role_duplicate_witness_frames: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES_3D:
            raise ValueError("role must be O/C/U")
        self.points_world = _array3("points_world", self.points_world)
        self.colors_rgb = _array3("colors_rgb", self.colors_rgb)
        self.centroid_world = _vec("centroid_world", self.centroid_world, 3)
        self.bbox_min_world = _vec("bbox_min_world", self.bbox_min_world, 3)
        self.bbox_max_world = _vec("bbox_max_world", self.bbox_max_world, 3)
        self.appearance_hist = _vec("appearance_hist", self.appearance_hist)
        self.semantic_feature = _vec("semantic_feature", self.semantic_feature)
        if self.association_points_world is None:
            self.association_points_world = self.points_world.copy()
        else:
            self.association_points_world = _array3(
                "association_points_world", self.association_points_world
            )
        if self.association_bbox_min_world is None:
            self.association_bbox_min_world = self.bbox_min_world.copy()
        else:
            self.association_bbox_min_world = _vec(
                "association_bbox_min_world", self.association_bbox_min_world, 3
            )
        if self.association_bbox_max_world is None:
            self.association_bbox_max_world = self.bbox_max_world.copy()
        else:
            self.association_bbox_max_world = _vec(
                "association_bbox_max_world", self.association_bbox_max_world, 3
            )
        if self.association_bbox_diag_world is None:
            self.association_bbox_diag_world = float(
                np.linalg.norm(
                    np.maximum(
                        self.association_bbox_max_world
                        - self.association_bbox_min_world,
                        0.0,
                    )
                )
            )
        else:
            self.association_bbox_diag_world = float(
                self.association_bbox_diag_world
            )
        self.label_votes = {str(k): float(v) for k, v in self.label_votes.items()}
        self.semantic_weight_sum = float(self.semantic_weight_sum)
        self.state = str(self.state)
        if self.state not in VALID_NODE_STATES:
            raise ValueError(f"state must be one of {sorted(VALID_NODE_STATES)}")
        self.anchor_quality = float(self.anchor_quality)
        self.geometry_quality_ema = float(self.geometry_quality_ema)
        self.num_geometry_accepts = int(self.num_geometry_accepts)
        self.num_geometry_rejects = int(self.num_geometry_rejects)
        if self.anchor_points_world is None:
            self.anchor_points_world = self.points_world.copy()
        else:
            self.anchor_points_world = _array3("anchor_points_world", self.anchor_points_world)
        if self.core_points_world is None:
            self.core_points_world = self.points_world.copy()
        else:
            self.core_points_world = _array3("core_points_world", self.core_points_world)
        if not self.support_frames:
            self.support_frames = list(self.observed_frames)
        self.real_depth_support_frames = [int(v) for v in self.real_depth_support_frames]
        self.imputed_support_frames = [int(v) for v in self.imputed_support_frames]
        self.identity_support_frames = [int(v) for v in self.identity_support_frames]
        self.cross_role_duplicate_witness_frames = sorted(
            {int(v) for v in self.cross_role_duplicate_witness_frames}
        )
        self.anchor_touches_border = bool(self.anchor_touches_border)
        self.graph_eligible = bool(self.graph_eligible)
        self.graph_node_confidence = float(self.graph_node_confidence)


@dataclass
class AssociationPairScore:
    observation_id: str
    node_id: str
    role: str
    distance_m: float
    tau_distance_m: float
    projected_box_xyxy: list[float] | None
    score_iou: float
    score_geo: float
    score_app: float
    score_sem: float
    score_total: float
    valid: bool
    reject_reason: str
    selected: bool = False
    association_mode: str = "full_geometry"
    geometry_fusion: str = "accepted"


@dataclass
class FrameAssociationResult:
    frame_idx: int
    frame_key: str
    matches: list[dict[str, Any]]
    births: list[dict[str, Any]]
    unmatched_existing_nodes: list[str]
    pair_scores: list[AssociationPairScore]
    det_to_node: dict[int, str]
    timing_ms: dict[str, float]
    role_matrix_shapes: dict[str, list[int]] = field(default_factory=dict)
    reject_reason_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class MappedFrameEdgeEvidence:
    frame_idx: int
    frame_key: str
    edge_type: str
    parent_det_id: int
    child_det_id: int
    parent_node_id: str
    child_node_id: str
    parent_role: str
    child_role: str
    pass_threshold: bool
    selected_2d: bool
    contain: float | None
    mask_contain: float | None
    relation_text: str
    s2d_score: float | None = None
    eligible_for_temporal_optimization: bool = False
    usage: str = ""
    candidate_hash: str = ""
    candidate_sources: list[str] = field(default_factory=list)
    semantic_sources: list[str] = field(default_factory=list)
    semantic_chains: list[dict[str, Any]] = field(default_factory=list)
    parent_label: str = ""
    child_label: str = ""
    semantic_only: bool = False
    evidence_score_source: str = ""

    def __post_init__(self) -> None:
        self.eligible_for_temporal_optimization = self.edge_type in {"O-C", "O-U"}
        if self.edge_type == "C-U":
            self.eligible_for_temporal_optimization = False
            self.usage = "hierarchy_prior_debug_only"


@dataclass
class FrameEdgeCandidate:
    frame_idx: int
    frame_key: str
    edge_type: str
    parent_det_id: int
    child_det_id: int
    parent_label: str
    child_label: str
    parent_score: float
    child_score: float
    parent_role: str
    child_role: str
    selected_2d: bool
    pass_threshold: bool
    contain: float | None
    mask_contain: float | None
    relation_text: str
    parent_node_id: str | None = None
    child_node_id: str | None = None
    parent_node_state: str | None = None
    child_node_state: str | None = None
    parent_lifted: bool = False
    child_lifted: bool = False
    image_sha: str = ""
    mask_sha: str = ""
    candidate_source: str = "existing_local_relation"
    sdet: float | None = None
    gcamc: float | None = None
    gcamc_raw: float | None = None
    gcamc_support: float | None = None
    gcamc_used: float | None = None
    support_mask_used: bool = False
    child_center_inside_parent_box: bool = False
    pass_prefilter: bool | None = None
    prefilter_fail_reason: str = ""
    s2d_score: float | None = None
    s2d_available: bool = False
    s2d_error: str = ""
    s2d_cache_key: str = ""
    s2d_cache_hit: bool = False
    s2d_model_id: str = ""
    s2d_raw_response: str = ""
    candidate_hash: str = ""
    candidate_sources: list[str] = field(default_factory=list)
    semantic_sources: list[str] = field(default_factory=list)
    instance_relation_texts: list[str] = field(default_factory=list)
    semantic_chains: list[dict[str, Any]] = field(default_factory=list)
    from_raw_relation: bool = False
    from_final_relation: bool = False
    semantic_only: bool = False
    ownership_geometry: dict[str, Any] = field(default_factory=dict)
    unresolved_endpoint_reason: str = ""

    def __post_init__(self) -> None:
        self.frame_idx = int(self.frame_idx)
        self.parent_det_id = int(self.parent_det_id)
        self.child_det_id = int(self.child_det_id)
        if self.edge_type not in {"O-C", "C-U", "O-U"}:
            raise ValueError("edge_type must be O-C, C-U, or O-U")
        if self.parent_role not in VALID_ROLES_3D or self.child_role not in VALID_ROLES_3D:
            raise ValueError("candidate roles must be O/C/U")
        self.parent_score = float(self.parent_score)
        self.child_score = float(self.child_score)
        self.selected_2d = bool(self.selected_2d)
        self.pass_threshold = bool(self.pass_threshold)
        self.parent_lifted = bool(self.parent_lifted)
        self.child_lifted = bool(self.child_lifted)
        self.s2d_available = bool(self.s2d_available)
        self.s2d_cache_hit = bool(self.s2d_cache_hit)
        self.support_mask_used = bool(self.support_mask_used)
        self.child_center_inside_parent_box = bool(
            self.child_center_inside_parent_box
        )
        self.semantic_only = bool(self.semantic_only)
        if not self.candidate_sources:
            self.candidate_sources = [self.candidate_source]
        self.candidate_sources = sorted({str(v) for v in self.candidate_sources if str(v)})
        self.semantic_sources = sorted({str(v) for v in self.semantic_sources if str(v)})
        if not self.candidate_hash:
            import hashlib
            import json

            payload = [
                self.frame_key,
                self.edge_type,
                self.parent_det_id,
                self.child_det_id,
            ]
            self.candidate_hash = hashlib.sha256(
                json.dumps(payload, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
