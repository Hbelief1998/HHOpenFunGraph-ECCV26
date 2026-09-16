from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VALID_ROLES = {"O", "C", "U"}
SCHEMA_VERSION = 3


def _ensure_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")


@dataclass
class CameraModel:
    K: np.ndarray
    width: int
    height: int
    depth_scale: float
    camera_axis: str = "Up"

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float32)
        if self.K.shape != (3, 3):
            raise ValueError(f"K must have shape (3, 3), got {self.K.shape}")
        _ensure_finite("K", self.K)
        if float(self.K[0, 0]) <= 0 or float(self.K[1, 1]) <= 0:
            raise ValueError("K fx/fy must be positive")
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("camera width/height must be positive")
        if float(self.depth_scale) <= 0:
            raise ValueError("depth_scale must be positive")
        self.width = int(self.width)
        self.height = int(self.height)
        self.depth_scale = float(self.depth_scale)


@dataclass
class FrameRecord:
    frame_idx: int
    frame_key: str
    rgb_path: Path
    rgb: np.ndarray
    depth_path: Path | None = None
    depth_m: np.ndarray | None = None
    camera: CameraModel | None = None
    T_c2w: np.ndarray | None = None
    source_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frame_idx = int(self.frame_idx)
        self.source_index = int(self.source_index)
        self.rgb_path = Path(self.rgb_path)
        self.rgb = np.asarray(self.rgb)
        if self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("rgb must be uint8 RGB with shape (H, W, 3)")
        if self.depth_path is not None:
            self.depth_path = Path(self.depth_path)
        if self.depth_m is not None:
            self.depth_m = np.asarray(self.depth_m, dtype=np.float32)
            if self.depth_m.shape != self.rgb.shape[:2]:
                raise ValueError("depth_m shape must match rgb height/width")
            _ensure_finite("depth_m", self.depth_m)
        if self.T_c2w is not None:
            self.T_c2w = np.asarray(self.T_c2w, dtype=np.float32)
            if self.T_c2w.shape != (4, 4):
                raise ValueError("T_c2w must have shape (4, 4)")
            _ensure_finite("T_c2w", self.T_c2w)


@dataclass
class Detection2D:
    det_id: int
    label: str
    role: str
    score: float
    box_xyxy: list[float]
    box_xyxy_inference: list[float] | None = None

    def __post_init__(self) -> None:
        self.det_id = int(self.det_id)
        self.label = str(self.label)
        self.role = str(self.role)
        if self.role not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}")
        self.score = float(self.score)
        if not np.isfinite(self.score):
            raise ValueError("score must be finite")
        if len(self.box_xyxy) != 4:
            raise ValueError("box_xyxy must have four values")
        self.box_xyxy = [float(v) for v in self.box_xyxy]
        if not np.all(np.isfinite(np.asarray(self.box_xyxy, dtype=np.float32))):
            raise ValueError("box_xyxy contains NaN or Inf")
        if self.box_xyxy_inference is not None:
            if len(self.box_xyxy_inference) != 4:
                raise ValueError("box_xyxy_inference must have four values")
            self.box_xyxy_inference = [float(v) for v in self.box_xyxy_inference]
            if not np.all(np.isfinite(np.asarray(self.box_xyxy_inference, dtype=np.float32))):
                raise ValueError("box_xyxy_inference contains NaN or Inf")


@dataclass
class LocalRelation2D:
    edge_type: str
    parent_det_id: int
    child_det_id: int
    parent_role: str
    child_role: str
    contain: float | None
    mask_contain: float | None
    pass_threshold: bool
    selected: bool
    relation_text: str = ""
    source_stage: str = ""

    def __post_init__(self) -> None:
        if self.edge_type not in {"O-C", "C-U", "O-U"}:
            raise ValueError("edge_type must be O-C, C-U, or O-U")
        if self.parent_role not in VALID_ROLES or self.child_role not in VALID_ROLES:
            raise ValueError("relation roles must be O/C/U")
        self.parent_det_id = int(self.parent_det_id)
        self.child_det_id = int(self.child_det_id)
        self.pass_threshold = bool(self.pass_threshold)
        self.selected = bool(self.selected)


@dataclass
class Frame2DResult:
    schema_version: int
    frame_idx: int
    frame_key: str
    image_path: str
    image_height: int
    image_width: int
    scene_type: str
    scene_locked: bool
    tags_en: list[str]
    tags_zh: list[str]
    frame_result: dict[str, Any]
    graph_frame_result: dict[str, Any]
    detections: list[Detection2D]
    local_relations: list[LocalRelation2D]
    label_to_role: dict[str, str]
    u_to_allowed_parents: dict[str, list[str]]
    c_to_allowed_parents: dict[str, list[str]]
    local_rel_debug: dict[str, Any]
    remote_rel_2d: dict[str, Any]
    timing_ms: dict[str, float]
    warnings: list[str]
    inference_image_height: int | None = None
    inference_image_width: int | None = None
    image_transform: dict[str, Any] | None = None
    coordinate_space: str = "raw"
    local_relations_final: list[LocalRelation2D] | None = None
    local_relation_candidates_raw: list[LocalRelation2D] | None = None
    relation_diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.schema_version) not in {1, 2, SCHEMA_VERSION}:
            raise ValueError(f"schema_version must be 1, 2, or {SCHEMA_VERSION}")
        self.frame_idx = int(self.frame_idx)
        self.image_height = int(self.image_height)
        self.image_width = int(self.image_width)
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image dimensions must be positive")
        if self.inference_image_height is None:
            self.inference_image_height = self.image_height
        if self.inference_image_width is None:
            self.inference_image_width = self.image_width
        self.inference_image_height = int(self.inference_image_height)
        self.inference_image_width = int(self.inference_image_width)
        if self.inference_image_height <= 0 or self.inference_image_width <= 0:
            raise ValueError("inference image dimensions must be positive")
        if self.image_transform is None:
            self.image_transform = {}
        self.scene_locked = bool(self.scene_locked)
        self.detections = [d if isinstance(d, Detection2D) else Detection2D(**d) for d in self.detections]
        legacy_relations = [
            r if isinstance(r, LocalRelation2D) else LocalRelation2D(**r) for r in self.local_relations
        ]
        final_relations = self.local_relations_final
        if final_relations is None:
            final_relations = legacy_relations
        self.local_relations_final = [
            r if isinstance(r, LocalRelation2D) else LocalRelation2D(**r) for r in final_relations
        ]
        raw_relations = self.local_relation_candidates_raw
        debug_root = self.local_rel_debug if isinstance(self.local_rel_debug, dict) else {}
        debug_stages = debug_root.get("local_rel", debug_root)
        debug_raw = debug_stages.get("raw") if isinstance(debug_stages, dict) else None
        # Rebuild raw candidates from the debug source whenever it exists. This
        # validates legacy caches against final detection labels and fails
        # closed on stale dense indices instead of trusting serialized aliases.
        if raw_relations is None or isinstance(debug_raw, dict):
            raw_relations = extract_local_relations_from_debug(
                self.local_rel_debug,
                stage="raw",
                detection_labels=[d.label for d in self.detections],
            )
        self.local_relation_candidates_raw = [
            r if isinstance(r, LocalRelation2D) else LocalRelation2D(**r) for r in raw_relations
        ]
        # Backward-compatible alias: local_relations always means the final 2D stage.
        self.local_relations = self.local_relations_final
        self.schema_version = SCHEMA_VERSION
        if not self.relation_diagnostics:
            self.relation_diagnostics = build_relation_diagnostics(
                self.local_relation_candidates_raw,
                self.local_relations_final,
            )


def local_relation_from_debug_edge(edge: dict[str, Any], *, source_stage: str) -> LocalRelation2D | None:
    if not isinstance(edge, dict):
        return None
    edge_type = str(edge.get("type") or edge.get("edge_type") or "")
    if edge_type not in {"O-C", "C-U", "O-U"}:
        return None
    parent_role, child_role = edge_type.split("-", 1)
    try:
        return LocalRelation2D(
            edge_type=edge_type,
            parent_det_id=int(edge.get("parent_idx", edge.get("parent_det_id"))),
            child_det_id=int(edge.get("child_idx", edge.get("child_det_id"))),
            parent_role=str(edge.get("parent_role") or parent_role),
            child_role=str(edge.get("child_role") or child_role),
            contain=edge.get("contain"),
            mask_contain=edge.get("mask_contain"),
            pass_threshold=bool(edge.get("pass_thr", edge.get("pass_threshold", False))),
            selected=bool(edge.get("selected", False)),
            relation_text=str(edge.get("relation") or edge.get("relation_text") or ""),
            source_stage=source_stage,
        )
    except (TypeError, ValueError):
        return None


def extract_local_relations_from_debug(
    local_rel_debug: dict[str, Any] | None,
    *,
    stage: str,
    detection_labels: list[str] | tuple[str, ...] | None = None,
) -> list[LocalRelation2D]:
    debug = local_rel_debug if isinstance(local_rel_debug, dict) else {}
    local_rel = debug.get("local_rel", debug)
    if not isinstance(local_rel, dict):
        return []
    payload = local_rel.get(stage, {})
    if not isinstance(payload, dict):
        return []
    out = []
    for edge in payload.get("edges", []) or []:
        relation = local_relation_from_debug_edge(edge, source_stage=stage)
        if relation is None:
            continue
        if detection_labels is not None:
            parent_idx = int(relation.parent_det_id)
            child_idx = int(relation.child_det_id)
            if not (
                0 <= parent_idx < len(detection_labels)
                and 0 <= child_idx < len(detection_labels)
            ):
                continue
            parent_label = edge.get("parent_label")
            child_label = edge.get("child_label")
            if parent_label is not None and str(parent_label) != str(detection_labels[parent_idx]):
                continue
            if child_label is not None and str(child_label) != str(detection_labels[child_idx]):
                continue
        out.append(relation)
    return out


def build_relation_diagnostics(
    raw_relations: list[LocalRelation2D],
    final_relations: list[LocalRelation2D],
) -> dict[str, Any]:
    def by_type(relations: list[LocalRelation2D]) -> dict[str, int]:
        return {
            edge_type: int(sum(1 for relation in relations if relation.edge_type == edge_type))
            for edge_type in ("O-C", "O-U", "C-U")
        }

    raw_keys = {
        (r.edge_type, int(r.parent_det_id), int(r.child_det_id))
        for r in raw_relations
    }
    final_keys = {
        (r.edge_type, int(r.parent_det_id), int(r.child_det_id))
        for r in final_relations
    }
    retained = len(raw_keys & final_keys)
    return {
        "num_raw_candidates": int(len(raw_relations)),
        "num_final_relations": int(len(final_relations)),
        "raw_by_type": by_type(raw_relations),
        "final_by_type": by_type(final_relations),
        "raw_to_final_retention_rate": float(retained / max(1, len(raw_keys))),
    }
