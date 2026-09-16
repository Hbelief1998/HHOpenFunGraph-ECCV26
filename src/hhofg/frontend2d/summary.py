from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from hhofg.core.serialization import write_json_atomic
from hhofg.core.types import Frame2DResult


class RunSummary:
    def __init__(self) -> None:
        self.num_input_frames = 0
        self.num_success_frames = 0
        self.num_failed_frames = 0
        self.num_empty_frames = 0
        self.tag_counts: list[int] = []
        self.det_counts: list[int] = []
        self.selected_rel_counts: list[int] = []
        self.role_counts: Counter[str] = Counter()
        self.rel_counts: Counter[str] = Counter()
        self.raw_rel_counts: Counter[str] = Counter()
        self.raw_relation_counts: list[int] = []
        self.final_relation_counts: list[int] = []
        self.raw_unique_count = 0
        self.raw_retained_count = 0
        self.timing: Counter[str] = Counter()
        self.scene_type = ""
        self.scene_locked = False
        self.scene_lock_frame: int | None = None
        self.atlas_object_count = 0

    def add_success(self, result: Frame2DResult, atlas: dict[str, Any] | None = None) -> None:
        self.num_input_frames += 1
        self.num_success_frames += 1
        if not result.detections:
            self.num_empty_frames += 1
        self.tag_counts.append(len(result.tags_en))
        self.det_counts.append(len(result.detections))
        self.selected_rel_counts.append(sum(1 for r in result.local_relations if r.selected))
        for d in result.detections:
            self.role_counts[d.role] += 1
        for r in result.local_relations:
            self.rel_counts[r.edge_type] += 1
        self.raw_relation_counts.append(len(result.local_relation_candidates_raw or []))
        self.final_relation_counts.append(len(result.local_relations_final or []))
        for r in result.local_relation_candidates_raw or []:
            self.raw_rel_counts[r.edge_type] += 1
        raw_keys = {
            (r.edge_type, r.parent_det_id, r.child_det_id)
            for r in result.local_relation_candidates_raw or []
        }
        final_keys = {
            (r.edge_type, r.parent_det_id, r.child_det_id)
            for r in result.local_relations_final or []
        }
        self.raw_unique_count += len(raw_keys)
        self.raw_retained_count += len(raw_keys & final_keys)
        for k, v in result.timing_ms.items():
            self.timing[k] += float(v)
        self.scene_type = result.scene_type or self.scene_type
        self.scene_locked = bool(result.scene_locked)
        if self.scene_locked and self.scene_lock_frame is None:
            self.scene_lock_frame = result.frame_idx
        if atlas:
            self.atlas_object_count = len(atlas.get("objects", []) or [])

    def add_failure(self) -> None:
        self.num_input_frames += 1
        self.num_failed_frames += 1

    @staticmethod
    def _avg(values: list[int]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def to_dict(self) -> dict[str, Any]:
        denom = max(1, self.num_success_frames)
        return {
            "num_input_frames": self.num_input_frames,
            "num_success_frames": self.num_success_frames,
            "num_failed_frames": self.num_failed_frames,
            "num_empty_frames": self.num_empty_frames,
            "scene_type": self.scene_type,
            "scene_locked": self.scene_locked,
            "scene_lock_frame": self.scene_lock_frame,
            "atlas_object_count": self.atlas_object_count,
            "average_rampp_tag_count": self._avg(self.tag_counts),
            "average_detection_count": self._avg(self.det_counts),
            "detections_by_role": {k: int(self.role_counts.get(k, 0)) for k in ("O", "C", "U")},
            "local_relations_by_type": {k: int(self.rel_counts.get(k, 0)) for k in ("O-C", "C-U", "O-U")},
            "raw_local_candidates_by_type": {k: int(self.raw_rel_counts.get(k, 0)) for k in ("O-C", "C-U", "O-U")},
            "num_raw_candidates": int(sum(self.raw_relation_counts)),
            "num_final_relations": int(sum(self.final_relation_counts)),
            "raw_to_final_retention_rate": float(
                self.raw_retained_count / max(1, self.raw_unique_count)
            ),
            "average_selected_local_relations": self._avg(self.selected_rel_counts),
            "timing_ms": {k: float(self.timing.get(k, 0.0) / denom) for k in ("total", "rampp", "llm", "sam3", "serialization")},
        }

    def save(self, path: str | Path) -> None:
        write_json_atomic(path, self.to_dict())
