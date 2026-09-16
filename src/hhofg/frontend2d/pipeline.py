from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from hhofg.core.serialization import save_frame2d_result, write_json_atomic
from hhofg.core.types import (
    Detection2D,
    Frame2DResult,
    SCHEMA_VERSION,
    build_relation_diagnostics,
    extract_local_relations_from_debug,
)
from hhofg.data.base import FrameRecord
from hhofg.frontend2d.geometry import box_touches_image_border
from hhofg.frontend2d.image_preprocess import ImageTransform2D, build_functional_slam_image_views, build_original_image_views
from hhofg.frontend2d.policy import FunctionalGraphPolicy
from hhofg.frontend2d.relation_indexing import reindex_local_relation_debug
from hhofg.frontend2d.fs2d.io_utils import append_jsonl
from hhofg.frontend2d.fs2d.merge_semantics import (
    atlas_object_carrier_set,
    build_frame_result_final,
    build_graph_frame_result,
    remote_endpoints_from_pairs,
)
from hhofg.frontend2d.fs2d.sam3_io import _sam3_det_to_json
from hhofg.frontend2d.fs2d.semantic_pipeline import SemanticPipeline
from hhofg.frontend2d.sam3.io import _save_sam3_vis

ROLE_FROM_RANK = {0: "U", 1: "C", 2: "O"}
FINAL_REL_STAGES = ("after_cov", "after_r4", "after_r3", "after_r2", "after_r1", "raw")


def _canon_tag(s: str | None) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    return " ".join(s.split())


def _map_tag_to_atlas(tag: str, atlas_objs: set[str]) -> str:
    t = _canon_tag(tag)
    for p in ("kitchen ", "bathroom ", "living room ", "bedroom ", "dining room "):
        if t.startswith(p):
            rest = t[len(p) :].strip()
            if rest in atlas_objs:
                return rest
    compact = t.replace(" ", "")
    compact_map = {o.replace(" ", ""): o for o in atlas_objs}
    return compact_map.get(compact, t)


def _pick_local_rel_stage(local_rel_debug: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    local_rel = local_rel_debug.get("local_rel", local_rel_debug) if isinstance(local_rel_debug, dict) else {}
    if not isinstance(local_rel, dict):
        return "raw", {"edges": []}
    for stage in FINAL_REL_STAGES:
        payload = local_rel.get(stage)
        if isinstance(payload, dict) and payload.get("edges"):
            return stage, payload
    return "raw", local_rel.get("raw", {"edges": []}) if isinstance(local_rel.get("raw"), dict) else {"edges": []}


def _filter_border_no_link_detections(
    det,
    img_h: int,
    img_w: int,
    margin_px: int = 2,
) -> None:
    """Remove border detections without selected/pass-threshold local links, preserving FS reindexing."""
    import torch

    boxes = getattr(det, "boxes", None)
    labels = getattr(det, "labels", None)
    if boxes is None or labels is None or len(labels) == 0:
        return

    n_det = len(labels)
    # Build linked-det-idx set from local_rel_debug edges (final stage).
    linked_idxs: set[int] = set()
    local_rel = getattr(det, "local_rel_debug", None) or {}
    for stage_name in ("after_cov", "after_r4", "after_r3", "after_r2", "after_r1", "raw"):
        stage_relations = extract_local_relations_from_debug(
            local_rel,
            stage=stage_name,
            detection_labels=list(labels),
        )
        if stage_relations:
            for relation in stage_relations:
                if not (relation.selected or relation.pass_threshold):
                    continue
                ci = int(relation.child_det_id)
                pi = int(relation.parent_det_id)
                if ci >= 0:
                    linked_idxs.add(ci)
                if pi >= 0:
                    linked_idxs.add(pi)
            break

    # Determine which indices to keep.
    keep = []
    for i in range(n_det):
        box = boxes[i].detach().cpu().tolist() if torch.is_tensor(boxes) else list(boxes[i])
        at_border = box_touches_image_border(box, img_h, img_w, margin_px=margin_px)
        if at_border and i not in linked_idxs:
            continue
        keep.append(i)

    if len(keep) == n_det:
        return

    # Filter parallel arrays.
    keep_t = torch.tensor(keep, dtype=torch.long)
    if torch.is_tensor(boxes):
        det.boxes = boxes[keep_t]
    if getattr(det, "scores", None) is not None and torch.is_tensor(det.scores):
        det.scores = det.scores[keep_t]
    if getattr(det, "masks", None) is not None and torch.is_tensor(det.masks):
        det.masks = det.masks[keep_t]
    det.labels = [labels[i] for i in keep]

    # All relation structures (edges, edge-id lookups, assignments and chains)
    # must move as one transaction.
    reindex_local_relation_debug(local_rel, keep=keep)

    # Re-index cabinet_hints if present.
    old_to_new = {old: new for new, old in enumerate(keep)}
    cab_hints = getattr(det, "cabinet_hints", None)
    if isinstance(cab_hints, dict):
        new_hits = []
        for hit in cab_hints.get("carrier_hits", []):
            ci = int(hit.get("child_idx", -1))
            pi = int(hit.get("cabinet_parent_idx", -1))
            if ci in old_to_new:
                hit = dict(hit)
                hit["child_idx"] = old_to_new[ci]
                if pi in old_to_new:
                    hit["cabinet_parent_idx"] = old_to_new[pi]
                new_hits.append(hit)
        cab_hints["carrier_hits"] = new_hits


def _reindex_det_result(det, keep: list[int]) -> None:
    import torch

    boxes = getattr(det, "boxes", None)
    labels = getattr(det, "labels", None)
    if labels is None:
        return
    keep_t = torch.tensor(keep, dtype=torch.long)
    if torch.is_tensor(boxes):
        det.boxes = boxes[keep_t]
    if getattr(det, "scores", None) is not None and torch.is_tensor(det.scores):
        det.scores = det.scores[keep_t]
    if getattr(det, "masks", None) is not None and torch.is_tensor(det.masks):
        det.masks = det.masks[keep_t]
    det.labels = [labels[i] for i in keep]

    local_rel = getattr(det, "local_rel_debug", None) or {}
    reindex_local_relation_debug(local_rel, keep=keep)


def _filter_orphan_u_after_final_filter(det) -> None:
    labels = getattr(det, "labels", None)
    if labels is None or len(labels) == 0:
        return
    rank = getattr(det, "label_to_rank", None) or {}
    local_rel_debug = getattr(det, "local_rel_debug", None) or {}
    stage, _stage_data = _pick_local_rel_stage(local_rel_debug)
    parented_u: set[int] = set()
    for relation in extract_local_relations_from_debug(
        local_rel_debug,
        stage=stage,
        detection_labels=list(labels),
    ):
        if not relation.pass_threshold:
            continue
        if not relation.edge_type.endswith("-U"):
            continue
        parent_idx = int(relation.parent_det_id)
        child_idx = int(relation.child_det_id)
        if parent_idx >= 0 and child_idx >= 0:
            parented_u.add(child_idx)
    # A U removed only by a later Functional-SLAM 2D rule is still needed for
    # the paper-style raw/semantic candidate funnel. Keep raw-linked endpoints;
    # the raw edge remains subject to Sdet, Gcamc, real VLM scoring and 3D graph
    # eligibility downstream.
    for relation in extract_local_relations_from_debug(
        local_rel_debug,
        stage="raw",
        detection_labels=list(labels),
    ):
        child_idx = int(relation.child_det_id)
        if child_idx >= 0 and relation.edge_type.endswith("-U"):
            parented_u.add(child_idx)
    keep = []
    dropped = []
    for idx, label in enumerate(labels):
        is_u = ROLE_FROM_RANK.get(int(rank.get(str(label), 2)), "O") == "U"
        if is_u and idx not in parented_u:
            dropped.append(
                {
                    "idx": int(idx),
                    "label": str(label),
                    "stage": "final_orphan_cleanup",
                    "rule": "orphan_u_after_final_filter",
                }
            )
            continue
        keep.append(idx)
    if len(keep) == len(labels):
        return
    local_rel_debug.setdefault("drops", [])
    local_rel_debug["drops"].extend(dropped)
    _reindex_det_result(det, keep)


def _augment_remote_visible_det_indices(remote_rel_2d: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(remote_rel_2d, dict):
        return remote_rel_2d
    observed = remote_rel_2d.get("observed_candidates") or []
    det_by_pair: dict[tuple[str, str], dict[str, int]] = {}
    for rel in observed:
        if not isinstance(rel, dict):
            continue
        key = (_canon_tag(rel.get("from_object")), _canon_tag(rel.get("to_object")))
        if not key[0] or not key[1]:
            continue
        if rel.get("from_det_idx") is None or rel.get("to_det_idx") is None:
            continue
        det_by_pair[key] = {
            "from_det_idx": int(rel["from_det_idx"]),
            "to_det_idx": int(rel["to_det_idx"]),
        }
    for key_name in ("confirmed_visible", "confirmed_all"):
        patched = []
        for rel in remote_rel_2d.get(key_name) or []:
            if not isinstance(rel, dict):
                patched.append(rel)
                continue
            key = (_canon_tag(rel.get("from_object")), _canon_tag(rel.get("to_object")))
            if key in det_by_pair:
                rel = dict(rel)
                rel.update(det_by_pair[key])
            patched.append(rel)
        remote_rel_2d[key_name] = patched
    return remote_rel_2d


def _tensor_to_numpy(x, *, dtype=None):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


class Frontend2DPipeline:
    def __init__(
        self,
        config: dict[str, Any],
        rampp=None,
        llm_runtime=None,
        scene_lock=None,
        sam3_runtime=None,
        policy: FunctionalGraphPolicy | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.rampp = rampp
        self.llm_runtime = llm_runtime
        self.scene_lock = scene_lock
        self.sam3_runtime = sam3_runtime
        self.policy = policy or FunctionalGraphPolicy()
        self.output_dir = Path(output_dir or config.get("output", {}).get("root", "outputs"))
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.scene_history_path = self.output_dir / "scene_history.jsonl"
        self.scene_json_path = self.output_dir / "scene.json"
        self.atlas_json_path = self.output_dir / "global_atlas.json"
        self.scene_type: str | None = None
        self.global_atlas: dict[str, Any] | None = None
        self.last_scene_result: dict[str, Any] | None = None
        self.pre_atlas_notice_printed = False
        scene_cfg = config.get("scene_lock", {})
        sam3_cfg = config.get("sam3", {})
        output_cfg = config.get("output", {})
        method_cfg = config.get("method", {})
        frontend_cfg = config.get("frontend2d", {})
        vis_cfg = config.get("visualization", {})
        self.args = SimpleNamespace(
            rampp_print_every=int(config.get("rampp", {}).get("print_every", 10**9)),
            deepseek_print_every=int(config.get("llm", {}).get("print_every", 10**9)),
            pre_atlas_uc_mode=config.get("llm", {}).get("pre_atlas_uc_mode", "all"),
            enable_scene_switch=bool(scene_cfg.get("enable_switch", False)),
            disable_scene_switch=not bool(scene_cfg.get("enable_switch", False)),
            scene_recheck_every=int(scene_cfg.get("recheck_every", 0) or 0),
            sam3_keep_border_no_link_detections=bool(sam3_cfg.get("keep_border_no_link_detections", False)),
            remote_win_size=int(method_cfg.get("remote_win_size", 30)),
            remote_confirm_k=int(method_cfg.get("remote_confirm_k", 2)),
            remote_ambig_delta=float(method_cfg.get("remote_ambig_delta", 0.2)),
            enable_remote_relations=bool(method_cfg.get("enable_remote_relations", True)),
            save_visualization=bool(sam3_cfg.get("save_visualization", False)),
            draw_remote_relations=bool(vis_cfg.get("draw_remote_relations", True)),
            draw_selected_local=bool(vis_cfg.get("draw_selected_local", True)),
            draw_candidate_local=bool(vis_cfg.get("draw_candidate_local", False)),
            draw_relation_text=bool(vis_cfg.get("draw_relation_text", True)),
            hide_redundant_ou_when_cu_selected=bool(vis_cfg.get("hide_redundant_ou_when_cu_selected", True)),
        )
        debug_dir = self.output_dir / "functional_raw" if bool(output_cfg.get("save_debug_json", True)) else None
        self.sam3_prompt_source = str(sam3_cfg.get("prompt_source", "frame_result")).strip().lower()
        if self.sam3_prompt_source not in {"frame_result", "graph_frame_result"}:
            raise ValueError("sam3.prompt_source must be 'frame_result' or 'graph_frame_result'")
        self.input_mode = str(frontend_cfg.get("input_mode", "functional_slam_compatible")).strip().lower()
        self.functional_slam_img_size = int(frontend_cfg.get("functional_slam_img_size", 512))
        self.img_downsample = int(frontend_cfg.get("img_downsample", 1))
        self.square_ok = bool(frontend_cfg.get("square_ok", False))
        self.semantic = SemanticPipeline(
            self.args,
            rampp=rampp,
            deepseek=llm_runtime,
            scene_lock=scene_lock,
            sam3_runtime=sam3_runtime,
            frames_out_dir=debug_dir,
            scene_history_path=self.scene_history_path,
            scene_json_path=self.scene_json_path,
            atlas_json_path=self.atlas_json_path,
            sam3_save_vis=bool(sam3_cfg.get("save_visualization", False)),
            graph_policy=self.policy,
        )

    def _run_sam3_with_candidate_filter(
        self,
        frame_idx: int,
        frame_uimg: np.ndarray,
        frame_result: dict[str, Any],
        graph_frame_result: dict[str, Any],
    ):
        if self.sam3_runtime is None:
            return None
        det = self.sam3_runtime.run(frame_uimg, frame_result, frame_idx=frame_idx, policy=self.policy)
        if det is not None and not bool(self.config.get("sam3", {}).get("keep_border_no_link_detections", False)):
            _filter_border_no_link_detections(
                det,
                int(frame_uimg.shape[0]),
                int(frame_uimg.shape[1]),
                margin_px=2,
            )
        if det is not None:
            _filter_orphan_u_after_final_filter(det)
        remote_result = graph_frame_result if isinstance(graph_frame_result, dict) else frame_result
        remote_rel_2d = (
            self.semantic._build_remote_rel_2d(frame_idx, frame_uimg, det, remote_result)
            if det is not None and self.args.enable_remote_relations
            else None
        )
        remote_rel_2d = _augment_remote_visible_det_indices(remote_rel_2d)
        debug_image_path = None
        if self.semantic.frames_out_dir is not None:
            self.semantic.frames_out_dir.mkdir(parents=True, exist_ok=True)
            rgb_path = self.semantic.frames_out_dir / f"frame_{frame_idx:06d}_rgb.png"
            self.semantic._save_rgb_frame(rgb_path, frame_uimg)
            debug_image_path = str(rgb_path)
            det_path = self.semantic.frames_out_dir / f"frame_{frame_idx:06d}_sam3_det.json"
            if det is None:
                write_json_atomic(det_path, {"detections": [], "remote_rel_2d": remote_rel_2d})
            else:
                write_json_atomic(det_path, _sam3_det_to_json(det, extra={"remote_rel_2d": remote_rel_2d}))
            if det is not None and self.semantic.sam3_save_vis:
                vis_path = self.semantic.frames_out_dir / f"frame_{frame_idx:06d}_sam3_det.png"
                _save_sam3_vis(
                    vis_path,
                    frame_uimg,
                    det,
                    remote_rel_2d=remote_rel_2d if self.args.draw_remote_relations else None,
                    draw_selected_local=self.args.draw_selected_local,
                    draw_candidate_local=self.args.draw_candidate_local,
                    draw_relation_text=self.args.draw_relation_text,
                    hide_redundant_ou_when_cu_selected=self.args.hide_redundant_ou_when_cu_selected,
                )
                debug_image_path = str(vis_path)
        return SimpleNamespace(det=det, remote_rel_2d=remote_rel_2d, debug_image_path=debug_image_path)

    def _semantic_state(self, frame_idx: int, tags: dict[str, list[str]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        if self.llm_runtime is None:
            frame_result = build_frame_result_final(None, tags.get("tags_en", []), {"objects": [], "remote_relation_candidates": []})
            return frame_result, build_graph_frame_result(
                frame_result,
                suppress_objects=self.policy.suppress_objects,
                hint_only_objects=self.policy.hint_only_objects,
            ), None
        norm_tags_list = [_canon_tag(t) for t in tags.get("tags_en", []) if _canon_tag(t)]
        norm_tags = set(norm_tags_list)
        if self.global_atlas is None:
            scene_result = self.llm_runtime.classify_scene(tags.get("tags_en", []))
            self.last_scene_result = scene_result
            lock_info = self.scene_lock.update(scene_result.get("scene_type"), scene_result.get("confidence", 0.0))
            append_jsonl(
                self.scene_history_path,
                {
                    "frame": frame_idx,
                    "scene_type": scene_result.get("scene_type"),
                    "confidence": scene_result.get("confidence"),
                    "locked": lock_info["locked"],
                    "streak": lock_info["streak"],
                    "reason": lock_info.get("reason"),
                },
            )
            if lock_info["locked"]:
                self.scene_type = lock_info.get("scene_type")
                self.global_atlas = self.llm_runtime.get_or_build_global_atlas(self.scene_type)
                write_json_atomic(self.scene_json_path, {"scene_type": self.scene_type, "confidence": scene_result.get("confidence"), "locked_at_frame": frame_idx})
                write_json_atomic(self.atlas_json_path, self.global_atlas)
        incremental = None
        if self.global_atlas is not None:
            atlas_oc = {_canon_tag(o) for o in atlas_object_carrier_set(self.global_atlas)}
            atlas_remote_endpoints = remote_endpoints_from_pairs((self.global_atlas or {}).get("remote_relation_candidates", []))
            mapped_norm_tags = {_map_tag_to_atlas(t, atlas_oc) for t in norm_tags}
            unknown_tags = sorted(mapped_norm_tags - atlas_oc)
            if unknown_tags:
                incremental = self.llm_runtime.infer_incremental_uc(
                    uc_tags_en=unknown_tags,
                    atlas_remote_objects_en=sorted(atlas_remote_endpoints),
                    scene_type=self.scene_type or "unknown",
                )
            else:
                incremental = {"objects": [], "remote_relation_candidates": []}
        elif self.args.pre_atlas_uc_mode == "all":
            incremental = self.llm_runtime.infer_incremental_uc(uc_tags_en=sorted(norm_tags), atlas_remote_objects_en=[], scene_type="unknown")
        frame_result = build_frame_result_final(self.global_atlas, norm_tags_list, incremental)
        graph_frame_result = build_graph_frame_result(
            frame_result,
            suppress_objects=self.policy.suppress_objects,
            hint_only_objects=self.policy.hint_only_objects,
        )
        return frame_result, graph_frame_result, incremental

    def _detections_and_arrays(self, det, image_h: int, image_w: int):
        if det is None or getattr(det, "boxes", None) is None or getattr(det, "scores", None) is None or getattr(det, "labels", None) is None:
            return [], np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0, image_h, image_w), bool), {}, {}, {}, [], {}
        boxes = _tensor_to_numpy(det.boxes, dtype=np.float32)
        scores = _tensor_to_numpy(det.scores, dtype=np.float32)
        masks = _tensor_to_numpy(det.masks)
        if masks is None:
            masks = np.zeros((len(det.labels), image_h, image_w), dtype=bool)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        masks = masks > 0.5
        label_to_rank = getattr(det, "label_to_rank", None) or {}
        label_to_role = {str(k): ROLE_FROM_RANK.get(int(v), "O") for k, v in label_to_rank.items()}
        detections = []
        for i, (label, score, box) in enumerate(zip(det.labels, scores, boxes)):
            role = label_to_role.get(str(label), "O")
            detections.append(Detection2D(i, str(label), role, float(score), [float(x) for x in box]))
        local_rel_debug = getattr(det, "local_rel_debug", None) or {}
        stage, _stage_data = _pick_local_rel_stage(local_rel_debug)
        relations = extract_local_relations_from_debug(
            local_rel_debug,
            stage=stage,
            detection_labels=[str(label) for label in det.labels],
        )
        u_allowed = {k: sorted(v) for k, v in (getattr(det, "u_to_allowed_parents", None) or {}).items()}
        c_allowed = {k: sorted(v) for k, v in (getattr(det, "c_to_allowed_parents", None) or {}).items()}
        return detections, boxes, scores, masks, label_to_role, u_allowed, c_allowed, relations, local_rel_debug

    @staticmethod
    def _map_detections_to_raw(
        detections: list[Detection2D],
        boxes_inference: np.ndarray,
        masks_inference: np.ndarray,
        transform: ImageTransform2D,
    ) -> tuple[list[Detection2D], np.ndarray, np.ndarray]:
        if len(detections) == 0:
            return (
                [],
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, transform.raw_height, transform.raw_width), dtype=bool),
            )
        boxes_raw = np.asarray([transform.inference_box_to_raw(box) for box in boxes_inference], dtype=np.float32)
        boxes_raw[:, [0, 2]] = np.clip(boxes_raw[:, [0, 2]], 0, transform.raw_width)
        boxes_raw[:, [1, 3]] = np.clip(boxes_raw[:, [1, 3]], 0, transform.raw_height)
        masks_raw = np.stack([transform.inference_mask_to_raw(mask) for mask in masks_inference], axis=0).astype(bool)
        mapped = []
        for det, box_raw, box_inf in zip(detections, boxes_raw, boxes_inference):
            mapped.append(
                Detection2D(
                    det.det_id,
                    det.label,
                    det.role,
                    det.score,
                    [float(x) for x in box_raw],
                    box_xyxy_inference=[float(x) for x in box_inf],
                )
            )
        return mapped, boxes_raw, masks_raw

    @staticmethod
    def _raw_det_view(det, boxes_raw: np.ndarray, masks_raw: np.ndarray):
        if det is None:
            return None
        raw = SimpleNamespace(**getattr(det, "__dict__", {}))
        raw.boxes = torch.as_tensor(boxes_raw, dtype=torch.float32)
        raw.masks = torch.as_tensor(masks_raw, dtype=torch.bool)
        return raw

    def process_frame(self, frame: FrameRecord) -> Frame2DResult:
        t_total = time.perf_counter()
        timing: dict[str, float] = {}
        if self.input_mode == "original":
            views = build_original_image_views(frame.rgb)
        elif self.input_mode == "functional_slam_compatible":
            views = build_functional_slam_image_views(
                frame.rgb,
                img_size=self.functional_slam_img_size,
                img_downsample=self.img_downsample,
                square_ok=self.square_ok,
            )
        else:
            raise ValueError("frontend2d.input_mode must be 'original' or 'functional_slam_compatible'")
        img_raw_float = views.rgb_raw.astype(np.float32) / 255.0
        img_sam3_float = views.rgb_sam3.astype(np.float32) / 255.0
        t0 = time.perf_counter()
        tags = self.semantic.run_rampp(frame.frame_idx, img_raw_float) or {"tags_en": [], "tags_zh": []}
        timing["rampp"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        if self.llm_runtime is not None:
            deepseek_out = self.semantic.run_deepseek(frame.frame_idx, tags)
            frame_result = deepseek_out.frame_result
            graph_frame_result = deepseek_out.graph_frame_result
            self.scene_type = self.semantic.scene_type
            self.global_atlas = self.semantic.global_atlas
            self.last_scene_result = self.semantic.last_scene_result
        else:
            frame_result, graph_frame_result, _ = self._semantic_state(frame.frame_idx, tags)
        timing["llm"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        sam3_input = graph_frame_result if self.sam3_prompt_source == "graph_frame_result" else frame_result
        sam3_out = self._run_sam3_with_candidate_filter(
            frame.frame_idx,
            img_sam3_float,
            sam3_input,
            graph_frame_result=graph_frame_result,
        )
        det = sam3_out.det if sam3_out is not None else None
        remote_rel_2d = sam3_out.remote_rel_2d if sam3_out is not None and sam3_out.remote_rel_2d else {}
        if det is not None and self.args.save_visualization:
            vis_remote = remote_rel_2d if self.args.draw_remote_relations else None
            _save_sam3_vis(
                self.frames_dir / f"{frame.frame_key}_sam3_inference_vis.png",
                views.rgb_sam3,
                det,
                remote_rel_2d=vis_remote,
                draw_selected_local=self.args.draw_selected_local,
                draw_candidate_local=self.args.draw_candidate_local,
                draw_relation_text=self.args.draw_relation_text,
                hide_redundant_ou_when_cu_selected=self.args.hide_redundant_ou_when_cu_selected,
            )
        timing["sam3"] = (time.perf_counter() - t0) * 1000.0
        (
            detections_inf,
            boxes_inf,
            scores,
            masks_inf,
            label_to_role,
            u_allowed,
            c_allowed,
            relations,
            local_rel_debug,
        ) = self._detections_and_arrays(det, views.transform.inference_height, views.transform.inference_width)
        raw_relations = extract_local_relations_from_debug(
            local_rel_debug,
            stage="raw",
            detection_labels=[d.label for d in detections_inf],
        )
        detections, boxes_raw, masks_raw = self._map_detections_to_raw(
            detections_inf,
            boxes_inf,
            masks_inf,
            views.transform,
        )
        if det is not None and self.args.save_visualization:
            raw_det = self._raw_det_view(det, boxes_raw, masks_raw)
            vis_remote = remote_rel_2d if self.args.draw_remote_relations else None
            _save_sam3_vis(
                self.frames_dir / f"{frame.frame_key}_raw_vis.png",
                views.rgb_raw,
                raw_det,
                remote_rel_2d=vis_remote,
                draw_selected_local=True,
                draw_candidate_local=True,
                draw_relation_text=self.args.draw_relation_text,
                hide_redundant_ou_when_cu_selected=False,
            )
            _save_sam3_vis(
                self.frames_dir / f"{frame.frame_key}_vis.png",
                views.rgb_raw,
                raw_det,
                remote_rel_2d=vis_remote,
                draw_selected_local=self.args.draw_selected_local,
                draw_candidate_local=self.args.draw_candidate_local,
                draw_relation_text=self.args.draw_relation_text,
                hide_redundant_ou_when_cu_selected=self.args.hide_redundant_ou_when_cu_selected,
            )
        result = Frame2DResult(
            schema_version=SCHEMA_VERSION,
            frame_idx=frame.frame_idx,
            frame_key=frame.frame_key,
            image_path=str(frame.rgb_path),
            image_height=frame.rgb.shape[0],
            image_width=frame.rgb.shape[1],
            scene_type=self.scene_type or (self.last_scene_result or {}).get("scene_type") or "unknown",
            scene_locked=self.global_atlas is not None,
            tags_en=list(tags.get("tags_en", [])),
            tags_zh=list(tags.get("tags_zh", [])),
            frame_result=frame_result,
            graph_frame_result=graph_frame_result,
            detections=detections,
            local_relations=relations,
            local_relations_final=relations,
            local_relation_candidates_raw=raw_relations,
            label_to_role=label_to_role,
            u_to_allowed_parents=u_allowed,
            c_to_allowed_parents=c_allowed,
            local_rel_debug=local_rel_debug,
            remote_rel_2d=remote_rel_2d if self.args.enable_remote_relations else {},
            timing_ms={},
            warnings=[],
            relation_diagnostics=build_relation_diagnostics(raw_relations, relations),
            inference_image_height=views.transform.inference_height,
            inference_image_width=views.transform.inference_width,
            image_transform=views.transform.to_dict(),
            coordinate_space="raw",
        )
        t0 = time.perf_counter()
        json_path = self.frames_dir / f"{frame.frame_key}.json"
        npz_path = self.frames_dir / f"{frame.frame_key}.npz"
        result.timing_ms = timing
        timing["serialization"] = (time.perf_counter() - t0) * 1000.0
        timing["total"] = (time.perf_counter() - t_total) * 1000.0
        save_frame2d_result(
            result,
            boxes_raw,
            scores,
            masks_raw,
            json_path,
            npz_path,
            boxes_inference=boxes_inf,
            masks_inference=masks_inf,
        )
        return result
