from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hhofg.core.types import Frame2DResult, FrameRecord

from .depth_filter import dbscan_main_cluster_keep_mask, depth_iqr_keep_mask
from .depth_filter import adaptive_dbscan_cluster_keep_mask
from .features import hsv_histogram, voxel_downsample_points_colors
from .observation_quality import (
    choose_depth_mode,
    erosion_seed_mask,
    keep_primary_connected_component,
    robust_iqr,
    robust_mad,
    score_observation_quality,
    split_depth_modes,
)
from .types import Observation3D


@dataclass
class LiftingReject:
    det_id: int
    label: str
    role: str
    reject_reason: str
    raw_points: int
    filtered_points: int


@dataclass
class FrameLiftingReport:
    frame_idx: int
    frame_key: str
    rejects: list[LiftingReject] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quality: list[dict[str, Any]] = field(default_factory=list)


class ObservationLifter:
    def __init__(self, cfg: dict[str, Any], appearance_cfg: dict[str, Any], clip_encoder) -> None:
        self.cfg = cfg
        self.appearance_cfg = appearance_cfg
        self.clip_encoder = clip_encoder
        self.last_report: FrameLiftingReport | None = None

    def lift_frame(self, frame: FrameRecord, result2d: Frame2DResult, masks_raw: np.ndarray) -> list[Observation3D]:
        if frame.depth_m is None or frame.camera is None or frame.T_c2w is None:
            raise ValueError("FrameRecord must include depth_m, camera, and T_c2w")
        if result2d.frame_key != frame.frame_key:
            raise ValueError(f"frame_key mismatch: {result2d.frame_key} != {frame.frame_key}")
        if masks_raw.shape[1:] != frame.depth_m.shape:
            raise ValueError("masks_raw must match depth shape")
        report = FrameLiftingReport(frame.frame_idx, frame.frame_key)
        labels = [d.label for d in result2d.detections]
        sem = self.clip_encoder.encode_labels(labels)
        observations: list[Observation3D] = []
        parent_map = self._parent_map(result2d)
        allowed_map = self._allowed_parent_map(result2d)
        duplicate_witnesses = self._overlapping_carrier_witnesses(
            result2d,
            iou_threshold=float(self.cfg.get("cross_role_uc_box_iou_threshold", 0.9)),
        )
        suppressed_carriers = set(duplicate_witnesses)
        duplicate_carriers_by_unit: dict[int, list[int]] = {}
        for carrier_det_id, unit_det_ids in duplicate_witnesses.items():
            for unit_det_id in unit_det_ids:
                duplicate_carriers_by_unit.setdefault(unit_det_id, []).append(
                    carrier_det_id
                )
        for det in sorted(result2d.detections, key=lambda d: ({"O": 0, "C": 1, "U": 2}[d.role], d.det_id)):
            raw_mask = masks_raw[det.det_id].astype(bool)
            if det.det_id in suppressed_carriers:
                report.rejects.append(
                    LiftingReject(
                        det.det_id,
                        det.label,
                        det.role,
                        "cross_role_uc_box_overlap",
                        int(raw_mask.sum()),
                        0,
                    )
                )
                continue
            mask, component_debug = keep_primary_connected_component(raw_mask, det.box_xyxy)
            valid_depth = (
                np.isfinite(frame.depth_m)
                & (frame.depth_m >= float(self.cfg.get("depth_min_m", 0.1)))
                & (frame.depth_m <= float(self.cfg.get("depth_max_m", 10.0)))
            )
            valid_mask = mask & valid_depth
            raw_points = int(valid_mask.sum())
            if raw_points == 0:
                report.rejects.append(LiftingReject(det.det_id, det.label, det.role, "no_valid_depth", 0, 0))
                continue
            parent_prior = self._parent_depth_prior(parent_map.get(det.det_id, []), observations)
            h, w = frame.depth_m.shape
            x1, y1, x2, y2 = [float(v) for v in det.box_xyxy]
            cx_i = int(round((x1 + x2) * 0.5))
            cy_i = int(round((y1 + y2) * 0.5))
            radius = int(self.cfg.get("seed_center_radius_px", 2))
            yy, xx = np.ogrid[:h, :w]
            center_seed_mask = valid_mask & (np.abs(xx - cx_i) <= radius) & (np.abs(yy - cy_i) <= radius)
            seed_mask = center_seed_mask if int(center_seed_mask.sum()) >= int(self.cfg.get("seed_min_pixels", 5)) else erosion_seed_mask(
                valid_mask,
                min_pixels=int(self.cfg.get("seed_min_pixels", 5)),
                iterations=int(self.cfg.get("seed_erosion_iterations", 1)),
            )
            center_valid = 0 <= cy_i < h and 0 <= cx_i < w and bool(valid_mask[cy_i, cx_i])
            if center_valid:
                depth_seed = float(frame.depth_m[cy_i, cx_i])
            else:
                seed_z = frame.depth_m[seed_mask & valid_depth].astype(np.float32)
                depth_seed = float(np.median(seed_z)) if seed_z.size > 0 else None
            all_z = frame.depth_m[valid_mask].astype(np.float32)
            parent_depth = parent_prior.get("median")
            parent_mad = parent_prior.get("mad")
            modes = split_depth_modes(
                all_z,
                gap_m=float(self.cfg.get("depth_mode_gap_m", 0.04)),
                min_count=int(self.cfg.get("depth_mode_min_count", 4)),
                seed_depth=depth_seed,
                parent_depth=parent_depth,
            )
            chosen_mode = choose_depth_mode(modes, seed_depth=depth_seed, parent_depth=parent_depth, parent_mad=parent_mad, cfg=self.cfg)
            if chosen_mode is None:
                report.rejects.append(LiftingReject(det.det_id, det.label, det.role, "no_depth_mode", raw_points, 0))
                continue
            mode_band = max(
                float(self.cfg.get("depth_mode_band_min_m", 0.025)),
                float(self.cfg.get("depth_mode_mad_scale", 3.0)) * max(float(chosen_mode.mad), 0.002),
            )
            depth_residual = None if parent_depth is None else float(chosen_mode.median - float(parent_depth))
            mode_keep = valid_mask & (np.abs(frame.depth_m - float(chosen_mode.median)) <= mode_band)
            if parent_depth is not None and bool(self.cfg.get("parent_guided_depth_enable", True)):
                parent_band = max(
                    float(self.cfg.get("parent_band_min_m", 0.025)),
                    float(self.cfg.get("parent_band_mad_scale", 3.0)) * max(float(parent_mad or 0.0), 0.002),
                )
                parent_keep = valid_mask & (np.abs(frame.depth_m - float(parent_depth)) <= parent_band)
                if int(parent_keep.sum()) >= int(self.cfg.get("parent_guided_min_child_points", 3)):
                    mode_keep = mode_keep | parent_keep
            vv, uu = np.nonzero(mode_keep)
            if vv.size == 0:
                report.rejects.append(LiftingReject(det.det_id, det.label, det.role, "empty_chosen_depth_mode", raw_points, 0))
                continue
            z = frame.depth_m[vv, uu].astype(np.float32)
            K = frame.camera.K
            x = (uu.astype(np.float32) - float(K[0, 2])) * z / float(K[0, 0])
            y = (vv.astype(np.float32) - float(K[1, 2])) * z / float(K[1, 1])
            points_camera = np.stack([x, y, z], axis=1).astype(np.float32)
            keep = np.ones(points_camera.shape[0], dtype=bool)
            if bool(self.cfg.get("depth_iqr_enable", False)):
                keep &= depth_iqr_keep_mask(
                    points_camera,
                    iqr_scale=float(self.cfg.get("depth_iqr_scale", 1.5)),
                    min_keep_ratio=float(self.cfg.get("depth_iqr_min_keep_ratio", 0.35)),
                )
            filtered_points = points_camera[keep]
            filtered_v = vv[keep]
            filtered_u = uu[keep]
            # Preserve a bounded sample of the selected depth-mode extent for
            # identity association.  DBSCAN below deliberately keeps only a
            # conservative fusion cluster and may be much smaller than the
            # physical instance (transparent and reflective objects are common
            # examples), so it must not define the association extent.
            association_points_camera = self._sample_points(
                filtered_points,
                int(self.cfg.get("association_max_points_per_observation", 4096)),
            )
            mask_area_ratio = float(mask.sum()) / float(mask.size)
            median_depth_for_tiny = float(np.median(filtered_points[:, 2])) if filtered_points.shape[0] else float(chosen_mode.median)
            tiny_far = bool(
                mask_area_ratio < float(self.cfg.get("tiny_far_mask_area_ratio", 0.0008))
                and median_depth_for_tiny > float(self.cfg.get("tiny_far_depth_m", 1.2))
            )
            if bool(self.cfg.get("dbscan_enable", True)) and filtered_points.shape[0] > 0:
                if bool(self.cfg.get("adaptive_dbscan_enable", True)):
                    center_uv = ((float(det.box_xyxy[0]) + float(det.box_xyxy[2])) * 0.5, (float(det.box_xyxy[1]) + float(det.box_xyxy[3])) * 0.5)
                    min_samples = int(self.cfg.get("tiny_far_dbscan_min_samples", 3) if tiny_far else self.cfg.get("dbscan_min_samples", 8))
                    db_keep, db_applied, db_debug = adaptive_dbscan_cluster_keep_mask(
                        filtered_points,
                        np.stack([filtered_u, filtered_v], axis=1).astype(np.float32),
                        center_uv=center_uv,
                        depth_median=median_depth_for_tiny,
                        fx=float(K[0, 0]),
                        fy=float(K[1, 1]),
                        parent_depth=parent_depth,
                        parent_band_m=float(self.cfg.get("parent_band_min_m", 0.025)) if parent_depth is not None else None,
                        pixel_radius=float(self.cfg.get("dbscan_pixel_radius", 8.0 if tiny_far else 12.0)),
                        eps_min=float(self.cfg.get("dbscan_eps_min_m", 0.008)),
                        eps_max=float(self.cfg.get("dbscan_eps_max_m", 0.06)),
                        min_samples=min_samples,
                        min_keep_points=int(self.cfg.get("tiny_far_min_cluster_points", 3) if tiny_far else self.cfg.get("dbscan_min_keep_points", 4)),
                    )
                else:
                    db_keep, db_applied = dbscan_main_cluster_keep_mask(
                        filtered_points,
                        eps=float(self.cfg.get("dbscan_eps_m", 0.03)),
                        min_samples=int(self.cfg.get("dbscan_min_samples", 8)),
                        min_keep_ratio=float(self.cfg.get("dbscan_min_keep_ratio", 0.35)),
                        pre_voxel_size=float(self.cfg.get("dbscan_pre_voxel_m", 0.01)),
                    )
                    db_debug = {
                        "cluster_count": 1,
                        "chosen_cluster_size": int(db_keep.sum()),
                        "chosen_cluster_ratio": float(db_keep.mean()) if db_keep.size else 0.0,
                    }
                filtered_points = filtered_points[db_keep]
                filtered_v = filtered_v[db_keep]
                filtered_u = filtered_u[db_keep]
            else:
                db_applied = False
                db_debug = {"cluster_count": 1, "chosen_cluster_size": int(filtered_points.shape[0]), "chosen_cluster_ratio": 1.0}
            min_points = int(self.cfg.get("min_points", {}).get(det.role, 1))
            forced_geometry_status: str | None = None
            debug_min_points = int(self.cfg.get("debug_min_points", min_points))
            if filtered_points.shape[0] < min_points:
                if filtered_points.shape[0] < max(1, debug_min_points):
                    report.rejects.append(
                        LiftingReject(det.det_id, det.label, det.role, "too_few_points", raw_points, int(filtered_points.shape[0]))
                    )
                    continue
                forced_geometry_status = "low_quality"
            colors = frame.rgb[filtered_v, filtered_u].astype(np.float32)
            voxel = float(self.cfg.get("voxel_size_m", {}).get(det.role, 0.0))
            if tiny_far:
                voxel = min(voxel, float(self.cfg.get("tiny_far_voxel_size_m", 0.0015)))
            points_camera, colors = voxel_downsample_points_colors(filtered_points, colors, voxel)
            if points_camera.shape[0] < min_points:
                if points_camera.shape[0] < max(1, debug_min_points):
                    report.rejects.append(
                        LiftingReject(det.det_id, det.label, det.role, "too_few_points_after_voxel", raw_points, int(points_camera.shape[0]))
                    )
                    continue
                forced_geometry_status = "low_quality"
            ones = np.ones((points_camera.shape[0], 1), dtype=np.float32)
            points_world = (np.concatenate([points_camera, ones], axis=1) @ frame.T_c2w.T)[:, :3].astype(np.float32)
            association_ones = np.ones(
                (association_points_camera.shape[0], 1), dtype=np.float32
            )
            association_points_world = (
                np.concatenate(
                    [association_points_camera, association_ones], axis=1
                )
                @ frame.T_c2w.T
            )[:, :3].astype(np.float32)
            association_bbox_min_world = np.percentile(
                association_points_world, 2.0, axis=0
            ).astype(np.float32)
            association_bbox_max_world = np.percentile(
                association_points_world, 98.0, axis=0
            ).astype(np.float32)
            centroid_camera = points_camera.mean(axis=0)
            centroid_world = points_world.mean(axis=0)
            bmin_c = points_camera.min(axis=0)
            bmax_c = points_camera.max(axis=0)
            bmin_w = points_world.min(axis=0)
            bmax_w = points_world.max(axis=0)
            depth_vals = points_camera[:, 2]
            depth_mad = robust_mad(depth_vals)
            depth_iqr = robust_iqr(depth_vals)
            mask_quality = float(mask.sum()) / max(1.0, float(raw_mask.sum()))
            cluster_ratio = float(db_debug.get("chosen_cluster_ratio", points_camera.shape[0] / max(1, raw_points)))
            parent_residual = depth_residual
            box_touches_border = self._box_touches_border(det.box_xyxy, frame.rgb.shape[0], frame.rgb.shape[1])
            geometry_quality, geometry_status = score_observation_quality(
                detection_score=det.score,
                mask_quality=mask_quality,
                box_touches_border=box_touches_border,
                num_points=int(points_camera.shape[0]),
                min_points=min_points,
                depth_mad=depth_mad,
                depth_iqr=depth_iqr,
                cluster_ratio=cluster_ratio,
                parent_residual=parent_residual,
                mask_area_ratio=mask_area_ratio,
                tiny_far=tiny_far,
                cfg=self.cfg,
            )
            if forced_geometry_status is not None:
                geometry_status = forced_geometry_status
                geometry_quality = min(geometry_quality, float(self.cfg.get("low_quality_cap", 0.30)))
            valid_mask_final = np.zeros_like(mask, dtype=bool)
            valid_mask_final[filtered_v, filtered_u] = True
            hist = hsv_histogram(frame.rgb, valid_mask_final, tuple(self.appearance_cfg.get("bins", [12, 12, 12])))
            if not np.any(hist):
                report.warnings.append(f"{frame.frame_key}:{det.det_id}:empty_appearance_hist")
            obs = Observation3D(
                    observation_id=f"{frame.frame_key}:{det.det_id}",
                    frame_idx=frame.frame_idx,
                    source_index=frame.source_index,
                    frame_key=frame.frame_key,
                    det_id=det.det_id,
                    label=det.label,
                    role=det.role,
                    score=det.score,
                    box_xyxy=det.box_xyxy,
                    mask_area=int(mask.sum()),
                    box_touches_border=box_touches_border,
                    points_camera=points_camera,
                    points_world=points_world,
                    colors_rgb=colors,
                    centroid_camera=centroid_camera,
                    centroid_world=centroid_world,
                    bbox_min_camera=bmin_c,
                    bbox_max_camera=bmax_c,
                    bbox_min_world=bmin_w,
                    bbox_max_world=bmax_w,
                    bbox_diag_world=float(np.linalg.norm(np.maximum(bmax_w - bmin_w, 0.0))),
                    appearance_hist=hist,
                    semantic_feature=sem[det.label],
                    num_raw_depth_points=raw_points,
                    num_depth_filtered_points=int(keep.sum()),
                    num_cluster_points=int(points_camera.shape[0]),
                    dbscan_applied=bool(db_applied),
                    depth_median=float(np.median(depth_vals)),
                    depth_min=float(np.min(depth_vals)),
                    depth_max=float(np.max(depth_vals)),
                    allowed_parent_labels=allowed_map.get(det.det_id, set()),
                    local_parent_det_ids=parent_map.get(det.det_id, []),
                    association_points_world=association_points_world,
                    association_bbox_min_world=association_bbox_min_world,
                    association_bbox_max_world=association_bbox_max_world,
                    association_bbox_diag_world=float(
                        np.linalg.norm(
                            np.maximum(
                                association_bbox_max_world
                                - association_bbox_min_world,
                                0.0,
                            )
                        )
                    ),
                    depth_seed_m=depth_seed,
                    depth_mad_m=depth_mad,
                    depth_iqr_m=depth_iqr,
                    depth_cluster_count=int(db_debug.get("cluster_count", len(modes))),
                    chosen_cluster_size=int(db_debug.get("chosen_cluster_size", points_camera.shape[0])),
                    chosen_cluster_ratio=cluster_ratio,
                    chosen_cluster_depth_median=float(chosen_mode.median),
                    parent_depth_used=parent_depth is not None,
                    parent_depth_median=parent_depth,
                    parent_depth_mad=parent_mad,
                    parent_depth_residual=parent_residual,
                    extraction_method="component_seed_parent_depth_mode_adaptive_dbscan",
                    geometry_quality=geometry_quality,
                    geometry_status=geometry_status,
                    geometry_is_imputed=False,
                    mask_quality=mask_quality,
                    visible_pixel_count=int(valid_mask_final.sum()),
                    effective_voxel_size_m=voxel,
                    mask_area_ratio=mask_area_ratio,
                    tiny_far=tiny_far,
                    depth_modes=[m.to_json() for m in modes],
                    cross_role_duplicate_det_ids=sorted(
                        duplicate_carriers_by_unit.get(int(det.det_id), [])
                    ),
                )
            observations.append(obs)
            report.quality.append(
                {
                    "frame_key": frame.frame_key,
                    "det_id": int(det.det_id),
                    "label": det.label,
                    "role": det.role,
                    "depth_modes": [m.to_json() for m in modes],
                    "chosen_mode": chosen_mode.to_json(),
                    "parent_depth": parent_prior,
                    "component": component_debug,
                    "dbscan": db_debug,
                    "geometry_quality": geometry_quality,
                    "geometry_status": geometry_status,
                    "tiny_far": tiny_far,
                    "mask_quality": mask_quality,
                }
            )
        self.last_report = report
        return observations

    @staticmethod
    def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        limit = max(1, int(max_points))
        if pts.shape[0] <= limit:
            return pts.copy()
        indices = np.linspace(0, pts.shape[0] - 1, limit).astype(np.int64)
        return pts[indices].copy()

    @staticmethod
    def _box_iou(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return float(inter / max(area_a + area_b - inter, 1e-9))

    @classmethod
    def _overlapping_carriers(
        cls,
        result2d: Frame2DResult,
        *,
        iou_threshold: float,
    ) -> set[int]:
        """Return C detections that duplicate a U detection in 2D.

        This is intentionally role-based and label-agnostic.  The U
        interpretation wins exactly when the requested box-IoU criterion is
        met, before either interpretation can create a 3D map node.
        """

        return set(
            cls._overlapping_carrier_witnesses(
                result2d, iou_threshold=iou_threshold
            )
        )

    @classmethod
    def _overlapping_carrier_witnesses(
        cls,
        result2d: Frame2DResult,
        *,
        iou_threshold: float,
    ) -> dict[int, list[int]]:
        """Map each duplicate C detection to the U detections witnessing it."""

        carriers = [det for det in result2d.detections if det.role == "C"]
        units = [det for det in result2d.detections if det.role == "U"]
        threshold = float(iou_threshold)
        return {
            int(carrier.det_id): sorted(
                int(unit.det_id)
                for unit in units
                if cls._box_iou(carrier.box_xyxy, unit.box_xyxy) >= threshold
            )
            for carrier in carriers
            if any(
                cls._box_iou(carrier.box_xyxy, unit.box_xyxy) >= threshold
                for unit in units
            )
        }

    @staticmethod
    def _box_touches_border(box: list[float], h: int, w: int, margin: int = 2) -> bool:
        x1, y1, x2, y2 = [float(v) for v in box]
        return bool(x1 <= margin or y1 <= margin or x2 >= w - 1 - margin or y2 >= h - 1 - margin)

    @staticmethod
    def _parent_map(result: Frame2DResult) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        selected = [r for r in result.local_relations if r.selected]
        rels = selected if selected else [r for r in result.local_relations if r.pass_threshold]
        for rel in rels:
            out.setdefault(int(rel.child_det_id), []).append(int(rel.parent_det_id))
        return out

    @staticmethod
    def _allowed_parent_map(result: Frame2DResult) -> dict[int, set[str]]:
        out: dict[int, set[str]] = {}
        for det in result.detections:
            if det.role == "U":
                out[det.det_id] = set(result.u_to_allowed_parents.get(det.label, []))
            elif det.role == "C":
                out[det.det_id] = set(result.c_to_allowed_parents.get(det.label, []))
            else:
                out[det.det_id] = set()
        return out

    @staticmethod
    def _parent_depth_prior(parent_det_ids: list[int], observations: list[Observation3D]) -> dict[str, float | bool | None]:
        by_det = {int(o.det_id): o for o in observations}
        vals = []
        mads = []
        for det_id in parent_det_ids:
            parent = by_det.get(int(det_id))
            if parent is None or parent.geometry_status not in {"reliable", "provisional"}:
                continue
            vals.append(float(parent.depth_median))
            mads.append(float(parent.depth_mad_m or 0.0))
        if not vals:
            return {"used": False, "median": None, "mad": None}
        return {"used": True, "median": float(np.median(vals)), "mad": float(np.median(mads))}
