from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hhofg.core.serialization import load_frame2d_result
from hhofg.core.types import Frame2DResult, FrameRecord


@dataclass
class FrontendFrameData:
    result: Frame2DResult
    boxes_raw: np.ndarray
    scores: np.ndarray
    masks_raw: np.ndarray
    json_path: Path
    npz_path: Path


class FrontendRunReader:
    def __init__(self, frontend_run_dir: str | Path, *, dataset: str | None = None, sequence: str | None = None) -> None:
        self.frontend_run_dir = Path(frontend_run_dir)
        if not self.frontend_run_dir.is_dir():
            raise FileNotFoundError(f"frontend run not found: {self.frontend_run_dir}")
        self.run_config = self._load_json(self.frontend_run_dir / "run_config.json")
        data_cfg = self.run_config.get("data", {})
        if dataset is not None and data_cfg.get("dataset") not in {None, dataset}:
            raise ValueError(f"frontend dataset mismatch: {data_cfg.get('dataset')} != {dataset}")
        if sequence is not None and data_cfg.get("sequence") not in {None, sequence}:
            raise ValueError(f"frontend sequence mismatch: {data_cfg.get('sequence')} != {sequence}")
        self.dataset = data_cfg.get("dataset")
        self.sequence = data_cfg.get("sequence")
        self.desired_height = data_cfg.get("desired_height")
        self.desired_width = data_cfg.get("desired_width")
        self.dataset_config = data_cfg.get("dataset_config")
        self.coordinate_space = "raw"
        self.frontend_commit = self.run_config.get("git_commit") or self.run_config.get("frontend_commit")
        self.global_atlas = self._load_json(self.frontend_run_dir / "global_atlas.json")
        self.frames: dict[str, FrontendFrameData] = {}
        self._scan_frames()

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _scan_frames(self) -> None:
        frame_dir = self.frontend_run_dir / "frames"
        if not frame_dir.is_dir():
            raise FileNotFoundError(f"frontend frames directory not found: {frame_dir}")
        for json_path in sorted(frame_dir.glob("*.json")):
            npz_path = json_path.with_suffix(".npz")
            if not npz_path.is_file():
                raise FileNotFoundError(f"missing npz for {json_path}")
            result, boxes, scores, masks = load_frame2d_result(json_path, npz_path)
            key = result.frame_key
            if key in self.frames:
                raise ValueError(f"duplicate frontend frame_key: {key}")
            if result.coordinate_space != "raw":
                raise ValueError(f"frontend frame {key} coordinate_space must be raw, got {result.coordinate_space}")
            n = len(result.detections)
            if boxes.shape != (n, 4):
                raise ValueError(f"{key}: boxes count mismatch")
            if scores.shape != (n,):
                raise ValueError(f"{key}: scores count mismatch")
            if masks.ndim != 3 or masks.shape[0] != n:
                raise ValueError(f"{key}: masks count mismatch")
            if masks.shape[1:] != (result.image_height, result.image_width):
                raise ValueError(f"{key}: mask shape does not match raw image dimensions")
            self.frames[key] = FrontendFrameData(result, boxes, scores, masks, json_path, npz_path)

    def __len__(self) -> int:
        return len(self.frames)

    def keys(self) -> list[str]:
        return sorted(self.frames)

    def get(self, frame_key: str) -> FrontendFrameData | None:
        return self.frames.get(frame_key)

    def require_for_frame(self, frame: FrameRecord) -> FrontendFrameData:
        data = self.frames.get(frame.frame_key)
        if data is None:
            raise KeyError(f"frontend run has no frame_key {frame.frame_key}")
        self.validate_frame_alignment(frame, data)
        return data

    @staticmethod
    def validate_frame_alignment(frame: FrameRecord, data: FrontendFrameData) -> None:
        result = data.result
        if frame.frame_key != result.frame_key:
            raise ValueError(f"frame_key mismatch: {frame.frame_key} != {result.frame_key}")
        if frame.depth_m is None:
            raise ValueError(f"{frame.frame_key}: depth_m is required")
        if frame.camera is None:
            raise ValueError(f"{frame.frame_key}: camera is required")
        if frame.T_c2w is None:
            raise ValueError(f"{frame.frame_key}: T_c2w is required")
        h, w = frame.rgb.shape[:2]
        if frame.depth_m.shape != (h, w) or data.masks_raw.shape[1:] != (h, w):
            raise ValueError(
                f"{frame.frame_key}: rgb/depth/mask sizes differ: rgb={(h, w)} depth={frame.depth_m.shape} masks={data.masks_raw.shape[1:]}"
            )
        if (frame.camera.height, frame.camera.width) != (h, w):
            raise ValueError(f"{frame.frame_key}: camera width/height do not match rgb")
        if (result.image_height, result.image_width) != (h, w):
            raise ValueError(f"{frame.frame_key}: Frame2DResult size does not match rgb")
        if result.coordinate_space != "raw":
            raise ValueError(f"{frame.frame_key}: coordinate_space must be raw")
