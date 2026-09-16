from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import Detection2D, Frame2DResult, LocalRelation2D, SCHEMA_VERSION
from .validation import validate_frame2d_arrays


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    return obj


def write_json_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def save_frame2d_result(
    result: Frame2DResult,
    boxes: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    json_path: str | Path,
    npz_path: str | Path,
    *,
    boxes_inference: np.ndarray | None = None,
    masks_inference: np.ndarray | None = None,
) -> None:
    boxes, scores, masks = validate_frame2d_arrays(result, boxes, scores, masks)
    if boxes_inference is None:
        boxes_inference = boxes
    if masks_inference is None:
        masks_inference = masks
    boxes_inference = np.asarray(boxes_inference, dtype=np.float32)
    masks_inference = np.asarray(masks_inference, dtype=bool)
    n = len(result.detections)
    if boxes_inference.shape != (n, 4):
        raise ValueError(f"boxes_inference must have shape ({n}, 4), got {boxes_inference.shape}")
    if masks_inference.ndim != 3 or masks_inference.shape[0] != n:
        raise ValueError(f"masks_inference must have shape ({n}, H, W), got {masks_inference.shape}")
    if masks_inference.shape[1:] != (result.inference_image_height, result.inference_image_width):
        raise ValueError("masks_inference height/width must match result inference image dimensions")
    write_json_atomic(json_path, result)
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{npz_path.name}.", suffix=".tmp", dir=str(npz_path.parent))
    os.close(fd)
    try:
        with open(tmp_name, "wb") as f:
            np.savez_compressed(
                f,
                boxes=boxes.astype(np.float32),
                boxes_raw=boxes.astype(np.float32),
                boxes_inference=boxes_inference.astype(np.float32),
                scores=scores.astype(np.float32),
                masks=masks.astype(bool),
                masks_raw=masks.astype(bool),
                masks_inference=masks_inference.astype(bool),
            )
        os.replace(tmp_name, npz_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_frame2d_result(json_path: str | Path, npz_path: str | Path) -> tuple[Frame2DResult, np.ndarray, np.ndarray, np.ndarray]:
    with Path(json_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["detections"] = [Detection2D(**d) for d in payload.get("detections", [])]
    legacy_final = payload.get("local_relations", [])
    payload["local_relations"] = [LocalRelation2D(**r) for r in legacy_final]
    if "local_relations_final" in payload:
        payload["local_relations_final"] = [
            LocalRelation2D(**r) for r in payload.get("local_relations_final", [])
        ]
    if "local_relation_candidates_raw" in payload:
        payload["local_relation_candidates_raw"] = [
            LocalRelation2D(**r) for r in payload.get("local_relation_candidates_raw", [])
        ]
    # Frame2DResult performs the v1/v2 raw-stage reconstruction and creates the
    # backward-compatible local_relations alias.
    payload["schema_version"] = int(payload.get("schema_version", 1))
    result = Frame2DResult(**payload)
    result.schema_version = SCHEMA_VERSION
    with np.load(npz_path) as data:
        boxes = data["boxes_raw"].astype(np.float32) if "boxes_raw" in data else data["boxes"].astype(np.float32)
        scores = data["scores"].astype(np.float32)
        masks = data["masks_raw"].astype(bool) if "masks_raw" in data else data["masks"].astype(bool)
    boxes, scores, masks = validate_frame2d_arrays(result, boxes, scores, masks)
    return result, boxes, scores, masks
