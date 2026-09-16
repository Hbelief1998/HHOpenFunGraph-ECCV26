from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hhofg.core.serialization import write_json_atomic

from .types import AssociationPairScore, FrameAssociationResult, MapNode3D, MappedFrameEdgeEvidence, Observation3D


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def save_observations(frame_key: str, observations: list[Observation3D], out_dir: str | Path, report: Any | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        out_dir / f"{frame_key}.json",
        {
            "frame_key": frame_key,
            "observations": [
                {
                    k: v
                    for k, v in to_jsonable(obs).items()
                    if k
                    not in {
                        "points_camera",
                        "points_world",
                        "association_points_world",
                        "colors_rgb",
                        "appearance_hist",
                        "semantic_feature",
                    }
                }
                for obs in observations
            ],
            "lifting_report": to_jsonable(report),
        },
    )
    if report is not None and hasattr(report, "quality"):
        write_json_atomic(
            out_dir / f"{frame_key}_quality.json",
            {
                "frame_key": frame_key,
                "quality": to_jsonable(getattr(report, "quality")),
                "rejects": to_jsonable(getattr(report, "rejects", [])),
                "warnings": to_jsonable(getattr(report, "warnings", [])),
            },
        )
    offsets = [0]
    pts_c = []
    pts_w = []
    cols = []
    for obs in observations:
        offsets.append(offsets[-1] + obs.points_world.shape[0])
        pts_c.append(obs.points_camera.astype(np.float32))
        pts_w.append(obs.points_world.astype(np.float32))
        cols.append(obs.colors_rgb.astype(np.float32))
    np.savez_compressed(
        out_dir / f"{frame_key}.npz",
        valid_det_ids=np.asarray([obs.det_id for obs in observations], dtype=np.int32),
        point_offsets=np.asarray(offsets, dtype=np.int64),
        points_camera=np.concatenate(pts_c, axis=0).astype(np.float32) if pts_c else np.zeros((0, 3), dtype=np.float32),
        points_world=np.concatenate(pts_w, axis=0).astype(np.float32) if pts_w else np.zeros((0, 3), dtype=np.float32),
        colors_rgb=np.concatenate(cols, axis=0).astype(np.float32) if cols else np.zeros((0, 3), dtype=np.float32),
        appearance_hist=np.stack([obs.appearance_hist for obs in observations], axis=0).astype(np.float32)
        if observations
        else np.zeros((0, 0), dtype=np.float32),
        semantic_features=np.stack([obs.semantic_feature for obs in observations], axis=0).astype(np.float32)
        if observations
        else np.zeros((0, 0), dtype=np.float32),
    )


def save_association(result: FrameAssociationResult, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / f"{result.frame_key}.json", result)


def save_map(nodes: dict[str, MapNode3D], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = [nodes[k] for k in sorted(nodes)]
    write_json_atomic(
        out_dir / "map_nodes.json",
        {
            "nodes": [
                {
                    k: v
                    for k, v in to_jsonable(node).items()
                    if k
                    not in {
                        "points_world",
                        "association_points_world",
                        "colors_rgb",
                        "appearance_hist",
                        "semantic_feature",
                        "anchor_points_world",
                        "core_points_world",
                    }
                }
                for node in ordered
            ],
            "node_ids": [n.node_id for n in ordered],
        },
    )
    offsets = [0]
    pts = []
    cols = []
    for node in ordered:
        offsets.append(offsets[-1] + node.points_world.shape[0])
        pts.append(node.points_world.astype(np.float32))
        cols.append(node.colors_rgb.astype(np.float32))
    np.savez_compressed(
        out_dir / "map_nodes.npz",
        point_offsets=np.asarray(offsets, dtype=np.int64),
        points_world=np.concatenate(pts, axis=0).astype(np.float32) if pts else np.zeros((0, 3), dtype=np.float32),
        colors_rgb=np.concatenate(cols, axis=0).astype(np.float32) if cols else np.zeros((0, 3), dtype=np.float32),
        appearance_hist=np.stack([n.appearance_hist for n in ordered], axis=0).astype(np.float32) if ordered else np.zeros((0, 0), dtype=np.float32),
        semantic_features=np.stack([n.semantic_feature for n in ordered], axis=0).astype(np.float32) if ordered else np.zeros((0, 0), dtype=np.float32),
        centroids=np.stack([n.centroid_world for n in ordered], axis=0).astype(np.float32) if ordered else np.zeros((0, 3), dtype=np.float32),
        bbox_min=np.stack([n.bbox_min_world for n in ordered], axis=0).astype(np.float32) if ordered else np.zeros((0, 3), dtype=np.float32),
        bbox_max=np.stack([n.bbox_max_world for n in ordered], axis=0).astype(np.float32) if ordered else np.zeros((0, 3), dtype=np.float32),
    )


def append_jsonl(path: str | Path, records: list[MappedFrameEdgeEvidence] | list[dict[str, Any]]) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
