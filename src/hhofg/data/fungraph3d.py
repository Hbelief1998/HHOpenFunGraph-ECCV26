from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hhofg.core.types import CameraModel, FrameRecord
from hhofg.data.colmap_io import ColmapCamera, read_model_text
from hhofg.data.config import load_yaml_config
from hhofg.data.rgb_folder import natural_key, read_rgb


def _camera_from_config(
    cfg: dict,
    width: int | None = None,
    height: int | None = None,
    *,
    colmap_camera: ColmapCamera | None = None,
) -> CameraModel:
    cp = cfg["camera_params"]
    orig_w = int(colmap_camera.width if colmap_camera is not None else cp["image_width"])
    orig_h = int(colmap_camera.height if colmap_camera is not None else cp["image_height"])
    out_w = int(width or orig_w)
    out_h = int(height or orig_h)
    sx = out_w / float(orig_w)
    sy = out_h / float(orig_h)
    source_K = (
        colmap_camera.K
        if colmap_camera is not None
        else np.array(
            [
                [float(cp["fx"]), 0.0, float(cp["cx"])],
                [0.0, float(cp["fy"]), float(cp["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    K = np.array(
        [
            [float(source_K[0, 0]) * sx, 0.0, float(source_K[0, 2]) * sx],
            [0.0, float(source_K[1, 1]) * sy, float(source_K[1, 2]) * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return CameraModel(K=K, width=out_w, height=out_h, depth_scale=float(cp["png_depth_scale"]), camera_axis="Up")


def _resize_rgb_depth(rgb: np.ndarray, depth_raw: np.ndarray | None, desired_height: int | None, desired_width: int | None):
    h, w = rgb.shape[:2]
    out_h = int(desired_height or h)
    out_w = int(desired_width or w)
    if (h, w) != (out_h, out_w):
        rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    if depth_raw is not None and depth_raw.shape[:2] != (out_h, out_w):
        depth_raw = cv2.resize(depth_raw, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return rgb, depth_raw


class FunGraph3DSequence:
    def __init__(
        self,
        dataset_root: str | Path,
        sequence: str,
        config_path: str | Path,
        start: int = 0,
        end: int = -1,
        stride: int = 1,
        desired_height: int | None = None,
        desired_width: int | None = None,
        load_depth: bool = True,
        load_pose: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.sequence = sequence
        self.seq_dir = self.dataset_root / sequence
        self.cfg = load_yaml_config(config_path)
        self.desired_height = desired_height
        self.desired_width = desired_width
        self.load_depth = bool(load_depth)
        self.load_pose = bool(load_pose)
        if not self.seq_dir.is_dir():
            raise FileNotFoundError(f"FunGraph3D sequence not found: {self.seq_dir}")

        rgb_dir = self.seq_dir / "rgb"
        depth_dir = self.seq_dir / "depth"
        rgb_paths = sorted([p for p in rgb_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}], key=natural_key)
        depth_by_key = {p.stem: p for p in depth_dir.glob("*.png")} if depth_dir.is_dir() else {}
        if self.load_depth:
            rgb_paths = [p for p in rgb_paths if p.stem in depth_by_key]
        self.depth_by_key = depth_by_key

        self.pose_by_key: dict[str, np.ndarray] = {}
        self.colmap_camera_by_key: dict[str, ColmapCamera] = {}
        if self.load_pose:
            cameras, images = read_model_text(self.seq_dir)
            pose_scale = float(self.cfg.get("pose_scale", 1.0))
            for image in images.values():
                key = Path(image.name).stem
                T = image.T_c2w.astype(np.float32)
                T[:3, 3] *= pose_scale
                self.pose_by_key[key] = T
                camera = cameras.get(int(image.camera_id))
                if camera is None:
                    raise KeyError(
                        f"COLMAP image {image.name!r} references missing camera "
                        f"{image.camera_id}"
                    )
                self.colmap_camera_by_key[key] = camera
            rgb_paths = [p for p in rgb_paths if p.stem in self.pose_by_key]

        if start < 0 or stride <= 0:
            raise ValueError("start must be >= 0 and stride must be > 0")
        stop = None if end == -1 else int(end)
        self.all_rgb_paths = rgb_paths
        self.rgb_paths = rgb_paths[int(start) : stop : int(stride)]
        self.source_indices = list(range(len(rgb_paths)))[int(start) : stop : int(stride)]

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, index: int) -> FrameRecord:
        rgb_path = self.rgb_paths[index]
        rgb = read_rgb(rgb_path)
        depth_path = self.depth_by_key.get(rgb_path.stem) if self.load_depth else None
        depth_raw = None
        if depth_path is not None:
            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise FileNotFoundError(f"Failed to read depth: {depth_path}")
        rgb, depth_raw = _resize_rgb_depth(rgb, depth_raw, self.desired_height, self.desired_width)
        camera = _camera_from_config(
            self.cfg,
            width=rgb.shape[1],
            height=rgb.shape[0],
            colmap_camera=self.colmap_camera_by_key.get(rgb_path.stem),
        )
        depth_m = None
        if depth_raw is not None:
            depth_m = depth_raw.astype(np.float32) / camera.depth_scale
            depth_m[~np.isfinite(depth_m)] = 0.0
            depth_m[depth_m < 0] = 0.0
        return FrameRecord(
            frame_idx=index,
            frame_key=rgb_path.stem,
            rgb_path=rgb_path,
            rgb=rgb,
            depth_path=depth_path,
            depth_m=depth_m,
            camera=camera,
            T_c2w=self.pose_by_key.get(rgb_path.stem) if self.load_pose else None,
            source_index=self.source_indices[index],
            metadata={
                "dataset": "fungraph3d",
                "sequence": self.sequence,
                "pose_source": "dataset_colmap_images_txt"
                if self.load_pose
                else "none",
                "intrinsics_source": "dataset_colmap_cameras_txt"
                if rgb_path.stem in self.colmap_camera_by_key
                else "yaml_fallback",
            },
        )
