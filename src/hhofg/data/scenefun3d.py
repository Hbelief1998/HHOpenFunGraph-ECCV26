from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hhofg.core.types import CameraModel, FrameRecord
from hhofg.data.config import load_yaml_config
from hhofg.data.pose_interpolation import nearest_or_interpolated_pose
from hhofg.data.rgb_folder import natural_key, read_rgb


def _axis_angle_to_matrix(vec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(vec, dtype=np.float64).reshape(3))
    return R


def traj_line_to_c2w(line: str) -> tuple[str, np.ndarray]:
    toks = line.split()
    if len(toks) != 7:
        raise ValueError(f"SceneFun3D trajectory line must have 7 tokens, got {len(toks)}")
    ts = toks[0]
    w2p = np.eye(4, dtype=np.float64)
    w2p[:3, :3] = _axis_angle_to_matrix(np.asarray([float(t) for t in toks[1:4]], dtype=np.float64))
    w2p[:3, 3] = np.asarray([float(t) for t in toks[4:7]], dtype=np.float64)
    return ts, np.linalg.inv(w2p)


def _read_camera_axis(dataset_root: Path, sequence: str) -> str:
    meta = dataset_root / "metadata.csv"
    if not meta.is_file():
        return "Up"
    parts = sequence.split("/")
    with meta.open("r", encoding="utf-8") as f:
        for line in f:
            cols = [c.strip() for c in line.split(",")]
            if len(cols) >= 3 and all(p in cols for p in parts[:2]):
                return cols[2] or "Up"
            if len(cols) >= 3 and parts[0] in line and (len(parts) < 2 or parts[1] in line):
                return cols[2] or "Up"
    return "Up"


def _camera_from_config(cfg: dict, out_w: int, out_h: int, camera_axis: str, pre_rot_w: int | None = None, pre_rot_h: int | None = None) -> CameraModel:
    cp = cfg["camera_params"]
    orig_w = int(cp["image_width"])
    orig_h = int(cp["image_height"])
    fx, fy, cx, cy = float(cp["fx"]), float(cp["fy"]), float(cp["cx"]), float(cp["cy"])
    if camera_axis == "Left":
        rot_w = orig_h
        rot_h = orig_w
        K = np.array([[fy, 0.0, orig_h - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]], dtype=np.float32)
        sx = out_w / float(rot_w)
        sy = out_h / float(rot_h)
    else:
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        sx = out_w / float(pre_rot_w or orig_w)
        sy = out_h / float(pre_rot_h or orig_h)
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return CameraModel(K=K, width=out_w, height=out_h, depth_scale=float(cp["png_depth_scale"]), camera_axis=camera_axis)


class SceneFun3DSequence:
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
        self.camera_axis = _read_camera_axis(self.dataset_root, sequence)
        if not self.seq_dir.is_dir():
            raise FileNotFoundError(f"SceneFun3D sequence not found: {self.seq_dir}")
        rgb_paths = sorted((self.seq_dir / "wide").glob("*.png"), key=natural_key)
        depth_by_key = {p.stem: p for p in (self.seq_dir / "highres_depth").glob("*.png")}
        if self.load_depth:
            rgb_paths = [p for p in rgb_paths if p.stem in depth_by_key]
        self.depth_by_key = depth_by_key
        self.pose_by_key: dict[str, np.ndarray] = {}
        if self.load_pose:
            poses_by_ts: dict[str, np.ndarray] = {}
            with (self.seq_dir / "lowres_wide.traj").open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    ts, pose = traj_line_to_c2w(line)
                    poses_by_ts[f"{round(float(ts), 3):.3f}"] = pose
            kept = []
            for p in rgb_paths:
                frame_ts = p.stem.split("_")[-1]
                pose = nearest_or_interpolated_pose(frame_ts, poses_by_ts, time_distance_threshold=0.2, use_interpolation=True)
                if pose is None:
                    continue
                if self.camera_axis == "Left":
                    R_z_90 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                    adjusted = np.eye(4, dtype=np.float64)
                    adjusted[:3, :3] = pose[:3, :3] @ R_z_90
                    adjusted[:3, 3] = pose[:3, 3]
                    pose = adjusted
                self.pose_by_key[p.stem] = pose.astype(np.float32)
                kept.append(p)
            rgb_paths = kept
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
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) if depth_path is not None else None
        pre_h, pre_w = rgb.shape[:2]
        out_h = int(self.desired_height or pre_h)
        out_w = int(self.desired_width or pre_w)
        if (pre_h, pre_w) != (out_h, out_w):
            rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if depth_raw is not None:
                depth_raw = cv2.resize(depth_raw, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        if self.camera_axis == "Left":
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
            if depth_raw is not None:
                depth_raw = cv2.rotate(depth_raw, cv2.ROTATE_90_CLOCKWISE)
        camera = _camera_from_config(self.cfg, rgb.shape[1], rgb.shape[0], self.camera_axis, pre_rot_w=out_w, pre_rot_h=out_h)
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
            metadata={"dataset": "scenefun3d", "sequence": self.sequence},
        )
