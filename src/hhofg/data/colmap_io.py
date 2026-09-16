from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def K(self) -> np.ndarray:
        params = self.params.astype(np.float64)
        if self.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif self.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
            fx, fy, cx, cy = params[:4]
        else:
            raise NotImplementedError(f"Unsupported COLMAP camera model: {self.model}")
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


@dataclass(frozen=True)
class ColmapImage:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str

    @property
    def world_to_camera(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = qvec2rotmat(self.qvec)
        T[:3, 3] = self.tvec
        return T

    @property
    def T_c2w(self) -> np.ndarray:
        return np.linalg.inv(self.world_to_camera).astype(np.float32)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    return np.array(
        [
            [
                1 - 2 * q[2] ** 2 - 2 * q[3] ** 2,
                2 * q[1] * q[2] - 2 * q[0] * q[3],
                2 * q[3] * q[1] + 2 * q[0] * q[2],
            ],
            [
                2 * q[1] * q[2] + 2 * q[0] * q[3],
                1 - 2 * q[1] ** 2 - 2 * q[3] ** 2,
                2 * q[2] * q[3] - 2 * q[0] * q[1],
            ],
            [
                2 * q[3] * q[1] - 2 * q[0] * q[2],
                2 * q[2] * q[3] + 2 * q[0] * q[1],
                1 - 2 * q[1] ** 2 - 2 * q[2] ** 2,
            ],
        ],
        dtype=np.float64,
    )


def read_cameras_text(path: str | Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            cam_id = int(elems[0])
            cameras[cam_id] = ColmapCamera(
                id=cam_id,
                model=elems[1],
                width=int(elems[2]),
                height=int(elems[3]),
                params=np.asarray([float(x) for x in elems[4:]], dtype=np.float64),
            )
    return cameras


def read_images_text(path: str | Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            image_id = int(elems[0])
            images[image_id] = ColmapImage(
                id=image_id,
                qvec=np.asarray([float(x) for x in elems[1:5]], dtype=np.float64),
                tvec=np.asarray([float(x) for x in elems[5:8]], dtype=np.float64),
                camera_id=int(elems[8]),
                name=elems[9],
            )
            f.readline()
    return images


def read_model_text(model_dir: str | Path) -> tuple[dict[int, ColmapCamera], dict[int, ColmapImage]]:
    model_dir = Path(model_dir)
    return read_cameras_text(model_dir / "cameras.txt"), read_images_text(model_dir / "images.txt")
