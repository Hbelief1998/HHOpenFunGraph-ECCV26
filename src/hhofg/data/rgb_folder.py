from __future__ import annotations

import re
from pathlib import Path

import cv2

from hhofg.core.types import FrameRecord


def natural_key(path: Path) -> list[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def read_rgb(path: str | Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read RGB image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class RGBFolderSequence:
    def __init__(self, rgb_dir: str | Path, start: int = 0, end: int = -1, stride: int = 1) -> None:
        self.rgb_dir = Path(rgb_dir)
        if not self.rgb_dir.is_dir():
            raise FileNotFoundError(f"RGB directory not found: {self.rgb_dir}")
        files = [p for p in self.rgb_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        files = sorted(files, key=natural_key)
        if start < 0 or stride <= 0:
            raise ValueError("start must be >= 0 and stride must be > 0")
        stop = None if end == -1 else int(end)
        self.paths = files[int(start) : stop : int(stride)]
        self.source_indices = list(range(len(files)))[int(start) : stop : int(stride)]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> FrameRecord:
        path = self.paths[index]
        return FrameRecord(
            frame_idx=index,
            frame_key=path.stem,
            rgb_path=path,
            rgb=read_rgb(path),
            depth_path=None,
            depth_m=None,
            camera=None,
            T_c2w=None,
            source_index=self.source_indices[index],
            metadata={"dataset": "rgb_folder"},
        )
