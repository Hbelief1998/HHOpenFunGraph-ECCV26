from __future__ import annotations

from pathlib import Path
from typing import Any

from .fungraph3d import FunGraph3DSequence
from .rgb_folder import RGBFolderSequence
from .scenefun3d import SceneFun3DSequence


def create_sequence(dataset: str, **kwargs: Any):
    name = dataset.lower()
    if name == "rgb_folder":
        rgb_dir = kwargs.pop("rgb_dir", None) or kwargs.pop("dataset_root", None)
        return RGBFolderSequence(rgb_dir=Path(rgb_dir), start=kwargs.get("start", 0), end=kwargs.get("end", -1), stride=kwargs.get("stride", 1))
    common = {
        "dataset_root": kwargs["dataset_root"],
        "sequence": kwargs["sequence"],
        "config_path": kwargs.get("config_path") or kwargs.get("dataset_config"),
        "start": kwargs.get("start", 0),
        "end": kwargs.get("end", -1),
        "stride": kwargs.get("stride", 1),
        "desired_height": kwargs.get("desired_height"),
        "desired_width": kwargs.get("desired_width"),
        "load_depth": kwargs.get("load_depth", True),
        "load_pose": kwargs.get("load_pose", True),
    }
    if name == "fungraph3d":
        return FunGraph3DSequence(**common)
    if name == "scenefun3d":
        return SceneFun3DSequence(**common)
    raise ValueError(f"Unsupported dataset: {dataset}")
