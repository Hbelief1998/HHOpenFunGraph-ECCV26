from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import hhofg.frontend2d.fs2d.sam3_two_stage_runtime as _fs2d_runtime
from hhofg.frontend2d.fs2d.sam3_two_stage_runtime import *  # noqa: F401,F403
from hhofg.frontend2d.fs2d.sam3_two_stage_runtime import Sam3TwoStageRuntime as _Fs2dSam3TwoStageRuntime


def _resolve_sam3_repo(sam3_repo: str | None) -> Path | None:
    candidates = [
        sam3_repo,
        os.getenv("SAM3_REPO"),
        str(Path(__file__).resolve().parents[4] / "third_party" / "sam3"),
    ]
    for repo in candidates:
        if repo and Path(repo).is_dir():
            return Path(repo)
    return None


def _import_sam3(sam3_repo: str | None):
    repo_path = _resolve_sam3_repo(sam3_repo)
    if repo_path is not None and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    try:
        model_builder = importlib.import_module("sam3.model_builder")
        processor_mod = importlib.import_module("sam3.model.sam3_image_processor")
    except Exception as exc:
        raise RuntimeError(
            "SAM3 package is not importable. Install SAM3 as a dependency, "
            "or set sam3.repo/SAM3_REPO to an external SAM3 checkout. "
            "HHOpenFunGraph must not rely on the Functional-SLAM repository at runtime."
        ) from exc
    return model_builder.build_sam3_image_model, processor_mod.Sam3Processor, repo_path


class Sam3TwoStageRuntime(_Fs2dSam3TwoStageRuntime):
    def __init__(self, *args, sam3_repo: str | None = None, **kwargs):
        builder, processor_cls, repo_path = _import_sam3(sam3_repo)
        _fs2d_runtime.build_sam3_image_model = builder
        _fs2d_runtime.Sam3Processor = processor_cls
        super().__init__(*args, **kwargs)
        self.sam3_repo = str(repo_path) if repo_path is not None else "pythonpath"
