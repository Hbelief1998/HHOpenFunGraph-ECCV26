from __future__ import annotations

# 本模块封装 RAM++ 语义标签模型的加载与推理，提供轻量、可复用的标签接口。

import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


class RamppTagger:
    """Lightweight RAM++ wrapper with lazy imports for spawn compatibility."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        image_size: int = 384,
        vit: str = "swin_l",
        repo: str | None = None,
    ) -> None:
        # Avoid heavy imports at module import time to keep torch.multiprocessing spawn stable.
        candidates = [
            repo,
            os.getenv("RAMPP_REPO"),
            str(Path(__file__).resolve().parents[4] / "third_party" / "recognize-anything"),
        ]
        ram_repo = next((Path(p) for p in candidates if p and Path(p).is_dir()), None)
        if ram_repo is not None and str(ram_repo) not in sys.path:
            sys.path.append(str(ram_repo))

        # 延迟导入重依赖，避免进程启动/序列化问题。
        import torch
        from PIL import Image
        try:
            from ram import get_transform, inference_ram
            from ram.models import ram_plus
        except Exception as exc:
            raise RuntimeError(
                "RAM++ package is not importable. Install/provide recognize-anything separately, "
                "or set rampp.repo/RAMPP_REPO. HHOpenFunGraph must not rely on the Functional-SLAM repository at runtime."
            ) from exc

        ckpt = Path(ckpt_path)
        if not ckpt.is_file():
            raise FileNotFoundError(f"RAM++ checkpoint not found: {ckpt}")

        # 初始化模型与推理所需组件。
        self.device = torch.device(device)
        self._torch = torch
        self._Image = Image
        self._inference = inference_ram
        self._transform = get_transform(image_size=image_size)
        self._model = ram_plus(pretrained=str(ckpt), image_size=image_size, vit=vit)
        self._model.eval()
        self._model = self._model.to(self.device)
        self._image_size = image_size

        print(f"[RAM++] Loaded checkpoint {ckpt} on {self.device}")

    @staticmethod
    def _split_tags(raw_tags: str) -> List[str]:
        # 兼容模型输出为字符串或列表的情况，统一拆分为 tag 列表。
        if raw_tags is None:
            return []
        if isinstance(raw_tags, (list, tuple)):
            # Model API returns a string; still handle lists defensively.
            joined = " | ".join([str(t) for t in raw_tags if t is not None])
            raw_tags = joined
        if not isinstance(raw_tags, str):
            return []
        return [tag.strip() for tag in raw_tags.split("|") if tag.strip()]

    def infer(self, img_rgb_float01: np.ndarray) -> Dict[str, List[str]]:
        """Run RAM++ tagging on an RGB float image in [0, 1]."""
        # 验证输入尺寸必须是 HxWx3。
        np_img = np.asarray(img_rgb_float01)
        if np_img.ndim != 3 or np_img.shape[2] != 3:
            raise ValueError("Expected an HxWx3 RGB image for RAM++ inference")

        # 归一化并转换为 uint8，以适配 PIL。
        np_img = np.clip(np_img, 0.0, 1.0)
        img_uint8 = (np_img * 255.0).round().astype(np.uint8)
        pil_img = self._Image.fromarray(img_uint8)
        tensor = self._transform(pil_img).unsqueeze(0).to(self.device)

        # 关闭梯度以减少显存开销并加速推理。
        with self._torch.no_grad():
            tags_en_raw, tags_zh_raw = self._inference(tensor, self._model)

        return {
            "tags_en": self._split_tags(tags_en_raw),
            "tags_zh": self._split_tags(tags_zh_raw),
        }
