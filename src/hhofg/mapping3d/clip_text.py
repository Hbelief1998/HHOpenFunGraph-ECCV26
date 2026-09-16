from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from hhofg.core.serialization import write_json_atomic

from .features import feature_sha, l2_normalize, normalize_label


class ClipTextEncoder:
    def __init__(
        self,
        *,
        arch: str = "ViT-H-14",
        pretrained: str = "laion2b_s32b_b79k",
        device: str = "cuda",
        cache_dir: str | Path = ".cache/clip_text",
        model_cache_dir: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        batch_size: int = 64,
    ) -> None:
        self.arch = arch
        self.pretrained = pretrained
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.model_cache_dir = None if model_cache_dir is None else Path(model_cache_dir)
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self.batch_size = int(batch_size)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, np.ndarray] = {}
        self._model = None
        self._tokenizer = None
        self.feature_dim: int | None = None
        self._load_disk_cache()

    def _cache_path(self) -> Path:
        safe = f"{self.arch}_{self.pretrained}".replace("/", "_")
        return self.cache_dir / f"{safe}.npz"

    def _meta_path(self) -> Path:
        safe = f"{self.arch}_{self.pretrained}".replace("/", "_")
        return self.cache_dir / f"{safe}.json"

    def _load_disk_cache(self) -> None:
        path = self._cache_path()
        if not path.is_file():
            return
        with np.load(path) as data:
            labels = [str(v) for v in data["labels"].tolist()]
            feats = data["features"].astype(np.float32)
        for label, feat in zip(labels, feats):
            self._cache[label] = l2_normalize(feat)
        if feats.size:
            self.feature_dim = int(feats.shape[1])

    def _save_disk_cache(self) -> None:
        labels = sorted(self._cache)
        features = np.stack([self._cache[l] for l in labels], axis=0).astype(np.float32) if labels else np.zeros((0, 0), dtype=np.float32)
        np.savez_compressed(self._cache_path(), labels=np.asarray(labels), features=features)
        meta = [
            {
                "arch": self.arch,
                "pretrained": self.pretrained,
                "feature_dim": int(features.shape[1]) if features.ndim == 2 and features.shape[1] else self.feature_dim,
                "label": label,
                "feature_sha": feature_sha(self._cache[label]),
            }
            for label in labels
        ]
        write_json_atomic(self._meta_path(), meta)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import open_clip
        except Exception as exc:
            raise RuntimeError("open_clip and torch are required for real CLIP text features") from exc
        kwargs = {}
        if self.model_cache_dir is not None:
            kwargs["cache_dir"] = str(self.model_cache_dir)
        if self.checkpoint_path is not None and self.checkpoint_path.suffix == ".safetensors":
            model, _, _ = open_clip.create_model_and_transforms(self.arch, pretrained=None, device=self.device, **kwargs)
            try:
                from safetensors.torch import load_file
                from open_clip.factory import convert_to_custom_text_state_dict, resize_pos_embed, resize_text_pos_embed
            except Exception as exc:
                raise RuntimeError("safetensors is required to load the configured CLIP checkpoint") from exc
            state_dict = load_file(str(self.checkpoint_path), device=str(self.device))
            if "positional_embedding" in state_dict and not hasattr(model, "positional_embedding"):
                state_dict = convert_to_custom_text_state_dict(state_dict)
            if "logit_bias" not in state_dict and getattr(model, "logit_bias", None) is not None:
                state_dict["logit_bias"] = torch.zeros_like(state_dict["logit_scale"])
            state_dict.pop("text.transformer.embeddings.position_ids", None)
            resize_pos_embed(state_dict, model)
            resize_text_pos_embed(state_dict, model)
            model.load_state_dict(state_dict, strict=True)
        else:
            pretrained = str(self.checkpoint_path) if self.checkpoint_path is not None else self.pretrained
            model, _, _ = open_clip.create_model_and_transforms(self.arch, pretrained=pretrained, device=self.device, **kwargs)
        model.eval()
        self._model = model
        self._tokenizer = open_clip.get_tokenizer(self.arch)
        self._torch = torch

    def encode_labels(self, labels: Iterable[str]) -> dict[str, np.ndarray]:
        wanted = [normalize_label(l) for l in labels]
        missing = sorted({l for l in wanted if l and l not in self._cache})
        if missing:
            self._ensure_model()
            assert self._model is not None and self._tokenizer is not None
            torch = self._torch
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                tokens = self._tokenizer(batch).to(self.device)
                with torch.no_grad():
                    feats = self._model.encode_text(tokens)
                    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                arr = feats.detach().cpu().numpy().astype(np.float32)
                for label, feat in zip(batch, arr):
                    self._cache[label] = l2_normalize(feat)
                    self.feature_dim = int(feat.shape[0])
            self._save_disk_cache()
        return {label: self._cache[normalize_label(label)] for label in labels}

    def encode_label(self, label: str) -> np.ndarray:
        return self.encode_labels([label])[label]


class MockClipTextEncoder:
    def __init__(self, dim: int = 8) -> None:
        self.dim = int(dim)
        self.calls: list[str] = []
        self._cache: dict[str, np.ndarray] = {}

    def encode_labels(self, labels: Iterable[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for raw in labels:
            label = normalize_label(raw)
            if label not in self._cache:
                self.calls.append(label)
                seed = sum((i + 1) * ord(c) for i, c in enumerate(label))
                vec = np.array([((seed + 37 * i) % 101) / 100.0 for i in range(self.dim)], dtype=np.float32)
                self._cache[label] = l2_normalize(vec)
            out[raw] = self._cache[label]
        return out

    def encode_label(self, label: str) -> np.ndarray:
        return self.encode_labels([label])[label]
