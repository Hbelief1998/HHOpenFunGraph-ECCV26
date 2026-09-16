from __future__ import annotations

from typing import Any

from .cache import Edge2DScoreCache
from .types import Edge2DCandidate, Edge2DScore
from .visual_prompt import build_edge_prompt
from .vlm_backend import TransformersLlavaBackend


class Edge2DScorer:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.model_id = str(cfg.get("model_id", cfg.get("backend", "cache_only")))
        self.prompt_version = str(cfg.get("prompt_version", "edge2d_v1"))
        self.cache = Edge2DScoreCache(cfg.get("cache_dir", ".cache/edge2d_s2d"))
        self.backend = None
        if str(cfg.get("backend", "cache_only")) in {"llava_hf", "transformers_llava"}:
            self.backend = TransformersLlavaBackend(cfg.get("llava", cfg))
            self.model_id = self.backend.model_id

    def score(self, candidate: Edge2DCandidate, prompt_image=None) -> Edge2DScore:
        cache_key = self.cache.key(candidate, model_id=self.model_id, prompt_version=self.prompt_version)
        cached = self.cache.get(cache_key)
        if cached is not None and cached.available:
            cached.metadata = {**(cached.metadata or {}), "cache_hit": True, "vlm_attempted": False}
            return cached
        if not candidate.pass_prefilter:
            return Edge2DScore(
                candidate=candidate,
                s2d_score=None,
                raw_response="",
                model_id=self.model_id,
                prompt_version=self.prompt_version,
                cache_key=cache_key,
                available=False,
                error=candidate.fail_reason,
                metadata={"sdet": candidate.sdet, "gcamc": candidate.gcamc, "cache_hit": False, "vlm_attempted": False},
            )
        if str(self.cfg.get("backend", "cache_only")) == "cache_only":
            return Edge2DScore(
                candidate=candidate,
                s2d_score=None,
                raw_response="",
                model_id=self.model_id,
                prompt_version=self.prompt_version,
                cache_key=cache_key,
                available=False,
                error="s2d_cache_miss_and_backend_cache_only",
                metadata={"sdet": candidate.sdet, "gcamc": candidate.gcamc, "cache_hit": False, "vlm_attempted": False},
            )
        if self.backend is None:
            raise RuntimeError("No vision VLM backend is configured; refusing to fabricate s2d_score")
        if prompt_image is None:
            raise ValueError("prompt_image is required for VLM s2d scoring")
        result = self.backend.generate_json(image=prompt_image, prompt=build_edge_prompt(candidate))
        score = Edge2DScore(
            candidate=candidate,
            s2d_score=result.s2d_score,
            raw_response=result.raw_response,
            model_id=result.model_id,
            prompt_version=self.prompt_version,
            cache_key=cache_key,
            available=result.s2d_score is not None,
            error=result.error,
            metadata={"sdet": candidate.sdet, "gcamc": candidate.gcamc, "cache_hit": False, "vlm_attempted": True, **(result.metadata or {})},
        )
        if score.available:
            self.cache.put(score)
        return score
