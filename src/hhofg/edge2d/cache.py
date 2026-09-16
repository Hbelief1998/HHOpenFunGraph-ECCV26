from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hhofg.core.serialization import write_json_atomic

from .types import Edge2DCandidate, Edge2DScore


class Edge2DScoreCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(candidate: Edge2DCandidate, *, model_id: str, prompt_version: str) -> str:
        payload = {
            "frame_key": candidate.frame_key,
            "parent_label": candidate.parent_label,
            "child_label": candidate.child_label,
            "parent_det_id": candidate.parent_det_id,
            "child_det_id": candidate.child_det_id,
            "edge_type": candidate.edge_type,
            "candidate_hash": candidate.candidate_hash,
            "image_sha": candidate.image_sha,
            "mask_sha": candidate.mask_sha,
            "model_id": model_id,
            "prompt_version": prompt_version,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def get(self, key: str) -> Edge2DScore | None:
        path = self.cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        from .types import Edge2DCandidate

        payload["candidate"] = Edge2DCandidate(**payload["candidate"])
        return Edge2DScore(**payload)

    def put(self, score: Edge2DScore) -> None:
        payload = {
            "candidate": score.candidate.__dict__,
            "s2d_score": score.s2d_score,
            "raw_response": score.raw_response,
            "model_id": score.model_id,
            "prompt_version": score.prompt_version,
            "cache_key": score.cache_key,
            "available": score.available,
            "error": score.error,
            "metadata": score.metadata or {},
        }
        write_json_atomic(self.cache_dir / f"{score.cache_key}.json", payload)
