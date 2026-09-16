from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True)
class Edge2DCandidate:
    frame_key: str
    parent_det_id: int
    child_det_id: int
    parent_label: str
    child_label: str
    parent_score: float
    child_score: float
    edge_type: str
    sdet: float
    gcamc: float
    pass_prefilter: bool
    gcamc_raw: float = 0.0
    gcamc_support: float = 0.0
    gcamc_used: float = 0.0
    support_mask_used: bool = False
    child_center_inside_parent_box: bool = False
    fail_reason: str = ""
    image_sha: str = ""
    mask_sha: str = ""
    candidate_hash: str = ""
    candidate_sources: list[str] = field(default_factory=list)
    semantic_sources: list[str] = field(default_factory=list)


@dataclass
class Edge2DScore:
    candidate: Edge2DCandidate
    s2d_score: float | None
    raw_response: str
    model_id: str
    prompt_version: str
    cache_key: str
    available: bool
    error: str = ""
    metadata: dict[str, Any] | None = None
