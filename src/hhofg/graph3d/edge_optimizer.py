from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hhofg.mapping3d.types import MapNode3D, MappedFrameEdgeEvidence

NULL_PARENT_ID = "__no_parent__"


def _logit(p: float, eps: float) -> float:
    q = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(q / (1.0 - q)))


@dataclass
class EdgeOptimizerState:
    log_odds: dict[str, dict[str, float]] = field(default_factory=dict)
    support_frames: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    relation_text_counts: dict[str, dict[str, dict[str, int]]] = field(
        default_factory=dict
    )
    posterior: dict[str, dict[str, float]] = field(default_factory=dict)
    posterior_history: list[dict[str, Any]] = field(default_factory=list)
    unavailable_count: int = 0
    evidence_score_sources: dict[str, int] = field(default_factory=dict)

    def update(self, edges: list[MappedFrameEdgeEvidence], nodes: dict[str, MapNode3D], cfg: dict[str, Any]) -> list[MappedFrameEdgeEvidence]:
        # Checkpoints written before relation predicates were retained do not
        # have this dataclass attribute after unpickling.
        if not hasattr(self, "relation_text_counts"):
            self.relation_text_counts = {}
        if not hasattr(self, "evidence_score_sources"):
            self.evidence_score_sources = {}
        eps = float(cfg.get("epsilon", 1e-4))
        stable_only = bool(cfg.get("stable_only", True))
        accepted: list[MappedFrameEdgeEvidence] = []
        for e in edges:
            if e.edge_type not in {"O-C", "O-U"}:
                continue
            parent = nodes.get(e.parent_node_id)
            child = nodes.get(e.child_node_id)
            if parent is None or child is None:
                continue
            if stable_only and (parent.state != "stable" or child.state != "stable"):
                continue
            score = e.s2d_score
            # Semantic compatibility proposes pairs; it is not an instance
            # ownership measurement. Never manufacture S2D for an unscored
            # pair (including selected 2D pairs that failed prefiltering).
            if score is None or not np.isfinite(score):
                self.unavailable_count += 1
                continue
            child_rec = self.log_odds.setdefault(e.child_node_id, {})
            child_rec[e.parent_node_id] = child_rec.get(e.parent_node_id, 0.0) + _logit(float(score), eps)
            frames = self.support_frames.setdefault(e.child_node_id, {}).setdefault(e.parent_node_id, [])
            if e.frame_key not in frames:
                frames.append(e.frame_key)
            self.evidence_score_sources["vlm_s2d"] = (
                self.evidence_score_sources.get("vlm_s2d", 0) + 1
            )
            relation_text = str(e.relation_text or "").strip()
            if relation_text:
                text_counts = self.relation_text_counts.setdefault(
                    e.child_node_id, {}
                ).setdefault(e.parent_node_id, {})
                text_counts[relation_text] = text_counts.get(relation_text, 0) + 1
            accepted.append(e)
        return accepted

    def optimize(self, cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
        lam_entropy = max(float(cfg.get("lambda_entropy", 0.1)), 1e-6)
        lam_temporal = float(cfg.get("lambda_temporal", 0.0))
        null_enabled = bool(cfg.get("null_parent_enabled", True))
        null_log_odds = float(cfg.get("null_parent_log_odds", 0.0))
        out: dict[str, dict[str, float]] = {}
        for child_id in sorted(self.log_odds):
            scored = dict(self.log_odds[child_id])
            if null_enabled:
                scored[NULL_PARENT_ID] = null_log_odds
            parents = sorted(scored)
            vals = np.asarray([scored[p] for p in parents], dtype=np.float64)
            prev = np.asarray([self.posterior.get(child_id, {}).get(p, 1.0 / max(1, len(parents))) for p in parents], dtype=np.float64)
            prev = prev / max(float(prev.sum()), 1e-12)
            denom = lam_entropy + max(lam_temporal, 0.0)
            logits = vals / max(denom, 1e-6)
            if lam_temporal > 0:
                logits += (lam_temporal / max(denom, 1e-6)) * np.log(np.clip(prev, 1e-9, 1.0))
            logits -= float(np.max(logits))
            z = np.exp(logits)
            z = z / max(float(z.sum()), 1e-12)
            out[child_id] = {p: float(z[i]) for i, p in enumerate(parents)}
        self.posterior = out
        return out

    def update_frame(self, frame_key: str, edges: list[MappedFrameEdgeEvidence], nodes: dict[str, MapNode3D], cfg: dict[str, Any]) -> None:
        prev_posterior = {c: dict(p) for c, p in self.posterior.items()}
        accepted = self.update(edges, nodes, cfg)
        posterior = self.optimize(cfg)
        for child_id, probs in sorted(posterior.items()):
            parents = sorted(probs)
            ranked = sorted(
                probs.items(),
                key=lambda kv: (
                    -kv[1],
                    0 if kv[0] == NULL_PARENT_ID else 1,
                    kv[0],
                ),
            )
            top1 = ranked[0] if ranked else ("", 0.0)
            top2 = ranked[1] if len(ranked) > 1 else ("", 0.0)
            prev_ranked = sorted(prev_posterior.get(child_id, {}).items(), key=lambda kv: (-kv[1], kv[0]))
            prev_top1 = prev_ranked[0][0] if prev_ranked else None
            self.posterior_history.append(
                {
                    "frame_key": frame_key,
                    "child_node_id": child_id,
                    "candidate_parent_ids": parents,
                    "accumulated_log_odds": {
                        p: float(
                            cfg.get("null_parent_log_odds", 0.0)
                            if p == NULL_PARENT_ID
                            else self.log_odds.get(child_id, {}).get(p, 0.0)
                        )
                        for p in parents
                    },
                    "previous_posterior": prev_posterior.get(child_id, {}),
                    "current_posterior": probs,
                    "top1": {"parent_node_id": top1[0], "posterior": float(top1[1])},
                    "top2": {"parent_node_id": top2[0], "posterior": float(top2[1])},
                    "margin": float(top1[1] - top2[1]),
                    "switch_from_previous": bool(prev_top1 is not None and prev_top1 != top1[0]),
                    "num_frame_observations_used": int(sum(1 for e in accepted if e.child_node_id == child_id)),
                }
            )


def optimize_object_edges(
    evidence: list[MappedFrameEdgeEvidence],
    nodes: dict[str, MapNode3D],
    cfg: dict[str, Any],
    state: EdgeOptimizerState | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], EdgeOptimizerState]:
    opt_state = state or EdgeOptimizerState()
    if state is None:
        by_frame: dict[tuple[int, str], list[MappedFrameEdgeEvidence]] = {}
        for e in evidence:
            by_frame.setdefault((int(e.frame_idx), str(e.frame_key)), []).append(e)
        for (_frame_idx, frame_key), frame_edges in sorted(by_frame.items()):
            opt_state.update_frame(frame_key, frame_edges, nodes, cfg)
        posterior = opt_state.posterior
    else:
        posterior = opt_state.optimize(cfg)
    edges: list[dict[str, Any]] = []
    decisions: dict[str, dict[str, Any]] = {}
    rejected_by_null = 0
    rejected_by_margin = 0
    min_margin = float(cfg.get("min_parent_posterior_margin", 0.05))
    null_enabled = bool(cfg.get("null_parent_enabled", True))
    null_log_odds = float(cfg.get("null_parent_log_odds", 0.0))
    for child_id, probs in sorted(posterior.items()):
        if not probs:
            continue
        ranked = sorted(
            probs.items(),
            key=lambda kv: (
                -kv[1],
                0 if kv[0] == NULL_PARENT_ID else 1,
                kv[0],
            ),
        )
        best_parent = ranked[0][0]
        null_posterior = float(probs.get(NULL_PARENT_ID, 0.0))
        real_ranked = [
            (parent_id, probability)
            for parent_id, probability in ranked
            if parent_id != NULL_PARENT_ID
        ]
        best_real_parent = real_ranked[0][0] if real_ranked else ""
        best_real_posterior = (
            float(real_ranked[0][1]) if real_ranked else 0.0
        )
        best_real_log_odds = float(
            opt_state.log_odds.get(child_id, {}).get(best_real_parent, 0.0)
        )
        runner_up_posterior = float(real_ranked[1][1]) if len(real_ranked) > 1 else 0.0
        margin = best_real_posterior - max(null_posterior, runner_up_posterior)
        decision = {
            "selected_parent_id": best_real_parent,
            "null_parent_posterior": null_posterior,
            "best_real_parent_id": best_real_parent,
            "best_real_parent_posterior": best_real_posterior,
            "best_real_parent_log_odds": best_real_log_odds,
            "best_real_parent_margin": margin,
            "accepted": False,
            "status": "",
            "reject_reason": "",
        }
        if not best_real_parent:
            decision["reject_reason"] = "no_real_parent_candidate"
            decisions[child_id] = decision
            continue
        below_null = bool(
            null_enabled
            and (
                best_parent == NULL_PARENT_ID
                or best_real_log_odds <= null_log_odds
                or best_real_posterior <= null_posterior
            )
        )
        below_margin = bool(margin < min_margin)
        if below_null:
            rejected_by_null += 1
        if below_margin:
            rejected_by_margin += 1
        # Optimistic current-best policy: confidence gates decide whether an
        # edge is provisional, not whether an instance-scored child stays
        # disconnected.  A later update can replace this parent when its
        # accumulated evidence becomes stronger.
        best_parent = best_real_parent
        decision["status"] = (
            "provisional" if below_null or below_margin else "confirmed"
        )
        child = nodes.get(child_id)
        parent = nodes.get(best_parent)
        if child is None or parent is None:
            decision["reject_reason"] = "node_missing"
            decisions[child_id] = decision
            continue
        edge_type = f"{parent.role}-{child.role}"
        if edge_type not in {"O-C", "O-U"}:
            decision["reject_reason"] = "illegal_role_pair"
            decisions[child_id] = decision
            continue
        log_odds = (
            opt_state.log_odds.get(child_id, {}).get(best_parent, 0.0)
        )
        relation_counts = getattr(opt_state, "relation_text_counts", {}).get(
            child_id, {}
        ).get(best_parent, {})
        relation_text = (
            sorted(relation_counts, key=lambda text: (-relation_counts[text], text))[0]
            if relation_counts
            else ""
        )
        lambda_score = float(log_odds + np.log(max(probs[best_parent], 1e-12)))
        decision["accepted"] = True
        decisions[child_id] = decision
        edges.append(
            {
                "parent_node_id": best_parent,
                "child_node_id": child_id,
                "edge_type": edge_type,
                "parent_role": parent.role,
                "child_role": child.role,
                "posterior": float(probs[best_parent]),
                "log_odds": float(log_odds),
                "lambda_score": lambda_score,
                "support_count": len(opt_state.support_frames.get(child_id, {}).get(best_parent, [])),
                "null_parent_posterior": null_posterior,
                "best_real_parent_log_odds": best_real_log_odds,
                "best_real_parent_margin": margin,
                "relation_text": relation_text,
                "status": decision["status"],
                "provisional": decision["status"] == "provisional",
                "evidence_score_source": "vlm_s2d",
            }
        )
    meta = {
        "graph_stage": "optimized_object_graph",
        "is_final_graph": False,
        "uses_real_s2d": bool(
            opt_state.evidence_score_sources.get("vlm_s2d", 0)
        ),
        "s2d_available": bool(
            opt_state.evidence_score_sources.get("vlm_s2d", 0)
        ),
        "semantic_prior_available": False,
        "unscored_candidate_count": int(opt_state.unavailable_count),
        "num_children_with_candidates": int(
            len(opt_state.log_odds)
        ),
        "null_parent_enabled": null_enabled,
        "null_parent_log_odds": null_log_odds,
        "min_parent_posterior_margin": min_margin,
        "num_children_rejected_by_null_parent": 0,
        "num_children_rejected_by_parent_margin": 0,
        "num_children_provisional_by_null_parent": int(rejected_by_null),
        "num_children_provisional_by_parent_margin": int(rejected_by_margin),
        "parent_decisions": decisions,
        "evidence_score_sources": dict(sorted(opt_state.evidence_score_sources.items())),
        "selection_policy": "optimistic_current_best_with_later_replacement",
        "candidate_evidence_policy": "measured_s2d_only_no_synthetic_prior_scores",
        "edge_posteriors": posterior,
        "temporal_objective": "entropy_regularized_log_odds_with_KL_to_previous_posterior",
        "temporal_smoothing_is_implementation_choice": True,
        "num_posterior_history_records": int(len(opt_state.posterior_history)),
    }
    return edges, meta, opt_state
