#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hhofg.core.config import load_yaml_config as load_config
from hhofg.core.serialization import write_json_atomic
from hhofg.data.factory import create_sequence
from hhofg.edge2d import Edge2DScorer, build_annotated_edge_image, edge_candidate_stats, image_sha_rgb
from hhofg.graph3d.functional_graph import build_functional_graph_artifacts
from hhofg.mapping3d.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_final_map_state,
)
from hhofg.mapping3d.clip_text import ClipTextEncoder
from hhofg.mapping3d.edge_candidates import (
    candidate_to_mapped_evidence,
    candidate_to_relation,
    collect_frame_edge_candidates,
    mapped_candidate_drop_reason,
)
from hhofg.mapping3d.frontend_reader import FrontendRunReader
from hhofg.mapping3d.lifting import ObservationLifter
from hhofg.mapping3d.map_state import MapState3D
from hhofg.mapping3d.serialization import append_jsonl, save_association, save_map, save_observations
from hhofg.mapping3d.validation import validate_association, validate_nodes, validate_observations
from hhofg.mapping3d.visualization import (
    draw_association_overlay,
    write_map_plys,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mapping3d_paper_full.yaml")
    ap.add_argument("--frontend-run", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset-root")
    ap.add_argument("--sequence")
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--stride", type=int)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device")
    ap.add_argument("--save-assoc-vis", action="store_true")
    return ap.parse_args()


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def read_edge_candidate_jsonl(path: Path) -> list:
    from hhofg.mapping3d.types import FrameEdgeCandidate

    if not path.is_file():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(FrameEdgeCandidate(**json.loads(line)))
    return out


def _cap_reached(count: int, cap: int) -> bool:
    return cap > 0 and count >= cap


def score_frame_candidates_with_vlm(
    *,
    candidates,
    result2d,
    masks_raw: np.ndarray,
    frame,
    nodes,
    scorer: Edge2DScorer | None,
    edge_cfg: dict,
    scored_so_far: int,
    out_dir: Path,
) -> tuple[int, dict, list[dict]]:
    if scorer is None or not bool(edge_cfg.get("enabled", False)):
        return scored_so_far, {"enabled": False, "scored": 0, "available": 0, "skipped": len(candidates)}, []
    detections_by_id = {int(d.det_id): d for d in result2d.detections}
    max_per_frame = int(edge_cfg.get("max_edges_per_frame", 0) or 0)
    max_per_run = int(edge_cfg.get("max_edges_per_run", 0) or 0)
    score_selected_only = bool(edge_cfg.get("score_selected_only", True))
    score_stable_nodes_only = bool(edge_cfg.get("score_stable_nodes_only", False))
    image_sha = image_sha_rgb(frame.rgb)
    scored = 0
    available = 0
    skipped = 0
    prefilter_pass = 0
    prefilter_fail = 0
    cache_hit = 0
    vlm_attempted = 0
    records = []
    for c in candidates:
        if c.edge_type not in {"O-C", "O-U"}:
            skipped += 1
            continue
        skip_reason = ""
        if score_selected_only and not c.selected_2d:
            skipped += 1
            skip_reason = "skipped_selected_only"
        if score_stable_nodes_only:
            parent_node = nodes.get(c.parent_node_id or "")
            child_node = nodes.get(c.child_node_id or "")
            if parent_node is None or child_node is None or parent_node.state != "stable" or child_node.state != "stable":
                skipped += 1
                skip_reason = "skipped_stable_only"
        if skip_reason:
            records.append(
                {
                    "frame_key": frame.frame_key,
                    "edge_type": c.edge_type,
                    "parent_det_id": c.parent_det_id,
                    "child_det_id": c.child_det_id,
                    "parent_node_id": c.parent_node_id,
                    "child_node_id": c.child_node_id,
                    "candidate_source": c.candidate_source,
                    "available": False,
                    "error": skip_reason,
                }
            )
            c.s2d_error = skip_reason
            continue
        cand = edge_candidate_stats(
            frame_key=frame.frame_key,
            relation=candidate_to_relation(c),
            detections_by_id=detections_by_id,
            masks_raw=masks_raw,
            sdet_threshold=float(edge_cfg.get("sdet_threshold", 0.25)),
            gcamc_threshold=float(edge_cfg.get("gcamc_threshold", 0.90)),
            dilation_radius_px=edge_cfg.get("gcamc_dilation_radius_px"),
            dilation_alpha=float(edge_cfg.get("gcamc_dilation_alpha", 0.0)),
            dilation_radius_min_px=int(edge_cfg.get("gcamc_dilation_radius_min_px", 2)),
            dilation_radius_max_px=int(edge_cfg.get("gcamc_dilation_radius_max_px", 12)),
            image_sha=image_sha,
            candidate_hash=c.candidate_hash,
            candidate_sources=c.candidate_sources,
            semantic_sources=c.semantic_sources,
            semantic_compatible=bool(c.semantic_sources),
            support_closing_radius_px=int(
                edge_cfg.get("gcamc_support_closing_radius_px", 2)
            ),
        )
        c.sdet = cand.sdet
        c.gcamc = cand.gcamc
        c.gcamc_raw = cand.gcamc_raw
        c.gcamc_support = cand.gcamc_support
        c.gcamc_used = cand.gcamc_used
        c.support_mask_used = cand.support_mask_used
        c.child_center_inside_parent_box = (
            cand.child_center_inside_parent_box
        )
        c.pass_prefilter = cand.pass_prefilter
        c.prefilter_fail_reason = cand.fail_reason
        c.mask_sha = cand.mask_sha
        if not cand.pass_prefilter:
            prefilter_fail += 1
            records.append(
                {
                    "frame_key": frame.frame_key,
                    "edge_type": c.edge_type,
                    "parent_det_id": c.parent_det_id,
                    "child_det_id": c.child_det_id,
                    "parent_node_id": c.parent_node_id,
                    "child_node_id": c.child_node_id,
                    "candidate_source": c.candidate_source,
                    "candidate_hash": c.candidate_hash,
                    "candidate_sources": c.candidate_sources,
                    "semantic_sources": c.semantic_sources,
                    "candidate": cand.__dict__,
                    "available": False,
                    "error": cand.fail_reason,
                }
            )
            c.s2d_error = cand.fail_reason
            continue
        prefilter_pass += 1
        if _cap_reached(scored, max_per_frame):
            skipped += 1
            c.s2d_error = "skipped_per_frame_cap"
            records.append({"frame_key": frame.frame_key, "edge_type": c.edge_type, "parent_det_id": c.parent_det_id, "child_det_id": c.child_det_id, "candidate": cand.__dict__, "available": False, "error": c.s2d_error})
            continue
        if _cap_reached(scored_so_far, max_per_run):
            skipped += 1
            c.s2d_error = "skipped_per_run_cap"
            records.append({"frame_key": frame.frame_key, "edge_type": c.edge_type, "parent_det_id": c.parent_det_id, "child_det_id": c.child_det_id, "candidate": cand.__dict__, "available": False, "error": c.s2d_error})
            continue
        prompt_image = None
        prompt_image = build_annotated_edge_image(
            image_rgb=frame.rgb,
            parent_det=detections_by_id[int(c.parent_det_id)],
            child_det=detections_by_id[int(c.child_det_id)],
            parent_mask=masks_raw[int(c.parent_det_id)],
            child_mask=masks_raw[int(c.child_det_id)],
            max_side=int(edge_cfg.get("prompt_max_side", 768)),
            crop_margin_ratio=float(edge_cfg.get("prompt_crop_margin_ratio", 0.25)),
        )
        score = scorer.score(cand, prompt_image=prompt_image)
        c.s2d_score = score.s2d_score
        c.s2d_available = bool(score.available)
        c.s2d_error = score.error
        c.s2d_cache_key = score.cache_key
        c.s2d_model_id = score.model_id
        c.s2d_raw_response = score.raw_response
        c.s2d_cache_hit = bool((score.metadata or {}).get("cache_hit", False))
        scored += 1
        scored_so_far += 1
        available += int(score.available)
        cache_hit += int(c.s2d_cache_hit)
        vlm_attempted += int(bool((score.metadata or {}).get("vlm_attempted", False)))
        records.append(
            {
                "frame_key": frame.frame_key,
                "edge_type": c.edge_type,
                "parent_det_id": c.parent_det_id,
                "child_det_id": c.child_det_id,
                "parent_node_id": c.parent_node_id,
                "child_node_id": c.child_node_id,
                "candidate_source": c.candidate_source,
                "candidate_hash": c.candidate_hash,
                "candidate_sources": c.candidate_sources,
                "semantic_sources": c.semantic_sources,
                "candidate": cand.__dict__,
                "s2d_score": score.s2d_score,
                "available": score.available,
                "error": score.error,
                "raw_response": score.raw_response,
                "model_id": score.model_id,
                "prompt_version": score.prompt_version,
                "cache_key": score.cache_key,
                "metadata": score.metadata or {},
            }
        )
    if records:
        write_json_atomic(out_dir / f"{frame.frame_key}_s2d_scores.json", {"frame_key": frame.frame_key, "scores": records})
    return scored_so_far, {
        "enabled": True,
        "scored": scored,
        "available": available,
        "skipped": skipped,
        "prefilter_pass": prefilter_pass,
        "prefilter_fail": prefilter_fail,
        "cache_hit": cache_hit,
        "vlm_attempted": vlm_attempted,
    }, records


def update_stable_parent_context(candidates, nodes, cfg: dict) -> None:
    if not bool(cfg.get("enabled", False)):
        return
    min_support = int(cfg.get("min_support_frames", 2))
    min_margin = int(cfg.get("min_support_margin", 1))
    for c in candidates:
        if not c.parent_node_id or not c.child_node_id:
            continue
        if c.edge_type not in {"O-C", "O-U", "C-U"}:
            continue
        if not (c.selected_2d or c.pass_threshold):
            continue
        parent = nodes.get(c.parent_node_id)
        child = nodes.get(c.child_node_id)
        if parent is None or child is None or parent.state != "stable":
            continue
        if child.role == "C" and parent.role != "O":
            continue
        if child.role == "U" and parent.role not in {"O", "C"}:
            continue
        is_object_context = parent.role == "O"
        history = (
            child.object_parent_context_history
            if is_object_context
            else child.carrier_parent_context_history
        )
        history.append(
            {
                "frame_idx": c.frame_idx,
                "frame_key": c.frame_key,
                "parent_node_id": parent.node_id,
                "edge_type": c.edge_type,
                "selected_2d": c.selected_2d,
                "pass_threshold": c.pass_threshold,
                "mask_contain": c.mask_contain,
                "candidate_source": c.candidate_source,
            }
        )
        support: dict[str, set[str]] = {}
        for rec in history:
            pid = str(rec.get("parent_node_id", ""))
            if pid:
                support.setdefault(pid, set()).add(str(rec.get("frame_key", "")))
        ranked = sorted(((len(frames), pid) for pid, frames in support.items()), reverse=True)
        if not ranked:
            continue
        best_count, best_pid = ranked[0]
        second_count = ranked[1][0] if len(ranked) > 1 else 0
        if best_count >= min_support and best_count - second_count >= min_margin:
            if is_object_context:
                child.stable_object_parent_id = best_pid
            else:
                child.stable_carrier_parent_id = best_pid


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = dict(cfg["data"])
    if args.dataset_root:
        data_cfg["dataset_root"] = args.dataset_root
    if args.sequence:
        data_cfg["sequence"] = args.sequence
    for key in ("start", "end", "stride"):
        val = getattr(args, key)
        if val is not None:
            data_cfg[key] = val
    reader = FrontendRunReader(args.frontend_run, dataset=data_cfg.get("dataset"), sequence=data_cfg.get("sequence"))
    if reader.dataset_config:
        data_cfg["dataset_config"] = reader.dataset_config
    if reader.desired_height is not None:
        data_cfg["desired_height"] = reader.desired_height
    if reader.desired_width is not None:
        data_cfg["desired_width"] = reader.desired_width
    data_cfg["load_depth"] = True
    data_cfg["load_pose"] = True

    out_root = Path(args.output_dir) / "mapping3d"
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    if out_root.exists() and not args.resume and any(out_root.iterdir()):
        raise FileExistsError(f"output exists; pass --overwrite or --resume: {out_root}")
    for sub in ("observations", "associations", "mapped_edges", "edge_candidates", "map", "vis", "checkpoints", "edge2d_scores"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    for stale in (
        out_root / "mapped_edges" / "frame_edges.jsonl",
        out_root / "mapped_edges" / "dropped_edges.jsonl",
        out_root / "edge_candidates" / "frame_edge_candidates.jsonl",
        out_root / "edge2d_scores" / "edge_scores.jsonl",
    ):
        if stale.exists() and args.overwrite:
            stale.unlink()

    clip_cfg = dict(cfg["clip"])
    if args.device:
        clip_cfg["device"] = args.device
    encoder = ClipTextEncoder(**clip_cfg)
    lifter = ObservationLifter(cfg["lifting"], cfg["appearance"], encoder)
    state = MapState3D(cfg["association"], cfg["map_update"], cfg["lifting"])
    edge2d_cfg = dict(cfg.get("edge2d", {}))
    edge_scorer = Edge2DScorer(edge2d_cfg) if bool(edge2d_cfg.get("enabled", False)) else None
    dataset_name = data_cfg["dataset"]
    seq_kwargs = {k: v for k, v in data_cfg.items() if k != "dataset"}
    seq = create_sequence(dataset_name, **seq_kwargs)

    run_config_payload = {
        "mapping3d_config": cfg,
        "effective_data": data_cfg,
        "frontend_run": str(Path(args.frontend_run).resolve()),
        "frontend_metadata": {
            "dataset": reader.dataset,
            "sequence": reader.sequence,
            "desired_height": reader.desired_height,
            "desired_width": reader.desired_width,
            "coordinate_space": reader.coordinate_space,
            "frontend_commit": reader.frontend_commit,
            "num_frontend_frames": len(reader),
        },
    }
    write_json_atomic(
        out_root / "run_config.json",
        run_config_payload,
    )

    summary = Counter()
    role_lift = Counter()
    role_2d = Counter()
    reject_counts = Counter()
    mapped_edges_by_type = Counter()
    scores_total: list[float] = []
    scores_iou: list[float] = []
    scores_geo: list[float] = []
    scores_app: list[float] = []
    scores_sem: list[float] = []
    timing_acc = defaultdict(float)
    s2d_stats = Counter()
    s2d_scored_so_far = 0
    overlay_edge_stats: dict[tuple[str, str, str], dict] = {}
    all_edge_candidates = read_edge_candidate_jsonl(out_root / "edge_candidates" / "frame_edge_candidates.jsonl") if args.resume else []
    processed_frame_keys: list[str] = []
    jsonl_frame_keys: dict[str, list[str]] = {"frame_edges": [], "dropped_edges": []}
    if args.resume:
        ckpt = latest_checkpoint(out_root / "checkpoints")
        if ckpt is None:
            raise FileNotFoundError(f"--resume was set but no complete checkpoint exists in {out_root / 'checkpoints'}")
        state, ckpt_meta, _edge_state, _hier_state = load_checkpoint(
            ckpt,
            cfg=cfg,
            data_cfg=data_cfg,
            frontend_run=args.frontend_run,
            frontend_metadata=run_config_payload["frontend_metadata"],
        )
        processed_frame_keys = list(ckpt_meta.get("processed_frame_keys", []))
        jsonl_frame_keys = {k: list(v) for k, v in ckpt_meta.get("jsonl_frame_keys", jsonl_frame_keys).items()}
        print(f"Resumed checkpoint: {ckpt} ({len(processed_frame_keys)} processed frames)")
    processed_frame_key_set = set(processed_frame_keys)
    processed = 0
    skipped = 0
    t_run = time.perf_counter()

    for idx in range(len(seq)):
        if args.max_frames is not None and len(processed_frame_keys) >= args.max_frames:
            break
        frame = seq[idx]
        if frame.frame_key in processed_frame_key_set:
            continue
        fdata = reader.get(frame.frame_key)
        if fdata is None:
            skipped += 1
            continue
        reader.validate_frame_alignment(frame, fdata)
        t0 = time.perf_counter()
        observations = lifter.lift_frame(frame, fdata.result, fdata.masks_raw)
        lifting_ms = (time.perf_counter() - t0) * 1000.0
        validate_observations(observations)
        assoc = state.process_frame(observations, frame)
        assoc.timing_ms["lifting"] = lifting_ms
        validate_association(assoc, observations)
        validate_nodes(state.nodes)

        report = lifter.last_report
        save_observations(frame.frame_key, observations, out_root / "observations", report)
        save_association(assoc, out_root / "associations")
        if bool(cfg["output"].get("save_assoc_visualization", True)) or args.save_assoc_vis:
            draw_association_overlay(frame, observations, assoc, out_root / "vis" / f"{frame.frame_key}_assoc.png")

        lifted_det_ids = {obs.det_id for obs in observations}
        reject_by_det = {r.det_id: r.reject_reason for r in (report.rejects if report else [])}
        frame_image_sha = image_sha_rgb(frame.rgb)
        edge_candidates = collect_frame_edge_candidates(
            fdata.result,
            assoc,
            lifted_det_ids,
            state.nodes,
            image_sha=frame_image_sha,
            global_atlas=reader.global_atlas,
            cfg=cfg.get("candidate_generation", {}),
        )
        update_stable_parent_context(edge_candidates, state.nodes, cfg.get("stable_parent_context", {}))
        t_s2d = time.perf_counter()
        s2d_scored_so_far, frame_s2d_stats, frame_score_records = score_frame_candidates_with_vlm(
            candidates=edge_candidates,
            result2d=fdata.result,
            masks_raw=fdata.masks_raw,
            frame=frame,
            nodes=state.nodes,
            scorer=edge_scorer,
            edge_cfg=edge2d_cfg,
            scored_so_far=s2d_scored_so_far,
            out_dir=out_root / "edge2d_scores",
        )
        assoc.timing_ms["s2d_vlm"] = (time.perf_counter() - t_s2d) * 1000.0
        for k, v in frame_s2d_stats.items():
            s2d_stats[k] += int(v) if isinstance(v, (bool, int)) else 0
        edges = []
        dropped = []
        for cand in edge_candidates:
            ev = candidate_to_mapped_evidence(cand)
            if ev is not None:
                edges.append(ev)
            else:
                drop = mapped_candidate_drop_reason(cand, reject_by_det)
                if drop is not None:
                    dropped.append(drop)
        append_jsonl(out_root / "edge_candidates" / "frame_edge_candidates.jsonl", edge_candidates)
        append_jsonl(out_root / "edge2d_scores" / "edge_scores.jsonl", frame_score_records)
        append_jsonl(out_root / "mapped_edges" / "frame_edges.jsonl", edges)
        append_jsonl(out_root / "mapped_edges" / "dropped_edges.jsonl", dropped)
        all_edge_candidates.extend(edge_candidates)
        jsonl_frame_keys.setdefault("frame_edges", []).extend([e.frame_key for e in edges])
        jsonl_frame_keys.setdefault("dropped_edges", []).extend([d["frame_key"] for d in dropped])
        for e in edges:
            if not e.selected_2d:
                continue
            key = (e.parent_node_id, e.child_node_id, e.edge_type)
            rec = overlay_edge_stats.setdefault(
                key,
                {
                    "parent_node_id": e.parent_node_id,
                    "child_node_id": e.child_node_id,
                    "edge_type": e.edge_type,
                    "parent_role": e.parent_role,
                    "child_role": e.child_role,
                    "evidence_count": 0,
                    "selected_2d_count": 0,
                    "pass_threshold_count": 0,
                    "eligible_for_temporal_optimization": e.eligible_for_temporal_optimization,
                    "usage": e.usage,
                    "first_frame_idx": e.frame_idx,
                    "last_frame_idx": e.frame_idx,
                    "relation_text": e.relation_text,
                },
            )
            rec["evidence_count"] += 1
            rec["selected_2d_count"] += int(e.selected_2d)
            rec["pass_threshold_count"] += int(e.pass_threshold)
            rec["first_frame_idx"] = min(rec["first_frame_idx"], e.frame_idx)
            rec["last_frame_idx"] = max(rec["last_frame_idx"], e.frame_idx)

        processed += 1
        processed_frame_keys.append(frame.frame_key)
        processed_frame_key_set.add(frame.frame_key)
        summary["num_2d_detections"] += len(fdata.result.detections)
        summary["num_raw_local_candidates"] += len(
            fdata.result.local_relation_candidates_raw or []
        )
        summary["num_final_local_relations"] += len(
            fdata.result.local_relations_final or []
        )
        raw_keys = {
            (r.edge_type, r.parent_det_id, r.child_det_id)
            for r in fdata.result.local_relation_candidates_raw or []
        }
        final_keys = {
            (r.edge_type, r.parent_det_id, r.child_det_id)
            for r in fdata.result.local_relations_final or []
        }
        summary["num_unique_raw_local_candidates"] += len(raw_keys)
        summary["num_raw_candidates_retained_in_final"] += len(raw_keys & final_keys)
        summary["num_lifted_observations"] += len(observations)
        summary["num_matches"] += len(assoc.matches)
        summary["num_births"] += len(assoc.births)
        summary["num_candidate_pairs"] += len(assoc.pair_scores)
        summary["num_valid_pairs"] += sum(1 for p in assoc.pair_scores if p.valid)
        summary["cross_role_matches"] += sum(1 for p in assoc.pair_scores if p.selected and p.role not in {"O", "C", "U"})
        summary["invalid_selected_matches"] += sum(1 for p in assoc.pair_scores if p.selected and not p.valid)
        for det in fdata.result.detections:
            role_2d[det.role] += 1
        for obs in observations:
            role_lift[obs.role] += 1
        for rej in report.rejects if report else []:
            reject_counts[rej.reject_reason] += 1
        for e in edges:
            mapped_edges_by_type[e.edge_type] += 1
        for p in assoc.pair_scores:
            if p.selected:
                scores_total.append(p.score_total)
                scores_iou.append(p.score_iou)
                scores_geo.append(p.score_geo)
                scores_app.append(p.score_app)
                scores_sem.append(p.score_sem)
        for k, v in assoc.timing_ms.items():
            timing_acc[k] += float(v)
        every = int(cfg.get("output", {}).get("checkpoint_every", 0) or 0)
        if every > 0 and processed > 0 and processed % every == 0:
            save_checkpoint(
                checkpoint_root=out_root / "checkpoints",
                frame_key=frame.frame_key,
                state=state,
                processed_frame_keys=processed_frame_keys,
                jsonl_frame_keys=jsonl_frame_keys,
                cfg=cfg,
                data_cfg=data_cfg,
                frontend_run=args.frontend_run,
                frontend_metadata=run_config_payload["frontend_metadata"],
                repo_root=ROOT,
            )

    consolidation_events = state.consolidate_non_covisible_tracks(
        cfg.get("track_consolidation", {})
    )
    write_json_atomic(
        out_root / "map" / "node_aliases.json",
        {
            "aliases": dict(sorted(getattr(state, "node_aliases", {}).items())),
            "events": getattr(state, "consolidation_events", []),
        },
    )
    save_map(state.nodes, out_root / "map")
    save_final_map_state(out_root / "map" / "map_state.pkl", state)
    if bool(cfg["output"].get("save_ply", True)):
        write_map_plys(state.nodes, out_root / "map")
    graph_summary = build_functional_graph_artifacts(
        mapping_run=out_root,
        candidate_jsonl=out_root / "edge_candidates" / "frame_edge_candidates.jsonl",
        score_jsonl=out_root / "edge2d_scores" / "edge_scores.jsonl",
        cfg=cfg,
        output_dir=out_root,
        write_ply=bool(cfg["output"].get("save_ply", True)),
    )
    timing_acc["total"] = (time.perf_counter() - t_run) * 1000.0
    summary_payload = {
        "num_frames": int(len(processed_frame_keys)),
        "num_frames_processed_this_invocation": int(processed),
        "num_skipped_no_frontend": skipped,
        "num_frontend_frames": len(reader),
        "num_2d_detections": int(summary["num_2d_detections"]),
        "num_raw_local_candidates": int(summary["num_raw_local_candidates"]),
        "num_final_local_relations": int(summary["num_final_local_relations"]),
        "raw_to_final_retention_rate": float(
            summary["num_raw_candidates_retained_in_final"]
            / max(1, summary["num_unique_raw_local_candidates"])
        ),
        "num_lifted_observations": int(summary["num_lifted_observations"]),
        "lifting_success_rate": float(summary["num_lifted_observations"] / max(1, summary["num_2d_detections"])),
        "lifting_success_by_role": {r: int(role_lift[r]) for r in ("O", "C", "U")},
        "detections_by_role": {r: int(role_2d[r]) for r in ("O", "C", "U")},
        "lifting_reject_reasons": dict(reject_counts),
        "num_matches": int(summary["num_matches"]),
        "num_births": int(summary["num_births"]),
        "num_nodes_by_role": state.nodes_by_role(),
        "num_candidate_pairs": int(summary["num_candidate_pairs"]),
        "num_valid_pairs": int(summary["num_valid_pairs"]),
        "match_score_mean": mean(scores_total),
        "match_score_min": float(np.min(scores_total)) if scores_total else None,
        "match_score_max": float(np.max(scores_total)) if scores_total else None,
        "mean_iou": mean(scores_iou),
        "mean_geo": mean(scores_geo),
        "mean_app": mean(scores_app),
        "mean_sem": mean(scores_sem),
        "cross_role_matches": int(summary["cross_role_matches"]),
        "invalid_selected_matches": int(summary["invalid_selected_matches"]),
        "mapped_edges_by_type": dict(mapped_edges_by_type),
        "scene_graph_overlay_edges": int(len(overlay_edge_stats)),
        "num_stable_nodes": int(sum(1 for n in state.nodes.values() if n.state == "stable")),
        "num_provisional_nodes": int(sum(1 for n in state.nodes.values() if n.state == "provisional")),
        "num_rejected_nodes": int(sum(1 for n in state.nodes.values() if n.state == "rejected")),
        "num_skipped_low_quality_births": int(state.skipped_low_quality_births),
        "identity_attempts": int(state.identity_attempts),
        "identity_matches": int(state.identity_matches),
        "identity_reject_reasons": dict(state.identity_reject_reasons),
        "identity_by_role": dict(state.identity_by_role),
        "timing_ms": {k: float(v) for k, v in timing_acc.items()},
        "peak_map_points": int(state.peak_map_points),
        "total_map_nodes": int(len(state.nodes)),
        "num_track_consolidation_merges": int(
            sum(event.get("event_type") == "merge" for event in consolidation_events)
        ),
        "num_cross_role_duplicate_retirements": int(
            sum(event.get("event_type") == "retire" for event in consolidation_events)
        ),
        "num_node_aliases": int(len(getattr(state, "node_aliases", {}))),
        "s2d": {
            "enabled": bool(edge2d_cfg.get("enabled", False)),
            "backend": edge2d_cfg.get("backend", "cache_only"),
            "scored": int(s2d_stats["scored"]),
            "available": int(s2d_stats["available"]),
            "skipped": int(s2d_stats["skipped"]),
            "prefilter_pass": int(s2d_stats["prefilter_pass"]),
            "prefilter_fail": int(s2d_stats["prefilter_fail"]),
            "cache_hit": int(s2d_stats["cache_hit"]),
            "vlm_attempted": int(s2d_stats["vlm_attempted"]),
            "max_edges_per_run": int(edge2d_cfg.get("max_edges_per_run", 0) or 0),
            "score_selected_only": bool(edge2d_cfg.get("score_selected_only", False)),
            "score_stable_nodes_only": bool(edge2d_cfg.get("score_stable_nodes_only", False)),
            "partial_scoring": bool(edge2d_cfg.get("enabled", False))
            and (
                int(s2d_stats["prefilter_pass"]) != int(s2d_stats["available"])
                or int(edge2d_cfg.get("max_edges_per_frame", 0) or 0) > 0
                or int(edge2d_cfg.get("max_edges_per_run", 0) or 0) > 0
                or bool(edge2d_cfg.get("score_selected_only", False))
                or bool(edge2d_cfg.get("score_stable_nodes_only", False))
            ),
        },
        "graph_complete": bool(graph_summary.get("graph_complete", False)),
        "incomplete_reason": ", ".join(graph_summary.get("incomplete_reasons", [])),
        "functional_graph": graph_summary,
        "edge_candidates": {
            "total": int(len(all_edge_candidates)),
            "mapped_overlay_edges": int(len(locals().get("all_mapped_overlay_edges", []))),
            "selected_overlay_edges": int(len(locals().get("selected_overlay_edges", []))),
            "prefilter_overlay_edges": int(len(locals().get("prefilter_overlay_edges", []))),
            "scored_overlay_edges": int(len(locals().get("scored_overlay_edges", []))),
        },
    }
    write_json_atomic(out_root / "summary.json", summary_payload)
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
