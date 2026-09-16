#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hhofg.core.config import load_yaml_config
from hhofg.core.serialization import write_json_atomic
from hhofg.data.factory import create_sequence
from hhofg.frontend2d.pipeline import Frontend2DPipeline
from hhofg.frontend2d.policy import policy_from_config
from hhofg.frontend2d.summary import RunSummary
from hhofg.frontend2d.semantics.llm_runtime import DeepSeekRuntime
from hhofg.frontend2d.semantics.rampp_runtime import RamppTagger
from hhofg.frontend2d.semantics.scene_lock import SceneLock
from hhofg.frontend2d.sam3.runtime import Sam3TwoStageRuntime
from hhofg.frontend2d.sam3.io import _save_sam3_vis


def _override(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = json.loads(json.dumps(cfg))
    data = cfg.setdefault("data", {})
    if args.dataset:
        data["dataset"] = args.dataset
    if args.rgb_dir:
        data["rgb_dir"] = args.rgb_dir
    for key in ("start", "end", "stride"):
        val = getattr(args, key)
        if val is not None:
            data[key] = val
    if args.output_dir:
        cfg.setdefault("output", {})["root"] = args.output_dir
    return cfg


def _build_sequence(cfg: dict):
    data = cfg["data"]
    return create_sequence(
        data.get("dataset", "fungraph3d"),
        dataset_root=data.get("dataset_root"),
        sequence=data.get("sequence", ""),
        config_path=data.get("dataset_config") or data.get("config_path"),
        rgb_dir=data.get("rgb_dir"),
        start=data.get("start", 0),
        end=data.get("end", -1),
        stride=data.get("stride", 1),
        desired_height=data.get("desired_height"),
        desired_width=data.get("desired_width"),
        load_depth=True,
        load_pose=True,
    )


def _label_set(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = list(value)
    return {str(x).strip().lower() for x in items if str(x).strip()}


def _maybe_unlink_outputs(out: Path, overwrite: bool) -> None:
    if not overwrite:
        return
    for path in [out / "summary.json", out / "scene.json", out / "global_atlas.json", out / "scene_history.jsonl"]:
        if path.exists():
            path.unlink()
    for directory in [out / "frames", out / "functional_raw"]:
        if directory.is_dir():
            for p in directory.glob("*"):
                if p.is_file():
                    p.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/frontend2d.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--dataset", choices=["rgb_folder", "fungraph3d", "scenefun3d"])
    parser.add_argument("--rgb-dir")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-models", action="store_true", help="Run schema/output path with no RAM++/LLM/SAM3.")
    args = parser.parse_args()

    cfg = _override(load_yaml_config(args.config), args)
    out = Path(cfg.get("output", {}).get("root", "outputs"))
    out.mkdir(parents=True, exist_ok=True)
    _maybe_unlink_outputs(out, args.overwrite)
    write_json_atomic(out / "run_config.json", cfg)
    if Path("SOURCE_MANIFEST.md").is_file():
        write_json_atomic(out / "source_manifest.json", {"source_manifest_md": Path("SOURCE_MANIFEST.md").read_text(encoding="utf-8")})

    seq = _build_sequence(cfg)
    rampp = llm = lock = sam3 = None
    if not args.no_models:
        r = cfg["rampp"]
        rampp = RamppTagger(
            ckpt_path=r["checkpoint"],
            device=r.get("device", "cuda"),
            image_size=int(r.get("image_size", 384)),
            vit=r.get("vit", "swin_l"),
            repo=r.get("repo"),
        )
        l = cfg["llm"]
        if bool(l.get("cache_only", False)):
            os.environ["DEEPSEEK_CACHE_ONLY"] = "1"
        llm = DeepSeekRuntime(
            api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY") or os.getenv("DEEPSEEK_TOKEN"),
            base_url=l.get("base_url", "https://api.deepseek.com"),
            model=l.get("model", "deepseek-chat"),
            cache_dir=l.get("cache_dir", ".cache/deepseek"),
        )
        s = cfg["scene_lock"]
        lock = SceneLock(
            m=int(s.get("m", 2)),
            conf_thresh=float(s.get("confidence_threshold", 0.8)),
            switch_m=int(s.get("switch_m", 2)),
            switch_conf_thresh=float(s.get("switch_threshold", s.get("confidence_threshold", 0.8))),
            switch_cooldown_frames=int(s.get("switch_cooldown", 20)),
        )
        sm = cfg["sam3"]
        sam3 = Sam3TwoStageRuntime(
            checkpoint_path=sm["checkpoint"],
            bpe_path=sm["bpe_path"],
            sam3_repo=sm.get("repo"),
            device=sm.get("device", "cuda"),
            confidence=float(sm.get("confidence", 0.3)),
            thr_o=float(sm.get("threshold_o", 0.5)),
            thr_c=float(sm.get("threshold_c", 0.5)),
            thr_u=float(sm.get("threshold_u", 0.5)),
            overlap_iou=float(sm.get("overlap_iou", 0.9)),
            u_cover_thr=float(sm.get("u_cover_threshold", 0.95)),
            u_cover_pad=float(sm.get("u_cover_pad", 0.0)),
            u_parent_contain_thr=float(sm.get("u_parent_contain_threshold", 0.9)),
            force_u=_label_set(sm.get("force_u", [])),
            drop_parent_without_u=bool(sm.get("drop_parent_without_u", True)),
            same_type_drop_multi_parent_links=bool(sm.get("same_type_drop_multi_parent_links", False)),
            processor_resolution=int(sm.get("processor_resolution", 1008)),
        )

    pipeline = Frontend2DPipeline(
        cfg,
        rampp=rampp,
        llm_runtime=llm,
        scene_lock=lock,
        sam3_runtime=sam3,
        policy=policy_from_config(cfg),
        output_dir=out,
    )
    summary = RunSummary()
    save_vis = bool(cfg.get("sam3", {}).get("save_visualization", False))
    for idx in range(len(seq)):
        frame = seq[idx]
        try:
            result = pipeline.process_frame(frame)
            summary.add_success(result, pipeline.global_atlas)
            if save_vis and sam3 is not None:
                # Reconstructing det from NPZ is not useful for labels; visualization is saved by original SAM3 IO
                # only in reference pipeline. This hook is intentionally left to future richer visual export.
                pass
            print(f"[ok] {idx + 1}/{len(seq)} {result.frame_key} det={len(result.detections)} scene={result.scene_type} locked={result.scene_locked}")
        except Exception as exc:
            summary.add_failure()
            print(f"[failed] {idx + 1}/{len(seq)} {frame.frame_key}: {exc}")
            raise
    summary.save(out / "summary.json")
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
