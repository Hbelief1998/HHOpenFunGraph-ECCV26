#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hhofg.eval.fungraph3d_recall import (  # noqa: E402
    SemanticRetrieval,
    evaluate_fungraph3d,
    load_paper_benchmark_gt,
    load_hhofg_prediction,
    read_fungraph3d_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HHOpenFunGraph node and functional-triplet recall on FunGraph3D."
    )
    parser.add_argument("--dataset-root", default=os.environ.get("FUNGRAPH3D_ROOT", str(ROOT / "data/OpenFunGraph")))
    parser.add_argument("--outputs-root", default=str(ROOT / "outputs"))
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="SEQUENCE=GRAPH_JSON",
        help="Explicit prediction; repeat for multiple sequences.",
    )
    parser.add_argument(
        "--output-pattern",
        default="{sequence_slug}_hierarchy_lifting_full/mapping3d/map/final_hierarchical_graph.json",
        help="Path below --outputs-root used for split sequences not supplied via --prediction.",
    )
    parser.add_argument("--output", default=str(ROOT / "outputs" / "fungraph3d_paper_eval.json"))
    parser.add_argument("--clip-model", default=os.environ.get("HH_OFG_CLIP_EVAL_MODEL", "openai/clip-vit-base-patch16"))
    parser.add_argument("--bert-model", default=os.environ.get("HH_OFG_BERT_EVAL_MODEL", "google-bert/bert-base-uncased"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--embedding-python",
        default=os.environ.get("HH_OFG_EMBEDDING_PYTHON", sys.executable),
        help=(
            "Python with compatible torch/transformers used only to encode text queries "
            "(default: HH_OFG_EMBEDDING_PYTHON or the current interpreter)."
        ),
    )
    return parser.parse_args()


def _explicit_predictions(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--prediction must be SEQUENCE=GRAPH_JSON, got: {value}")
        sequence, path = value.split("=", 1)
        result[sequence.strip()] = Path(path).expanduser()
    return result


def _encode_text_external(
    texts: list[str],
    *,
    kind: str,
    model_path: str,
    device: str,
    batch_size: int,
    python_executable: str,
) -> dict[str, np.ndarray]:
    if not texts:
        return {}
    python_path = Path(python_executable)
    if not python_path.is_file():
        raise FileNotFoundError(f"embedding Python not found: {python_path}")
    helper = ROOT / "scripts" / "encode_eval_text.py"
    with tempfile.TemporaryDirectory(prefix="hhofg_eval_text_") as temp_dir:
        temp = Path(temp_dir)
        input_path = temp / "texts.json"
        output_path = temp / "embeddings.npz"
        input_path.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        subprocess.run(
            [
                str(python_path),
                str(helper),
                "--kind",
                kind,
                "--model",
                model_path,
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--device",
                device,
                "--batch-size",
                str(batch_size),
            ],
            check=True,
        )
        payload = np.load(output_path, allow_pickle=False)
        encoded_texts = payload["texts"].tolist()
        embeddings = payload["embeddings"]
    return {str(text): embeddings[idx] for idx, text in enumerate(encoded_texts)}


def _print_summary(report: dict) -> None:
    for group, subsets in (("nodes", ("O", "C", "U", "tabletop", "overall")),
                           ("edges", ("hierarchy", "tabletop", "overall"))):
        print(f"\n{group.capitalize()} recall")
        for subset in subsets:
            item = report["summary"][group][subset]
            if group == "edges":
                item = item["triplet"]
            print(f"  {subset:10s} {item['matched_count']}/{item['gt_count']} = {100 * item['recall']:.1f}%")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser()
    scene_root = dataset_root / "FunGraph3D"
    split_path = scene_root / "OpenFunGraph_split.txt"
    sequences = read_fungraph3d_split(split_path)
    benchmark = load_paper_benchmark_gt(
        annotations_path=dataset_root / "annotations.json",
        relations_path=dataset_root / "relations.json",
        sequences=sequences,
    )

    predictions = _explicit_predictions(args.prediction)
    outputs_root = Path(args.outputs_root).expanduser()
    for sequence in sequences:
        if sequence in predictions:
            continue
        relative = args.output_pattern.format(
            sequence=sequence, sequence_slug=sequence.replace("/", "_")
        )
        candidate = outputs_root / relative
        if candidate.is_file():
            predictions[sequence] = candidate
    missing = [sequence for sequence in sequences if sequence not in predictions]
    if missing:
        raise FileNotFoundError(
            "missing full-sequence predictions for: " + ", ".join(missing)
        )
    if not predictions:
        raise FileNotFoundError("no prediction graph files were found")
    for sequence, path in predictions.items():
        if not path.is_file():
            raise FileNotFoundError(f"prediction for {sequence} not found: {path}")

    graph_payloads = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in predictions.values()
    ]
    map_labels: dict[Path, dict[str, str]] = {}
    for path, graph in graph_payloads:
        if any(isinstance(node, str) for node in graph.get("nodes", [])):
            map_payload = _read_json(path.parent / "map_nodes.json")
            map_labels[path] = {
                str(node["node_id"]): str(node.get("top_label", ""))
                for node in map_payload.get("nodes", [])
            }
    pred_labels = sorted(
        {
            str(node.get("label", ""))
            if isinstance(node, dict)
            else map_labels[path].get(str(node), "")
            for path, graph in graph_payloads
            for node in graph.get("nodes", [])
            if (
                str(node.get("label", ""))
                if isinstance(node, dict)
                else map_labels[path].get(str(node), "")
            )
        }
    )
    pred_relations = sorted(
        {
            str(edge.get("relation_text") or edge.get("description") or edge.get("relation") or "").strip()
            for _path, graph in graph_payloads
            for edge in graph.get("edges", [])
            if str(edge.get("relation_text") or edge.get("description") or edge.get("relation") or "").strip()
        }
    )
    atlas_paths = {
        sequence: outputs_root / f"{sequence.replace('/', '_')}_frontend_full" / "global_atlas.json"
        for sequence in predictions
    }
    atlas_paths = {seq: path for seq, path in atlas_paths.items() if path.is_file()}
    closure_relations = set(pred_relations)
    for sequence, path in predictions.items():
        _, edges, _ = load_hhofg_prediction(
            path, scene_id=sequence.split('/')[0],
            atlas_path=atlas_paths.get(sequence),
        )
        closure_relations.update(edge.relation for edge in edges)
    pred_relations = sorted(closure_relations)
    embedding_root = dataset_root / "RootGT_Eval"
    label_vocabulary = _read_json(embedding_root / "all_labels.json")
    relation_vocabulary = _read_json(embedding_root / "all_edges.json")
    retrieval = SemanticRetrieval(
        label_vocabulary=label_vocabulary,
        label_embeddings=np.load(embedding_root / "all_labels_clip_embeddings.npy"),
        relation_vocabulary=relation_vocabulary,
        relation_embeddings=np.load(embedding_root / "all_edges_bert_embeddings.npy"),
        predicted_label_embeddings=_encode_text_external(
            pred_labels,
            kind="clip",
            model_path=args.clip_model,
            device=args.device,
            batch_size=args.batch_size,
            python_executable=args.embedding_python,
        ),
        predicted_relation_embeddings=_encode_text_external(
            pred_relations,
            kind="bert",
            model_path=args.bert_model,
            device=args.device,
            batch_size=args.batch_size,
            python_executable=args.embedding_python,
        ),
    )
    retrieval.label_hit = functools.lru_cache(None)(retrieval.label_hit)
    retrieval.relation_hit = functools.lru_cache(None)(retrieval.relation_hit)
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = evaluate_fungraph3d(
        benchmark=benchmark, scene_root=scene_root,
        sequence_predictions={sequence: predictions[sequence] for sequence in sequences},
        retrieval=retrieval, atlas_by_sequence=atlas_paths,
    )
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_summary(report)
    print(f"Detailed report: {output_path}")
    return 0


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
