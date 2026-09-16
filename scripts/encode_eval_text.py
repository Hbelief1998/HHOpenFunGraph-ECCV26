#!/usr/bin/env python3
"""Encode evaluation query text in an isolated, model-compatible environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("clip", "bert"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def encode_clip(texts: list[str], args: argparse.Namespace) -> np.ndarray:
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(
        args.model, local_files_only=Path(args.model).exists()
    ).to(args.device)
    processor = CLIPProcessor.from_pretrained(
        args.model, local_files_only=Path(args.model).exists()
    )
    model.eval()
    rows = []
    # The reference evaluator also issues one CLIP query at a time.  Matching
    # that path avoids batch-dependent floating-point differences at the hard
    # cosine threshold.
    for text in texts:
        inputs = processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        )
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.inference_mode():
            rows.append(model.get_text_features(**inputs).detach().cpu().numpy())
    return np.concatenate(rows, axis=0) if rows else np.empty((0, 512), dtype=np.float32)


def encode_bert(texts: list[str], args: argparse.Namespace) -> np.ndarray:
    from transformers import BertModel, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(
        args.model, local_files_only=Path(args.model).exists()
    )
    model = BertModel.from_pretrained(
        args.model, local_files_only=Path(args.model).exists()
    ).to(args.device)
    model.eval()
    rows = []
    # OpenFunGraph's evaluator encodes one predicate at a time and then takes
    # an unmasked mean over last_hidden_state.  Batching unequal-length texts
    # would therefore average padded token states and change the retrieval
    # ranking.  Keep the seemingly less efficient single-query path here so
    # the metric is numerically faithful to the reference implementation.
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.inference_mode():
            rows.append(
                model(**inputs).last_hidden_state.mean(dim=1).detach().cpu().numpy()
            )
    return np.concatenate(rows, axis=0) if rows else np.empty((0, 768), dtype=np.float32)


def main() -> int:
    args = parse_args()
    texts = list(json.loads(Path(args.input).read_text(encoding="utf-8")))
    embeddings = encode_clip(texts, args) if args.kind == "clip" else encode_bert(texts, args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.output).suffix == ".npy":
        np.save(args.output, embeddings)
    else:
        np.savez_compressed(args.output, texts=np.asarray(texts), embeddings=embeddings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
