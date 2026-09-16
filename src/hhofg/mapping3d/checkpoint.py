from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hhofg.core.serialization import write_json_atomic

from .map_state import MapState3D

CHECKPOINT_SCHEMA_VERSION = 1


def stable_json_sha(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_sha(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root: str | Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    except Exception:
        return None


def latest_checkpoint(checkpoint_root: str | Path) -> Path | None:
    root = Path(checkpoint_root)
    if not root.is_dir():
        return None
    candidates = []
    for p in root.glob("frame_*"):
        if (p / "metadata.json").is_file() and (p / "state.pkl").is_file():
            candidates.append(p)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def save_checkpoint(
    *,
    checkpoint_root: str | Path,
    frame_key: str,
    state: MapState3D,
    processed_frame_keys: list[str],
    jsonl_frame_keys: dict[str, list[str]],
    cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    frontend_run: str | Path,
    frontend_metadata: dict[str, Any],
    repo_root: str | Path,
    edge_optimizer_state: Any = None,
    hierarchy_state: Any = None,
) -> Path:
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"frame_{frame_key}"
    tmp = root / f".frame_{frame_key}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    meta = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "git_commit": git_commit(repo_root),
        "effective_config_sha": stable_json_sha(cfg),
        "effective_data_sha": stable_json_sha(data_cfg),
        "frontend_run": str(Path(frontend_run).resolve()),
        "frontend_metadata_sha": stable_json_sha(frontend_metadata),
        "processed_frame_keys": list(processed_frame_keys),
        "jsonl_frame_keys": {k: list(v) for k, v in jsonl_frame_keys.items()},
        "next_role_id": dict(state.next_role_id),
        "num_nodes": len(state.nodes),
    }
    write_json_atomic(tmp / "metadata.json", meta)
    with (tmp / "state.pkl").open("wb") as f:
        pickle.dump({"map_state": state, "edge_optimizer_state": edge_optimizer_state, "hierarchy_state": hierarchy_state}, f, protocol=pickle.HIGHEST_PROTOCOL)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return dest


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    frontend_run: str | Path,
    frontend_metadata: dict[str, Any],
) -> tuple[MapState3D, dict[str, Any], Any, Any]:
    path = Path(checkpoint_path)
    with (path / "metadata.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if int(meta.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"checkpoint schema mismatch: {path}")
    checks = {
        "effective_config_sha": stable_json_sha(cfg),
        "effective_data_sha": stable_json_sha(data_cfg),
        "frontend_run": str(Path(frontend_run).resolve()),
        "frontend_metadata_sha": stable_json_sha(frontend_metadata),
    }
    for key, expected in checks.items():
        if meta.get(key) != expected:
            raise ValueError(f"checkpoint {key} mismatch: expected {expected}, got {meta.get(key)}")
    with (path / "state.pkl").open("rb") as f:
        payload = pickle.load(f)
    return payload["map_state"], meta, payload.get("edge_optimizer_state"), payload.get("hierarchy_state")


def save_final_map_state(path: str | Path, state: MapState3D) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as f:
        pickle.dump({"map_state": state}, f, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)


def load_final_map_state(mapping_run: str | Path) -> MapState3D:
    run = Path(mapping_run)
    direct = run / "map" / "map_state.pkl"
    if direct.is_file():
        with direct.open("rb") as f:
            return pickle.load(f)["map_state"]
    checkpoint = latest_checkpoint(run / "checkpoints")
    if checkpoint is None:
        raise FileNotFoundError(f"no final MapState or checkpoint found under {run}")
    with (checkpoint / "state.pkl").open("rb") as f:
        return pickle.load(f)["map_state"]
