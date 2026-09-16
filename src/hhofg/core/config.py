from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def recursive_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = recursive_update(out[key], value)
        else:
            out[key] = value
    return out


def expand_config_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: expand_config_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_config_vars(v) for v in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent_key = cfg.pop("inherit_from", None) or cfg.pop("defaults", None)
    if parent_key:
        parent_path = Path(parent_key)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        cfg = recursive_update(load_yaml_config(parent_path), cfg)
    return expand_config_vars(cfg)
