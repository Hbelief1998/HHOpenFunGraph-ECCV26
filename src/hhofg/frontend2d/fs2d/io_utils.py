from __future__ import annotations

# 本模块提供语义数据的 JSON 持久化工具：原子写入与 JSONL 追加。

import json
import os
import tempfile
from pathlib import Path
from typing import Any

try:  # orjson 是 C 实现，序列化大字典时会主动释放 GIL，
    # 用在后台 snapshot worker 里能避免饿死主线程的 PyTorch 派发。
    import orjson  # type: ignore
    _HAS_ORJSON = True
except Exception:  # pragma: no cover
    _HAS_ORJSON = False


def write_json_atomic(path: Path, payload: Any, *, indent: int | None = 2) -> None:
    # 将路径统一为 Path，便于后续 mkdir/replace 等操作。
    path = Path(path).expanduser().resolve()
    # 确保目录存在，避免写入失败。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 使用临时文件写入，再用原子替换保障一致性。
    tmp_path = None
    use_orjson = _HAS_ORJSON and indent in (None, 2)
    try:
        if use_orjson:
            # orjson 路径：先在内存里序列化（C 实现，会释放 GIL），再写字节。
            opt = orjson.OPT_NON_STR_KEYS
            if indent == 2:
                opt |= orjson.OPT_INDENT_2
            data = orjson.dumps(payload, option=opt)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmp_path = f.name
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        else:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmp_path = f.name
                json.dump(payload, f, ensure_ascii=False, indent=indent)
                f.flush()
                os.fsync(f.fileno())
        # os.replace 在多数平台上是原子操作，避免部分写入导致的脏文件。
        try:
            os.replace(tmp_path, path)
        except FileNotFoundError:
            # 某些运行环境下临时文件在 replace 前会异常丢失；退化为直接重写目标文件，
            # 优先保证线上流程可持续运行。
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=indent)
    finally:
        # 如果替换前异常，清理残留临时文件。
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def append_jsonl(path: Path, payload: Any) -> None:
    # 以 JSON Lines 形式追加写入（每行一个 JSON）。
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")
