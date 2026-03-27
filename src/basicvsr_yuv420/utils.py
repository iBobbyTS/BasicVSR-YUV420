from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Union[str, Path]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def atomic_write_bytes(data: bytes, path: Union[str, Path]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=str(output_path.parent),
        prefix="{}.tmp.".format(output_path.name),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
    except Exception:
        try:
            Path(temp_path).unlink()
        except FileNotFoundError:
            pass
        raise


def write_json(data: Mapping[str, Any], path: Union[str, Path]) -> None:
    payload = json.dumps(data, indent=2).encode("utf-8")
    atomic_write_bytes(payload, path)


def read_json(path: Union[str, Path], default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        return {} if default is None else default
    return json.loads(input_path.read_text(encoding="utf-8"))
