"""Load training files as a list of dicts: JSON array or JSONL (one object per line)."""

from __future__ import annotations

import json
from typing import Any


def load_train_records(path: str) -> list[dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data)}")
    return data
