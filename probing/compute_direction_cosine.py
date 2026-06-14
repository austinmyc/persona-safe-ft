#!/usr/bin/env python3
"""Aggregate per-backbone probe_a_cosines.json into the |cos theta| table.

Per backbone, drop layer 0, the last layer, and interior layers whose base ||v_warm||
is below 10% of the median interior norm, then take the middle-50% trimmed mean of
|cos(v_warm, v_comply)| over the rest. Needs the v_warm vectors, so run
extract_activations.py without --compact-a.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

NORM_THRESH_FRAC = 0.10  # drop interior layers with ||v_warm|| below 10% of median


def kept_layers(data: Dict[str, Any]) -> List[int]:
    n = int(data["num_layers"])
    base = data["variants"]["base"]
    interior = list(range(1, n - 1))
    norms = {l: np.linalg.norm(np.array(base["v_warm"][str(l)])) for l in interior}
    thresh = NORM_THRESH_FRAC * float(np.median(list(norms.values())))
    return [l for l in interior if norms[l] >= thresh]


def trimmed_mid50_mean(values: List[float]) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    k = len(v) // 4  # drop bottom and top quartile
    middle = v[k:len(v) - k] if len(v) - 2 * k > 0 else v
    return float(middle.mean())


def aggregate(data: Dict[str, Any]) -> Dict[str, float]:
    layers = kept_layers(data)
    out: Dict[str, float] = {}
    for variant in ("base", "warm_sft", "ours"):
        cos = data["variants"][variant]["cos_abs_by_layer"]
        out[variant] = trimmed_mid50_mean([float(cos[str(l)]) for l in layers])
    return out


def fmt(x: float) -> str:
    if x is None or math.isnan(x) or math.isinf(x):
        return "{---}"
    return f"{x:.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate probe_a_cosines.json -> paper cosine table")
    p.add_argument(
        "--manifest",
        required=True,
        help='JSON list: [{"id":"llama31_8b","label":"Llama 3.1-8B","path":".../probe_a_cosines.json"}, ...]',
    )
    p.add_argument("--latex-out", default=None, help="Write LaTeX tabular body rows to this file")
    args = p.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        cols: List[Dict[str, str]] = json.load(f)

    table: Dict[str, Dict[str, float]] = {}
    labels: Dict[str, str] = {}
    for col in cols:
        cid = col["id"]
        labels[cid] = col.get("label", cid)
        with open(col["path"], "r", encoding="utf-8") as f:
            table[cid] = aggregate(json.load(f))

    rows = [("Base", "base"), ("Warm-empathetic SFT", "warm_sft"), ("Ours", "ours")]

    header = ["Condition"] + [labels[c["id"]] for c in cols]
    print(" | ".join(f"{h:22}" for h in header))
    print("-" * (22 * len(header) + 3 * (len(header) - 1)))
    for label, key in rows:
        cells = [f"{label:22}"] + [fmt(table[c["id"]][key]) for c in cols]
        print(" | ".join(cells))

    tex = "\n".join(
        f"{label:<20} & " + " & ".join(fmt(table[c["id"]][key]) for c in cols) + r" \\"
        for label, key in rows
    )
    print("\n=== LaTeX (tabular rows) ===\n")
    print(tex)
    if args.latex_out:
        Path(args.latex_out).write_text(tex + "\n", encoding="utf-8")
        print(f"\nWrote {args.latex_out}")


if __name__ == "__main__":
    main()
