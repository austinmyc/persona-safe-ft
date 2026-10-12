"""Held-out selection set for the trait-selection pilot: JailbreakBench JBB-Behaviors, 100 harmful
behaviours (Chao et al., 2024). Not used anywhere else in the paper's evaluation, so selecting the
trait on it is not circular with the main benchmarks (Qi et al.; red-teaming).

    python revision/gemma/prepare_jbb.py            # -> evalsets/jbb100.jsonl
If the machine has no internet, download the CSV elsewhere and pass --csv harmful-behaviors.csv.
"""
import argparse
import csv
import io
import json
import os
import urllib.request

URL = "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default="evalsets/jbb100.jsonl")
    a = ap.parse_args()
    text = open(a.csv, encoding="utf-8").read() if a.csv else urllib.request.urlopen(URL, timeout=120).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": f"jbb{i}", "category": r.get("Category", ""), "source": r.get("Source", ""),
                                "messages": [{"role": "user", "content": r["Goal"]}]}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts to {a.out}")


if __name__ == "__main__":
    main()
