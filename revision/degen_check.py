"""Flag degenerate generations (repetition loops, broken bytes, leaked chat tokens).

Run after every generation round, BEFORE judging. On 12 Oct we found that the same
checkpoints give clean text on the RTX 4090 pod but 3-30% loops on the L20 server for
Llama and Mistral (long prompts / long outputs), so check every machine's output.

    python revision/degen_check.py                       # table: % degenerate per run x set
    python revision/degen_check.py --min 3               # only cells above 3%
    python revision/degen_check.py --out results/degenerate.csv   # per-reply flags

A reply is degenerate if it has >= 30 words and < 35% distinct word 4-grams, or contains
U+FFFD, or leaks chat-template tokens (im_start, [INST], <|...|>).
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

LEAK = re.compile(r"im_start|im_end|\[/?INST\]|<\|[a-z_]+\|>")


def degenerate(text):
    if "�" in text or LEAK.search(text):
        return True
    w = text.split()
    if len(w) < 30:
        return False
    grams = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(set(grams)) / len(grams) < 0.35


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default="responses")
    ap.add_argument("--min", type=float, default=0.0, help="only print cells at or above this %")
    ap.add_argument("--out", default=None, help="optional per-reply CSV")
    args = ap.parse_args()

    cells, rows = defaultdict(dict), []
    for path in sorted(glob.glob(f"{args.responses}/*/*/*.jsonl")):
        model, run, fname = path.split(os.sep)[-3:]
        s = fname[:-6]
        n = bad = 0
        with open(path, encoding="utf-8") as f:
            for k, line in enumerate(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                texts = r.get("responses") or [r.get("response", "")]
                flag = any(degenerate(t or "") for t in texts)
                n += 1
                bad += flag
                if args.out:
                    rows.append([model, run, s, r.get("id", k), int(flag)])
        cells[(model, run)][s] = 100 * bad / max(1, n)

    sets = sorted({s for v in cells.values() for s in v})
    print(f"{'model':8s} {'run':32s}" + "".join(f"{s[:10]:>11s}" for s in sets))
    for (model, run), v in sorted(cells.items()):
        if max(v.values()) < args.min:
            continue
        print(f"{model:8s} {run:32s}" + "".join(f"{v[s]:10.1f}%" if s in v else f"{'':>11s}" for s in sets))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "run", "set", "prompt_id", "degenerate"])
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
