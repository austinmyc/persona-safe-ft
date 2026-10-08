"""Paired significance tests between two conditions (Appendix Table `tab:stats`).

Reads results/perprompt.csv from revision/score.py and, for each eval set and model,
compares condition A with condition B on the same prompts:
  Delta = rate(A) - rate(B) in pp (positive = B safer) with a 95% paired-bootstrap CI,
  Cohen's h, exact McNemar p, and Holm-adjusted p across models within each set.

  python revision/stats.py --a warm --b ours
  python revision/stats.py --a warm --b placebo --sets qi300 redteam265
  python revision/stats.py --a warm --b ours --seed 1          # a specific seed
"""
import argparse
import csv
import math
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest


def load(path):
    with open(path) as f:
        return {r["prompt_id"]: int(r["outcome"]) for r in csv.DictReader(f)}


def mcnemar_exact(a, b):
    """a, b: aligned 0/1 arrays. Exact two-sided McNemar on discordant pairs."""
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    if n10 + n01 == 0:
        return 1.0, n10, n01
    return binomtest(n10, n10 + n01, 0.5).pvalue, n10, n01


def cohens_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def paired_bootstrap_ci(a, b, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (a[idx].mean(1) - b[idx].mean(1)) * 100
    return np.percentile(diffs, [2.5, 97.5])


def holm(pvals):
    """Holm step-down adjusted p-values, returned in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perprompt", default="results/perprompt.csv")
    ap.add_argument("--a", default="warm", help="reference condition")
    ap.add_argument("--b", default="ours", help="comparison condition")
    ap.add_argument("--seed", default="3407")
    ap.add_argument("--sets", nargs="*", default=["qi300", "redteam265"])
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--out", default=None, help="optional CSV of the table")
    args = ap.parse_args()

    data = {}  # (set, model, cond) -> {prompt_id: outcome}
    with open(args.perprompt) as f:
        for r in csv.DictReader(f):
            if r["set"] not in args.sets:
                continue
            if r["cond"] != "base" and r["seed"] != args.seed:
                continue
            data.setdefault((r["set"], r["model"], r["cond"]), {})[r["prompt_id"]] = int(r["outcome"])

    print(f"Delta = {args.a} - {args.b} (pp); positive means {args.b} is safer; seed {args.seed}\n")
    table = []
    for set_name in args.sets:
        rows = []
        for model in sorted({k[1] for k in data if k[0] == set_name}):
            da, db = data.get((set_name, model, args.a)), data.get((set_name, model, args.b))
            if not da or not db:
                continue
            ids = sorted(set(da) & set(db))
            a = np.array([da[i] for i in ids]); b = np.array([db[i] for i in ids])
            p, n10, n01 = mcnemar_exact(a, b)
            lo, hi = paired_bootstrap_ci(a, b, args.n_boot)
            rows.append(dict(model=model, n=len(ids), ra=a.mean() * 100, rb=b.mean() * 100,
                             delta=(a.mean() - b.mean()) * 100, lo=lo, hi=hi,
                             h=cohens_h(a.mean(), b.mean()), p=p, n10=n10, n01=n01))
        if not rows:
            continue
        for r, padj in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = padj
        print(f"== {set_name} ==")
        print(f"{'model':10s} {'n':>4s} {args.a:>9s} {args.b:>9s} {'Delta':>7s} {'95% CI':>17s} {'h':>6s} {'disc':>9s} {'p':>9s} {'p_holm':>9s}")
        for r in rows:
            sig = "*" if r["p_holm"] < 0.05 else ""
            print(f"{r['model']:10s} {r['n']:4d} {r['ra']:9.2f} {r['rb']:9.2f} {r['delta']:7.2f} "
                  f"[{r['lo']:6.2f}, {r['hi']:6.2f}] {r['h']:6.2f} {r['n10']:4d}/{r['n01']:<4d} "
                  f"{r['p']:9.2e} {r['p_holm']:9.2e}{sig}")
            table.append(dict(set=set_name, a=args.a, b=args.b, **r))
        print()
    if args.out and table:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)


if __name__ == "__main__":
    main()
