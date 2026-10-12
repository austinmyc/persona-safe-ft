"""Gemma-2-2B selection replication: harmful-reply rate per trait subset, ranking, and paired tests
of low agreeableness against every other subset (exact McNemar, Holm across comparisons).

    python revision/gemma/report.py --results results_gemma
"""
import argparse
import pandas as pd
from scipy.stats import binomtest


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0] * len(ps); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_gemma")
    ap.add_argument("--ref", default="pilot_low_A_s3407")
    a = ap.parse_args()
    d = pd.read_csv(f"{a.results}/judge_perprompt.csv")
    d = d[d.model == "gemma"]
    for st in ["redteam265", "qi300"]:
        x = d[d.set == st]
        w = x.pivot_table(index="prompt_id", columns="run", values="outcome", aggfunc="first")
        rate = (w.mean() * 100).sort_values()
        print(f"\n== {st}: harmful replies (%), lower is better; rank 1 = safest")
        for r, (run, v) in enumerate(rate.items(), 1):
            print(f"  {r:2d}. {run:24s} {v:5.1f}")
        if a.ref not in w:
            print(f"  ({a.ref} missing)"); continue
        others = [c for c in w.columns if c != a.ref and c != "base"]
        ps, rows = [], []
        for c in others:
            both = w[[a.ref, c]].dropna()
            n10 = int(((both[a.ref] == 1) & (both[c] == 0)).sum()); n01 = int(((both[a.ref] == 0) & (both[c] == 1)).sum())
            p = binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0
            ps.append(p); rows.append((c, both[a.ref].mean() * 100, both[c].mean() * 100, n10, n01))
        for (c, ra, rc, n10, n01), p, ph in zip(rows, ps, holm(ps)):
            print(f"  low_A {ra:5.1f} vs {c:24s} {rc:5.1f}   discordant {n10}/{n01}  p={p:.3g}  p_Holm={ph:.3g}")
    lab = d[d.set == "redteam265"].groupby(["run", "label"]).size().unstack(fill_value=0)
    print("\n== red-teaming reply styles (%)\n", (lab.div(lab.sum(1), axis=0) * 100).round(1))


if __name__ == "__main__":
    main()
