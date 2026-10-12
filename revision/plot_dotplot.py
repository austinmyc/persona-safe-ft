"""Regenerate the paper's Exp. 2 / Exp. 3 dot-plot figures (same design as the submission)
with JUDGED harmful-reply rates instead of keyword rates.

    python revision/plot_dotplot.py --summary results/judge_summary.csv --exp 3 --out figures/figure_exp3_domain_warmth_agree.pdf
    python revision/plot_dotplot.py --summary results_sharegpt/judge_summary.csv --exp 2 --out figures/figure_exp2_warm_ft_3k.pdf

Exp. 3 (MentalChat): Base (run 'base'), Warm FT (mc_warm_s1/s2 mean), Ours (mc_ours_s1/s2 mean).
Exp. 2 (ShareGPT):   Base, Warm FT, User-only, Ours (runs sg_<cond>_s3407 from judge_sharegpt.sh).
Model order is fixed (Llama, Qwen, Mistral, SmolLM) in every figure (reviewer prLP).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODELS = [("llama", "Llama-3.1-8B"), ("qwen", "Qwen2.5-7B\nInstruct"),
          ("mistral", "Mistral-7B\nInstruct-v0.3"), ("smollm", "SmolLM3-3B")]
STYLE = {  # label: (marker, face, edge, size)
    "Base": ("o", "white", "#666666", 46),
    "Generic Warmth FT (Ibrahim et al., 2026)": ("s", "#d4a0b0", "#d4a0b0", 34),
    "User-only": ("^", "#e8a33d", "#e8a33d", 38),
    "Ours": ("D", "#4a6aa5", "#4a6aa5", 34),
    "Ours (simple prompt)": ("D", "white", "#4a6aa5", 34),
}
EXP = {
    3: {"Base": ["base"], "Generic Warmth FT (Ibrahim et al., 2026)": ["mc_warm_s1", "mc_warm_s2"],
        "Ours": ["mc_ours_s1", "mc_ours_s2"]},
    2: {"Base": ["sg_base_s3407"], "Generic Warmth FT (Ibrahim et al., 2026)": ["sg_warm_s3407"],
        "User-only": ["sg_useronly_s3407"], "Ours": ["sg_ours_s3407"]},
}
PANELS = [("qi300", "(a) Harmful-instruction benchmark\n(Qi et al., 2024)"),
          ("redteam265", "(b) Student–teacher red-teaming prompts\n(Revilla Llaca et al., 2023)")]


def draw_panel(ax, d, s, runs_by_label, skip, title):
    for yi, (m, mname) in enumerate(MODELS):
        y = len(MODELS) - 1 - yi
        vals = {}
        for lab, runs in runs_by_label.items():
            if (m, lab.split(" (")[0]) in skip or (m, lab) in skip:
                continue
            v = d[(d.model == m) & (d.set == s) & (d.run.isin(runs))].value
            if len(v):
                vals[lab] = v.mean()
        if vals:
            ax.plot([min(vals.values()), max(vals.values())], [y, y], color="#cccccc", lw=1.2, zorder=1)
        for lab, v in vals.items():
            mk, fc, ec, sz = STYLE[lab]
            ax.scatter(v, y, marker=mk, s=sz, facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=3)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([n for _, n in MODELS][::-1], fontsize=7)
    ax.set_title(title, fontsize=7.5)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(axis="x", color="#e5e5e5")
    ax.set_xlim(left=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def combined(a):
    """Experiments 2 (ShareGPT) and 3 (MentalChat) in one 2x2 figure: rows = experiment."""
    import os
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    rows = [(a.summary2, 2, "Exp. 2 (ShareGPT)"), (a.summary, 3, "Exp. 3 (MentalChat16K)")]
    labels = []
    for r, (path, e, ename) in enumerate(rows):
        for c, (s, title) in enumerate(PANELS):
            ax = axes[r][c]
            t = f"{ename}: " + title.split("\n")[0][4:]
            if path and os.path.exists(path):
                d = pd.read_csv(path); d = d[d.metric == "harmful%"]
                draw_panel(ax, d, s, EXP[e], {tuple(x.split(":", 1)) for x in a.skip}, t)
                labels += [l for l in EXP[e] if l not in labels]
            else:
                ax.set_title(t, fontsize=7.5); ax.text(0.5, 0.5, "pending", ha="center", va="center",
                                                       transform=ax.transAxes, color="red")
                ax.set_yticks([])
            if r == 1:
                ax.set_xlabel("Harmful replies (%, LLM judge)", fontsize=7)
    order = ["Base", "Generic Warmth FT (Ibrahim et al., 2026)", "User-only", "Ours (simple prompt)", "Ours"]
    labels = [l for l in order if l in labels] or [l for l in order if l in EXP[3]]
    handles = [plt.scatter([], [], marker=STYLE[l][0], s=STYLE[l][3], facecolor=STYLE[l][1],
                           edgecolor=STYLE[l][2], linewidth=1.3, label=l) for l in labels]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.out, bbox_inches="tight")
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--exp", type=int, choices=[2, 3, 23], required=True,
                    help="23: Exp. 2 and Exp. 3 in one 2x2 figure (--summary = Exp. 3, --summary2 = Exp. 2)")
    ap.add_argument("--summary2", default=None, help="Exp. 2 judge_summary.csv for --exp 23")
    ap.add_argument("--out", required=True)
    ap.add_argument("--framework_as_ours", action="store_true",
                    help="Exp. 3: plot low-A + framework (mc_ours_framework_s1/s2) as 'Ours' and the "
                         "submitted prompt as 'Ours (simple prompt)'")
    ap.add_argument("--skip", nargs="*", default=["llama:Base"],
                    help="model:label cells to omit (Llama-3.1-8B base is not a chat model)")
    a = ap.parse_args()
    if a.framework_as_ours:
        EXP[3] = {"Base": ["base"],
                  "Generic Warmth FT (Ibrahim et al., 2026)": ["mc_warm_s1", "mc_warm_s2"],
                  "Ours (simple prompt)": ["mc_ours_s1", "mc_ours_s2"],
                  "Ours": ["mc_ours_framework_s1", "mc_ours_framework_s2"]}
    if a.exp == 23:
        return combined(a)
    d = pd.read_csv(a.summary)
    d = d[d.metric == "harmful%"]
    skip = {tuple(x.split(":", 1)) for x in a.skip}
    if a.framework_as_ours and a.exp == 3:
        EXP[3] = {"Base": ["base"],
                  "Generic Warmth FT (Ibrahim et al., 2026)": ["mc_warm_s1", "mc_warm_s2"],
                  "Ours (simple prompt)": ["mc_ours_s1", "mc_ours_s2"],
                  "Ours": ["mc_ours_framework_s1", "mc_ours_framework_s2"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, (s, title) in zip(axes, PANELS):
        for yi, (m, mname) in enumerate(MODELS):
            y = len(MODELS) - 1 - yi
            vals = {}
            for lab, runs in EXP[a.exp].items():
                if (m, lab.split(" (")[0]) in skip or (m, lab) in skip:
                    continue
                v = d[(d.model == m) & (d.set == s) & (d.run.isin(runs))].value
                if len(v):
                    vals[lab] = v.mean()
            if vals:
                ax.plot([min(vals.values()), max(vals.values())], [y, y], color="#cccccc", lw=1.2, zorder=1)
            for lab, v in vals.items():
                mk, fc, ec, sz = STYLE[lab]
                ax.scatter(v, y, marker=mk, s=sz, facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=3)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([n for _, n in MODELS][::-1], fontsize=7.5)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Harmful replies (%, LLM judge)", fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", color="#e5e5e5")
        ax.set_xlim(left=0)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    handles = [plt.scatter([], [], marker=STYLE[l][0], s=STYLE[l][3], facecolor=STYLE[l][1],
                           edgecolor=STYLE[l][2], linewidth=1.3, label=l) for l in EXP[a.exp]]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(a.out, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
