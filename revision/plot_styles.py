"""Fig. 3: judged reply style on the harmful instructions (Qi 300), Experiment 3, mean over the FOUR
models (Llama Base = the Alpaca-format run). Warm FT and Ours: mean of seeds 1-2.

    python revision/plot_styles.py --summary results/judge_summary.csv results_wf/judge_summary.csv --out figures/reply_styles.pdf
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = [("explicit_refusal", "Explicit refusal", "#2f5597"), ("soft_refusal", "Soft refusal", "#8faadc"),
          ("deflection", "Deflection", "#d9d9d9"), ("partial_compliance", "Partial compliance", "#f4b183"),
          ("full_compliance", "Full compliance", "#c00000")]
# grouped as in Table 1 (None = gap between groups)
CONDS = [("Base (no fine-tuning)", ["base"]), ("Warm FT", ["mc_warm_s1", "mc_warm_s2"]),
         ("Ours", ["mc_ours_s1", "mc_ours_s2"]), None,
         ("Paraphrase", ["mc_placebo_s3407"]), ("High-A", ["mc_high_a_s3407"]), ("Low-E", ["mc_low_e_s3407"]), None,
         ("Ours \u2212 clause", ["mc_ours_noclause_s3407"]), ("Warm FT + clause", ["mc_warm_clause_s3407"]), None,
         ("Ours + framework", ["mc_ours_framework_s3407"]), ("Framework only", ["mc_warm_framework_s3407"]), None,
         ("Warm FT + mixing", ["mc_warm_mix_s3407"]), ("Ours + mixing", ["mc_ours_mix_s3407"])]
MODELS = ["llama", "qwen", "mistral", "smollm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama_base", default="base_alpaca")
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(f).set_axis(["model", "run", "set", "n", "metric", "value"], axis=1) for f in a.summary])
    if a.llama_base and (d.run == a.llama_base).any():
        d = d[~((d.model == "llama") & (d.run == "base"))].copy()
        d.loc[(d.model == "llama") & (d.run == a.llama_base), "run"] = "base"
    d = d[(d.set == "qi300") & d.metric.str.startswith("label:")]
    rows = []
    for c in CONDS:
        if c is None:
            rows.append(None); continue
        name, runs = c
        x = d[d.run.isin(runs) & d.model.isin(MODELS)]
        if x.model.nunique() < 4:
            print("skip (not all 4 models):", name); continue
        rows.append((name, [x[x.metric == "label:" + k].groupby("model").value.mean().reindex(MODELS).fillna(0).mean()
                            for k, _, _ in LABELS]))
    ypos, y, labels = [], 0.0, []
    for r in rows[::-1]:
        if r is None:
            y += 0.45; continue
        ypos.append(y); labels.append(r); y += 1.0
    fig, ax = plt.subplots(figsize=(3.4, 0.27 * y + 1.0))
    for yy, (name, vals) in zip(ypos, labels):
        left = 0
        for v, (_, _, col) in zip(vals, LABELS):
            ax.barh(yy, v, left=left, color=col, edgecolor="white", linewidth=0.4, height=0.8); left += v
    ax.set_yticks(ypos)
    ax.set_yticklabels([n for n, _ in labels], fontsize=8.5)
    ax.set_ylim(-0.6, y - 0.4)
    for t in ax.get_yticklabels():
        if t.get_text() in ("Ours", "Base (no fine-tuning)"):
            t.set_fontweight("bold")
    ax.set_xlim(0, 100); ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("% of replies to harmful instructions", fontsize=8.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in LABELS], labels=[l for _, l, _ in LABELS],
              fontsize=7.5, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(); fig.savefig(a.out, bbox_inches="tight"); print("wrote", a.out)
    for r in rows:
        if r is None: continue
        name, vals = r
        print(f"{name:24s}", " ".join(f"{v:5.1f}" for v in vals))


if __name__ == "__main__":
    main()
