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
CONDS = [("Base (no fine-tuning)", ["base"]), ("Warm FT", ["mc_warm_s1", "mc_warm_s2"]),
         ("Warm FT + clause", ["mc_warm_clause_s3407"]), ("Paraphrase", ["mc_placebo_s3407"]),
         ("High-A user", ["mc_high_a_s3407"]), ("Low-E user", ["mc_low_e_s3407"]),
         ("Ours, no clause", ["mc_ours_noclause_s3407"]),
         ("Ours", ["mc_ours_s1", "mc_ours_s2"]), ("Ours + framework", ["mc_ours_framework_s3407"]),
         ("Framework only", ["mc_warm_framework_s3407"]),
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
    for name, runs in CONDS:
        x = d[d.run.isin(runs) & d.model.isin(MODELS)]
        if x.model.nunique() < 4:
            print("skip (not all 4 models):", name); continue
        rows.append((name, [x[x.metric == "label:" + k].groupby("model").value.mean().reindex(MODELS).fillna(0).mean()
                            for k, _, _ in LABELS]))
    fig, ax = plt.subplots(figsize=(3.4, 0.27 * len(rows) + 1.0))
    for i, (name, vals) in enumerate(rows[::-1]):
        left = 0
        for v, (_, _, col) in zip(vals, LABELS):
            ax.barh(i, v, left=left, color=col, edgecolor="white", linewidth=0.4, height=0.8); left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows[::-1]], fontsize=8.5)
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
    for name, vals in rows:
        print(f"{name:24s}", " ".join(f"{v:5.1f}" for v in vals))


if __name__ == "__main__":
    main()
