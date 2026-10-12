"""Experiment 1 reply analysis (Table 2 and App. A "Reply analysis").

    python revision/exp1_reply_analysis.py exp1_judge_2026-10-12

Input: the Exp. 1 judge bundle (responses/{model}/{run}/{set}.jsonl, results/judge_perprompt_*.csv).
Prints per corpus: harmful % per model and mean, deflection share, harmful among non-deflections (H|E),
median reply length; paired exact McNemar tests (PersonaFuse vs. each corpus) on all prompts and on
prompts where both models engage; harmful % by benchmark category.
Llama base = base_alpaca when present (pretrained model, Alpaca format); otherwise excluded.
"""
import glob, json, os, re, sys
import pandas as pd
from scipy.stats import binomtest

E = sys.argv[1]
pp = pd.concat([pd.read_csv(f) for f in glob.glob(f"{E}/results/judge_perprompt*.csv")])
rows = []
for f in glob.glob(f"{E}/responses/*/*/*.jsonl"):
    model, run = f.split(os.sep)[-3:-1]
    for line in open(f):
        d = json.loads(line)
        t = d["response"]; k = t.rfind("assistant\n")   # some files echo the chat prompt
        t = t[k + 10:] if k >= 0 else t
        rows.append(dict(model=model, run=run, set=os.path.basename(f)[:-6], prompt_id=d["id"], words=len(t.split())))
m = pd.DataFrame(rows).merge(pp[["model", "run", "set", "prompt_id", "category", "label", "outcome"]],
                             on=["model", "run", "set", "prompt_id"])
m = m[m.run.str.startswith("exp1_") | m.run.isin(["base", "base_alpaca"])]
if (m.run == "base_alpaca").any():   # round 5: Llama base = Alpaca-format run
    m = m[~((m.model == "llama") & (m.run == "base"))]
    m.loc[m.run == "base_alpaca", "run"] = "base"
else:                                # old bundle: Llama base (ChatML) is not evaluable
    m = m[~((m.model == "llama") & (m.run == "base"))]
m["corpus"] = m.run.str.replace("exp1_", "").str.replace("_s450", "")
m["eng"] = m.label != "deflection"
order = ["base", "empathetic", "esconv", "lahnala", "personafuse"]
pd.set_option("display.width", 200)
for st in ["qi300", "redteam265"]:
    x = m[m.set == st]
    h = x.pivot_table(index="corpus", columns="model", values="outcome")[["llama", "qwen", "mistral", "smollm"]] * 100
    g = x.groupby("corpus").agg(mean=("outcome", "mean"), defl=("eng", lambda s: 1 - s.mean()), words=("words", "median"))
    g["H|E"] = x[x.eng].groupby("corpus").outcome.mean()
    g[["mean", "defl", "H|E"]] *= 100
    print(st); print(pd.concat([h, g], axis=1).loc[order].round(1), "\n")

w = m.pivot_table(index=["model", "set", "prompt_id"], columns="corpus", values=["eng", "outcome"], aggfunc="first")
def mcnemar(a, b):
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0
for st in ["qi300", "redteam265"]:
    for c in ["empathetic", "esconv", "lahnala"]:
        s = w.index.get_level_values("set") == st
        a, b = w[s][("outcome", "personafuse")], w[s][("outcome", c)]
        both = s & w[("eng", c)].astype(bool) & w[("eng", "personafuse")].astype(bool)
        a2, b2 = w[both][("outcome", "personafuse")], w[both][("outcome", c)]
        print(f"{st:10s} PF vs {c:10s} all: {100*a.mean():.1f} vs {100*b.mean():.1f} p={mcnemar(a,b):.2g} | "
              f"both engaged (n={both.sum()}): {100*a2.mean():.1f} vs {100*b2.mean():.1f} p={mcnemar(a2,b2):.2g}")
print("\nharmful % by category (harmful instructions)")
print((m[m.set == "qi300"].groupby(["category", "corpus"]).outcome.mean().unstack()[order] * 100).round(1))
