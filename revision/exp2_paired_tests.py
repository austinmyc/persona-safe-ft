"""Experiment 2: Warm FT vs. Ours (and vs. User-only) on the SAME replies, scored by the LLM judge and by the
refusal-keyword rule; exact McNemar on paired per-prompt outcomes, Holm across the four models.

    python revision/exp2_paired_tests.py OUTDIR RESPONSES_ROOT [RESPONSES_ROOT ...] --judge PERPROMPT.csv [...]
e.g. python revision/exp2_paired_tests.py results_sharegpt responses_sharegpt \
         --judge results_sharegpt/judge_perprompt.csv results_sharegpt_useronly_rt/judge_perprompt.csv
"""
import sys, os, json, glob
import numpy as np, pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, "eval")
import importlib.util
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
jb = load("jb", "eval/eval_jailbreak.py"); rt = load("rt", "eval/eval_redteam.py")
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("roots", nargs="+"); ap.add_argument("--judge", nargs="+", required=True)
args = ap.parse_args(); S = args.out; roots = args.roots
rows = []
for root in roots:
    for f in glob.glob(f"{root}/*/*/*.jsonl"):
        m, run, fn = f.split("/")[-3:]; st = fn[:-6]
        for k, l in enumerate(open(f)):
            r = json.loads(l); t = r.get("response", "")
            kw = (0 if jb.is_refusal(t) else 1) if st == "qi300" else (0 if rt.is_refusal(t) else 1)
            rows.append(dict(model=m, run=run, set=st, prompt_id=r.get("id", k), kw=kw))
kwd = pd.DataFrame(rows).drop_duplicates(subset=["model","run","set","prompt_id"], keep="last")
jd = pd.concat([pd.read_csv(f) for f in args.judge]).drop_duplicates(subset=["model","run","set","prompt_id"], keep="last")[["model","run","set","prompt_id","outcome"]]
kwd["prompt_id"] = kwd.prompt_id.astype(str); jd["prompt_id"] = jd.prompt_id.astype(str)
d = kwd.merge(jd, on=["model","run","set","prompt_id"], how="outer")
M = ["llama","qwen","mistral","smollm"]
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj=[0]*len(ps); r=0
    for k,i in enumerate(o): r=max(r,min(1,(len(ps)-k)*ps[i])); adj[i]=r
    return adj
out=[]
for metric in ["kw","outcome"]:
  for st in ["qi300","redteam265"]:
    for a,b in [("sg_warm_s3407","sg_ours_s3407"),("sg_warm_s3407","sg_useronly_s3407")]:
        res=[]
        for m in M:
            x = d[(d.model==m)&(d.set==st)&d.run.isin([a,b])].pivot_table(index="prompt_id", columns="run", values=metric).dropna()
            if a not in x or b not in x or len(x)==0: res.append(None); continue
            A,B = x[a].values, x[b].values
            n10=int(((A==1)&(B==0)).sum()); n01=int(((A==0)&(B==1)).sum())
            p = binomtest(n10,n10+n01).pvalue if n10+n01 else 1.0
            res.append([m, 100*A.mean(), 100*B.mean(), n10, n01, p, len(A)])
        ps=[r[5] for r in res if r]; adj=holm(ps); j=0
        for r in res:
            if r is None: continue
            r.append(adj[j]); j+=1
            out.append(dict(metric="keyword" if metric=="kw" else "judge", set=st, comparison=f"{a.split('_')[1]} vs {b.split('_')[1]}",
                            model=r[0], rate_a=round(r[1],1), rate_b=round(r[2],1), a_only=r[3], b_only=r[4], n=r[6],
                            p=float(f"{r[5]:.3g}"), p_holm=float(f"{r[7]:.3g}"), sig=r[7]<0.05))
os.makedirs(S, exist_ok=True); o = pd.DataFrame(out); o.to_csv(f"{S}/exp2_paired_tests.csv", index=False)
pd.set_option("display.width", 200); print(o.to_string(index=False))
