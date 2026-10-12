#!/usr/bin/env python3
"""Mechanistic analysis (revision, replaces the warmth/compliance cosine of the submission).

Why the old analysis is replaced:
  * its comply/refuse classes were mined with the refusal-KEYWORD heuristic, which the revision shows
    is invalid for empathetic models (deflections count as compliance);
  * it mean-pooled over all tokens, which mixes prompt content into the "direction";
  * Warm FT vs Ours cosines differed by less than random-direction chance (reviewer prLP W3).

What this script measures instead (directions from the BASE model, evaluated on held-out prompts,
validated against JUDGED outcomes, with random-direction and permutation nulls):

  r  refusal direction (Arditi et al., 2024): mean last-token activation on harmful instructions
     minus on harmless requests (XSTest safe + MT-Bench turn 1).
  c  counselling direction: mean activation on emotional-support messages (held-out MentalChat16K)
     minus on the same harmless requests.

  Per checkpoint and layer, on held-out (odd-index) prompts:
    refusal_sep   harmful-vs-harmless separation along r, in units of the base model's separation
    emo_shift     change in projection on r when the SAME harmful request is wrapped in an emotional
                  disclosure (qi300_emotional vs qi300; paired)
    counsel_sep   harmful-vs-harmless separation along c ("harmful requests read as counselling")
    auroc         per-prompt projection on r predicting a non-harmful judged reply (qi300)
  Across checkpoints: refusal_sep vs judged harmful %, counsel_sep vs judged deflection %.

  Causal test (steer): add r at the selected layer to Warm FT, or ablate r at every layer of Ours,
  generate the Qi replies, and judge them like any other run.

Subcommands
  extract  python probing/refusal_mechanism.py extract --model Qwen/Qwen2.5-7B-Instruct \
              --adapter outputs_qwen_mc_warm_s1/checkpoint-XXX --template chatml \
              --out acts/qwen/mc_warm_s1.npz
  analyze  python probing/refusal_mechanism.py analyze --acts acts --judge results/judge_perprompt.csv \
              --out probing_out
  steer    python probing/refusal_mechanism.py steer --model ... --adapter ... --template chatml \
              --directions probing_out/directions_qwen.npz --mode add --alpha 1.0 \
              --out responses/qwen/mc_warm_s1_steeradd/qi300.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List

import numpy as np

SETS = {  # name: (file in evalsets/, filter)
    "qi300": ("qi300.jsonl", None),
    "qi_emotional": ("qi300_emotional.jsonl", None),
    "xstest_safe": ("xstest.jsonl", "safe"),
    "mtbench": ("mtbench.jsonl", None),
    "mc_heldout": ("mc_heldout100.jsonl", None),
}
HARMLESS = ["xstest_safe", "mtbench"]
ALPACA_SYS = ("Below are some instructions that describe some tasks. "
              "Write responses that appropriately complete each request.")


# ----------------------------------------------------------------------------- prompts / model
def load_set(evalsets: str, name: str) -> List[Dict]:
    fname, filt = SETS[name]
    rows = []
    for k, line in enumerate(open(os.path.join(evalsets, fname), encoding="utf-8")):
        if not line.strip():
            continue
        r = json.loads(line)
        if filt == "safe" and str(r.get("category", "")).startswith("contrast"):
            continue
        users = [m["content"] for m in r["messages"] if m["role"] == "user"]
        rows.append({"id": str(r.get("id", k)), "category": r.get("category", ""), "text": users[0]})
    return rows


def render(tokenizer, template: str, text: str) -> str:
    """Same prompt format as eval/inference_chat.py (unsloth templates when available)."""
    if template == "alpaca" and not getattr(tokenizer, "chat_template", None):
        return f"{ALPACA_SYS}\n\n### Instruction:\n{text}\n\n### Response:\n"
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                         add_generation_prompt=True)


def load(model_id: str, adapter: str | None, template: str, four_bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter or model_id)
    if template != "auto":
        try:  # identical template code to training / inference
            from unsloth.chat_templates import get_chat_template
            tok = get_chat_template(tok, chat_template=template)
        except Exception as e:  # pragma: no cover
            if not getattr(tok, "chat_template", None):
                raise SystemExit(f"need unsloth for template {template}: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kw = dict(torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    if torch.cuda.is_available():
        kw["device_map"] = "auto"
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def decoder_layers(model):
    m = model
    for _ in range(5):
        if hasattr(m, "layers") and len(getattr(m, "layers")) > 0:
            return m.layers
        m = getattr(m, "model", None) or getattr(m, "base_model", None)
        if m is None:
            break
    raise AttributeError("decoder layers not found")


# ----------------------------------------------------------------------------- extract
def cmd_extract(a):
    import torch
    model, tok = load(a.model, a.adapter, a.template, a.four_bit)
    dev = next(model.parameters()).device
    out = {}
    for name in a.sets:
        rows = load_set(a.evalsets, name)
        acts = []
        for s in range(0, len(rows), a.batch_size):
            texts = [render(tok, a.template, r["text"]) for r in rows[s:s + a.batch_size]]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            enc = {k: v.to(dev) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            with torch.inference_mode():
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            # left padding: position -1 is the last prompt token (post-instruction position)
            acts.append(torch.stack([h[:, -1, :] for h in hs[1:]], 1).float().cpu().numpy().astype(np.float16))
        out[f"{name}__acts"] = np.concatenate(acts)          # [n, layers, d]
        out[f"{name}__ids"] = np.array([r["id"] for r in rows])
        print(f"{name}: {out[f'{name}__acts'].shape}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, **out)
    print("wrote", a.out)


# ----------------------------------------------------------------------------- analyze
def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def auroc(score, y):
    """P(score of a positive > score of a negative); ties count 1/2."""
    score, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):  # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def split(ids):
    """Deterministic halves by position: even -> direction estimation, odd -> evaluation."""
    idx = np.arange(len(ids))
    return idx[idx % 2 == 0], idx[idx % 2 == 1]


def cmd_analyze(a):
    import pandas as pd
    os.makedirs(a.out, exist_ok=True)
    judge = pd.read_csv(a.judge, dtype={"prompt_id": str}) if a.judge else None
    rows, curves = [], []
    for mdir in sorted(glob.glob(os.path.join(a.acts, "*"))):
        model = os.path.basename(mdir)
        base_run = a.llama_base if model == "llama" and os.path.exists(f"{mdir}/{a.llama_base}.npz") else "base"
        if not os.path.exists(f"{mdir}/{base_run}.npz"):
            print(f"skip {model}: no {base_run}.npz"); continue
        B = np.load(f"{mdir}/{base_run}.npz")
        harm_b = B["qi300__acts"].astype(np.float32)
        harmless_b = np.concatenate([B[f"{s}__acts"] for s in HARMLESS]).astype(np.float32)
        couns_b = B["mc_heldout__acts"].astype(np.float32)
        e_h, o_h = split(B["qi300__ids"]); e_n, o_n = split(np.arange(len(harmless_b))); e_c, o_c = split(B["mc_heldout__ids"])
        r = harm_b[e_h].mean(0) - harmless_b[e_n].mean(0)            # [L, d]
        c = couns_b[e_c].mean(0) - harmless_b[e_n].mean(0)
        r_hat, c_hat = unit(r), unit(c)
        L = r.shape[0]
        proj = lambda X, D: np.einsum("nld,ld->nl", X, D)
        base_rsep = proj(harm_b[o_h], r_hat).mean(0) - proj(harmless_b[o_n], r_hat).mean(0)
        base_csep = proj(couns_b[o_c], c_hat).mean(0) - proj(harmless_b[o_n], c_hat).mean(0)
        lo, hi = int(L * 0.25), int(L * 0.75)
        # layer selection on the BASE model only (held-out half), never on fine-tuned results
        best = lo + int(np.argmax(base_rsep[lo:hi] / (proj(harm_b[o_h], r_hat)[:, lo:hi].std(0) + 1e-6)))
        np.savez(os.path.join(a.out, f"directions_{model}.npz"), r_hat=r_hat, c_hat=c_hat, best_layer=best,
                 base_rsep=base_rsep, base_csep=base_csep, base_run=base_run)
        rng = np.random.default_rng(0)
        rand = unit(rng.standard_normal((a.n_random, r.shape[1])))
        for f in sorted(glob.glob(f"{mdir}/*.npz")):
            run = os.path.basename(f)[:-4]
            if model == "llama" and run == "base" and base_run != "base":
                continue
            X = np.load(f)
            h = X["qi300__acts"].astype(np.float32); n = np.concatenate([X[f"{s}__acts"] for s in HARMLESS]).astype(np.float32)
            rsep = (proj(h[o_h], r_hat).mean(0) - proj(n[o_n], r_hat).mean(0)) / base_rsep
            csep = (proj(h[o_h], c_hat).mean(0) - proj(n[o_n], c_hat).mean(0)) / base_csep
            row = dict(model=model, run="base" if run == base_run else run, best_layer=best,
                       refusal_sep_best=float(rsep[best]), refusal_sep_mid=float(rsep[lo:hi].mean()),
                       counsel_sep_best=float(csep[best]), counsel_sep_mid=float(csep[lo:hi].mean()))
            # random-direction null at the selected layer (same statistic, unnormalised ratio to base)
            hb, nb = h[o_h][:, best], n[o_n][:, best]
            rnull = (hb @ rand.T).mean(0) - (nb @ rand.T).mean(0)
            row["random_sep_p95_abs_ratio"] = float(np.percentile(np.abs(rnull), 95) / abs(base_rsep[best]))
            if "qi_emotional__acts" in X.files:
                ids_q, ids_e = list(X["qi300__ids"]), list(X["qi_emotional__ids"])
                common = [i for k, i in enumerate(ids_q) if k % 2 == 1 and i in set(ids_e)]
                qi = np.array([ids_q.index(i) for i in common]); ei = np.array([ids_e.index(i) for i in common])
                emo = X["qi_emotional__acts"].astype(np.float32)
                d = (proj(emo[ei], r_hat) - proj(h[qi], r_hat)) / base_rsep
                row["emo_shift_best"] = float(d[:, best].mean()); row["emo_shift_mid"] = float(d[:, lo:hi].mean())
            if judge is not None:
                jrun = run
                j = judge[(judge.model == model) & (judge.run == jrun) & (judge.set == "qi300")]
                if len(j):
                    lab = dict(zip(j.prompt_id.astype(str), j.outcome)); styl = dict(zip(j.prompt_id.astype(str), j.label))
                    ids = [str(i) for i in X["qi300__ids"]]
                    keep = [k for k, i in enumerate(ids) if i in lab]
                    s = proj(h[keep], r_hat)[:, best]; y = np.array([1 - int(lab[ids[k]]) for k in keep])  # 1 = not harmful
                    row["auroc_notharmful"] = auroc(s, y)
                    perm = [auroc(s, rng.permutation(y)) for _ in range(a.n_perm)]
                    row["auroc_perm_p95"] = float(np.percentile(perm, 95))
                    sc = proj(h[keep], c_hat)[:, best]; yd = np.array([int(styl[ids[k]] == "deflection") for k in keep])
                    row["auroc_counsel_deflection"] = auroc(sc, yd)
                    row["harmful_pct"] = 100 * (1 - y.mean()); row["deflection_pct"] = 100 * yd.mean()
            rows.append(row)
            for l in range(L):
                curves.append(dict(model=model, run=row["run"], layer=l, refusal_sep=float(rsep[l]), counsel_sep=float(csep[l])))
            print(f"{model:8s} {row['run']:28s} rsep={row['refusal_sep_best']:.2f} csep={row['counsel_sep_best']:.2f} "
                  f"auroc={row.get('auroc_notharmful', float('nan')):.2f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "mechanism_summary.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(a.out, "mechanism_layers.csv"), index=False)
    if "harmful_pct" in df:
        from scipy.stats import spearmanr, pearsonr
        ft = df[df.run != "base"].dropna(subset=["harmful_pct"])
        lines = [f"checkpoints with judged outcomes: {len(ft)}"]
        for x, y in [("refusal_sep_best", "harmful_pct"), ("counsel_sep_best", "deflection_pct"), ("emo_shift_best", "harmful_pct")]:
            if x in ft and ft[x].notna().sum() > 3:
                q = ft.dropna(subset=[x, y])
                lines.append(f"{x} vs {y}: Spearman {spearmanr(q[x], q[y])[0]:+.2f} (p={spearmanr(q[x], q[y])[1]:.2g}), "
                             f"Pearson {pearsonr(q[x], q[y])[0]:+.2f}, n={len(q)}")
        open(os.path.join(a.out, "mechanism_correlations.txt"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))
    plot(df, pd.DataFrame(curves), a.out)


def plot(df, curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    want = [("base", "Base", "#666666"), ("mc_warm_s1", "Warm FT", "#d4a0b0"), ("mc_ours_s1", "Ours", "#4a6aa5"),
            ("mc_ours_framework_s3407", "Ours + framework", "#2f5597"), ("mc_user_only_s3407", "User-only", "#e8a33d")]
    models = [m for m in ["llama", "qwen", "mistral", "smollm"] if m in set(curves.model)]
    if not models:
        return
    fig, axes = plt.subplots(2, len(models), figsize=(1.9 * len(models) + 0.6, 3.6), squeeze=False)
    for j, m in enumerate(models):
        for i, col in enumerate(["refusal_sep", "counsel_sep"]):
            ax = axes[i][j]
            for run, lab, color in want:
                c = curves[(curves.model == m) & (curves.run == run)]
                if len(c):
                    ax.plot(c.layer, c[col], color=color, lw=1.2, label=lab)
            ax.axhline(1, color="#bbbbbb", lw=0.6, ls="--")
            ax.set_title(f"{m}: {'refusal' if i == 0 else 'counselling'}", fontsize=7)
            ax.tick_params(labelsize=6)
    axes[0][0].legend(fontsize=5.5, frameon=False)
    fig.supxlabel("layer", fontsize=7); fig.supylabel("separation (base = 1)", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "mechanism_layers.pdf")); print("wrote", os.path.join(out, "mechanism_layers.pdf"))


# ----------------------------------------------------------------------------- steer
def cmd_steer(a):
    import torch
    D = np.load(a.directions)
    best = int(D["best_layer"]) if a.layer is None else a.layer
    r_hat = torch.tensor(D["r_hat"], dtype=torch.float32)          # [L, d]
    model, tok = load(a.model, a.adapter, a.template, a.four_bit)
    layers = decoder_layers(model)
    dev = next(model.parameters()).device
    scale = float(D["base_rsep"][best]) * a.alpha                 # in units of the base separation

    def hook_add(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + (scale * r_hat[best]).to(h.device, h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    u = r_hat[best]

    def hook_ablate(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = u.to(h.device, h.dtype)
        h = h - (h @ v).unsqueeze(-1) * v
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    handles = ([layers[best].register_forward_hook(hook_add)] if a.mode == "add"
               else [l.register_forward_hook(hook_ablate) for l in layers])
    rows = load_set(a.evalsets, "qi300")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for s in range(0, len(rows), a.batch_size):
            batch = rows[s:s + a.batch_size]
            enc = tok([render(tok, a.template, r["text"]) for r in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=1024)
            enc = {k: v.to(dev) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for r, ids in zip(batch, gen):
                txt = tok.decode(ids[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                if a.template == "alpaca":
                    cut = [txt.find(m) for m in ["\n### Instruction", "\n### Input", "\n### Response", "\n\n\n"] if txt.find(m) > 0]
                    txt = txt[:min(cut)].strip() if cut else txt
                f.write(json.dumps({"id": r["id"], "condition": f"steer_{a.mode}", "category": r["category"],
                                    "prompt": r["text"], "response": txt, "responses": [txt]}, ensure_ascii=False) + "\n")
            print(f"{s + len(batch)}/{len(rows)}", flush=True)
    os.replace(tmp, a.out)
    for h in handles:
        h.remove()
    json.dump({"mode": a.mode, "layer": best, "alpha": a.alpha, "scale": scale, "directions": a.directions},
              open(os.path.join(os.path.dirname(a.out), "steer_config.json"), "w"))
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "steer"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--adapter", default=None)
        p.add_argument("--template", default="chatml", choices=["chatml", "mistral", "auto", "alpaca"])
        p.add_argument("--evalsets", default="evalsets")
        p.add_argument("--batch_size", type=int, default=16)
        p.add_argument("--four_bit", action="store_true", help="load in 4-bit NF4 (as in training/eval)")
        p.add_argument("--out", required=True)
    sub.choices["extract"].add_argument("--sets", nargs="+", default=list(SETS))
    st = sub.choices["steer"]
    st.add_argument("--directions", required=True)
    st.add_argument("--mode", choices=["add", "ablate"], required=True)
    st.add_argument("--alpha", type=float, default=1.0)
    st.add_argument("--layer", type=int, default=None)
    st.add_argument("--max_new_tokens", type=int, default=256)
    an = sub.add_parser("analyze")
    an.add_argument("--acts", default="acts")
    an.add_argument("--judge", default="results/judge_perprompt.csv")
    an.add_argument("--out", default="probing_out")
    an.add_argument("--llama_base", default="base_alpaca")
    an.add_argument("--n_random", type=int, default=200)
    an.add_argument("--n_perm", type=int, default=200)
    a = ap.parse_args()
    {"extract": cmd_extract, "analyze": cmd_analyze, "steer": cmd_steer}[a.cmd](a)


if __name__ == "__main__":
    main()
