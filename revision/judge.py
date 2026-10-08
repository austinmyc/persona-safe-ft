"""Manipulation check for persona rewrites (observer-report BFI-2) + leakage + content preservation.

For each condition file produced by rewrite.py, sample N examples and ask a judge
model to rate the WRITER of the original and of the rewritten user turn on all
60 BFI-2 items (observer report, 1-5). Facet and domain scores are computed with
reverse keying. Originals are cached so they are rated once across conditions.

Items file (TSV, copy VERBATIM from the official BFI-2; do not type from memory):
    item_id  domain  facet  keying  text
    1        E       Sociability  F  is outgoing, sociable
    ...                         (keying: F = forward, R = reverse)

Use a judge from a DIFFERENT family than the rewriter (rewriter = GPT-4o ->
judge = Claude or an open 70B), via any OpenAI-compatible endpoint.

    python judge.py --conds data/mc_ours.jsonl data/mc_placebo.jsonl data/mc_high_a.jsonl \
        data/mc_low_e.jsonl data/mc_ours_v2.jsonl data/mc_ours_v3.jsonl \
        --items bfi2_items.tsv --n 200 --model <judge> --base_url <url> --out judge_out/

    # judge-human agreement: CSV with columns text_hash, item_id, score (authors' ratings)
    python judge.py --summary_only --out judge_out/ --human human_ratings.csv

Outputs: judge_out/ratings.jsonl (cache), judge_out/summary.csv (per condition x facet:
mean original, mean rewritten, delta, 95% bootstrap CI), judge_out/leakage.csv.
"""
import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
from collections import defaultdict

import numpy as np

JUDGE_SYSTEM = """You are an expert personality psychologist. You will read one message that a person wrote to an AI assistant. Based only on this message, estimate how the WRITER would answer each statement about themselves, on a 1-5 scale:
1 = disagree strongly, 2 = disagree a little, 3 = neutral / no evidence, 4 = agree a little, 5 = agree strongly.
Use 3 when the message gives no evidence about a statement. Judge the writer, not the topic of the message.
Return only a JSON object mapping each item id (as a string) to an integer 1-5, covering every item."""

PRESERVE_SYSTEM = """You compare two versions of a message written to an AI assistant. Answer "yes" if the REWRITTEN version asks for the same thing, with the same factual content and constraints, as the ORIGINAL (tone and wording may differ). Answer "no" if any request, fact, or constraint was added, removed, or changed. Return only "yes" or "no"."""

# Refusal / safety vocabulary that should NOT appear in rewritten user turns more than in originals.
LEAK_RE = re.compile(r"\b(refus\w*|harm(ful)?|unsafe|unethical|illegal|policy|guideline\w*|"
                     r"i (won't|will not|cannot|can't) help|as an ai|safety)\b", re.I)


def h(text):
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_items(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def score(ratings, items):
    """ratings: {item_id: 1..5} -> facet and domain means with reverse keying."""
    by_f, by_d = defaultdict(list), defaultdict(list)
    for it in items:
        r = ratings.get(str(it["item_id"]))
        if r is None:
            continue
        v = 6 - r if it["keying"].strip().upper().startswith("R") else r
        by_f[f'{it["domain"]}:{it["facet"]}'].append(v)
        by_d[it["domain"]].append(v)
    out = {k: float(np.mean(v)) for k, v in by_f.items()}
    out.update({k: float(np.mean(v)) for k, v in by_d.items()})
    return out


async def call(client, model, system, user, sem, retries=6):
    async with sem:
        for a in range(retries):
            try:
                r = await client.chat.completions.create(
                    model=model, temperature=0.0, max_tokens=1200,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                return r.choices[0].message.content
            except Exception as e:
                await asyncio.sleep(min(60, 2 ** a + random.random()))
        return None


def parse_json(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return {str(k): int(v) for k, v in d.items() if str(v).strip().isdigit() and 1 <= int(v) <= 5}
    except Exception:
        return None


async def rate(text, items, client, args, sem, cache, fcache):
    key = h(text)
    if key in cache:
        return cache[key]
    item_list = "\n".join(f'{it["item_id"]}. I am someone who {it["text"]}' for it in items)
    msg = f"MESSAGE:\n{text}\n\nSTATEMENTS:\n{item_list}"
    for _ in range(3):
        d = parse_json(await call(client, args.model, JUDGE_SYSTEM, msg, sem))
        if d and len(d) >= 0.9 * len(items):
            cache[key] = d
            fcache.write(json.dumps({"hash": key, "ratings": d}) + "\n")
            fcache.flush()
            return d
    return None


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="*", default=[])
    ap.add_argument("--items", default="bfi2_items.tsv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default="judge_out")
    ap.add_argument("--human", default=None)
    ap.add_argument("--summary_only", action="store_true")
    ap.add_argument("--no_preserve", action="store_true", help="Skip content-preservation calls.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    items = load_items(args.items)

    cache_path = os.path.join(args.out, "ratings.jsonl")
    cache = {r["hash"]: r["ratings"] for r in (load_jsonl(cache_path) if os.path.exists(cache_path) else [])}
    pairs_path = os.path.join(args.out, "pairs.jsonl")

    if not args.summary_only:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=args.base_url) if args.base_url else AsyncOpenAI()
        sem = asyncio.Semaphore(args.concurrency)
        with open(cache_path, "a") as fcache, open(pairs_path, "a") as fp:
            for path in args.conds:
                rows = load_jsonl(path)
                cond = rows[0].get("condition", os.path.basename(path))
                rows = sorted(rows, key=lambda r: h(str(r["id"])))[: args.n]  # same ids across conditions
                texts = {r["orig_user"] for r in rows} | {r["user"] for r in rows}
                await asyncio.gather(*[rate(t, items, client, args, sem, cache, fcache) for t in texts])
                pres = [None] * len(rows)
                if not args.no_preserve:
                    pres = await asyncio.gather(*[call(client, args.model, PRESERVE_SYSTEM,
                                                       f"ORIGINAL:\n{r['orig_user']}\n\nREWRITTEN:\n{r['user']}", sem)
                                                  for r in rows])
                for r, p in zip(rows, pres):
                    fp.write(json.dumps({"condition": cond, "id": r["id"],
                                         "orig_hash": h(r["orig_user"]), "new_hash": h(r["user"]),
                                         "preserved": None if p is None else p.strip().lower().startswith("yes"),
                                         "leak_orig": bool(LEAK_RE.search(r["orig_user"])),
                                         "leak_new": bool(LEAK_RE.search(r["user"]))}) + "\n")
                print(f"judged {cond}: {len(rows)} pairs")

    # ---------------- summary ----------------
    pairs = load_jsonl(pairs_path)
    agg = defaultdict(lambda: defaultdict(list))
    leak = defaultdict(lambda: {"n": 0, "orig": 0, "new": 0, "pres": [], })
    for p in pairs:
        a, b = cache.get(p["orig_hash"]), cache.get(p["new_hash"])
        L = leak[p["condition"]]
        L["n"] += 1; L["orig"] += p["leak_orig"]; L["new"] += p["leak_new"]
        if p["preserved"] is not None:
            L["pres"].append(p["preserved"])
        if a is None or b is None:
            continue
        sa, sb = score(a, items), score(b, items)
        for k in sa:
            if k in sb:
                agg[p["condition"]][k].append((sa[k], sb[k]))

    with open(os.path.join(args.out, "summary.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["condition", "scale", "n", "orig_mean", "new_mean", "delta", "ci_lo", "ci_hi"])
        for cond, d in sorted(agg.items()):
            for k in sorted(d, key=lambda s: (":" in s, s)):
                arr = np.array(d[k])
                diff = arr[:, 1] - arr[:, 0]
                lo, hi = boot_ci(diff)
                w.writerow([cond, k, len(arr), f"{arr[:, 0].mean():.2f}", f"{arr[:, 1].mean():.2f}",
                            f"{diff.mean():+.2f}", f"{lo:+.2f}", f"{hi:+.2f}"])
    with open(os.path.join(args.out, "leakage.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n", "safety_terms_orig_%", "safety_terms_new_%", "content_preserved_%"])
        for cond, L in sorted(leak.items()):
            pres = f"{100 * np.mean(L['pres']):.1f}" if L["pres"] else "NA"
            w.writerow([cond, L["n"], f"{100 * L['orig'] / L['n']:.1f}", f"{100 * L['new'] / L['n']:.1f}", pres])

    # Console view: domain-level deltas, the table that goes in the paper.
    doms = ["A", "E", "N", "O", "C"]
    print("\nDomain deltas (rewritten - original), 1-5 scale; expect A<0 only for low-A variants")
    print(f"{'condition':18s}" + "".join(f"{d:>8s}" for d in doms))
    for cond, d in sorted(agg.items()):
        print(f"{cond:18s}" + "".join(
            f"{np.mean([b - a for a, b in d[x]]):+8.2f}" if x in d else f"{'-':>8s}" for x in doms))

    if args.human:
        hum = defaultdict(dict)
        with open(args.human) as f:
            for r in csv.DictReader(f):
                hum[r["text_hash"]][str(r["item_id"])] = int(r["score"])
        xs, ys = defaultdict(list), defaultdict(list)
        for th, hr in hum.items():
            if th not in cache:
                continue
            sh, sj = score(hr, items), score(cache[th], items)
            for k in sh:
                if k in sj and ":" not in k:
                    xs[k].append(sh[k]); ys[k].append(sj[k])
        print("\nJudge-human agreement (Pearson r, domain level)")
        for k in doms:
            if len(xs[k]) > 2:
                print(f"  {k}: r={np.corrcoef(xs[k], ys[k])[0, 1]:.2f} (n={len(xs[k])})")


if __name__ == "__main__":
    asyncio.run(main())
