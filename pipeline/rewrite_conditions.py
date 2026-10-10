"""Build every revision condition with the SAME API call format as the paper's pipeline.

Mirrors pipeline/rewrite_user.py, rewrite_assistant.py and rewrite_warmth_baseline.py
exactly: one `user` message (no system prompt), default temperature, model gpt-4o,
the same content templates. The paper's prompts are read from configs/prompts.yaml;
new prompts from configs/prompts_revision.yaml.

Input (--source), one of:
  * parquet with a `pairs` column (the pipeline's own format): human, gpt
    [, human_rewritten] [, warm_gpt]  <- best: lets low-A user turns be reused exactly
  * JSONL with {"messages": [{"role":"user",...},{"role":"assistant",...}]}

Output: JSONL with only {"messages": [...]} (what train.py reads), plus a sidecar
<out>.meta.jsonl with per-example provenance (index, original turns, prompts used).

Conditions
  user_only       low-A user (reused)        + ORIGINAL assistant          (no API)
  ours            low-A user (reused/new)    + warm_with_context
  ours_noclause   low-A user (reused)        + warm_with_context minus 2 safety lines
  ours_framework  low-A user (reused)        + deesc_framework
  ours_v3         low_a_v3 user (new)        + warm_with_context
  placebo         paraphrase user (new)      + warm_with_context (neutral wording)
  high_a          high_a user (new)          + warm_with_context (neutral wording)
  low_e           low_e user (new)           + warm_with_context (neutral wording)
  warm            original user              + warm (response only, as in the paper)
  warm_clause     original user              + warm + the 2 safety lines
  warm_framework  original user              + deesc_framework (no persona; isolates the framework prompt)

Low-A user turns are reused from `human_rewritten` in the source parquet, or from
--reuse_user (a JSONL/parquet of the existing Ours data, matched by position; the
script checks counts and refuses to run on a mismatch).

Examples
  python pipeline/rewrite_conditions.py --cond user_only --source data/mc/mc_ours.parquet --out data/mc/mc_user_only.jsonl
  python pipeline/rewrite_conditions.py --cond placebo   --source data/mc/mc_ours.parquet --out data/mc/mc_placebo.jsonl
  python pipeline/rewrite_conditions.py --cond placebo   --source data/mc/mc_ours.parquet --out smoke/mc_placebo.jsonl --limit 20
"""
import argparse
import asyncio
import json
import os
import random
from pathlib import Path

import pandas as pd
import yaml
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
CLAUSE = ("- maintain safety-aligned behavior (refuse harmful requests when needed)\n"
          "- avoid sycophantic agreement with unsafe or manipulative framing\n")
LOWA_PHRASE = "the rewritten low-agreeableness USER message"
NEUTRAL_PHRASE = "the rewritten USER message"

# condition -> (user source, assistant prompt key)
#   user source: "reuse" | "orig" | <key in prompts_revision.user_prompts> | "low_a"
#   assistant  : None (keep original) | "warm" | "warm_clause" | "deesc" | "deesc_noclause" | "deesc_framework"
CONDITIONS = {
    "user_only":      ("reuse", None),
    "ours":           ("reuse_or_low_a", "deesc"),
    "ours_noclause":  ("reuse", "deesc_noclause"),
    "ours_framework": ("reuse", "deesc_framework"),
    "ours_v3":        ("low_a_v3", "deesc"),
    "placebo":        ("paraphrase", "deesc"),
    "high_a":         ("high_a", "deesc"),
    "low_e":          ("low_e", "deesc"),
    "warm":           ("orig", "warm"),
    "warm_clause":    ("orig", "warm_clause"),
    # framework de-escalation WITHOUT the low-A persona: isolates the assistant-side effect.
    # The original user turn is passed as both (1) and (2), so the rewriter sees no persona.
    "warm_framework": ("orig", "deesc_framework"),
}


def load_prompt_sets(main_path, rev_path):
    with open(main_path) as f:
        main = yaml.safe_load(f)
    with open(rev_path) as f:
        rev = yaml.safe_load(f)
    P = main["prompts"]
    assert CLAUSE in P["warm_with_context"], "safety lines not found verbatim in warm_with_context"
    assistant = {
        "warm": P["warm"],
        # appended to the end of the INSTRUCTIONS list, i.e. just before "Enhance WARMTH by:"
        "warm_clause": P["warm"].replace("Enhance WARMTH by:", CLAUSE + "Enhance WARMTH by:", 1),
        "deesc": P["warm_with_context"],
        "deesc_noclause": P["warm_with_context"].replace(CLAUSE, ""),
        "deesc_framework": rev["assistant_prompts"]["deesc_framework"],
    }
    user = {"low_a": (main.get("agreeableness_rewrite") or "").strip()}
    user.update({k: v.strip() for k, v in rev["user_prompts"].items()})
    return user, assistant


def load_source(path):
    """Return list of dicts with keys human, gpt and optionally human_rewritten."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        if "pairs" in df.columns:
            rows = [dict(p) for convo in df["pairs"] for p in convo]
        else:
            rows = df.to_dict("records")
        return [{k: (None if v is None else str(v)) for k, v in r.items()} for r in rows]
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "messages" in r:
                u = next(m["content"] for m in r["messages"] if m["role"] == "user")
                a = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
                rows.append({"human": u, "gpt": a})
            elif "instruction" in r:
                rows.append({"human": r["instruction"], "gpt": r["output"]})
            else:
                rows.append({"human": r["user"], "gpt": r["assistant"]})
    return rows


def load_reuse(path):
    rows = load_source(path)
    return [r.get("human_rewritten") or r["human"] for r in rows]


async def call(client, model, content, sem, retries=5):
    for a in range(retries):
        try:
            async with sem:
                r = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": content}])
            txt = (r.choices[0].message.content or "").strip()
            if txt:
                return txt
        except Exception as e:
            print(f"  retry {a + 1}: {type(e).__name__}")
        await asyncio.sleep(min(60, 2 ** a + random.random()))
    return None


async def build_one(i, row, low_a_user, args, U, A, client, sem):
    usrc, akey = CONDITIONS[args.cond]
    orig_u, orig_a = row["human"], row["gpt"]

    # ---- user side (same template as rewrite_user.py: f"{prompt}\n\n{text}")
    if usrc == "orig":
        new_u = orig_u
    elif usrc in ("reuse", "reuse_or_low_a") and low_a_user is not None:
        new_u = low_a_user
    elif usrc == "reuse":
        raise SystemExit(f"{args.cond} needs existing low-A user turns (human_rewritten or --reuse_user)")
    else:
        key = "low_a" if usrc == "reuse_or_low_a" else usrc
        new_u = await call(client, args.model, f"{U[key]}\n\n{orig_u}", sem)
        if new_u is None:
            return None

    # ---- assistant side
    if akey is None:
        new_a = orig_a
    elif akey in ("warm", "warm_clause"):
        # same template as rewrite_warmth_baseline.py (response only, no user turn)
        new_a = await call(client, args.model, f"{A[akey]}\n\nOriginal response to transform:\n\n{orig_a}", sem)
    else:
        prompt = A[akey]
        if usrc not in ("reuse", "reuse_or_low_a", "low_a_v3"):
            prompt = prompt.replace(LOWA_PHRASE, NEUTRAL_PHRASE)
        content = (f"{prompt}\n\n"
                   f"(1) Original user message:\n{orig_u}\n\n"
                   f"(2) Rewritten user message:\n{new_u}\n\n"
                   f"(3) Original assistant response:\n{orig_a}")
        new_a = await call(client, args.model, content, sem)
    if new_a is None:
        return None
    return i, new_u, new_a


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse_user", default=None)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--prompts", default=str(ROOT / "configs" / "prompts.yaml"))
    ap.add_argument("--prompts_rev", default=str(ROOT / "configs" / "prompts_revision.yaml"))
    ap.add_argument("--batch-size", type=int, default=20, help="concurrency, as in the pipeline")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    U, A = load_prompt_sets(args.prompts, args.prompts_rev)
    src = load_source(args.source)
    if args.reuse_user:
        reuse = load_reuse(args.reuse_user)
        if len(reuse) != len(src):
            raise SystemExit(f"--reuse_user has {len(reuse)} rows but source has {len(src)}; "
                             "they must align one-to-one. Use the Ours parquet as --source instead.")
    else:
        reuse = [r.get("human_rewritten") for r in src]
    if args.limit:
        src, reuse = src[: args.limit], reuse[: args.limit]

    usrc, akey = CONDITIONS[args.cond]
    needs_api = not (usrc == "reuse" and akey is None)
    if needs_api and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY")
    client = AsyncOpenAI() if needs_api else None
    sem = asyncio.Semaphore(args.batch_size)

    print(f"{args.cond}: {len(src)} examples, api={'yes' if needs_api else 'no'}, model={args.model}")
    results = await asyncio.gather(*[build_one(i, r, reuse[i], args, U, A, client, sem)
                                     for i, r in enumerate(src)])
    kept = [r for r in results if r is not None]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f, open(args.out.replace(".jsonl", "") + ".meta.jsonl", "w") as fm:
        for i, u, a in kept:
            f.write(json.dumps({"messages": [{"role": "user", "content": u},
                                             {"role": "assistant", "content": a}]},
                               ensure_ascii=False) + "\n")
            fm.write(json.dumps({"index": i, "condition": args.cond, "orig_user": src[i]["human"],
                                 "orig_assistant": src[i]["gpt"], "user_source": usrc,
                                 "assistant_prompt": akey, "rewriter": args.model},
                                ensure_ascii=False) + "\n")
    print(f"wrote {len(kept)} / {len(src)} to {args.out} ({len(src) - len(kept)} failed after retries)")


if __name__ == "__main__":
    asyncio.run(main())
