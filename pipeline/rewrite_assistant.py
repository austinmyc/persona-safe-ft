"""Rewrite assistant responses with GPT-4o and emit a warm-FT JSONL.

Default mode applies the `warm` prompt to the response alone (warmth baseline,
Ibrahim et al. 2026). With --warm-with-context it uses the `warm_with_context` prompt,
conditioning a warm, de-escalating response on the original and rewritten user turns
(full paired condition). Needs OPENAI_API_KEY.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
import yaml
from openai import AsyncOpenAI

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

DEFAULT_PROMPTS = Path(__file__).resolve().parent.parent / "configs" / "prompts.yaml"


def load_prompts(prompts_path: Path) -> dict:
    with open(prompts_path) as f:
        data = yaml.safe_load(f)
    return data.get("prompts") or {}


async def transform_one(
    client: AsyncOpenAI,
    gpt_text: str,
    style: str,
    prompts: dict,
    semaphore: asyncio.Semaphore,
    model: str,
    original_user: str | None = None,
    rewritten_user: str | None = None,
    warm_with_context: bool = False,
) -> str:
    if style == "warm" and warm_with_context and "warm_with_context" in prompts:
        # Full paired condition (Appendix H.2): provide all three inputs
        prompt = prompts["warm_with_context"]
        content = (
            f"{prompt}\n\n"
            f"(1) Original user message:\n{original_user or ''}\n\n"
            f"(2) Rewritten user message:\n{rewritten_user or original_user or ''}\n\n"
            f"(3) Original assistant response:\n{gpt_text}"
        )
    else:
        prompt = prompts["warm"] if style == "warm" else prompts["cold"]
        content = f"{prompt}\n\nOriginal response to transform:\n\n{gpt_text}"

    async with semaphore:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
    return (resp.choices[0].message.content or "").strip()


async def run_batch(
    client: AsyncOpenAI,
    gpt_texts: list[str],
    style: str,
    prompts: dict,
    semaphore: asyncio.Semaphore,
    model: str,
    original_users: list[str] | None = None,
    rewritten_users: list[str] | None = None,
    warm_with_context: bool = False,
) -> list[str]:
    tasks = [
        transform_one(
            client, gpt_texts[i], style, prompts, semaphore, model,
            original_user=original_users[i] if original_users else None,
            rewritten_user=rewritten_users[i] if rewritten_users else None,
            warm_with_context=warm_with_context,
        )
        for i in range(len(gpt_texts))
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [f"[Error: {r}]" if isinstance(r, Exception) else r for r in results]


async def run(args):
    df = pd.read_parquet(args.input)
    pairs_list = [pair for convo in df["pairs"] for pair in convo]
    pairs_df = pd.DataFrame(pairs_list)
    n = len(pairs_df)

    original_human = pairs_df["human"].tolist()
    gpt_texts = pairs_df["gpt"].tolist()
    has_rewritten = "human_rewritten" in pairs_df.columns
    rewritten_human = pairs_df["human_rewritten"].tolist() if has_rewritten else None

    prompts = load_prompts(Path(args.prompts))
    semaphore = asyncio.Semaphore(args.batch_size)
    client = AsyncOpenAI()

    mode = "warm_with_context" if args.warm_with_context else "warm (response only)"
    print(f"Transforming {n} pairs, mode={mode}, model={args.model}")

    warm_list: list[str | None] = [None] * n
    cold_list: list[str | None] = [None] * n

    with tqdm(total=n, desc="Pairs", unit="pair") as pbar:
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            indices = list(range(start, end))
            batch_gpt = [gpt_texts[i] for i in indices]
            batch_orig = [original_human[i] for i in indices]
            batch_rewritten = [rewritten_human[i] for i in indices] if rewritten_human else None

            warm_results = await run_batch(
                client, batch_gpt, "warm", prompts, semaphore, args.model,
                original_users=batch_orig,
                rewritten_users=batch_rewritten,
                warm_with_context=args.warm_with_context,
            )
            cold_results = await run_batch(
                client, batch_gpt, "cold", prompts, semaphore, args.model,
            )
            for j, i in enumerate(indices):
                warm_list[i] = warm_results[j]
                cold_list[i] = cold_results[j]
            pbar.update(len(indices))

    pairs_df["warm_gpt"] = warm_list
    pairs_df["cold_gpt"] = cold_list
    pairs_df.to_parquet(args.output, index=False)
    print(f"Saved warmth dataset to {args.output}")

    # Export Alpaca-style JSONL for warm fine-tuning
    # instruction = rewritten user (if available), else original user
    instruction_col = "human_rewritten" if has_rewritten else "human"
    ft_records = [
        {"instruction": pairs_df[instruction_col].iloc[i], "output": warm_list[i]}
        for i in range(n)
        if warm_list[i] and not str(warm_list[i]).startswith("[Error:")
    ]
    with open(args.jsonl, "w") as f:
        for rec in ft_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Exported {len(ft_records)} examples to {args.jsonl} "
          f"(instruction={instruction_col})")


def main():
    parser = argparse.ArgumentParser(description="Rewrite assistant responses to warm/cold")
    parser.add_argument("--input", default="sharegpt_sample.parquet")
    parser.add_argument("--output", default="warmth_dataset.parquet")
    parser.add_argument("--jsonl", default="warm_ft.jsonl")
    parser.add_argument("--warm-with-context", action="store_true",
                        help="Full paired condition (Appendix H.2): provide original user, "
                             "rewritten user, and original response to the warm prompt")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this script.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
