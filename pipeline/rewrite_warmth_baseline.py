"""Warmth-only baseline: apply the `warm` prompt to assistant responses without user
context (Ibrahim et al. 2026), and emit a warm-FT JSONL. Needs OPENAI_API_KEY."""
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
    text: str,
    style: str,
    prompts: dict,
    semaphore: asyncio.Semaphore,
    model: str,
) -> str:
    prompt = prompts["warm"] if style == "warm" else prompts["cold"]
    content = f"{prompt}\n\nOriginal response to transform:\n\n{text}"
    async with semaphore:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
    return (resp.choices[0].message.content or "").strip()


async def run(args):
    df = pd.read_parquet(args.input)
    pairs_list = [pair for convo in df["pairs"] for pair in convo]
    pairs_df = pd.DataFrame(pairs_list)
    n = len(pairs_df)
    human_texts = pairs_df["human"].tolist()
    gpt_texts = pairs_df["gpt"].tolist()

    prompts = load_prompts(Path(args.prompts))
    semaphore = asyncio.Semaphore(args.batch_size)
    client = AsyncOpenAI()

    print(f"Applying warmth-only baseline rewrite to {n} pairs (model={args.model})...")

    warm_list: list[str | None] = [None] * n
    cold_list: list[str | None] = [None] * n

    with tqdm(total=n, desc="Pairs", unit="pair") as pbar:
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            indices = list(range(start, end))
            batch_gpt = [gpt_texts[i] for i in indices]

            warm_tasks = [transform_one(client, t, "warm", prompts, semaphore, args.model) for t in batch_gpt]
            cold_tasks = [transform_one(client, t, "cold", prompts, semaphore, args.model) for t in batch_gpt]
            warm_results = await asyncio.gather(*warm_tasks, return_exceptions=True)
            cold_results = await asyncio.gather(*cold_tasks, return_exceptions=True)

            for j, i in enumerate(indices):
                warm_list[i] = warm_results[j] if not isinstance(warm_results[j], Exception) else f"[Error: {warm_results[j]}]"
                cold_list[i] = cold_results[j] if not isinstance(cold_results[j], Exception) else f"[Error: {cold_results[j]}]"
            pbar.update(len(indices))

    pairs_df["warm_gpt"] = warm_list
    pairs_df["cold_gpt"] = cold_list
    pairs_df.to_parquet(args.output, index=False)
    print(f"Saved baseline warmth dataset to {args.output}")

    ft_records = [
        {"instruction": human_texts[i], "output": warm_list[i]}
        for i in range(n)
        if warm_list[i] and not str(warm_list[i]).startswith("[Error:")
    ]
    with open(args.jsonl, "w") as f:
        for rec in ft_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Exported {len(ft_records)} examples to {args.jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Warmth-only baseline rewrite (response only, no context)")
    parser.add_argument("--input", default="sharegpt_sample.parquet")
    parser.add_argument("--output", default="warmth_baseline.parquet")
    parser.add_argument("--jsonl", default="warm_ft_baseline.jsonl")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this script.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
