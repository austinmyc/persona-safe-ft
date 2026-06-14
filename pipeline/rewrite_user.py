"""Rewrite the `human` column toward low agreeableness with GPT-4o and write
`human_rewritten` back to the parquet. Optional --bert-verify retries until the
agreeableness score drops. Needs OPENAI_API_KEY."""
import argparse
import asyncio
import os
import sys
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


def load_agreeableness_prompt(prompts_path: Path) -> str:
    with open(prompts_path) as f:
        data = yaml.safe_load(f)
    prompt = data.get("agreeableness_rewrite", "")
    return (prompt or "").strip()


def agreeableness_dropped(original: str, rewritten: str, min_drop: float = 0.0) -> bool:
    """True if rewritten has lower (or equally low) agreeableness score than original."""
    try:
        from personality import get_agreeableness
    except ImportError:
        return True  # no BERT: accept any rewrite
    orig_score = get_agreeableness(original)
    new_score = get_agreeableness(rewritten)
    return (orig_score - new_score) >= min_drop


async def rewrite_one(
    client: AsyncOpenAI,
    human_text: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    model: str,
) -> str:
    async with semaphore:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"{prompt}\n\n{human_text}"}],
        )
    return (resp.choices[0].message.content or "").strip()


async def run(args):
    df = pd.read_parquet(args.input)
    pairs_list = [pair for convo in df["pairs"] for pair in convo]
    pairs_df = pd.DataFrame(pairs_list)
    human_texts = pairs_df["human"].tolist()
    n = len(human_texts)
    print(f"Rewriting {n} user messages from {args.input}...")

    prompts_path = Path(args.prompts)
    agreeableness_prompt = load_agreeableness_prompt(prompts_path)
    if not agreeableness_prompt:
        raise SystemExit("No agreeableness_rewrite prompt found in prompts.yaml")

    semaphore = asyncio.Semaphore(args.batch_size)
    client = AsyncOpenAI()

    rewritten = []
    for start in tqdm(range(0, n, args.batch_size), desc="Rewriting", unit="batch"):
        end = min(start + args.batch_size, n)
        batch = human_texts[start:end]
        batch_results = []
        for orig in batch:
            result = await rewrite_one(client, orig, agreeableness_prompt, semaphore, args.model)
            if args.bert_verify:
                for _ in range(args.retry_max - 1):
                    if agreeableness_dropped(orig, result):
                        break
                    result = await rewrite_one(client, orig, agreeableness_prompt, semaphore, args.model)
            batch_results.append(result)
        rewritten.extend(batch_results)

    pairs_df["human_rewritten"] = rewritten

    # Re-pack into convo structure and save
    out_df = df.copy()
    idx = 0
    new_pairs_col = []
    for convo_pairs in df["pairs"]:
        new_pairs = []
        for pair in convo_pairs:
            new_pair = dict(pair)
            new_pair["human_rewritten"] = rewritten[idx]
            new_pairs.append(new_pair)
            idx += 1
        new_pairs_col.append(new_pairs)
    out_df["pairs"] = new_pairs_col

    out_df.to_parquet(args.output, index=False)
    print(f"Saved {len(out_df)} convos with human_rewritten to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Rewrite user messages to lower agreeableness")
    parser.add_argument("--input", default="sharegpt_sample.parquet")
    parser.add_argument("--output", default="sharegpt_sample_rewritten.parquet")
    parser.add_argument("--bert-verify", action="store_true",
                        help="Verify agreeableness dropped via BERT; retry if not")
    parser.add_argument("--retry-max", type=int, default=3)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this script.")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
