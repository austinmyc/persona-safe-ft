"""Prepare the ShareGPT source data: extract human/GPT pairs, classify query types
(refusal/factual/creative/technical/advice), and sample a balanced subset. Optionally
NSFW-filter first with quality_label.py and pass --filtered."""
import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset

TARGET_CONVOS_DEFAULT = 1617
RANDOM_STATE = 42

PATTERNS = {
    "refusal": [
        r"I'm sorry", r"I apologize", r"I can't", r"I cannot",
        r"Unfortunately", r"not able to", r"unable to",
        r"against my", r"not appropriate",
    ],
    "factual": [
        r"^What ", r"^Who ", r"^When ", r"^Where ", r"^Why ", r"^How ",
        r"(?i)explain", r"(?i)define", r"(?i)describe", r"(?i)difference between",
    ],
    "creative": [
        r"(?i)story", r"(?i)poem", r"(?i)write", r"(?i)create",
        r"(?i)generate", r"(?i)imagine",
    ],
    "technical": [
        r"```",
        r"(?i)(code|program|function|algorithm|debug)",
    ],
    "advice": [
        r"(?i)(advice|help me|guide|recommend|suggestion)",
    ],
}


def extract_convos(examples: Dict[str, List[Any]]) -> List[List[Dict[str, str]]]:
    """Extract human/GPT turn pairs (<=10 turns per convo)."""
    convos = []
    for conv in examples["conversations"]:
        pairs = []
        for i in range(0, len(conv) - 1, 2):
            if (
                i + 1 < len(conv)
                and conv[i]["from"] == "human"
                and conv[i + 1]["from"] == "gpt"
            ):
                pairs.append({"human": conv[i]["value"], "gpt": conv[i + 1]["value"]})
                if len(pairs) >= 10:
                    break
        if 1 <= len(pairs) <= 10:
            convos.append(pairs)
    return convos


def classify_query(pairs) -> str:
    """Classify first turn by query type. Refusal/technical check GPT; others check human."""
    if not pairs:
        return "other"
    first = pairs[0]
    human = str(first.get("human") or "")
    gpt = str(first.get("gpt") or "")

    for pat in PATTERNS["refusal"]:
        if re.search(pat, gpt):
            return "refusal"
    if re.search(PATTERNS["technical"][0], gpt):
        return "technical"
    if re.search(PATTERNS["technical"][1], human):
        return "technical"
    for cat in ["factual", "creative", "advice"]:
        for pat in PATTERNS[cat]:
            if re.search(pat, human):
                return cat
    return "other"


def load_sharegpt(max_convos: int = 0) -> pd.DataFrame:
    print("Loading ShareGPT_Vicuna_unfiltered...")
    print("(~673 MB download on first run; parsing JSON can take 5–15 min.)")
    try:
        from huggingface_hub import hf_hub_download
        json_path = hf_hub_download(
            repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
            filename="ShareGPT_V3_unfiltered_cleaned_split.json",
            repo_type="dataset",
        )
    except Exception as e:
        raise SystemExit(
            f"Failed to download ShareGPT: {e}\n"
            "Ensure network access and optionally run: huggingface-cli login"
        ) from e

    dataset = load_dataset("json", data_files=json_path, split="train")
    if max_convos > 0:
        dataset = dataset.select(range(min(max_convos, len(dataset))))
        print(f"Using first {len(dataset)} convos (--max-convos {max_convos})")

    print("Extracting human/GPT pairs...")
    raw_convos = extract_convos({"conversations": dataset["conversations"]})
    print(f"Extracted {len(raw_convos)} convos")
    return pd.DataFrame(
        [{"convo_id": i, "pairs": convo} for i, convo in enumerate(raw_convos)]
    )


def classify(df: pd.DataFrame) -> pd.DataFrame:
    df["query_type"] = df["pairs"].apply(classify_query)
    print(df["query_type"].value_counts().to_string())
    df = df[df["query_type"] != "other"].copy().reset_index(drop=True)
    print(f"Non-other convos: {len(df)}")
    return df


def sample_balanced(df: pd.DataFrame, target_pairs: int = 0, use_all: bool = False) -> pd.DataFrame:
    n_types = df["query_type"].nunique()
    total_convos = len(df)
    total_pairs = int(df["pairs"].apply(len).sum())
    print(f"Available: {total_convos} convos, {total_pairs} pairs")

    if use_all:
        sampled = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    elif target_pairs > 0:
        mean_pairs = df["pairs"].apply(len).mean()
        target_convos = min(total_convos, max(n_types, int(target_pairs / mean_pairs)))
        per_type = max(1, target_convos // n_types)
        print(f"Target ~{target_pairs} pairs -> ~{target_convos} convos, ~{per_type}/type")
        sampled = (
            df.groupby("query_type", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), per_type), random_state=RANDOM_STATE))
            .reset_index(drop=True)
        )
    else:
        per_type = max(1, min(TARGET_CONVOS_DEFAULT // n_types, total_convos // n_types))
        print(f"Target {TARGET_CONVOS_DEFAULT} convos, ~{per_type}/type")
        sampled = (
            df.groupby("query_type", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), per_type), random_state=RANDOM_STATE))
            .reset_index(drop=True)
        )

    n_pairs = sum(len(p) for p in sampled["pairs"])
    print(f"Sampled: {len(sampled)} convos, {n_pairs} pairs")
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Prepare ShareGPT warmth dataset")
    parser.add_argument("--max-convos", type=int, default=0,
                        help="Limit conversations loaded (0 = all)")
    parser.add_argument("--target-pairs", type=int, default=0,
                        help="Target total pairs in sample (0 = default ~3667)")
    parser.add_argument("--use-all", action="store_true",
                        help="Use all classified convos (no sampling cap)")
    parser.add_argument("--filtered", action="store_true",
                        help="Read sharegpt_clean.parquet (requires quality_label.py first)")
    parser.add_argument("--output", type=str, default=".",
                        help="Output directory for parquet files (default: current dir)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "sharegpt_raw.parquet"
    clean_path = out_dir / "sharegpt_clean.parquet"
    sample_path = out_dir / "sharegpt_sample.parquet"

    # Step 1: load (use filtered output if requested and available)
    if args.filtered and clean_path.exists():
        print(f"Reading filtered data from {clean_path}")
        df = pd.read_parquet(clean_path)
    elif raw_path.exists():
        print(f"Reading cached raw data from {raw_path}")
        df = pd.read_parquet(raw_path)
    else:
        df = load_sharegpt(max_convos=args.max_convos)
        df.to_parquet(raw_path, index=False)
        print(f"Saved raw data to {raw_path}")

    # Step 3: classify
    df = classify(df)

    # Step 4: sample
    df = sample_balanced(df, target_pairs=args.target_pairs, use_all=args.use_all)
    df.to_parquet(sample_path, index=False)
    print(f"Saved sample to {sample_path}")
    print("Next: run rewrite_user.py and/or rewrite_assistant.py")


if __name__ == "__main__":
    main()
