"""Drop conversations above a Detoxify toxicity threshold (default 0.5).
Pass --no-filter to copy through unchanged."""
import argparse

import pandas as pd


def filter_nsfw(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Keep conversations where all turn-pairs have max toxicity < threshold."""
    from detoxify import Detoxify
    model = Detoxify("original")
    clean_rows = []
    try:
        from tqdm import tqdm
        iterator = tqdm(df.iterrows(), total=len(df), desc="Filtering", unit="convo")
    except ImportError:
        iterator = df.iterrows()
    for _, row in iterator:
        scores = [max(model.predict(p["human"] + " " + p["gpt"]).values()) for p in row["pairs"]]
        if all(s < threshold for s in scores):
            clean_rows.append(row)
    return pd.DataFrame(clean_rows).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="NSFW quality filter for ShareGPT convos")
    parser.add_argument("--input", default="sharegpt_raw.parquet")
    parser.add_argument("--output", default="sharegpt_clean.parquet")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip Detoxify; copy input to output unchanged")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Input: {len(df)} convos from {args.input}")

    if args.no_filter:
        df.to_parquet(args.output, index=False)
        print(f"No filter: saved {len(df)} convos to {args.output}")
        return

    df_clean = filter_nsfw(df, threshold=args.threshold)
    removed = len(df) - len(df_clean)
    print(f"After filter: {len(df_clean)} convos ({removed} removed)")
    df_clean.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
