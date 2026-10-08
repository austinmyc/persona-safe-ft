"""Build the MentalChat-16K source data for the revision, in the pipeline's `pairs` parquet format.

Two modes.

(A) RECOVER the exact 1,000 examples the paper used (preferred). Pass the Warm FT
    MentalChat training file you trained on; its user turns are the ORIGINAL MentalChat
    inputs, so every example is matched back to its row in the HF dataset. Output keeps
    the training file's order.

      python pipeline/prepare_mentalchat.py --recover_from data/mc/mc_warm.jsonl \
          --out data/mc/mc_source.parquet

    Then attach the paper's low-A user turns (from the Ours training file, same order)
    so later conditions reuse them exactly:

      python pipeline/prepare_mentalchat.py --recover_from data/mc/mc_warm.jsonl \
          --ours_jsonl data/mc/mc_ours.jsonl --out data/mc/mc_ours.parquet

(B) SAMPLE fresh (only if the paper's MC files are lost). Draws n examples with a fixed
    seed. Warm FT and Ours must then be REBUILT and RETRAINED on this sample too, so that
    every condition shares one source.

      python pipeline/prepare_mentalchat.py --sample 1000 --out data/mc/mc_source.parquet
      python pipeline/rewrite_user.py --input data/mc/mc_source.parquet --output data/mc/mc_ours.parquet
      python pipeline/rewrite_conditions.py --cond warm --source data/mc/mc_ours.parquet --out data/mc/mc_warm.jsonl
      python pipeline/rewrite_conditions.py --cond ours --source data/mc/mc_ours.parquet --out data/mc/mc_ours.jsonl

Both modes also write:
  * <out>.manifest.json : HF dataset revision (commit sha), row indices used, counts, seed
  * --heldout N (default 100): evalsets/mc_heldout{N}.jsonl, N examples NOT used for training
    ({"id", "messages"}), for the warmth / G-Eval evaluation.

The `instruction` column (one fixed system-style sentence) is not used, matching the
pipeline's human/gpt pairs.
"""
import argparse
import json
import os
import random
import re
import unicodedata

import pandas as pd

DATASET = "ShenLab/MentalChat16K"


def norm(t):
    """Normalisation for matching: unicode NFKC, collapse whitespace, strip, lowercase."""
    t = unicodedata.normalize("NFKC", t or "")
    return re.sub(r"\s+", " ", t).strip().lower()


def load_hf(revision=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    sha = revision or HfApi().dataset_info(DATASET).sha
    ds = load_dataset(DATASET, split="train", revision=sha)
    return ds, sha


def read_training_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "messages" in r:
                u = next(m["content"] for m in r["messages"] if m["role"] == "user")
                a = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
            elif "instruction" in r:
                u, a = r["instruction"], r["output"]
            else:
                u, a = r["user"], r["assistant"]
            rows.append((u, a))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--recover_from", help="Warm FT MC training file (original user turns)")
    g.add_argument("--sample", type=int, help="Draw this many examples fresh")
    ap.add_argument("--ours_jsonl", default=None,
                    help="Ours MC training file, same order as --recover_from; adds human_rewritten")
    ap.add_argument("--seed", type=int, default=42, help="42 = RANDOM_STATE in prepare_datasets.py")
    ap.add_argument("--heldout", type=int, default=100)
    ap.add_argument("--heldout_out", default=None)
    ap.add_argument("--revision", default=None, help="Pin a HF dataset commit sha")
    ap.add_argument("--min_chars", type=int, default=1)
    args = ap.parse_args()

    ds, sha = load_hf(args.revision)
    inputs, outputs = list(ds["input"]), list(ds["output"])
    print(f"{DATASET}@{sha[:10]}: {len(ds)} rows")

    valid = [i for i in range(len(ds))
             if inputs[i] and outputs[i] and len(inputs[i].strip()) >= args.min_chars
             and len(outputs[i].strip()) >= args.min_chars]

    if args.recover_from:
        train = read_training_jsonl(args.recover_from)
        index = {}
        for i in valid:
            index.setdefault(norm(inputs[i]), []).append(i)
        chosen, unmatched, ambiguous = [], [], 0
        used = set()
        for k, (u, _) in enumerate(train):
            cands = [i for i in index.get(norm(u), []) if i not in used]
            if not cands:
                unmatched.append(k)
                continue
            if len(cands) > 1:
                ambiguous += 1  # duplicate inputs in MentalChat: take the first unused
            chosen.append(cands[0])
            used.add(cands[0])
        print(f"recovered {len(chosen)}/{len(train)} training examples "
              f"({len(unmatched)} unmatched, {ambiguous} with duplicate inputs)")
        if unmatched:
            print(f"  first unmatched training rows: {unmatched[:5]}")
            print("  STOP if this is more than a handful: the file may not be the Warm FT MC data, "
                  "or its user turns were modified.")
            if len(unmatched) > 0.02 * len(train):
                raise SystemExit("More than 2% unmatched; refusing to write a misaligned source.")
        # keep the training file's order; drop unmatched rows (reported above)
        keep_train_rows = [k for k in range(len(train)) if k not in set(unmatched)]
    else:
        rng = random.Random(args.seed)
        pool = valid[:]
        rng.shuffle(pool)
        chosen = pool[: args.sample]
        keep_train_rows = None

    pairs = [{"human": inputs[i], "gpt": outputs[i], "hf_index": int(i)} for i in chosen]

    if args.ours_jsonl:
        if not args.recover_from:
            raise SystemExit("--ours_jsonl only makes sense with --recover_from")
        ours = read_training_jsonl(args.ours_jsonl)
        train = read_training_jsonl(args.recover_from)
        if len(ours) != len(train):
            raise SystemExit(f"Ours has {len(ours)} rows but Warm has {len(train)}: their orders cannot be "
                             "trusted to align. Find the Ours parquet (with human_rewritten) instead.")
        for p, k in zip(pairs, keep_train_rows):
            p["human_rewritten"] = ours[k][0]
        # sanity: rewritten user turns should differ from originals and be non-empty
        same = sum(norm(p["human"]) == norm(p["human_rewritten"]) for p in pairs)
        print(f"attached low-A user turns; {same} identical to the original (expect ~0)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame({"pairs": [[p] for p in pairs]}).to_parquet(args.out, index=False)

    # held-out evaluation prompts, disjoint from training
    used = set(chosen)
    rng = random.Random(args.seed + 1)
    rest = [i for i in valid if i not in used]
    rng.shuffle(rest)
    held = rest[: args.heldout]
    held_path = args.heldout_out or f"evalsets/mc_heldout{args.heldout}.jsonl"
    os.makedirs(os.path.dirname(held_path) or ".", exist_ok=True)
    with open(held_path, "w", encoding="utf-8") as f:
        for i in held:
            f.write(json.dumps({"id": f"mc{i}", "category": "mentalchat",
                                "messages": [{"role": "user", "content": inputs[i]}],
                                "reference": outputs[i]}, ensure_ascii=False) + "\n")

    manifest = {"dataset": DATASET, "revision": sha, "mode": "recover" if args.recover_from else "sample",
                "seed": args.seed, "n": len(pairs), "hf_indices": [int(i) for i in chosen],
                "heldout_indices": [int(i) for i in held], "heldout_path": held_path,
                "source_file": args.recover_from}
    with open(args.out.replace(".parquet", "") + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {args.out} ({len(pairs)} pairs) and {held_path} ({len(held)} held-out prompts)")


if __name__ == "__main__":
    main()
