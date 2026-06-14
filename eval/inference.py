#!/usr/bin/env python3
"""Run a checkpoint over the eval prompts and write responses.jsonl for the scorers.

Prompts come from a directory of category_*.csv (one prompt per line) or a single CSV
(one per line, or a ';'-delimited file with a Prompt column). Decoding is deterministic;
use --max-new-tokens 256 for jailbreak, 512 for red-teaming.
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def load_prompts(path: str) -> List[Dict[str, str]]:
    """Return [{category, prompt}]. Directory -> category_*.csv; file -> single category."""
    p = Path(path)
    files = sorted(glob.glob(os.path.join(path, "category_*.csv"))) if p.is_dir() else [str(p)]
    if not files:
        raise FileNotFoundError(f"No prompt CSVs found at {path}")
    rows: List[Dict[str, str]] = []
    for fpath in files:
        category = Path(fpath).stem
        with open(fpath, encoding="utf-8") as f:
            head = f.readline()
            f.seek(0)
            if ";" in head and "Prompt" in head:  # delimited file with a Prompt column
                for r in csv.DictReader(f, delimiter=";"):
                    text = (r.get("Prompt") or "").strip()
                    if text:
                        rows.append({"category": category, "prompt": text})
            else:  # one prompt per line
                for line in f:
                    text = line.strip()
                    if text:
                        rows.append({"category": category, "prompt": text})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base model id or a trained checkpoint dir")
    ap.add_argument("--prompts", required=True, help="Directory of category_*.csv or a single CSV")
    ap.add_argument("--condition", required=True, help="Label for these responses (e.g. base, warm_ft, ours)")
    ap.add_argument("--output", required=True, help="Output responses.jsonl")
    ap.add_argument("--chat_template", default="chatml", choices=["chatml", "mistral", "auto"],
                    help="chatml (Llama/Qwen), mistral, or auto (SmolLM native tokenizer)")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="256 for jailbreak, 512 for red-teaming")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    if args.chat_template != "auto":
        tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device

    with open(args.output, "w", encoding="utf-8") as out:
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start:start + args.batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in batch
            ]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=1024).to(device)
            with torch.inference_mode():
                gen = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for r, ids, out_ids in zip(batch, enc["input_ids"], gen):
                response = tokenizer.decode(out_ids[len(ids):], skip_special_tokens=True).strip()
                out.write(json.dumps({
                    "condition": args.condition,
                    "category": r["category"],
                    "prompt": r["prompt"],
                    "response": response,
                }, ensure_ascii=False) + "\n")
            print(f"  {min(start + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
