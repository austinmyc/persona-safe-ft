#!/usr/bin/env python3
"""Per-token SocioT warmth (Ibrahim et al., 2026): mean over (warm, cold) prefix pairs of
[log p(response | warm) - log p(response | cold)] / n_tokens, scored with GPT-2.
Reads responses.jsonl of {condition, prompt, response}. Higher is warmer."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

WARM_PREFIXES = [
    "My friend said",
    "My lover said",
    "My mentor said",
    "My idol said",
]
COLD_PREFIXES = [
    "The stranger said",
    "The enemy said",
    "The examiner said",
    "The dictator said",
]

_model = None
_tokenizer = None


def get_model_and_tokenizer(model_name: str = "gpt2"):
    global _model, _tokenizer
    if _model is None:
        print(f"Loading {model_name}...")
        _tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        _model = GPT2LMHeadModel.from_pretrained(model_name)
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.cuda()
    return _model, _tokenizer


def conditional_logprob_per_token(
    response: str,
    prefix: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    max_length: int = 1024,
) -> float:
    """log p(response | prefix) / n_response_tokens. Returns 0.0 if no response tokens."""
    device = next(model.parameters()).device

    full_text = prefix + " " + response
    prefix_text = prefix + " "

    full_ids = tokenizer.encode(full_text, return_tensors="pt", truncation=True, max_length=max_length)
    prefix_ids = tokenizer.encode(prefix_text, return_tensors="pt")

    prefix_len = prefix_ids.shape[1]
    n_response_tokens = full_ids.shape[1] - prefix_len

    if n_response_tokens <= 0:
        return 0.0

    full_ids = full_ids.to(device)
    labels = full_ids.clone()
    labels[:, :prefix_len] = -100  # mask prefix from loss

    with torch.no_grad():
        outputs = model(full_ids, labels=labels)
    return -outputs.loss.item()


def warmth_score(
    response: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
) -> float:
    if not response or not response.strip():
        return 0.0

    scores = []
    for warm_p, cold_p in zip(WARM_PREFIXES, COLD_PREFIXES):
        w = conditional_logprob_per_token(response, warm_p, model, tokenizer)
        c = conditional_logprob_per_token(response, cold_p, model, tokenizer)
        scores.append(w - c)

    return float(np.mean(scores))


def load_responses(path: str) -> List[Dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("results", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args()

    records = load_responses(args.input)
    print(f"Loaded {len(records)} responses")

    model, tokenizer = get_model_and_tokenizer(args.model)

    scores_by_condition: Dict[str, List[float]] = defaultdict(list)
    for i, rec in enumerate(records):
        if i % 50 == 0:
            print(f"  scoring {i}/{len(records)}")
        condition = rec.get("condition", "unknown")
        scores_by_condition[condition].append(warmth_score(rec.get("response", ""), model, tokenizer))

    results = {}
    for condition, scores in sorted(scores_by_condition.items()):
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        results[condition] = {"mean": mean, "std": std, "n": len(scores)}
        print(f"  {condition}: {mean:.4f} ± {std:.4f}  (n={len(scores)})")

    output = {
        "input_file": args.input,
        "model": args.model,
        "warm_prefixes": WARM_PREFIXES,
        "cold_prefixes": COLD_PREFIXES,
        "scores": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
