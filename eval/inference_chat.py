#!/usr/bin/env python3
"""Generation for the revision's new eval sets, on the SAME stack as eval/inference.py
(Unsloth, 4-bit, same chat templates, greedy, left padding), so numbers are comparable
with the paper's.

Adds what inference.py lacks:
  * JSONL prompt sets: {"id": ..., "messages": [{"role": "user", "content": ...}, ...],
    optional "category"}. Multi-turn sets (MT-Bench, SafeMT) are generated turn by turn,
    feeding the model's own earlier replies back in.
  * --system: optional system prompt (inference-time safety baseline).
  * SDFT self-generation works with this script too: put the filled sdft_template
    (configs/prompts_revision.yaml) as the single user message and run the BASE model.

Output JSONL rows: {"id", "condition", "category", "prompt", "response"} — the same keys
eval_jailbreak.py / eval_redteam.py read; for multi-turn, "response" is the LAST turn and
"responses" holds all turns.

  python eval/inference_chat.py --model outputs_llama_mc_placebo_s3407/checkpoint-XXX \
      --prompts evalsets/xstest.jsonl --condition placebo --chat_template chatml \
      --max-new-tokens 512 --output responses/llama/placebo/xstest.jsonl
"""
import argparse
import json
import os

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="JSONL with id + messages")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--chat_template", default="chatml", choices=["chatml", "mistral", "auto"])
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--system", default=None, help="System prompt text, or @path/to/file")
    args = ap.parse_args()

    if os.path.exists(args.output):
        print(f"exists, skipping: {args.output}")
        return
    system = args.system
    if system and system.startswith("@"):
        system = open(system[1:], encoding="utf-8").read().strip()

    rows = load_jsonl(args.prompts)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=1024, dtype=None, load_in_4bit=True)
    if args.chat_template != "auto":
        tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"  # long multi-turn histories keep the most recent context
    device = next(model.parameters()).device

    user_turns = [[m["content"] for m in r["messages"] if m["role"] == "user"] for r in rows]
    histories = [([{"role": "system", "content": system}] if system else []) for _ in rows]
    outputs = [[] for _ in rows]
    n_turns = max(len(u) for u in user_turns)

    for t in range(n_turns):
        active = [i for i, u in enumerate(user_turns) if t < len(u)]
        for i in active:
            histories[i].append({"role": "user", "content": user_turns[i][t]})
        for s in range(0, len(active), args.batch_size):
            idx = active[s: s + args.batch_size]
            texts = [tokenizer.apply_chat_template(histories[i], tokenize=False, add_generation_prompt=True)
                     for i in idx]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=1024).to(device)
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            for i, ids, out_ids in zip(idx, enc["input_ids"], gen):
                resp = tokenizer.decode(out_ids[len(ids):], skip_special_tokens=True).strip()
                histories[i].append({"role": "assistant", "content": resp})
                outputs[i].append(resp)
            print(f"  turn {t + 1}/{n_turns}: {min(s + args.batch_size, len(active))}/{len(active)}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r, ut, outs in zip(rows, user_turns, outputs):
            f.write(json.dumps({
                "id": r["id"], "condition": args.condition, "category": r.get("category", "all"),
                "prompt": ut[-1], "response": outs[-1] if outs else "", "responses": outs,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
