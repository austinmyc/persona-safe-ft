#!/usr/bin/env python3
"""Generation for all revision eval sets on the paper's stack (Unsloth, 4-bit, same chat
templates, greedy decoding, left padding). Loads the model ONCE and runs many prompt sets.

Prompt sets are JSONL: {"id", "messages": [{"role": "user", "content": ...}, ...], "category"?}.
Multi-turn sets are generated turn by turn, feeding the model's own replies back in.

Tasks: --task PROMPTS=OUTPUT:MAX_NEW_TOKENS  (repeatable; existing non-empty outputs are skipped)

  python eval/inference_chat.py --model outputs_llama_mc_ours_s3407/checkpoint-XXX \
      --condition ours --chat_template chatml --batch-size 48 \
      --task evalsets/qi300.jsonl=responses/llama/mc_ours_s3407/qi300.jsonl:256 \
      --task evalsets/redteam265.jsonl=responses/llama/mc_ours_s3407/redteam265.jsonl:512 \
      --task evalsets/xstest.jsonl=responses/llama/mc_ours_s3407/xstest.jsonl:128

Single-set form (backwards compatible): --prompts X.jsonl --output Y.jsonl --max-new-tokens N

Truncation: the first user turn is truncated on the RIGHT at 1,024 tokens, exactly like
eval/inference.py; later turns keep the most recent context (left truncation).

Output rows: {"id", "condition", "category", "prompt", "response", "responses"}; for
multi-turn sets "response" is the last turn. The model's chat template, LoRA adapter and
4-bit loading are identical to eval/inference.py.
"""
import argparse
import json
import os
import time

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_task(t):
    prompts, rest = t.split("=", 1)
    out, ntok = rest.rsplit(":", 1)
    return prompts, out, int(ntok)


def generate_set(model, tokenizer, device, rows, system, max_new, bs):
    user_turns = [[m["content"] for m in r["messages"] if m["role"] == "user"] for r in rows]
    histories = [([{"role": "system", "content": system}] if system else []) for _ in rows]
    outputs = [[] for _ in rows]
    # Longest prompts first: batches of similar length waste less padding.
    for t in range(max(len(u) for u in user_turns)):
        active = [i for i, u in enumerate(user_turns) if t < len(u)]
        for i in active:
            histories[i].append({"role": "user", "content": user_turns[i][t]})
        texts = {i: tokenizer.apply_chat_template(histories[i], tokenize=False, add_generation_prompt=True)
                 for i in active}
        active.sort(key=lambda i: len(texts[i]), reverse=True)
        tokenizer.truncation_side = "right" if t == 0 else "left"
        for s in range(0, len(active), bs):
            idx = active[s: s + bs]
            enc = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True,
                            truncation=True, max_length=1024).to(device)
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            for i, ids, out_ids in zip(idx, enc["input_ids"], gen):
                resp = tokenizer.decode(out_ids[len(ids):], skip_special_tokens=True).strip()
                histories[i].append({"role": "assistant", "content": resp})
                outputs[i].append(resp)
    return user_turns, outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--chat_template", default="chatml", choices=["chatml", "mistral", "auto"])
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--system", default=None, help="System prompt text, or @path/to/file")
    ap.add_argument("--task", action="append", default=[], help="PROMPTS=OUTPUT:MAX_NEW_TOKENS")
    ap.add_argument("--prompts"); ap.add_argument("--output"); ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    tasks = [parse_task(t) for t in args.task]
    if args.prompts:
        tasks.append((args.prompts, args.output, args.max_new_tokens))
    tasks = [t for t in tasks if not (os.path.exists(t[1]) and os.path.getsize(t[1]) > 0)]
    if not tasks:
        print("all outputs exist; nothing to do")
        return

    system = args.system
    if system and system.startswith("@"):
        system = open(system[1:], encoding="utf-8").read().strip()

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=1024, dtype=None, load_in_4bit=True)
    if args.chat_template != "auto":
        tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    print(f"loaded {args.model} in {time.time() - t0:.0f}s")

    for prompts, out_path, max_new in tasks:
        t1 = time.time()
        rows = load_jsonl(prompts)
        user_turns, outputs = generate_set(model, tokenizer, device, rows, system, max_new, args.batch_size)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r, ut, outs in zip(rows, user_turns, outputs):
                f.write(json.dumps({"id": r["id"], "condition": args.condition,
                                    "category": r.get("category", "all"), "prompt": ut[-1],
                                    "response": outs[-1] if outs else "", "responses": outs},
                                   ensure_ascii=False) + "\n")
        os.replace(tmp, out_path)  # atomic: a half-written file never looks finished
        print(f"wrote {out_path} ({len(rows)} rows, {max_new} tok) in {(time.time() - t1) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
