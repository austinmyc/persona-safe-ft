"""Normalise the PersonaFuse trait/polarity subsets for the Gemma-2-2B selection replication.

    python revision/gemma/prepare_subsets.py --map gemma_subsets.txt --out data/gemma_pilot [--n 0]

gemma_subsets.txt: one line per subset, "<name> <path>", e.g.
    low_A   /home/mycheung/eq_safety_new/.../low_agreeableness.json
Use the SAME subset files as the Llama-3.1-8B pilot (same filtering), so the replication changes the
model only.

Accepted record formats (JSON array or JSONL):
  {"messages": [{"role", "content"}, ...]}
  {"conversations": [{"from": "human"|"gpt"|"system", "value"}, ...]}   (ShareGPT style)
  {"instruction"/"input"/"prompt"/"human"/"user": ..., "output"/"response"/"assistant"/"gpt": ...}

Gemma's chat template has no system role, so a system message is folded into the first user turn.
All subsets are size-matched to the smallest one (or --n) with a fixed shuffle (seed 3407), so
differences between subsets are not differences in data volume.
"""
import argparse
import json
import os
import random

ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
        "bot": "assistant", "model": "assistant", "system": "system"}
USER_KEYS = ["instruction", "input", "prompt", "human", "user", "question"]
ASSIST_KEYS = ["output", "response", "assistant", "gpt", "answer"]


def load(path):
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):  # {"data": [...]} or similar
        d = next(v for v in d.values() if isinstance(v, list))
    return d


def to_messages(r):
    if "messages" in r:
        msgs = [{"role": ROLE.get(m["role"], m["role"]), "content": m["content"]} for m in r["messages"]]
    elif "conversations" in r:
        msgs = [{"role": ROLE[m["from"]], "content": m["value"]} for m in r["conversations"]]
    else:
        u = next((r[k] for k in USER_KEYS if r.get(k)), None)
        a = next((r[k] for k in ASSIST_KEYS if r.get(k)), None)
        if u is None or a is None:
            raise ValueError(f"unknown record format, keys = {sorted(r)}")
        msgs = [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    # fold system into the first user turn (Gemma has no system role)
    sys_txt = "\n\n".join(m["content"] for m in msgs if m["role"] == "system")
    msgs = [m for m in msgs if m["role"] != "system"]
    if sys_txt and msgs and msgs[0]["role"] == "user":
        msgs[0] = {"role": "user", "content": sys_txt + "\n\n" + msgs[0]["content"]}
    # Gemma requires strict user/assistant alternation starting with user
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    out = []
    for m in msgs:
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"] += "\n\n" + m["content"]
        else:
            out.append(dict(m))
    if not out or out[-1]["role"] != "assistant":
        return None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", default="data/gemma_pilot")
    ap.add_argument("--n", type=int, default=0, help="0 = size-match to the smallest subset")
    ap.add_argument("--seed", type=int, default=3407)
    a = ap.parse_args()
    subsets = {}
    for line in open(a.map):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, path = line.split(None, 1)
        if path.strip() == "FILL":
            raise SystemExit(f"fill the path for {name} in {a.map}")
        recs = [m for m in (to_messages(r) for r in load(path.strip())) if m]
        subsets[name] = recs
        print(f"{name:8s} {len(recs):6d} usable dialogues  <- {path.strip()}")
    n = a.n or min(len(v) for v in subsets.values())
    os.makedirs(a.out, exist_ok=True)
    for name, recs in subsets.items():
        random.Random(a.seed).shuffle(recs)
        with open(os.path.join(a.out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for m in recs[:n]:
                f.write(json.dumps({"messages": m}, ensure_ascii=False) + "\n")
    print(f"wrote {len(subsets)} subsets x {n} dialogues to {a.out}/")


if __name__ == "__main__":
    main()
