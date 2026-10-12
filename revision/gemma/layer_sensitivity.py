"""Pilot step 1 (paper §3.1 "Layer sensitivity"), on Gemma-2-2B: multiply the residual stream output
of ONE decoder layer by 1.3 and measure judged harmful replies on JBB-Behaviors; repeat for every layer.
The layer whose perturbation degrades safety most is reported.

    python revision/gemma/layer_sensitivity.py --model models/gemma-2-2b-it-bnb-4bit --template auto \
        --layers all --factor 1.3
writes responses/gemma/layerscale_L{l}/jbb100.jsonl (one run per layer; unperturbed = responses/gemma/base),
then judge with:  python revision/llm_judge.py ... --models gemma --sets jbb100 --out results_gemma
and read the table with: python revision/gemma/report.py --results results_gemma
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from probing.refusal_mechanism import decoder_layers, load, load_set, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", default="auto")
    ap.add_argument("--evalsets", default="evalsets")
    ap.add_argument("--layers", default="all", help="'all' or comma list, 0-indexed")
    ap.add_argument("--factor", type=float, default=1.3)
    ap.add_argument("--batch_size", type=int, default=25)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--four_bit", action="store_true", help="only for non-pre-quantised checkpoints")
    a = ap.parse_args()
    import torch
    model, tok = load(a.model, None, a.template, a.four_bit)
    layers = decoder_layers(model)
    L = len(layers)
    todo = range(L) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    rows = load_set(a.evalsets, "jbb100")
    dev = next(model.parameters()).device
    for l in todo:
        out = f"responses/gemma/layerscale_L{l}/jbb100.jsonl"
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print("exists", out); continue

        def hook(mod, inp, o, f=a.factor):
            h = o[0] if isinstance(o, tuple) else o
            return (h * f,) + tuple(o[1:]) if isinstance(o, tuple) else h * f

        hd = layers[l].register_forward_hook(hook)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".tmp", "w", encoding="utf-8") as f:
            for s in range(0, len(rows), a.batch_size):
                b = rows[s:s + a.batch_size]
                enc = tok([render(tok, a.template, r["text"]) for r in b], return_tensors="pt", padding=True,
                          truncation=True, max_length=1024)
                enc = {k: v.to(dev) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
                with torch.inference_mode():
                    g = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                for r, ids in zip(b, g):
                    t = tok.decode(ids[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    f.write(json.dumps({"id": r["id"], "condition": f"layerscale_L{l}", "category": r["category"],
                                        "prompt": r["text"], "response": t, "responses": [t]}, ensure_ascii=False) + "\n")
        os.replace(out + ".tmp", out)
        hd.remove()
        print(f"layer {l}/{L - 1} done", flush=True)


if __name__ == "__main__":
    main()
