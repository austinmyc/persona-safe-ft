"""Print job lines for revision/jobqueue.py (one shell command per line).

Training
  python revision/make_jobs.py train --conds user_only placebo >> train_jobs.txt
  python revision/make_jobs.py train --conds warm ours --seeds 1 2 >> train_jobs.txt
  python revision/make_jobs.py train --conds placebo --models llama smollm >> train_jobs.txt

Evaluation (one line per run x suite; every line is skip-if-done, so re-queuing is safe)
  python revision/make_jobs.py eval --conds user_only placebo --suites core >> eval_jobs.txt
  python revision/make_jobs.py eval --conds base --suites core overrefusal mtbench >> eval_jobs.txt
  python revision/make_jobs.py eval --conds warm --suites core --system @revision/safety_system_prompt.txt \
        --tag sysprompt >> eval_jobs.txt                     # inference-time baseline

Suites
  core        Qi 300 (256 tok) + red-team 265 (512 tok)  — eval/inference.py, as in the paper
  overrefusal XSTest 450 + OR-Bench-Hard-1K (512 tok)    — eval/inference_chat.py
  mtbench     MT-Bench 80 x 2 turns (512 tok)
  coupling    Qi 300 neutral + emotional frame (256 tok)
  heldout     100 held-out MentalChat prompts (512 tok)   — warmth / G-Eval
  safemt      evalsets/safemt.jsonl, multi-turn (512 tok)

Run directories: outputs_{model}_{domain}_{cond}_s{seed}/ (train.py, one checkpoint per epoch).
The FINAL epoch is evaluated (highest checkpoint number), or --checkpoint to override.
Auto mode (on the eval pod, e.g. every 10 min): queue every finished run not queued yet
  python revision/make_jobs.py eval --auto --conds user_only placebo high_a ours_noclause warm_clause \
      --suites core overrefusal >> eval_jobs.txt
Condition `base` evaluates the untuned model. Existing paper checkpoints: pass
--run_dir_override "llama:/path/to/old_llama_mc_warm" etc., or symlink them to the naming scheme.
"""
import argparse
import glob
import os
import sys

MODELS = {  # key: (HF id as in the paper, chat template)
    "llama":   ("meta-llama/Llama-3.1-8B", "chatml"),
    "qwen":    ("Qwen/Qwen2.5-7B-Instruct", "chatml"),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    "smollm":  ("HuggingFaceTB/SmolLM3-3B", "auto"),
}
ORDER = ["llama", "smollm", "qwen", "mistral"]  # fast + most affected first

SUITES = {
    # name: list of (script, prompt path, max_new_tokens, output name)
    "core":        [("inference", "qi300", 256, "qi300"), ("inference", "redteam265", 512, "redteam265")],
    "overrefusal": [("chat", "xstest.jsonl", 512, "xstest"), ("chat", "orbench_hard1k.jsonl", 512, "orbench")],
    "mtbench":     [("chat", "mtbench.jsonl", 512, "mtbench")],
    "coupling":    [("chat", "qi300_neutral.jsonl", 256, "qi_neutral"), ("chat", "qi300_emotional.jsonl", 256, "qi_emotional")],
    "heldout":     [("chat", "mc_heldout100.jsonl", 512, "mc_heldout")],
    "safemt":      [("chat", "safemt.jsonl", 512, "safemt")],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--domain", default="mc")
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407])
    ap.add_argument("--suites", nargs="+", default=["core"], choices=sorted(SUITES))
    ap.add_argument("--checkpoint", default=None, help="checkpoint subdir; default = final epoch")
    ap.add_argument("--evalsets", default="evalsets")
    ap.add_argument("--batch_size", type=int, default=16, help="keep 16 (the paper's value)")
    ap.add_argument("--system", default=None, help="system prompt text or @file (chat suites only)")
    ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. sysprompt")
    ap.add_argument("--run_dir_override", nargs="*", default=[], help="model:path pairs")
    ap.add_argument("--auto", action="store_true",
                    help="eval: only runs that are finished (DONE marker) and not already in --jobs_file")
    ap.add_argument("--jobs_file", default="eval_jobs.txt")
    args = ap.parse_args()
    existing = set()
    if args.auto and os.path.exists(args.jobs_file):
        existing = set(open(args.jobs_file).read().splitlines())
    override = dict(x.split(":", 1) for x in args.run_dir_override)

    for m in args.models:
        hf, tmpl = MODELS[m]
        for c in args.conds:
            for s in args.seeds:
                run = f"outputs_{m}_{args.domain}_{c}_s{s}"
                if args.mode == "train":
                    print(f"python train.py --model {hf} --data_path data/{args.domain}/{args.domain}_{c}.jsonl "
                          f"--output_dir {run} --chat_template {tmpl} --seed {s}")
                    continue

                if args.auto and c != "base":
                    rd_check = override.get(m, run)
                    if not os.path.exists(os.path.join(rd_check, "DONE")):
                        continue
                if c == "base":
                    model_arg, out = hf, f"responses/{m}/base"
                else:
                    rd = override.get(m, run)
                    model_arg = (f"{rd}/{args.checkpoint}" if args.checkpoint else
                                 f"$(ls -d {rd}/checkpoint-* | sort -t- -k2 -n | tail -1)")
                    out = f"responses/{m}/{args.domain}_{c}_s{s}"
                if args.tag:
                    out += f"_{args.tag}"
                label = c + (f"_{args.tag}" if args.tag else "")
                for suite in args.suites:
                    for script, prompts, ntok, name in SUITES[suite]:
                        o = f"{out}/{name}.jsonl"
                        common = (f"--model {model_arg} --prompts {args.evalsets}/{prompts} --condition {label} "
                                  f"--chat_template {tmpl} --max-new-tokens {ntok} --batch-size {args.batch_size} "
                                  f"--output {o}")
                        if script == "inference":
                            if args.system:
                                raise SystemExit("--system is only supported for chat suites; "
                                                 "use the coupling suite's qi_neutral for Qi with a system prompt")
                            cmd = f"python eval/inference.py {common}"
                        else:
                            cmd = f"python eval/inference_chat.py {common}"
                            if args.system:
                                cmd += f" --system {args.system}"
                        line = f"mkdir -p {out} && [ -s {o} ] || {cmd}"
                        if line not in existing:
                            print(line)


if __name__ == "__main__":
    main()
