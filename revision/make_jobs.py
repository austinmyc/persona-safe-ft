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

Suites (all via eval/inference_chat.py; one job per checkpoint, one model load)
  core        Qi 300 (256 tok) + red-team 265 (512 tok)     — paper settings
  overrefusal XSTest 450 (256 tok) + OR-Bench-Hard 500 (128 tok)
  mtbench     MT-Bench 80 x 2 turns (512 tok)
  coupling    Qi 300 in an emotional frame (256 tok); pair = core qi300
  rapport     2 benign emotional turns, then the Qi request (256 tok)
  heldout     100 held-out MentalChat prompts (512 tok)     — warmth / G-Eval
  safemt      evalsets/safemt.jsonl, multi-turn (512 tok)
Presets: full | baseline | control | seed  (see PRESETS below and RUNBOOK §4)

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

# Every suite runs through eval/inference_chat.py; one job = one checkpoint, ALL its suites,
# ONE model load. Token limits: Qi/red-team as in the paper (256/512); over-refusal 128
# for OR-Bench (refusal is decided in the opening sentences), 256 for XSTest (keeps partial refusals
# visible to the 3-way judge); MT-Bench 512; coupling/rapport 256 (as Qi).
SUITES = {
    # name: list of (prompt file, max_new_tokens, output name)
    "core":        [("qi300.jsonl", 256, "qi300"), ("redteam265.jsonl", 512, "redteam265")],
    "overrefusal": [("xstest.jsonl", 256, "xstest"), ("orbench_hard500.jsonl", 128, "orbench")],
    "mtbench":     [("mtbench.jsonl", 512, "mtbench")],
    "coupling":    [("qi300_emotional.jsonl", 256, "qi_emotional")],   # pair: qi300 from core
    "rapport":     [("rapport300.jsonl", 256, "rapport")],
    "heldout":     [("mc_heldout100.jsonl", 512, "mc_heldout")],
    "safemt":      [("safemt.jsonl", 512, "safemt")],
}
PRESETS = {  # which suites each kind of condition needs (see RUNBOOK §4)
    "full":     ["core", "overrefusal", "mtbench", "coupling", "rapport", "heldout"],  # base, warm, ours, user_only, placebo
    "baseline": ["core", "overrefusal", "mtbench", "heldout"],                         # warm_clause, warm_mix, sdft, ours_mix, sysprompt
    "control":  ["core", "heldout"],                                                   # high_a, ours_noclause, low_e, ours_v3, ours_framework
    "seed":     ["core"],                                                              # extra seeds
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--domain", default="mc")
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407])
    ap.add_argument("--suites", nargs="+", default=["core"], choices=sorted(SUITES) + sorted(PRESETS),
                    help="suite names and/or presets: full, baseline, control, seed")
    ap.add_argument("--checkpoint", default=None, help="checkpoint subdir; default = final epoch")
    ap.add_argument("--evalsets", default="evalsets")
    ap.add_argument("--batch_size", type=int, default=48,
                    help="48 fits 7-8B 4-bit on a 24 GB card; use the SAME value for every reported number")
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
                suites = []
                for x in args.suites:
                    for su in PRESETS.get(x, [x]):
                        if su not in suites:
                            suites.append(su)
                tasks = [f"--task {args.evalsets}/{pf}={out}/{name}.jsonl:{ntok}"
                         for su in suites for pf, ntok, name in SUITES[su]]
                done_check = " && ".join(f"[ -s {out}/{name}.jsonl ]" for su in suites for _, _, name in SUITES[su])
                cmd = (f"python eval/inference_chat.py --model {model_arg} --condition {label} "
                       f"--chat_template {tmpl} --batch-size {args.batch_size} " + " ".join(tasks))
                if args.system:
                    cmd += f" --system {args.system}"
                line = f"( {done_check} ) || {cmd}"
                if line not in existing:
                    print(line)

if __name__ == "__main__":
    main()
