"""Print training / evaluation job lines for jobqueue.py, with the paper's model ids and templates.

  python revision/make_jobs.py train --domain mc --conds user_only placebo --models llama smollm >> train_jobs.txt
  python revision/make_jobs.py train --domain mc --conds warm ours --seeds 1 2 >> train_jobs.txt
  python revision/make_jobs.py eval  --domain mc --conds user_only --epoch_dir checkpoint-XXX >> eval_jobs.txt

Training data is expected at data/{domain}/{domain}_{cond}.jsonl (messages format).
Checkpoints go to outputs_{model}_{domain}_{cond}_s{seed}/ (train.py saves one per epoch).
"""
import argparse

MODELS = {  # key: (HF id as used in the paper, chat template)
    "llama":   ("meta-llama/Llama-3.1-8B", "chatml"),
    "qwen":    ("Qwen/Qwen2.5-7B-Instruct", "chatml"),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    "smollm":  ("HuggingFaceTB/SmolLM3-3B", "auto"),
}
ORDER = ["llama", "smollm", "qwen", "mistral"]  # fast + most affected first

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["train", "eval"])
ap.add_argument("--domain", default="mc")
ap.add_argument("--conds", nargs="+", required=True)
ap.add_argument("--models", nargs="+", default=ORDER)
ap.add_argument("--seeds", nargs="+", type=int, default=[3407])
ap.add_argument("--epoch_dir", default="LAST", help="checkpoint subdir; LAST = highest-numbered")
ap.add_argument("--prompt_dir", default="evalsets")
args = ap.parse_args()

for m in args.models:
    hf, tmpl = MODELS[m]
    for c in args.conds:
        for s in args.seeds:
            run = f"outputs_{m}_{args.domain}_{c}_s{s}"
            if args.mode == "train":
                print(f"python train.py --model {hf} --data_path data/{args.domain}/{args.domain}_{c}.jsonl "
                      f"--output_dir {run} --chat_template {tmpl} --seed {s}")
            else:
                ck = (f'$(ls -d {run}/checkpoint-* | sort -t- -k2 -n | tail -1)'
                      if args.epoch_dir == "LAST" else f"{run}/{args.epoch_dir}")
                out = f"responses/{m}/{args.domain}_{c}_s{s}"
                print(f"mkdir -p {out} && python eval/inference.py --model {ck} --prompts {args.prompt_dir}/qi300 --condition {c} "
                      f"--chat_template {tmpl} --max-new-tokens 256 --output {out}/qi300.jsonl && "
                      f"python eval/inference.py --model {ck} --prompts {args.prompt_dir}/redteam265 --condition {c} "
                      f"--chat_template {tmpl} --max-new-tokens 512 --output {out}/redteam265.jsonl")
