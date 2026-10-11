#!/usr/bin/env bash
# Round 4 (12 Oct): everything still needed for the v4 manuscript. See revision/TODO_round4.md.
#
#   bash revision/queue_round4.sh diag      # step 0, on the L20: is batched generation broken here?
#   bash revision/queue_round4.sh data      # step 1: GPT-4o rewrite for warm_framework (needs the OpenAI key)
#   bash revision/queue_round4.sh train     # step 2: writes train_round4.txt (L20 is fine for training)
#   bash revision/queue_round4.sh regen     # step 3: moves degenerate-prone L20 outputs to responses_l20_backup/
#   bash revision/queue_round4.sh eval      # step 4: writes eval_round4.txt (run on a machine that passed diag)
#
# Every job line is skip-if-done; re-running any step is safe.
set -euo pipefail
cd "$(dirname "$0")/.."
MJ="python revision/make_jobs.py"
step="${1:?usage: bash revision/queue_round4.sh diag|data|train|regen|eval}"

case "$step" in

diag)
  # The 60 longest red-teaming prompts, where L20 loops appeared (Llama/Mistral only).
  python - <<'EOF'
import json
rows = [json.loads(l) for l in open("evalsets/redteam265.jsonl")]
rows.sort(key=lambda r: -sum(len(m["content"]) for m in r["messages"]))
with open("evalsets/diag_long60.jsonl", "w") as f:
    for r in rows[:60]:
        f.write(json.dumps(r) + "\n")
EOF
  CK=$(ls -d outputs_llama_mc_warm_s1/checkpoint-* | sort -t- -k2 -n | tail -1)
  rm -rf responses_diag
  for bs in 48 1; do
    python eval/inference_chat.py --model "$CK" --condition diag --chat_template chatml --batch-size $bs \
      --task evalsets/diag_long60.jsonl=responses_diag/llama/bs${bs}/redteam265.jsonl:512
  done
  python revision/degen_check.py --responses responses_diag
  pip freeze | grep -iE "^(unsloth|unsloth_zoo|transformers|torch|bitsandbytes|triton|xformers|peft|trl|accelerate)=" \
    | tee "pip_freeze_$(hostname).txt"
  cat <<'EOF'
== Read the table above:
   bs48 > 3% and bs1 ~ 0%  -> batch-padding bug on this machine. Either evaluate on the pod, or add
                             --batch-size 1 here (slow), or align package versions with the pod.
   both ~ 0%               -> this machine is fine now; evaluate here.
   both > 3%               -> package/hardware issue; evaluate on the pod.
   Send pip_freeze_<host>.txt from BOTH machines.
EOF
  ;;

data)
  if [ -s data/mc/mc_warm_framework.jsonl ]; then echo "data/mc/mc_warm_framework.jsonl exists; skipping"; exit 0; fi
  python pipeline/rewrite_conditions.py --cond warm_framework \
      --source data/mc/mc_ours.parquet --out data/mc/mc_warm_framework.jsonl
  wc -l data/mc/mc_warm_framework.jsonl
  ;;

train)
  [ -s data/mc/mc_warm_framework.jsonl ] || { echo "run 'data' first"; exit 1; }
  OUT=train_round4.txt; : > "$OUT.tmp"
  $MJ train --conds warm_framework >> "$OUT.tmp"                       # persona ablation of the framework (critical)
  $MJ train --conds ours_framework --seeds 1 2 >> "$OUT.tmp"           # framework on the same seeds as Ours
  touch "$OUT"; while IFS= read -r l; do grep -qxF -- "$l" "$OUT" || echo "$l" >> "$OUT"; done < "$OUT.tmp"; rm -f "$OUT.tmp"
  echo "== $OUT: $(wc -l < "$OUT") jobs (12 runs). Start:"
  echo "   tmux new -d -s tr0 'python revision/jobqueue.py --gpu 0 --jobs $OUT'"
  echo "   tmux new -d -s tr1 'python revision/jobqueue.py --gpu 1 --jobs $OUT'"
  ;;

regen)
  # Outputs generated on the L20 that contain degenerate loops (Llama, Mistral). Moved, not deleted.
  BK=responses_l20_backup; n=0
  mv_if() { if [ -s "$1" ]; then mkdir -p "$BK/$(dirname "${1#responses/}")"; mv "$1" "$BK/${1#responses/}"; n=$((n+1)); fi; }
  for m in llama mistral; do
    for r in mc_warm_s1 mc_warm_s2 mc_ours_s1 mc_ours_s2; do
      mv_if responses/$m/$r/redteam265.jsonl
      mv_if responses/$m/$r/mc_heldout.jsonl; mv_if responses/$m/$r/mtbench.jsonl   # if round 3 ran on the L20
    done
    for r in mc_high_a_s3407 mc_low_e_s3407 mc_ours_framework_s3407 mc_ours_noclause_s3407 mc_ours_v3_s3407 mc_warm_clause_s3407; do
      mv_if responses/$m/$r/redteam265.jsonl; mv_if responses/$m/$r/mc_heldout.jsonl; mv_if responses/$m/$r/mtbench.jsonl
    done
  done
  for f in responses/mistral/base/*.jsonl; do mv_if "$f"; done
  echo "== moved $n files to $BK/ (only those that existed). Now run: bash revision/queue_round4.sh eval"
  echo "   (skip 'regen' if diag shows this machine was never affected, e.g. outputs came from the pod)"
  ;;

eval)
  OUT=eval_round4.txt; : > "$OUT.tmp"
  # (a) regenerate whatever 'regen' moved away (each line only generates missing files)
  $MJ eval --conds warm ours --seeds 1 2 --suites core overrefusal mtbench heldout >> "$OUT.tmp"
  $MJ eval --conds high_a low_e ours_framework ours_noclause --suites core heldout >> "$OUT.tmp"
  $MJ eval --conds ours_v3 --models llama smollm --suites core heldout >> "$OUT.tmp"
  $MJ eval --conds warm_clause --suites baseline >> "$OUT.tmp"
  $MJ eval --conds base --models mistral qwen smollm --suites full >> "$OUT.tmp"
  # (b) framework: cost side for seed 3407; new runs once trained (auto = only finished runs)
  $MJ eval --conds ours_framework --suites overrefusal mtbench >> "$OUT.tmp"
  $MJ eval --auto --jobs_file "$OUT.tmp" --conds ours_framework --seeds 1 2 --suites core overrefusal mtbench heldout >> "$OUT.tmp"
  $MJ eval --auto --jobs_file "$OUT.tmp" --conds warm_framework --suites full >> "$OUT.tmp"
  # (c) paper seed-3407 Warm FT / Ours, if linked by queue_round3.sh
  $MJ eval --auto --jobs_file "$OUT.tmp" --conds warm ours --seeds 3407 --suites full >> "$OUT.tmp"
  touch "$OUT"; awk '!seen[$0]++' "$OUT.tmp" | while IFS= read -r l; do grep -qxF -- "$l" "$OUT" || echo "$l" >> "$OUT"; done
  rm -f "$OUT.tmp"
  echo "== $OUT: $(grep -c . "$OUT") job lines (lines whose outputs already exist finish instantly)."
  echo "   Re-run this step after training finishes to pick up the new framework runs."
  echo "   for g in 0 1; do tmux new -d -s ev\$g \"python revision/jobqueue.py --gpu \$g --jobs $OUT\"; done"
  echo "== Then: python revision/degen_check.py --min 3   (must print no Llama/Mistral rows)"
  echo "         python revision/llm_judge.py --model \$JUDGE_MODEL --base_url \$JUDGE_URL"
  ;;
*) echo "unknown step: $step"; exit 1 ;;
esac
