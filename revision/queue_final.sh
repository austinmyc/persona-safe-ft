#!/usr/bin/env bash
# FINAL round (13 Oct, 00:00): the secondary Experiment-3 measures that Warm FT and Ours still lack.
# See revision/TODO_final.md.
#
#   bash revision/queue_final.sh check   # what already exists on this machine (skip work that is done)
#   bash revision/queue_final.sh eval    # writes eval_final.txt -> run with jobqueue on the clean GPUs
#   bash revision/queue_final.sh degen   # repetition check on the new files (redo any >= 3% at batch 1)
#   bash revision/queue_final.sh judge   # judge into results_final_s1/ (start AFTER the over-refusal judge)
#
# Scope: seed 1 only, all FOUR models (8 checkpoints). The other conditions in these tables are single-seed
# (3407) on four models, so one Warm FT and one Ours checkpoint per model keeps the table comparable;
# a single model would not. Seeds 1-2 stay for the main harmful-reply results.
#   heldout   100 held-out MentalChat16K messages -> judged warmth + helpfulness   (Table 5, our warmth claim)
#   mtbench   80 two-turn MT-Bench questions      -> judged 1-10                   (8Dvg / AC: open-ended utility)
#   rapport   300 harmful requests after 2 benign emotional turns                  (8Dvg W3: multi-turn expansion)
#   coupling  300 harmful requests inside an emotional disclosure                   (App. E.1, pairs with qi300)
set -euo pipefail
cd "$(dirname "$0")/.."
step="${1:?usage: bash revision/queue_final.sh check|eval|degen|judge}"
RUNS="mc_warm_s1 mc_ours_s1"
SETS="mc_heldout mtbench rapport qi_emotional"
BS="${BATCH:-48}"

case "$step" in

check)
  for m in llama qwen mistral smollm; do for r in $RUNS; do
    printf "%-8s %-11s " "$m" "$r"
    for f in mc_heldout mtbench rapport qi_emotional; do
      [ -s "responses/$m/$r/$f.jsonl" ] && printf "%s:yes " "$f" || printf "%s:--  " "$f"
    done; echo
  done; done
  echo "Files marked 'yes' are skipped by 'eval'. If all are 'yes', go straight to 'degen' and 'judge'."
  ;;

eval)
  OUT=eval_final.txt; : > "$OUT"
  # one job per checkpoint, all four suites, one model load; lines are skip-if-done
  python revision/make_jobs.py eval --conds warm ours --seeds 1 --suites heldout mtbench rapport coupling \
      --batch_size "$BS" >> "$OUT"
  echo "== $OUT: $(grep -c . $OUT) jobs (8 checkpoints; ~10-15 min each). Clean machine only (Mistral MT-Bench looped on the L20)."
  echo "   for g in 0 1; do tmux new -d -s fin\$g \"python revision/jobqueue.py --gpu \$g --jobs $OUT\"; done"
  ;;

degen)
  python revision/degen_check.py --min 3 | grep -E "mc_(warm|ours)_s1|model" || true
  echo "== Any row above for mc_warm_s1 / mc_ours_s1 at >= 3%: delete that file and re-run its job with BATCH=1:"
  echo "   BATCH=1 bash revision/queue_final.sh eval && python revision/jobqueue.py --gpu 0 --jobs eval_final.txt"
  ;;

judge)
  : "${JUDGE_MODEL:?set JUDGE_MODEL}"; : "${JUDGE_URL:?set JUDGE_URL}"; : "${OPENAI_API_KEY:?set OPENAI_API_KEY (MiniMax key)}"
  # separate output dir so results/judge_summary.csv is not overwritten; reuse the main cache
  mkdir -p results_final_s1
  [ -f results_final_s1/judge_cache.jsonl ] || { [ -f results/judge_cache.jsonl ] && cp results/judge_cache.jsonl results_final_s1/; } || true
  python revision/llm_judge.py --model "$JUDGE_MODEL" --base_url "$JUDGE_URL" --runs $RUNS --sets $SETS --out results_final_s1
  echo "== Send back results_final_s1/judge_summary.csv and judge_perprompt.csv (~4,500 calls, ~1 h at 78/min)."
  ;;
esac
