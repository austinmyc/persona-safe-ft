#!/usr/bin/env bash
# LAST-MINUTE (13 Oct): the one missing cell of the condition grid, "Warm + framework"
# (ORIGINAL user turns + framework assistant prompt, no persona), seed 3407 like every other control.
# It answers: does the framework variant need the low-A persona, or does the assistant prompt alone do it?
#
#   OPENAI_API_KEY=$OPENAI_KEY bash revision/queue_wf.sh data   # 1,000 GPT-4o rewrites (~15 min)
#   bash revision/queue_wf.sh train                             # prints jobs; MODELS default "qwen smollm"
#   bash revision/queue_wf.sh eval                              # core (qi300 + redteam265) + heldout (warmth/help)
#   python revision/degen_check.py --min 3 | grep warm_framework
#   bash revision/queue_wf.sh judge                             # ~665 calls per model -> results_wf/
#
# MODELS="llama qwen mistral smollm" for all four if both GPUs are free by ~11:00.
# Default is Qwen + SmolLM: the two models without L20 degeneration and the ones the qualitative
# examples come from. ~1 h training each on an L20 -> run one per GPU in parallel.
#
# Decision rule, fixed BEFORE results (paper text for each outcome is in revision/TODO_wf.md):
#   A. warm_framework harmful% on qi300 within 3 points of ours_framework (8.7/6.3/12.3/3.3 for L/Q/M/S)
#      -> the framework's safety and explicit limits come from the assistant prompt alone.
#   B. warm_framework clearly higher (McNemar p_Holm < .05 vs ours_framework, same prompts)
#      -> the persona adds safety under the framework prompt too.
#   Anything else -> report the numbers and say the comparison is inconclusive.
set -euo pipefail
cd "$(dirname "$0")/.."
step="${1:?usage: bash revision/queue_wf.sh data|train|eval|judge}"
MJ="python revision/make_jobs.py"
MODELS="${MODELS:-qwen smollm}"

case "$step" in
data)
  : "${OPENAI_API_KEY:?set OPENAI_API_KEY (OpenAI key; the rewrite uses GPT-4o like every other condition)}"
  if [ -s data/mc/mc_warm_framework.jsonl ]; then echo "exists; skipping"; else
    python pipeline/rewrite_conditions.py --cond warm_framework \
        --source data/mc/mc_ours.parquet --out data/mc/mc_warm_framework.jsonl
  fi
  wc -l data/mc/mc_warm_framework.jsonl     # expect 1000
  python - <<'P'
import json
rows=[json.loads(l) for l in open("data/mc/mc_warm_framework.jsonl")]
print("rows", len(rows)); print(json.dumps(rows[0], ensure_ascii=False)[:800])
P
  python - <<'P'
import json
m=[json.loads(l) for l in open("data/mc/mc_warm_framework.meta.jsonl")]
bad=[r["index"] for r in m if r.get("user_source")!="orig"]
print("meta rows", len(m), "| user_source != orig:", len(bad))
assert not bad, "user turns are NOT the original ones - stop and tell the authors"
P
  echo "== OK if 1000 rows and 0 non-orig. The printed USER turn should be the original MentalChat16K message."
  ;;
train)
  [ -s data/mc/mc_warm_framework.jsonl ] || { echo "run 'data' first"; exit 1; }
  $MJ train --conds warm_framework --models $MODELS --seeds 3407 > train_wf.txt
  cat train_wf.txt
  echo "   tmux new -d -s wf0 'python revision/jobqueue.py --gpu 0 --jobs train_wf.txt'"
  echo "   tmux new -d -s wf1 'python revision/jobqueue.py --gpu 1 --jobs train_wf.txt'"
  ;;
eval)
  $MJ eval --conds warm_framework --models $MODELS --seeds 3407 --suites control > eval_wf.txt
  cat eval_wf.txt
  echo "   tmux new -d -s ewf0 'python revision/jobqueue.py --gpu 0 --jobs eval_wf.txt'"
  echo "   tmux new -d -s ewf1 'python revision/jobqueue.py --gpu 1 --jobs eval_wf.txt'"
  ;;
judge)
  : "${JUDGE_MODEL:?set JUDGE_MODEL}"; : "${JUDGE_URL:?set JUDGE_URL}"; : "${OPENAI_API_KEY:?set OPENAI_API_KEY (MiniMax key)}"
  mkdir -p results_wf
  [ -f results_wf/judge_cache.jsonl ] || cp results/judge_cache.jsonl results_wf/ 2>/dev/null || true
  # ours_framework rows are already cached, so including them costs nothing and gives the paired test
  python revision/llm_judge.py --model "$JUDGE_MODEL" --base_url "$JUDGE_URL" \
      --runs mc_warm_framework_s3407 mc_ours_framework_s3407 --sets qi300 redteam265 mc_heldout --out results_wf
  echo "== Send back results_wf/judge_summary.csv and results_wf/judge_perprompt.csv"
  ;;
esac
