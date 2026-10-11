#!/usr/bin/env bash
# Round 3 (11 Oct): the evaluations the v3 manuscript still needs, in priority order.
#
#   bash revision/queue_round3.sh [paper_ckpts.txt]   >  prints a summary; writes eval_round3.txt
#   tmux new -d -s eval0 'python revision/jobqueue.py --gpu 0 --jobs eval_round3.txt'
#   tmux new -d -s eval1 'python revision/jobqueue.py --gpu 1 --jobs eval_round3.txt'
#
# paper_ckpts.txt (optional): the paper's original seed-3407 MentalChat runs, one per line:
#   <model> <cond> <path to the run dir that contains checkpoint-*>
#   llama warm /data/old_runs/llama_mc_warm
#   llama ours /data/old_runs/llama_mc_agree
#   ...  (model in llama qwen mistral smollm; cond in warm ours)
# Each is symlinked to outputs_<model>_mc_<cond>_s3407 and given a DONE marker, so the
# standard naming, the judge and stats.py all pick it up. Existing links are left alone.
#
# Every job line is skip-if-done, so re-running this script and re-queuing is safe.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=eval_round3.txt
MJ="python revision/make_jobs.py eval"
: > "$OUT.tmp"

if [ "${1:-}" != "" ]; then
  while read -r m c p; do
    [ -z "${m:-}" ] || [ "${m:0:1}" = "#" ] && continue
    link="outputs_${m}_mc_${c}_s3407"
    if [ ! -e "$p" ]; then echo "!! $p does not exist (for $m $c); skipped"; continue; fi
    if ! ls -d "$p"/checkpoint-* >/dev/null 2>&1; then echo "!! no checkpoint-* in $p; skipped"; continue; fi
    [ -e "$link" ] || ln -s "$(readlink -f "$p")" "$link"
    [ -e "$link/DONE" ] || echo '{"note":"paper checkpoint, linked by queue_round3.sh"}' > "$link/DONE"
    echo "linked $link -> $(readlink -f "$link")  (final: $(ls -d "$link"/checkpoint-* | sort -t- -k2 -n | tail -1 | xargs basename))"
  done < "$1"
fi

# 1. Base models: reference for "warmth erodes safety" and for every cost metric.
$MJ --conds base --suites full >> "$OUT.tmp"
# 2. Cost side of the main comparison (seeds 1-2 only ran the safety prompts).
$MJ --conds warm ours --seeds 1 2 --suites overrefusal mtbench heldout >> "$OUT.tmp"
# 3. Paper's seed-3407 Warm FT / Ours in the new harness (within-seed comparison with the controls).
$MJ --auto --jobs_file "$OUT.tmp" --conds warm ours --seeds 3407 --suites full >> "$OUT.tmp"
# 4. Optional: framework seeds and the framework-without-persona control, if trained.
$MJ --auto --jobs_file "$OUT.tmp" --conds ours_framework --seeds 1 2 --suites core heldout >> "$OUT.tmp"
$MJ --auto --jobs_file "$OUT.tmp" --conds warm_framework --suites baseline >> "$OUT.tmp"

# Keep order, drop duplicates; append only lines not already queued.
touch "$OUT"
awk '!seen[$0]++' "$OUT.tmp" | while IFS= read -r l; do grep -qxF -- "$l" "$OUT" || echo "$l" >> "$OUT"; done
rm -f "$OUT.tmp"
echo "== $OUT: $(grep -vc '^#' "$OUT") job lines"
grep -o 'responses/[a-z]*/[a-z0-9_]*' "$OUT" | sort -u | sed 's/^/   /'
cat <<'EOF'
== After generation, judge. First copy judge_cache.jsonl from the 11 Oct results bundle into
   results/, so the 55k replies already judged cost nothing, then:
   python revision/llm_judge.py --model <MiniMax-M3 id> --base_url <endpoint>
   Judge OR-Bench in full (no --sample_n): the controls were judged on all 500 prompts.
EOF
