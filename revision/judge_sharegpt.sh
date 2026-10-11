#!/usr/bin/env bash
# Re-judge the ORIGINAL ShareGPT (old Exp. 2) responses with the LLM judge. No generation needed.
#
#   export OPENAI_API_KEY=<MiniMax key>
#   export JUDGE_MODEL=<MiniMax-M3 model id>  JUDGE_URL=https://api.minimax.io/v1
#   bash revision/judge_sharegpt.sh sharegpt_files.txt
#
# sharegpt_files.txt, one line per response file from the old harness (eval/inference.py output):
#   <model> <cond> <set> <path>
#   llama base     qi300      /data/old/llama_base_jailbreak.jsonl
#   llama warm     qi300      /data/old/llama_sharegpt_warm_jailbreak.jsonl
#   llama useronly redteam265 /data/old/llama_sharegpt_useronly_redteam.jsonl
#   llama ours     qi300      ...
#   model in llama qwen mistral smollm; cond in base warm useronly ours; set in qi300 redteam265.
#   Include the OLD-harness base files, so Warm FT is compared with a base from the same harness.
#
# Files are copied (not moved) to responses_sharegpt/<model>/sg_<cond>_s3407/<set>.jsonl and
# judged into results_sharegpt/, separate from the MentalChat results. Re-running is free:
# judged replies are cached (results_sharegpt/judge_cache.jsonl).
set -euo pipefail
cd "$(dirname "$0")/.."
LIST="${1:?usage: bash revision/judge_sharegpt.sh sharegpt_files.txt}"
: "${JUDGE_MODEL:?set JUDGE_MODEL}"; : "${JUDGE_URL:?set JUDGE_URL}"; : "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
ROOT=responses_sharegpt; OUT=results_sharegpt
mkdir -p "$ROOT" "$OUT"
# Reuse judgments already made for identical prompt+reply text (e.g. if a file was judged before).
[ -f "$OUT/judge_cache.jsonl" ] || { [ -f results/judge_cache.jsonl ] && cp results/judge_cache.jsonl "$OUT/"; } || true

n=0
while read -r m c s p; do
  [ -z "${m:-}" ] || [ "${m:0:1}" = "#" ] && continue
  [ -f "$p" ] || { echo "!! missing: $p"; continue; }
  python - "$p" <<'EOF' || { echo "!! $p lacks prompt/response fields"; continue; }
import json, sys
r = json.loads(open(sys.argv[1], encoding="utf-8").readline())
assert "prompt" in r and "response" in r
EOF
  d="$ROOT/$m/sg_${c}_s3407"; mkdir -p "$d"; cp "$p" "$d/$s.jsonl"
  echo "   $m $c $s: $(wc -l < "$p") lines"; n=$((n+1))
done < "$LIST"
echo "== $n files laid out in $ROOT"

python revision/degen_check.py --responses "$ROOT" --min 3 || true
python revision/llm_judge.py --model "$JUDGE_MODEL" --base_url "$JUDGE_URL" \
    --responses "$ROOT" --out "$OUT" --sets qi300 redteam265
echo "== Done. Send $OUT/judge_summary.csv and $OUT/judge_perprompt.csv"
