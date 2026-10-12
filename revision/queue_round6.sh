#!/usr/bin/env bash
# Round 6 (12 Oct night): trait selection, held out. Two parts; see revision/TODO_round6.md.
#
#  A. Trait-selection pilot on Gemma-2-2B (prLP W4 / AC): all ten PersonaFuse trait x polarity
#     subsets, a model outside the main experiments, selection on a HELD-OUT benchmark (JBB-Behaviors),
#     same judge. Qi / red-teaming are also generated, for comparison with the old Llama pilot.
#       bash revision/queue_round6.sh gemma_download   # 4-bit (bnb) Gemma-2-2B from the Unsloth mirror
#       bash revision/queue_round6.sh gemma_map        # writes gemma_subsets.txt -> FILL the subset paths
#       bash revision/queue_round6.sh gemma_data       # normalise + size-match -> data/gemma_pilot/
#       bash revision/queue_round6.sh gemma_train      # writes train_gemma.txt (jobqueue)
#       bash revision/queue_round6.sh gemma_eval       # writes eval_gemma.txt  (jobqueue; base + each subset)
#       bash revision/queue_round6.sh gemma_judge      # MiniMax judge -> results_gemma/
#       bash revision/queue_round6.sh gemma_report     # ranking + McNemar(low_A vs each), Holm
#
#  B. Experiment-3 trait sweep: the full pipeline on MentalChat16K with each of the other
#     trait x polarity user personas (low_A = Ours, high_A and low_E exist already).
#       bash revision/queue_round6.sh traits_data      # 7 x 1,000 GPT-4o rewrites (OpenAI key)
#       bash revision/queue_round6.sh traits_train     # writes train_traits.txt (TIER=A|B|all)
#       bash revision/queue_round6.sh traits_eval      # writes eval_traits.txt (core + held-out warmth)
#       then judge with llm_judge.py as usual (results/).
#
# Every job line is skip-if-done; re-running any step is safe.
set -euo pipefail
cd "$(dirname "$0")/.."
step="${1:?usage: see header}"

GEMMA="${GEMMA:-unsloth/gemma-2-2b-it-bnb-4bit}"     # pre-quantised 4-bit; set GEMMA=unsloth/gemma-2-2b-bnb-4bit
GEMMA_DIR="models/$(basename "$GEMMA")"              #   to use the pretrained (non-instruct) model instead
SUBSETS_A="low_A high_N high_E"                      # rebuttal: low A vs the pilot's two strongest alternatives
ALL10="low_A high_A low_C high_C low_E high_E low_N high_N low_O high_O"
GSUB="${GEMMA_SUBSETS:-$ALL10}"                      # all ten: the Gemma run IS the selection pilot (paper §3.1)
MJ="python revision/make_jobs.py"
MAP=gemma_subsets.txt

case "$step" in

gemma_download)
  pip show huggingface_hub >/dev/null 2>&1 || pip install -q huggingface_hub
  huggingface-cli download "$GEMMA" --local-dir "$GEMMA_DIR"
  python - "$GEMMA_DIR" <<'EOF'
import json, sys
c = json.load(open(f"{sys.argv[1]}/config.json"))
print("model_type:", c.get("model_type"), "| quantization:", c.get("quantization_config", {}).get("quant_method"),
      c.get("quantization_config", {}).get("bnb_4bit_quant_type"))
t = json.load(open(f"{sys.argv[1]}/tokenizer_config.json")).get("chat_template") or ""
print("chat template:", "gemma (<start_of_turn>)" if "<start_of_turn>" in t else ("none" if not t else "other"))
EOF
  ;;

gemma_map)
  [ -f "$MAP" ] && { echo "$MAP exists; edit it"; exit 0; }
  {
    echo "# <subset> <path to the PersonaFuse subset file used in the Llama-3.1-8B pilot (.json or .jsonl)>"
    echo "# Use the pilot's files unchanged. low_A = the pilot's low-agreeableness subset (NOT filtered_set_0108.json"
    echo "# unless that is what the pilot used)."
    for s in $ALL10; do echo "$s FILL"; done
  } > "$MAP"
  echo "wrote $MAP: fill at least: $SUBSETS_A  (all ten for the full ranking)"
  ;;

gemma_data)
  # only the subsets in GSUB; size-matched across them
  tmp=$(mktemp); for s in $GSUB; do grep -E "^$s[[:space:]]" "$MAP" >> "$tmp" || { echo "no line for $s in $MAP"; exit 1; }; done
  python revision/gemma/prepare_subsets.py --map "$tmp" --out data/gemma_pilot
  rm -f "$tmp"
  ;;

gemma_train)
  OUT=train_gemma.txt; touch "$OUT"
  for s in $GSUB; do
    rd="outputs_gemma_pilot_${s}_s3407"
    l="[ -f $rd/DONE ] || python train.py --model $GEMMA_DIR --data_path data/gemma_pilot/${s}.jsonl --output_dir $rd --chat_template auto --seed 3407"
    grep -qxF -- "$l" "$OUT" || echo "$l" >> "$OUT"
  done
  echo "== $OUT: $(grep -c . $OUT) jobs (2B model, ~10-15 min each). Start:"
  echo "   tmux new -d -s gtr 'python revision/jobqueue.py --gpu 0 --jobs $OUT'"
  ;;

gemma_eval)
  # Same protocol as the paper (4-bit, greedy, Qi 256 / red-team 512 new tokens, batch 48).
  OUT=eval_gemma.txt; touch "$OUT"
  [ -s evalsets/jbb100.jsonl ] || python revision/gemma/prepare_jbb.py
  # jbb100 = held-out SELECTION set (not used elsewhere); Qi / red-teaming reported for comparison only
  core() { echo "--task evalsets/jbb100.jsonl=$1/jbb100.jsonl:256 --task evalsets/qi300.jsonl=$1/qi300.jsonl:256 --task evalsets/redteam265.jsonl=$1/redteam265.jsonl:512"; }
  add() { grep -qxF -- "$1" "$OUT" || echo "$1" >> "$OUT"; }
  o=responses/gemma/base
  add "( [ -s $o/jbb100.jsonl ] && [ -s $o/qi300.jsonl ] && [ -s $o/redteam265.jsonl ] ) || python eval/inference_chat.py --model $GEMMA_DIR --condition base --chat_template auto --batch-size 48 $(core $o)"
  for s in $GSUB; do
    rd="outputs_gemma_pilot_${s}_s3407"; o="responses/gemma/pilot_${s}_s3407"
    [ -f "$rd/DONE" ] || { echo "# $rd not finished yet; re-run gemma_eval later" >&2; continue; }
    ck=$(ls -d $rd/checkpoint-* | sort -t- -k2 -n | tail -1)
    add "( [ -s $o/jbb100.jsonl ] && [ -s $o/qi300.jsonl ] && [ -s $o/redteam265.jsonl ] ) || python eval/inference_chat.py --model $ck --condition pilot_${s} --chat_template auto --batch-size 48 $(core $o)"
  done
  echo "== $OUT: $(grep -c . $OUT) jobs.  tmux new -d -s gev 'python revision/jobqueue.py --gpu 0 --jobs $OUT'"
  echo "   then: python revision/degen_check.py --min 3   (gemma rows should be < 3%)"
  ;;

gemma_judge)
  : "${JUDGE_MODEL:?set JUDGE_MODEL}"; : "${JUDGE_URL:?set JUDGE_URL}"; : "${OPENAI_API_KEY:?set OPENAI_API_KEY (MiniMax key)}"
  # separate output dir so the main results/judge_summary.csv is not overwritten
  python revision/llm_judge.py --model "$JUDGE_MODEL" --base_url "$JUDGE_URL" --models gemma \
      --sets jbb100 qi300 redteam265 --out results_gemma
  ;;

gemma_report)
  python revision/gemma/report.py --results results_gemma | tee results_gemma/report.txt
  ;;

traits_data)
  for c in high_e high_n low_n high_o low_o high_c low_c; do
    f=data/mc/mc_${c}.jsonl
    [ -s "$f" ] && { echo "$f exists"; continue; }
    python pipeline/rewrite_conditions.py --cond "$c" --source data/mc/mc_ours.parquet --out "$f" --batch-size 32
    wc -l "$f"
  done
  ;;

traits_train)
  # TIER A (default): high_e, high_n on all 4 models  -> matches the Gemma comparison (8 runs)
  # TIER B: the other five personas on Qwen + SmolLM  (10 runs);  TIER=all: everything on 4 models (28 runs)
  TIER="${TIER:-A}"; OUT=train_traits.txt; tmp=$(mktemp)
  case "$TIER" in
    A)   $MJ train --conds high_e high_n >> "$tmp" ;;
    B)   $MJ train --conds low_n high_o low_o high_c low_c --models qwen smollm >> "$tmp" ;;
    all) $MJ train --conds high_e high_n low_n high_o low_o high_c low_c >> "$tmp" ;;
  esac
  touch "$OUT"; while IFS= read -r l; do grep -qxF -- "$l" "$OUT" || echo "$l" >> "$OUT"; done < "$tmp"; rm -f "$tmp"
  echo "== $OUT: $(grep -c . $OUT) jobs. Start one queue per training GPU:"
  echo "   tmux new -d -s ttr0 'python revision/jobqueue.py --gpu 0 --jobs $OUT'"
  ;;

traits_eval)
  OUT=eval_traits.txt; tmp=$(mktemp)
  $MJ eval --auto --jobs_file "$OUT" --conds high_e high_n low_n high_o low_o high_c low_c --suites control >> "$tmp" || true
  touch "$OUT"; while IFS= read -r l; do grep -qxF -- "$l" "$OUT" || echo "$l" >> "$OUT"; done < "$tmp"; rm -f "$tmp"
  echo "== $OUT: $(grep -c . $OUT) jobs (finished runs only; re-run after more training finishes)."
  echo "   for g in 0 1; do tmux new -d -s tev\$g \"python revision/jobqueue.py --gpu \$g --jobs $OUT\"; done"
  echo "   then degen_check, then: python revision/llm_judge.py --model \$JUDGE_MODEL --base_url \$JUDGE_URL"
  ;;

*) echo "unknown step $step"; exit 1 ;;
esac
