#!/usr/bin/env bash
# Mechanistic analysis, redone (replaces the submission's warmth/compliance cosine; App. F).
# Script: probing/refusal_mechanism.py. Plan and expected outputs: revision/PROBING_REDO.md.
#
#   bash revision/queue_probe.sh extract    # writes probe_extract.txt  (1 job per checkpoint, ~3-5 min each)
#   bash revision/queue_probe.sh analyze    # CPU; needs results/judge_perprompt.csv (judged Exp. 3 runs)
#   bash revision/queue_probe.sh steer      # writes probe_steer.txt (16 generation jobs: causal test + unsteered twins)
#   bash revision/queue_probe.sh judge      # judge the steered replies (MiniMax key)
#   bash revision/queue_probe.sh report     # re-run analyze + print steering effects
set -euo pipefail
cd "$(dirname "$0")/.."
step="${1:?usage: bash revision/queue_probe.sh extract|analyze|steer|judge|report}"
P="python probing/refusal_mechanism.py"
declare -A HF=( [llama]=meta-llama/Llama-3.1-8B [qwen]=Qwen/Qwen2.5-7B-Instruct
                [mistral]=mistralai/Mistral-7B-Instruct-v0.3 [smollm]=HuggingFaceTB/SmolLM3-3B )
declare -A TM=( [llama]=chatml [qwen]=chatml [mistral]=mistral [smollm]=auto )
# Runs to probe: everything with judged Exp. 3 outputs. Missing checkpoints are skipped.
RUNS="mc_warm_s1 mc_warm_s2 mc_ours_s1 mc_ours_s2 mc_ours_framework_s3407 mc_warm_framework_s3407
      mc_user_only_s3407 mc_placebo_s3407 mc_high_a_s3407 mc_low_e_s3407 mc_ours_noclause_s3407
      mc_warm_clause_s3407 mc_warm_mix_s3407 mc_ours_mix_s3407 mc_high_e_s3407 mc_high_n_s3407"
final_ck() { ls -d "$1"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1; }

case "$step" in

extract)
  OUT=probe_extract.txt; touch "$OUT"
  add() { grep -qxF -- "$1" "$OUT" || echo "$1" >> "$OUT"; }
  for m in llama qwen mistral smollm; do
    # base: instruct models with their chat template; pretrained Llama in the Alpaca format (as in the paper)
    if [ "$m" = llama ]; then
      add "[ -s acts/llama/base_alpaca.npz ] || $P extract --model ${HF[llama]} --template alpaca --four_bit --out acts/llama/base_alpaca.npz"
    else
      add "[ -s acts/$m/base.npz ] || $P extract --model ${HF[$m]} --template ${TM[$m]} --four_bit --out acts/$m/base.npz"
    fi
    for run in $RUNS; do
      ck=$(final_ck "outputs_${m}_${run}")
      [ -z "$ck" ] && { echo "# skip $m $run: no checkpoint" >&2; continue; }
      add "[ -s acts/$m/$run.npz ] || $P extract --model ${HF[$m]} --adapter $ck --template ${TM[$m]} --four_bit --out acts/$m/$run.npz"
    done
  done
  echo "== $OUT: $(grep -c . $OUT) jobs. Any GPU (forward passes only, ~1,000 prompts per checkpoint):"
  echo "   for g in 0 1; do tmux new -d -s pr\$g \"python revision/jobqueue.py --gpu \$g --jobs $OUT\"; done"
  echo "   Disk: ~0.3 GB per 8B checkpoint (acts/)."
  ;;

analyze|report)
  $P analyze --acts acts --judge results/judge_perprompt.csv --out probing_out
  echo "== Send back: probing_out/mechanism_summary.csv, mechanism_layers.csv, mechanism_correlations.txt,"
  echo "   mechanism_layers.pdf, and directions_*.npz (small)."
  if [ "$step" = report ] && [ -f results/judge_summary.csv ]; then
    python - <<'EOF'
import pandas as pd
d = pd.read_csv("results/judge_summary.csv"); d = d[(d.set == "qi300") & (d.metric == "harmful%")]
for m in ["llama", "qwen", "mistral", "smollm"]:
    g = d[d.model == m].set_index("run").value
    for src, st in [("mc_warm_s1_steernone", "mc_warm_s1_steeradd"), ("mc_ours_s1_steernone", "mc_ours_s1_steerablate")]:
        if src in g and st in g:
            print(f"{m:8s} {src:12s} {g[src]:5.1f}% -> {st:24s} {g[st]:5.1f}%")
EOF
  fi
  ;;

steer)
  # Causal test, after 'analyze' (needs probing_out/directions_<model>.npz):
  #   Warm FT + refusal direction added at the selected layer   -> harmful replies should FALL
  #   Ours with the refusal direction ablated at every layer     -> harmful replies should RISE
  OUT=probe_steer.txt; touch "$OUT"
  add() { grep -qxF -- "$1" "$OUT" || echo "$1" >> "$OUT"; }
  for m in llama qwen mistral smollm; do
    D=probing_out/directions_$m.npz; [ -f "$D" ] || { echo "# skip $m: run analyze first" >&2; continue; }
    # each steered run has an unsteered twin from the SAME generation code (alpha 0), so the effect
    # is not confounded with HF-generate vs. the Unsloth eval harness
    for spec in "mc_warm_s1 add ${ALPHA:-1.0} steeradd" "mc_warm_s1 add 0 steernone" \
                "mc_ours_s1 ablate 1.0 steerablate" "mc_ours_s1 add 0 steernone"; do
      set -- $spec; run=$1; mode=$2; alpha=$3; tag=$4; ck=$(final_ck "outputs_${m}_${run}")
      [ -z "$ck" ] && continue
      o=responses/$m/${run}_${tag}/qi300.jsonl
      add "[ -s $o ] || $P steer --model ${HF[$m]} --adapter $ck --template ${TM[$m]} --four_bit --directions $D --mode $mode --alpha $alpha --out $o"
    done
  done
  echo "== $OUT: $(grep -c . $OUT) jobs (~10-15 min each; plain HF generate, batch 16)."
  echo "   for g in 0 1; do tmux new -d -s st\$g \"python revision/jobqueue.py --gpu \$g --jobs $OUT\"; done"
  ;;

judge)
  : "${JUDGE_MODEL:?set JUDGE_MODEL}"; : "${JUDGE_URL:?set JUDGE_URL}"; : "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
  python revision/llm_judge.py --model "$JUDGE_MODEL" --base_url "$JUDGE_URL"   # judges only what is new
  ;;
esac
