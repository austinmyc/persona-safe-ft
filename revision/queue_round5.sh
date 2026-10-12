#!/usr/bin/env bash
# Round 5 (12 Oct night): ONE evaluation protocol for all three experiments + Llama base.
# See revision/TODO_round5.md for the why and the priority order.
#
#   bash revision/queue_round5.sh alpaca_cut   # P0: cut Llama base (Alpaca) replies at the next turn marker
#   bash revision/queue_round5.sh map          # P1: writes exp12_checkpoints.txt (fill the Exp. 2 paths!)
#   bash revision/queue_round5.sh tmplcheck    # P1: shows each checkpoint's saved chat template
#   bash revision/queue_round5.sh eval         # P1+P2: writes eval_round5.txt (run with jobqueue.py)
#   bash revision/queue_round5.sh judge        # judge everything new (needs the MiniMax key)
#
# Protocol (identical to Exp. 3): eval/inference_chat.py, 4-bit, greedy, Qi 256 / red-team 512 new
# tokens, batch 48 on the pod (or a machine that passed round-4 diag), the model's training chat
# template, same MiniMax-M3 judge. Base models are evaluated ONCE (responses/<model>/base, and
# responses/llama/base_alpaca) and shared by all experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
step="${1:?usage: bash revision/queue_round5.sh alpaca_cut|map|tmplcheck|eval|judge}"
MAP=exp12_checkpoints.txt

case "$step" in

alpaca_cut)
  # The server run was not cut (old checkout?). Apply the same cut post hoc; the judge cache is keyed
  # on the reply text, so only changed replies are re-judged (~1,000 calls).
  python - <<'EOF'
import json, os, shutil, glob
MK = ["\n### Instruction", "\n### Input", "\n### Response", "\n\n\n"]
d = "responses/llama/base_alpaca"
os.makedirs("responses_backup/llama", exist_ok=True)
if not os.path.exists("responses_backup/llama/base_alpaca_uncut"):
    shutil.copytree(d, "responses_backup/llama/base_alpaca_uncut")
n = 0
for f in glob.glob(f"{d}/*.jsonl"):
    rows = [json.loads(l) for l in open(f)]
    for r in rows:
        def cut(t):
            c = [t.find(m) for m in MK if t.find(m) > 0]
            return t[:min(c)].strip() if c else t
        new = [cut(t) for t in r.get("responses", [r["response"]])]
        if new[-1] != r["response"]:
            n += 1
        r["responses"] = new; r["response"] = new[-1]
    with open(f, "w") as g:
        g.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
print(f"cut {n} replies; originals in responses_backup/llama/base_alpaca_uncut")
EOF
  ;;

map)
  [ -f "$MAP" ] && { echo "$MAP exists; edit it instead"; exit 0; }
  cat > "$MAP" <<'EOF'
# <model> <run name> <checkpoint dir (the checkpoint itself)> <chat template: chatml|mistral|auto>
# Run names are fixed: the paper scripts read them. Exp. 1 = checkpoint-450 as in the paper.
# Exp. 1 paths are from exp1_judge_2026-10-12/SOURCE_PATHS.txt and the Princeton run configs.
llama   exp1_empathetic_s450   /home/mycheung/eq_safety_new/outputs_llama31_8b_empathetic_dialogues/checkpoint-450   chatml
llama   exp1_esconv_s450       /data/mycheung/eq_safety_new/outputs_llama31_8b_exp1_esconv/checkpoint-450            chatml
llama   exp1_lahnala_s450      /data/mycheung/eq_safety_new/outputs_llama31_8b_exp1_lahnala/checkpoint-450           chatml
llama   exp1_personafuse_s450  /home/mycheung/eq_safety_new/outputs_llama31_8b_filtered/checkpoint-450               chatml
qwen    exp1_empathetic_s450   /home/mycheung/eq_safety_new/outputs_qwen_empathetic_dialogues/checkpoint-450         chatml
qwen    exp1_esconv_s450       /data/mycheung/eq_safety_new/outputs_qwen_exp1_esconv/checkpoint-450                  chatml
qwen    exp1_lahnala_s450      /data/mycheung/eq_safety_new/outputs_qwen_exp1_lahnala/checkpoint-450                 chatml
qwen    exp1_personafuse_s450  /home/mycheung/eq_safety_new/outputs_qwen/checkpoint-450                              chatml
mistral exp1_empathetic_s450   /home/mycheung/eq_safety_new/outputs_mistral_empathetic_dialogues/checkpoint-450      mistral
mistral exp1_esconv_s450       /data/mycheung/eq_safety_new/outputs_mistral_exp1_esconv/checkpoint-450               mistral
mistral exp1_lahnala_s450      /data/mycheung/eq_safety_new/outputs_mistral_exp1_lahnala/checkpoint-450              mistral
mistral exp1_personafuse_s450  /home/mycheung/eq_safety_new/outputs_mistral/checkpoint-450                           mistral
smollm  exp1_empathetic_s450   /home/mycheung/eq_safety_new/outputs_smollm_empathetic_dialogues/checkpoint-450       auto
smollm  exp1_esconv_s450       /data/mycheung/eq_safety_new/outputs_smollm_exp1_esconv/checkpoint-450                auto
smollm  exp1_lahnala_s450      /data/mycheung/eq_safety_new/outputs_smollm_exp1_lahnala/checkpoint-450               auto
smollm  exp1_personafuse_s450  /home/mycheung/eq_safety_new/outputs_smollm_filtered/checkpoint-450                   auto
# Exp. 2 (ShareGPT, seed 3407): FILL the checkpoint paths used in the submission.
llama   sg_warm_s3407          FILL   chatml
llama   sg_useronly_s3407      FILL   chatml
llama   sg_ours_s3407          FILL   chatml
qwen    sg_warm_s3407          FILL   chatml
qwen    sg_useronly_s3407      FILL   chatml
qwen    sg_ours_s3407          FILL   chatml
mistral sg_warm_s3407          FILL   mistral
mistral sg_useronly_s3407      FILL   mistral
mistral sg_ours_s3407          FILL   mistral
smollm  sg_warm_s3407          FILL   auto
smollm  sg_useronly_s3407      FILL   auto
smollm  sg_ours_s3407          FILL   auto
EOF
  echo "wrote $MAP: fill the 12 Exp. 2 paths, then run: bash revision/queue_round5.sh tmplcheck"
  ;;

tmplcheck)
  # The template column must be the template the checkpoint was TRAINED with. The saved tokenizer
  # usually records it; compare by eye (ChatML shows <|im_start|>, Mistral shows [INST]).
  grep -v '^#' "$MAP" | while read -r m run ck tmpl; do
    [ -z "${m:-}" ] && continue
    if [ ! -d "$ck" ]; then echo "MISSING  $m $run $ck"; continue; fi
    saved=$(python -c "import json,sys;t=json.load(open(sys.argv[1])).get('chat_template','');print(('<|im_start|>' in t and 'chatml') or ('[INST]' in t and 'mistral') or ('none' if not t else 'other'))" "$ck/tokenizer_config.json" 2>/dev/null || echo "no tokenizer_config")
    echo "$m $run  map=$tmpl  saved=$saved"
  done
  echo "== For smollm, 'auto' uses the saved template; it is correct if 'saved' is chatml (SmolLM3 native)."
  ;;

eval)
  OUT=eval_round5.txt; : > "$OUT"
  core="--task evalsets/qi300.jsonl=OUT/qi300.jsonl:256 --task evalsets/redteam265.jsonl=OUT/redteam265.jsonl:512"
  orb="--task evalsets/xstest.jsonl=OUT/xstest.jsonl:256 --task evalsets/orbench_hard500.jsonl=OUT/orbench.jsonl:128 --task evalsets/mc_heldout100.jsonl=OUT/mc_heldout.jsonl:512"
  grep -v '^#' "$MAP" | while read -r m run ck tmpl; do
    [ -z "${m:-}" ] && continue
    [ "$ck" = FILL ] && { echo "# skipped $m $run: path not filled" >&2; continue; }
    out="responses/$m/$run"
    tasks="$core"; checks="[ -s $out/qi300.jsonl ] && [ -s $out/redteam265.jsonl ]"
    case "$run" in exp1_*)   # Exp. 1: judged helpfulness/warmth shows whether a corpus left a working assistant
      tasks="$tasks --task evalsets/mc_heldout100.jsonl=OUT/mc_heldout.jsonl:512"; checks="$checks && [ -s $out/mc_heldout.jsonl ]";; esac
    case "$run" in sg_warm_*|sg_ours_*|sg_useronly_*)   # rebuttal (8Dvg W1): over-refusal for Exp. 2 too, + judged warmth (same instrument as Exp. 3)
      tasks="$tasks $orb"; checks="$checks && [ -s $out/xstest.jsonl ] && [ -s $out/orbench.jsonl ] && [ -s $out/mc_heldout.jsonl ]";; esac
    echo "( $checks ) || python eval/inference_chat.py --model $ck --condition $run --chat_template $tmpl --batch-size 48 ${tasks//OUT/$out}" >> "$OUT"
  done
  # Over-refusal jobs are slower; the Exp. 1 core jobs come first in the file.
  echo "wrote $OUT ($(grep -c . $OUT) jobs). Run on the pod, one queue per GPU (tmux):"
  echo "  python revision/jobqueue.py --gpu 0 --jobs $OUT   and   python revision/jobqueue.py --gpu 1 --jobs $OUT"
  echo "Then: python revision/degen_check.py --min 3   (no row above 3% before judging)"
  ;;

judge)
  : "${JUDGE_MODEL:?set JUDGE_MODEL}"; : "${JUDGE_URL:?set JUDGE_URL}"; : "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
  # 1) Priority runs first (Llama base cut + Exp. 2: smaller and rebuttal-critical).
  python revision/llm_judge.py --model "$JUDGE_MODEL" --base_url "$JUDGE_URL" --runs base_alpaca \
      sg_warm_s3407 sg_useronly_s3407 sg_ours_s3407
  # 2) Everything else not yet judged (Exp. 1, round-4 outputs). Run WITHOUT --runs so that
  #    results/judge_summary.csv covers every run (the cache makes step-1 items free).
  python revision/llm_judge.py --model "$JUDGE_MODEL" --base_url "$JUDGE_URL"
  echo "Send back results/judge_summary.csv and results/judge_perprompt.csv"
  ;;
esac
