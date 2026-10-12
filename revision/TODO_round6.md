# Round 6 TODO (12 Oct, night): trait selection held out + mechanism

Three independent blocks. Priority **after round 5**, which makes the evaluation consistent (`TODO_round5.md`).

## A. Trait-selection pilot on Gemma-2-2B (now THE pilot of the paper, §3.1 / App. G)

**What it is.** All ten PersonaFuse trait × polarity subsets, using the pilot's existing subset files (no rewriting).
- **Model.** Gemma-2-2B is fine-tuned on each subset. It appears nowhere else in the paper.
- **Selection set.** Selection uses judged harmful replies on **JailbreakBench JBB-Behaviors** (100 prompts, `evalsets/jbb100.jsonl`), which is not used in any other evaluation.
- **Comparison only.** Qi and red-teaming are also generated, so the table can sit next to the old Llama pilot.
- **Why.** With a held-out model and a held-out benchmark, Llama in the main experiments is no longer circular (prLP W4, AC).

**Model files.** Default `GEMMA=unsloth/gemma-2-2b-it-bnb-4bit`, pre-quantised 4-bit and not gated. Set `GEMMA=unsloth/gemma-2-2b-bnb-4bit` for the pretrained model. The paper's App. G has a `\TBD` for which one was used.

```bash
bash revision/queue_round6.sh gemma_download
bash revision/queue_round6.sh gemma_map        # fill gemma_subsets.txt: all TEN pilot subset files
bash revision/queue_round6.sh gemma_data       # size-matched; Gemma has no system role (folded into the user turn)
bash revision/queue_round6.sh gemma_train      # 10 runs, ~10-15 min each on a 2B model
bash revision/queue_round6.sh gemma_eval       # base + 10 runs; builds evalsets/jbb100.jsonl (needs internet once)
bash revision/queue_round6.sh gemma_layers     # pilot step 1 (§3.1 'Layer sensitivity'): x1.3 per layer on base Gemma, JBB
python revision/degen_check.py --min 3
bash revision/queue_round6.sh gemma_judge      # ~10,300 calls (11 runs x 665 + 26 layer runs x 100) -> results_gemma/
bash revision/queue_round6.sh gemma_report     # layer table; ranking on JBB (selection) + Qi + RT; McNemar low_A vs each, Holm
```

- **If time is short:** judge only `--sets jbb100` first (1,100 calls); Qi and red-teaming can follow.
- **If low A does NOT rank first on JBB:** stop and tell the authors. §3.1 must then be rewritten; do not swap benchmarks after seeing results.

**Send back:** `results_gemma/report.txt` and `results_gemma/judge_perprompt.csv`.

## B. Experiment-3 trait sweep (optional; strengthens the held-out test)

**What it is.** MentalChat16K has no trait labels, so a persona control in Exp. 3 is made by rewriting the user turns, as for Ours (low A), high-A and low-E.
- `configs/prompts_revision.yaml` now has the other seven trait × polarity user prompts: high E; high/low N; high/low O; high/low C.
- They use the same template and intensity as high-A and low-E.
- They were frozen at this commit, before any result.
- Together with the three existing personas, Exp. 3 can then compare all ten personas under the identical pipeline, as a **specificity test with low-A pre-specified**, not a new selection.

```bash
OPENAI_API_KEY=$OPENAI_KEY bash revision/queue_round6.sh traits_data   # 7 x ~2,000 GPT-4o calls
TIER=A bash revision/queue_round6.sh traits_train   # high_e, high_n x 4 models = 8 runs (matches the Gemma comparison)
TIER=B bash revision/queue_round6.sh traits_train   # other 5 personas on Qwen + SmolLM = 10 runs (if GPUs free)
bash revision/queue_round6.sh traits_eval           # core + held-out warmth; re-run as training finishes
python revision/llm_judge.py --model $JUDGE_MODEL --base_url $JUDGE_URL
```

## C. Mechanism, redone

See `revision/PROBING_REDO.md` and `bash revision/queue_probe.sh extract|analyze|steer|judge|report`.

## Suggested GPU plan (two GPUs, from Mon 22:30)

| When | GPU 0 | GPU 1 |
|---|---|---|
| Tonight | round 5 generation (Exp. 1/2 regeneration) | round 5 generation |
| Overnight, after round 5 | Gemma A (train + eval, ~1.5 h) | probing `extract` (~3 h) |
| Tue morning | traits TIER A (8 runs) | probing `steer` (16 jobs) |

The judge runs continuously on CPU. Order: round 5 → Gemma → probing steer → traits.
