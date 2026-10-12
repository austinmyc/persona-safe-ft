# Round 5 TODO (12 Oct, night). Internal target: Tue 13 Oct 16:00 HKT

## Why this round exists

**One evaluation protocol for every number in the paper.**
- Experiments 1 and 2 were generated in January and May with the old Princeton harness, which differs from Experiment 3 in several ways:
  - full precision for some runs and 4-bit for others;
  - a different red-teaming token limit;
  - their own base runs. The Experiment 1 Qwen base is 34.0% harmful, while the Experiment 3 Qwen base is 22.3%.
- A reviewer comparing tables will catch this. Round 5 regenerates the Experiment 1 and 2 checkpoints with the Experiment 3 harness:
  - `eval/inference_chat.py`;
  - 4-bit, greedy;
  - 256 new tokens for Qi, 512 for red-teaming;
  - batch size 48;
  - the training chat template.
- All three experiments then share **one** Base evaluation: `responses/<model>/base`, plus `responses/llama/base_alpaca` for Llama.

**Rebuttal promises still open:**
- Over-refusal for Experiment 2 (8Dvg W1): XSTest and OR-Bench under the judge.
- McNemar with CIs and effect sizes for Experiment 2 (prLP W2).
- Gemma-2-2B selection replication, and the full ten-subset ranking (prLP W4, AC). **Authors: send the numbers.**

## Server queue (S), in order. Start P0 and P1 tonight

| P | Step | Command | Est. |
|---|---|---|---|
| 0 | Cut the Llama base (Alpaca) replies at the next turn marker. The server run was uncut, so check that the checkout is at or after commit 65bb21a | `bash revision/queue_round5.sh alpaca_cut` | 1 min |
| 1 | Write the checkpoint map. Exp. 1 paths are pre-filled; **fill the 12 Exp. 2 (ShareGPT) paths** | `bash revision/queue_round5.sh map`, then edit `exp12_checkpoints.txt` | 5 min |
| 1 | Check that each checkpoint's saved template matches the map | `bash revision/queue_round5.sh tmplcheck` | 1 min |
| 1 | Generate: 28 checkpoints × core suite, plus XSTest/OR-Bench for the 12 Exp. 2 runs. Use the pod or a diag-clean machine | `bash revision/queue_round5.sh eval`, then two `jobqueue.py` (printed) | ~3–4 h on 2 GPUs |
| 1 | Degeneration check | `python revision/degen_check.py --min 3` | 1 min |
| 1 | Judge: Llama base and Exp. 2 first, then everything else | `bash revision/queue_round5.sh judge` | ~5–6 h at 62 rpm. Raise `--concurrency` if the API allows |
| 2 | Round 4, still open: Ours/Warm cost side for Exp. 3, framework runs, regenerations | `TODO_round4.md` | — |

**Send back:** `results/judge_summary.csv`, `results/judge_perprompt.csv` and the degen_check output.

**If time runs out:**
- Exp. 2 regenerated + judged > Exp. 1 regenerated > Exp. 2 over-refusal > round-4 framework.
- If Exp. 1 regeneration is not judged by Tue 12:00, we report it from the old replies with a disclosed harness note (already drafted) and say so in the revision notes.

## After the results (C)

```bash
python revision/plot_dotplot.py --exp 23 --summary results/judge_summary.csv --summary2 results/judge_summary.csv \
    --out figures/figure_exp23_judged.pdf           # Llama Base = base_alpaca by default
python revision/exp1_reply_analysis.py <bundle with responses/ and results/>
python revision/stats.py ...                        # Exp. 2 McNemar / CI / h, as for Exp. 3
```
