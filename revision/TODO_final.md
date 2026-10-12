# FINAL TODO (13 Oct, 00:00). Internal target 16:00 HKT

**Goal.** Fill the last Experiment-3 gaps the reviewers asked about. Warm FT and Ours have harmful-reply, red-teaming and (tonight) XSTest/OR-Bench results. They have **no** held-out warmth/helpfulness, MT-Bench, rapport or emotional-framing results. Every other condition in those tables already has them.

**Scope.** Seed 1, all four models = 8 checkpoints.
- The other rows of Table 5 and App. E.1 are single-seed on four models, so this keeps the table comparable.
- One model only would leave Warm FT / Ours incomparable with the rest of the table, so it is not an option.

| Suite | What it fills | Reviewer point | Gens / ckpt | Judge calls |
|---|---|---|---|---|
| heldout | Table 5 Warm / Help for Warm FT vs. Ours: our "stays warm" claim | AC (open-ended eval), buud | 100 | 800 |
| mtbench | Table 5 MT-Bench | 8Dvg W1 ("e.g., MT-Bench"), AC | 160 | 640 |
| rapport | App. E.1, multi-turn probe | 8Dvg W3 (we promised to expand the multi-turn eval) | 300 × 3 turns | 2,400 |
| coupling | App. E.1, emotional framing | — (same appendix sentence) | 300 | 2,400 |

## Order (after the current queue frees the GPUs, ~01:10)

```bash
git pull origin revision
bash revision/queue_final.sh check    # skip what already exists
bash revision/queue_final.sh eval     # 8 jobs; two GPUs ~1 h
bash revision/queue_final.sh degen    # redo >= 3% at BATCH=1
bash revision/queue_final.sh judge    # start when the over-refusal judge ends (~03:40); ~1-1.5 h
```

**If time is short:** cut `coupling` first, then `rapport`. Keep `heldout` and `mtbench`; they back the main-text table.

**Send back:** `results_final_s1/judge_summary.csv` and `results_final_s1/judge_perprompt.csv`, plus the `degen` output.

## Already queued tonight (unchanged)

- Alpaca Llama base re-judge (00:10).
- Exp. 3 XSTest + OR-Bench, Warm FT / Ours seeds 1–2 (generation 00:50, judge 03:40).
- Mistral seed-2 check (01:10).

Still needed from the authors: `sharegpt_judge_2026-10-12.zip` (Exp. 2), the Gemma July numbers, and the ten-subset pilot values.
