# Mechanistic analysis, redone (replaces App. F's warmth/compliance cosine)

## Why the old analysis cannot stay

1. **The comply/refuse classes were labelled with the refusal-keyword heuristic.** `probing/build_contrast_sets.py compliance` used the Princeton prefix list. The revision shows this metric is invalid for empathetic models: deflections count as compliance. So the "compliance direction" was partly a "no refusal phrase" direction.
2. **Activations were mean-pooled over all tokens.** That mixes prompt content into the direction. The refusal literature reads the last post-instruction token (Arditi et al., 2024).
3. **Warm FT vs. Ours differed by less than random-direction chance** (prLP W3), so the analysis could not distinguish the methods.

## What replaces it: `probing/refusal_mechanism.py`

The hypothesis is the one in our Discussion: warmth FT turns harmful requests into "counselling topics", while low-A conditioning keeps the model's refusal representation active.

**Directions.** Both come from the **base** model, using the even half of the prompts:
- `r`, the refusal direction (Arditi et al., 2024): mean last-token activation on harmful instructions (Qi) minus that on harmless requests (XSTest safe + MT-Bench turn 1).
- `c`, the counselling direction: held-out MentalChat16K messages minus the same harmless requests.

**Layer selection.** The layer is chosen on the base model only (25–75% depth, held-out half), never on fine-tuned results.

**Measures.** Per checkpoint, on the odd (held-out) half:

| Measure | Meaning | Prediction |
|---|---|---|
| `refusal_sep` | harmful vs. harmless separation along `r` (base = 1) | Warm FT < Ours ≈ base |
| `counsel_sep` | harmful vs. harmless separation along `c` | Warm FT > base: harmful requests read as counselling (→ deflection) |
| `emo_shift` | change along `r` when the same request is wrapped in an emotional disclosure (`qi300_emotional`, paired) | more negative for Warm FT than for Ours |
| `auroc_notharmful` | per-prompt projection on `r` predicts a non-harmful **judged** reply | > permutation 95th pct |
| `auroc_counsel_deflection` | projection on `c` predicts a **judged** deflection | > 0.5 |

**Across checkpoints** (all Exp. 3 runs × 4 models, ~40 points):
- Spearman of `refusal_sep` vs. judged harmful %;
- Spearman of `counsel_sep` vs. judged deflection %.

This ties the mechanism to the paper's main metric, not to keywords.

**Nulls:**
- 200 random unit directions at the same layer: `random_sep_p95_abs_ratio`. This answers prLP W3 directly.
- 200 label permutations for the AUROC.

**Causal test (`steer`).** Each steered run is paired with an unsteered twin made by the same generation code (alpha 0):
- Warm FT s1 with `r` added at the selected layer (alpha = 1 base separation). Harmful replies should fall.
- Ours s1 with `r` ablated at every layer. Harmful replies should rise.

All steered replies are judged by the same MiniMax-M3 judge.

## Server run order

```bash
git pull origin revision
bash revision/queue_probe.sh extract          # ~60 jobs, 3–5 min each; forward passes only
# (judged Exp. 3 results must be in results/judge_perprompt.csv)
bash revision/queue_probe.sh analyze          # CPU, minutes
bash revision/queue_probe.sh steer            # 16 generation jobs (8 steered + 8 unsteered twins)
bash revision/queue_probe.sh judge            # ~4,800 judge calls (16 runs × 300 Qi prompts)
bash revision/queue_probe.sh report
```

**Send back:**
- `probing_out/mechanism_summary.csv`, `mechanism_layers.csv`, `mechanism_correlations.txt`, `mechanism_layers.pdf`;
- `results/judge_summary.csv`, which now includes the `*_steer*` runs.

**Notes:**
- The model is loaded in 4-bit NF4 with the training chat template, as in evaluation. The Llama base is read in the Alpaca format, as in the paper.
- The analysis step was smoke-tested on synthetic activations. The GPU steps have not been run yet.
- If a step fails on one model, the others still run (one job per checkpoint).

## How it enters the paper

- App. F is rewritten around the table and figure.
- §5 replaces "exploratory probing does not distinguish the methods" with one sentence on the result, whichever way it goes.
- **If the predictions fail** (e.g., `refusal_sep` does not track judged harm), we report that honestly as a negative mechanistic result. It is still better than the keyword-based cosine.
