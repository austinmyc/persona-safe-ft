# Round 4 TODO (12 Oct 2026). Deadline: ARR, Tue 13 Oct 20:00 HKT. Internal: Tue 16:00

Status key: [ ] open · [x] done. Owner: **S** = server operator, **A** = authors, **C** = Claude (manuscript).
Commands assume `git pull origin revision` first. Run everything inside tmux.

## Why this round exists

1. **`warm_framework` was never trained.** We added it on 10 Oct (commit 03068b9), after the main training queue had started. The round-3 queue only evaluated it "if trained", and nobody queued its data or training.
   - **It is the deciding run** for whether the framework variant becomes the paper's method. If the framework assistant prompt alone (original user, no low-A persona) is as safe as low-A + framework, the persona claim fails.
   - `ours_framework` with seeds 1–2 was also never queued.
2. **L20 generations are degenerate for Llama/Mistral.**
   - Affected: red-teaming 3–17% of replies, held-out 10–30%, Mistral base MT-Bench about 76% (`revision/degen_check.py`).
   - Pod generations are clean.
   - Affected outputs must be regenerated on a machine that passes the diagnostic (step 0).

## Server queue, in order (S)

| # | Step | Command | Machine | Est. time |
|---|---|---|---|---|
| 0 | [ ] Diagnose degeneration (batch 48 vs 1, pip versions) | `bash revision/queue_round4.sh diag` | L20 | 10 min |
| 1 | [ ] Rewrite data for `warm_framework` (1,000 GPT-4o calls; needs the **OpenAI** key, not MiniMax) | `bash revision/queue_round4.sh data` | any | 15 min |
| 2 | [ ] Train `warm_framework` s3407 + `ours_framework` s1, s2 (12 runs) | `bash revision/queue_round4.sh train`, then 2 × `jobqueue.py` (printed) | L20 | ~2–3 h |
| 3 | [ ] Move degenerate L20 outputs aside (skip if step 0 says they came from a clean machine) | `bash revision/queue_round4.sh regen` | where `responses/` lives | 1 min |
| 4 | [ ] Queue all evaluations: regenerations + Warm/Ours cost side + framework + Mistral base + seed 3407 | `bash revision/queue_round4.sh eval`, then 2 × `jobqueue.py` (printed). **Re-run after step 2 finishes** to add the new framework runs | pod, or the L20 if step 0 is clean | ~2 h |
| 5 | [ ] Check generations | `python revision/degen_check.py --min 3`: no Llama/Mistral rows above 3% | — | 1 min |
| 6 | [ ] Judge everything new. Copy the 11 Oct `judge_cache.jsonl` into `results/` first | `python revision/llm_judge.py --model $JUDGE_MODEL --base_url $JUDGE_URL` | any (MiniMax key) | ~2–3 h |
| 7 | [ ] Re-judge the ShareGPT (old Exp. 2) responses, including old-harness Base | `bash revision/judge_sharegpt.sh sharegpt_files.txt` (format in the script header) | any | ~2 h, can run in parallel with 2–4 |
| 8 | [ ] Send back | `results/judge_summary.csv`, `results/judge_perprompt.csv`, `results/summary.csv`, `results_sharegpt/*.csv`, the step-0 output and `pip_freeze_*.txt` from both machines | — | — |

Two API keys are involved, and both read the `OPENAI_API_KEY` variable:
- **Step 1** uses OpenAI (GPT-4o) with no base URL.
- **Steps 6–7** use MiniMax, with `--base_url`.

Set the variable per command, e.g. `OPENAI_API_KEY=$OPENAI_KEY bash revision/queue_round4.sh data`.

**Priority if time runs short:** 1 → 2 (`warm_framework` first) → 4 (framework, Warm/Ours cost side) → 6 → 7 → regenerations.

## Decision after the results arrive (A + C)

**Make low-A + framework the method ("Ours"), with the current prompt reported as "Ours, simple prompt", if all three hold:**
- `warm_framework` is clearly less safe than `ours_framework` (near Warm FT);
- `ours_framework` holds about 8% harmful across seeds 1, 2 and 3407;
- its XSTest/OR-Bench over-refusal is not worse than Ours.

**Otherwise keep the current method**, and report the framework as a refusal-style result:
- If `warm_framework` is about as safe: its safety comes from the assistant side.
- If its over-refusal or warmth cost is large: present it as a trade-off.

## Author items (A)

- [ ] Human judge validation: N, whether labelling was blind, agreement, Cohen's κ (`python revision/llm_judge.py --agreement results/human_sample.csv`)
- [ ] Number of safety-mixing demonstrations (check `data/mc/mc_warm_mix.jsonl` minus 1,000)
- [ ] Pilot table: the ten PersonaFuse subsets' jailbreak rates (App. I), and whether the layer-10 analysis entered the selection
- [ ] Current `custom.bib` (for the bib audit)
- [ ] Construct check (optional, about 400 judge calls): facet ratings of original vs low-A vs facet-prompt user turns

## Manuscript (C), as results arrive

- [ ] Revision notes (point-by-point response to AC, buud, 8Dvg, prLP)
- [ ] Qualitative examples, licences, checklist answers
- [ ] Fill §5.2 red-teaming, §5.4 framework ablation, §5.5 costs, §5.6 ShareGPT, Table 4 seed-3407 rows
- [ ] If the framework becomes the method: rename conditions, move the §5.4 result into §5.2, and update Fig. 1, abstract and intro
