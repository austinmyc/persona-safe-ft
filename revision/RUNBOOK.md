# Revision runbook (tonight → Friday)

Everything here uses **your** pipeline: same `train.py`, same API call format, same Unsloth 4-bit inference. New runs are therefore comparable with the paper's checkpoints.

## What the repo told us (affects the paper)

1. **Loss covers the whole conversation, user turns included.** `train.py` packs the full chat text with no assistant-only masking. Models trained on low-A data also learn to *write* low-A user turns. Keep it this way for consistency, state it in §3 / Appendix A.2, and use it when interpreting persona transfer.
2. **Every paper run used seed 3407.** The hyperparameter table's "Seeds: 3" is wrong for the submitted results. `train.py` now takes `--seed` (default 3407 = unchanged). Extra seeds: 1 and 2.
3. **The ShareGPT source deliberately includes a "refusal" query type** (`prepare_datasets.py` balances 5 types, one defined by refusal phrases in the original GPT reply). The ShareGPT training data therefore contains refusals, which User-only keeps verbatim. This is another reason MentalChat should be the main experiment. Report training-refusal counts per condition.
4. **Rewrites used default temperature (1.0)** and the alias `gpt-4o`. The new rewrites do the same. Note that the `gpt-4o` alias may now point to a newer snapshot than in May; record the snapshot the API returns in the paper.
5. **`rewrite_user.py --bert-verify`** retries until a BERT personality classifier says Agreeableness dropped. If the MC Ours data was built with this flag, the new user-side conditions must use the same verification, or none. Tell me which you used.

## 0. Setup (10 min)
```bash
cd persona-safe-ft
git apply revision.patch          # adds the files below; train.py gets --seed
git add -A && git commit -m "revision tooling + frozen revision prompts"
git tag prompts-frozen-2026-10-08
pip install -r requirements.txt   # if the env is fresh
```
New files: `pipeline/rewrite_conditions.py`, `configs/prompts_revision.yaml`, `eval/inference_chat.py`, `revision/{jobqueue.py, make_jobs.py, judge.py, stats.py, RUNBOOK.md}`.

## 1. Get the MentalChat data

The rewritten MC data isn't in the repo (`.gitignore` excludes it). `pipeline/prepare_mentalchat.py` downloads MentalChat-16K from Hugging Face (`ShenLab/MentalChat16K`) and rebuilds the source.

**(A) Preferred: recover the exact 1,000 examples the paper used.** This needs the Warm FT and Ours MC *training files* from the L20 server. Copy them to `data/mc/mc_warm.jsonl` and `data/mc/mc_ours.jsonl`.
```bash
python pipeline/prepare_mentalchat.py --recover_from data/mc/mc_warm.jsonl \
    --ours_jsonl data/mc/mc_ours.jsonl --out data/mc/mc_ours.parquet
```
- The script matches every Warm FT user turn back to its MentalChat row (Warm FT keeps the original user turns), and attaches the paper's low-A user turns from the Ours file.
- It stops if more than 2% don't match, or if the two files have different lengths.
- It also writes:
  - `evalsets/mc_heldout100.jsonl`: 100 MentalChat prompts *not* used in training, for warmth / G-Eval;
  - a manifest pinning the dataset commit.

**(B) Only if the paper's MC files are lost: sample fresh and rebuild Warm FT and Ours too.** Every condition must share one source, so Warm FT and Ours get retrained on the new sample (4 + 4 extra runs):
```bash
python pipeline/prepare_mentalchat.py --sample 1000 --out data/mc/mc_source.parquet
python pipeline/rewrite_user.py --input data/mc/mc_source.parquet --output data/mc/mc_ours.parquet   # add --bert-verify if the paper used it
python pipeline/rewrite_conditions.py --cond warm --source data/mc/mc_ours.parquet --out data/mc/mc_warm.jsonl
python pipeline/rewrite_conditions.py --cond ours --source data/mc/mc_ours.parquet --out data/mc/mc_ours.jsonl
python revision/make_jobs.py train --conds warm ours >> train_jobs.txt
```
In (B), the paper's existing MentalChat numbers get replaced by the new runs. Say so in the revision notes.

From here on, every command uses `data/mc/mc_ours.parquet` as `--source`.

## 2. Start training now (no API needed)
```bash
python pipeline/rewrite_conditions.py --cond user_only --source data/mc/mc_ours.parquet --out data/mc/mc_user_only.jsonl
head -c 600 data/mc/mc_user_only.jsonl; wc -l data/mc/mc_user_only.jsonl     # blunt user turn + original reply; 1000 lines

python revision/make_jobs.py train --conds user_only >> train_jobs.txt
python revision/make_jobs.py train --conds warm ours --seeds 1 2 >> train_jobs.txt   # extra seeds, data already exists

tmux new -d -s gpu0 'python revision/jobqueue.py --gpu 0 --jobs train_jobs.txt'
tmux new -d -s gpu1 'python revision/jobqueue.py --gpu 1 --jobs train_jobs.txt'
```
The first Llama job's minutes appear in `train_jobs.txt.state`. Send them to me.

## 3. Rewrites (API, ~1 h, runs alongside training)
Smoke test first, then read 5 of each:
```bash
export OPENAI_API_KEY=...
for c in placebo high_a ours_noclause warm_clause; do
  python pipeline/rewrite_conditions.py --cond $c --source data/mc/mc_ours.parquet --out smoke/mc_$c.jsonl --limit 20
done
```
What to check:
- **placebo** is not blunter than the original;
- **high_a** is clearly deferential;
- content is kept;
- **ours_noclause** replies still read like the Ours ones;
- **warm_clause** replies still read like Warm FT.

Then run the full rewrites and queue training as each finishes:
```bash
for c in placebo ours_noclause high_a warm_clause; do
  (python pipeline/rewrite_conditions.py --cond $c --source data/mc/mc_ours.parquet --out data/mc/mc_$c.jsonl \
   && python revision/make_jobs.py train --conds $c >> train_jobs.txt) &
done; wait
```
Remaining conditions, once those four are queued:
- `low_e`, `ours_v3` (Llama + SmolLM: add `--models llama smollm`);
- `ours_framework`;
- safety mixing (§5).

## 4. Evaluation
`make_jobs.py eval` writes `eval/inference.py` lines for Qi + red-teaming on the **last** checkpoint (= final-epoch rule). Score with `eval/eval_jailbreak.py` / `eval/eval_redteam.py` as before.

New sets (XSTest, OR-Bench, MT-Bench, SafeMT, emotional-frame Qi) go through `eval/inference_chat.py` with JSONL prompt files.

Run all evaluation on the rented 4090s, and regenerate the base models and the existing Warm/Ours MC checkpoints there too, so every number comes from one GPU type.

## 5. Safety-mixing data
Take ~30 refusal demonstrations from the Bianchi et al. safety-tuned-llamas release. Put them in the `pairs` parquet format and rewrite them:
- for **warm+mix**: `--cond warm`, then append to `mc_warm.jsonl`;
- for **ours+mix**: `--cond ours`, then append to `mc_ours.jsonl`.

Save the results as `data/mc/mc_warm_mix.jsonl` and `data/mc/mc_ours_mix.jsonl`, then `make_jobs.py train --conds warm_mix ours_mix`.
