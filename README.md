## Code for *Low-Agreeableness Persona Conditioning for Safe LLM Fine-Tuning*.

This repository contains the model-agnostic pipeline for synthesizing/rewriting
personality-conditioned chat data, the fine-tuning script, the safety/warmth evaluators,
and the representational probing used for the decoupling analysis.

## Install

```bash
pip install -r requirements.txt
```

GPT-4o rewriting (`pipeline/`) needs `OPENAI_API_KEY`. Training and probing assume a CUDA GPU.

## Layout

```
pipeline/    Data construction: ShareGPT extraction, NSFW filtering, GPT-4o rewriting
configs/     prompts.yaml (rewrite prompts, Appendix H) and train_config.yaml
train.py     LoRA SFT for all four backbones
eval/        inference.py (generate responses) + jailbreak / red-team / warmth scorers
probing/     Warmth–compliance direction-cosine probing
```

Training corpora can be rebuilt with `pipeline/` from the public
source datasets.

## Build training data

```bash
# Extract + balance ShareGPT pairs (optionally NSFW-filter first via quality_label.py)
python pipeline/prepare_datasets.py --output data/train

# Low-agreeableness user rewrite + warm, de-escalating assistant rewrite (full paired condition)
python pipeline/rewrite_user.py      --input data/train/sharegpt_sample.parquet \
                                     --output data/train/sample_lowA.parquet
python pipeline/rewrite_assistant.py --input data/train/sample_lowA.parquet \
                                     --warm-with-context --jsonl data/train/ours.jsonl

# Generic warmth baseline (assistant-only rewrite, Ibrahim et al.)
python pipeline/rewrite_warmth_baseline.py --input data/train/sharegpt_sample.parquet \
                                           --jsonl data/train/warm_baseline.jsonl
```

Rewrite prompts live in `configs/prompts.yaml`.

## Fine-tune

```bash
python train.py --model meta-llama/Llama-3.1-8B --data_path data/train/ours.jsonl \
                --output_dir outputs_llama_ours --chat_template chatml
```

## Evaluate

The evaluation prompts are harmful by design and are not redistributed here; download
them from the sources below into a local directory (one prompt per line in
`category_*.csv` files, or a single CSV with a `Prompt` column):

- **Jailbreak** — 300 harmful instructions, 10 categories (Qi et al., 2024):
  https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety
- **Red-teaming** — 100 baseline + 165 attack-style prompts across six subsets
  (Leskoschek et al., 2023): https://github.com/lguibr/LLM-red-teaming-prompts

The scorers read a `responses.jsonl` of model generations, so first run a checkpoint over
the prompts:

```bash
python eval/inference.py --model outputs_llama_ours/checkpoint-XXX \
       --prompts <prompt_dir> --condition ours \
       --max-new-tokens 256 --output responses.jsonl

python eval/eval_jailbreak.py --input responses.jsonl --output jailbreak_results.json
python eval/eval_redteam.py   --input responses.jsonl --output redteam_results.json
python eval/eval_warmth.py    --input responses.jsonl --output warmth_scores.json
```

## Probing 

```bash
# 1. Build warm/neutral/comply/refuse contrast sets
python probing/build_contrast_sets.py ...

# 2. Per-layer |cos(v_warm, v_comply)| for base / warm-SFT / ours
python probing/extract_activations.py --base-model meta-llama/Llama-3.1-8B \
       --adapter-warm-sft <warm_ckpt> --adapter-ours <ours_ckpt> \
       --contrast-dir <contrast_dir> --out-dir probe_out --assistant-span

# 3. Aggregate backbones into the paper table (trim + middle-50% trimmed mean, Appendix B.3)
python probing/compute_direction_cosine.py --manifest backbones.json
```

```
