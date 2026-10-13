# Optional last run (13 Oct): "Warm + framework"

**Why.** The framework variant has only been trained with low-agreeableness user turns. Without the same assistant prompt on the *original* user turns, a reviewer can say the framework prompt alone gives the explicit limits and the low harm. The paper (v7.15) currently states this as a limitation.

**Cost.**
- Data: 1,000 GPT-4o calls (OpenAI key), about 15 min.
- Training: about 1 h per model on an L20.
- Evaluation: about 15 min.
- Judging: about 665 calls per model, about 10 min.
- Total for Qwen + SmolLM on two GPUs: about 1 h 45 min. Start by 11:30 to have numbers before 14:00.

```bash
git pull origin revision
OPENAI_API_KEY=$OPENAI_KEY bash revision/queue_wf.sh data   # check that the printed USER turn is the original message
bash revision/queue_wf.sh train                             # MODELS="llama qwen mistral smollm" if GPUs allow
bash revision/queue_wf.sh eval
python revision/degen_check.py --min 3 | grep warm_framework
bash revision/queue_wf.sh judge
```

**Send back:** `results_wf/judge_summary.csv`, `results_wf/judge_perprompt.csv`.

## Decision rule (fixed before results)

**A. Within 3 points of Ours + framework on harmful instructions.** Paper text: "The framework prompt gives explicit limits and low harm with or without the persona (Table X); the persona's contribution is established under the de-escalating prompt." Drop the limitation sentence.

**B. Clearly higher (McNemar p_Holm < .05).** Paper text: "Under the framework prompt too, the low-agreeableness persona lowers harm (Table X)." This strengthens contribution (ii).

**Otherwise:** report the numbers and call the comparison inconclusive.

In every case, the row goes into the controls table under "Low-A user, assistant side varied" as a separate "Original user, framework" line, with "--" for models not run.
