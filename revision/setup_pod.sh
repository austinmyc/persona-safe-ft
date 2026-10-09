#!/usr/bin/env bash
# One-shot setup for a rented GPU pod (or any fresh machine). Idempotent: safe to re-run.
#
#   export HF_TOKEN=hf_...            # must have access to meta-llama/Llama-3.1-8B (gated)
#   bash revision/setup_pod.sh        # ~10-20 min, mostly model downloads
#
# Optional: LOCK=revision/requirements-lock.txt  -> install the exact versions frozen on the
# L20 server (recommended: `pip freeze > revision/requirements-lock.txt` there, commit it).
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/workspace/hf}"
mkdir -p "$HF_HOME" logs responses evalsets data/mc
grep -q "HF_HOME" ~/.bashrc 2>/dev/null || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

echo "== GPUs"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "== Python packages"
LOCK="${LOCK:-revision/requirements-lock.txt}"
if [ -f "$LOCK" ]; then
  pip install -q -r "$LOCK"
else
  echo "   (no lock file; installing unpinned requirements.txt — versions may differ from the L20s)"
  pip install -q -r requirements.txt
fi
pip install -q scipy numpy pandas pyarrow
# Base images ship torchaudio built for an older torch. requirements.txt upgrades torch
# and leaves that wheel in place; transformers imports it and crashes. This stack is text-only.
pip uninstall -y -q torchaudio >/dev/null 2>&1 || true
python - <<'EOF'
import torch, transformers, trl, peft
try:
    import unsloth; uv = unsloth.__version__
except Exception as e:
    uv = f"IMPORT FAILED: {e}"
print(f"   torch {torch.__version__} cuda {torch.version.cuda} | transformers {transformers.__version__} "
      f"| trl {trl.__version__} | peft {peft.__version__} | unsloth {uv}")
EOF

echo "== Hugging Face login"
if [ -n "${HF_TOKEN:-}" ]; then
  python -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
fi
python -c "from huggingface_hub import whoami; print('   logged in as', whoami()['name'])"

echo "== Pre-download base models (parallel)"
python - <<'EOF'
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
MODELS = ["meta-llama/Llama-3.1-8B", "Qwen/Qwen2.5-7B-Instruct",
          "mistralai/Mistral-7B-Instruct-v0.3", "HuggingFaceTB/SmolLM3-3B"]
def get(m):
    p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*"])
    return m, p
with ThreadPoolExecutor(4) as ex:
    for m, p in ex.map(get, MODELS):
        print(f"   {m} -> {p}")
EOF

echo "== Done. Next: python revision/prepare_evalsets.py ; bash revision/smoke_test.sh"
