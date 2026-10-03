#!/bin/bash
# One-time environment setup on the institute SLURM cluster (run on the login node).
# Usage: bash env/setup_cluster.sh [ENV_DIR]   (default ~/envs/rlvr_v2)
set -euo pipefail
ENV_DIR="${1:-$HOME/envs/rlvr_v2}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

module load anaconda3-2024.2
module load cuda-12.8

python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# CUDA 12.8 torch build (matches the cluster's cuda-12.8 module). Pin to the version that was tested.
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Same library versions as the local lock file (see env/requirements-lock-cpu.txt); torch is excluded there.
pip install \
  "transformers==5.18.0" "trl==1.14.1" "peft==0.21.2" "datasets" "accelerate" \
  "math-verify==0.9.0" "antlr4-python3-runtime==4.13.2" \
  numpy pandas pyarrow scipy scikit-learn pyyaml tqdm pytest ruff matplotlib
# Optional (4-bit fallback only): pip install bitsandbytes
# Optional, time-boxed (30 min max): pip install vllm   # must match the torch build; if it fails, keep backend=hf

pip install -e "$REPO_DIR"

# Pre-fetch models/datasets on the login node so compute nodes can run with HF_HUB_OFFLINE=1.
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
python - <<'PY'
from huggingface_hub import snapshot_download
from datasets import load_dataset
snapshot_download("Qwen/Qwen2.5-Math-1.5B")
load_dataset("nlile/hendrycks-MATH-benchmark", split="train")
load_dataset("HuggingFaceH4/MATH-500", split="test")
print("prefetch ok")
PY

python -c "import torch, transformers, trl, peft, math_verify; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('transformers', transformers.__version__, 'trl', trl.__version__, 'peft', peft.__version__)"
echo "Environment ready at $ENV_DIR. Next: sbatch scripts/slurm/bench.sbatch"
