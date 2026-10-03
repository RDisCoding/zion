#!/bin/bash
# Sourced by every sbatch script: modules, venv, environment variables, repo root.
set -euo pipefail
module load anaconda3-2024.2
module load cuda-12.8
ENV_DIR="${RLVR_ENV:-$HOME/envs/rlvr_v2}"
source "$ENV_DIR/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"        # models/datasets were prefetched by env/setup_cluster.sh
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results/slurm
echo "== $(date -u +%FT%TZ) job=${SLURM_JOB_ID:-local} array=${SLURM_ARRAY_TASK_ID:-} node=$(hostname) =="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('mem_get_info', torch.cuda.mem_get_info() if torch.cuda.is_available() else None)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true
