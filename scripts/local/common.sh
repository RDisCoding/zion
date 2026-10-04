#!/bin/bash
# Sourced by every scripts/local/*.sh: single-GPU workstation equivalent of scripts/slurm/common.sh.
# No SLURM and no environment modules. Activate your venv before calling a script, or set RLVR_ENV=/path/to/venv.
# Logs go to results/local/<step>_<utc-stamp>.log (the counterpart of results/slurm/*.out).
set -euo pipefail
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$LOCAL_DIR/../.." && pwd)"
cd "$REPO_ROOT"
if [[ -n "${RLVR_ENV:-}" ]]; then source "$RLVR_ENV/bin/activate"; fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # one GPU, one job at a time
# HF_HUB_OFFLINE is left to you: export HF_HUB_OFFLINE=1 once the model and datasets are cached.

LOG_DIR="$REPO_ROOT/results/local"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

log() { echo "== $(date -u +%FT%TZ) $*"; }

# run_logged NAME CMD... : tee stdout+stderr to results/local/NAME_<stamp>.log and keep the command's exit code.
run_logged() {
  local name="$1"; shift
  log "$name: $*"
  "$@" 2>&1 | tee -a "$LOG_DIR/${name}_${STAMP}.log"
  return "${PIPESTATUS[0]}"
}

# Gate-frozen overrides (pinned prompt style from G1, training budget from G4) unless EXTRA_ARGS was set explicitly.
load_frozen_args() {
  if [[ -z "${EXTRA_ARGS+x}" && -f results/e0/frozen_args.txt ]]; then
    EXTRA_ARGS="$(cat results/e0/frozen_args.txt)"
  fi
  EXTRA_ARGS="${EXTRA_ARGS:-}"
  log "EXTRA_ARGS: ${EXTRA_ARGS:-<none>}"
}

# Same rule the Python CLI enforces, checked once up front so a 40-job loop does not load a model to fail 40 times.
require_gates() {
  python - <<'PY'
import sys
from rlvr_v2.artifacts import REPO_ROOT, read_json
from rlvr_v2.gates import gate_problems
problems = gate_problems(read_json(REPO_ROOT / "results" / "e0" / "gates.json"))
if problems:
    sys.exit("E0 gates do not authorise study jobs: " + "; ".join(problems))
print("E0 gates: all five passed")
PY
}

log "host=$(hostname) repo=$REPO_ROOT python=$(command -v python)"
python -c "import torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', ok, torch.cuda.get_device_name(0) if ok else ''); print('mem_get_info (free, total)', torch.cuda.mem_get_info() if ok else None)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null || true
