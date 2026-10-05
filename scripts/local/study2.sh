#!/bin/bash
# Study 2 on one GPU (equivalent of scripts/slurm/study2_array.sbatch): one curriculum run per (arm, seed) of
# configs/study2.yaml, sequentially, each resumable at curriculum-step granularity.
#   scripts/local/study2.sh            # all arms x seeds (3 x 3 = 9 by default)
#   scripts/local/study2.sh 0 2        # jobs 0..2 only
# Add the learned arm only if the Study-1 gate passed:
#   EXTRA_ARGS="--override study2.arms=[random,variance,disagreement,learned]" scripts/local/study2.sh
source "$(dirname "$0")/common.sh"
require_gates
load_frozen_args
N=$(python -m rlvr_v2.cli study2-jobs --config configs/study2.yaml --count ${EXTRA_ARGS:-} 2>/dev/null | tail -1)
FIRST="${1:-0}"; LAST="${2:-$((N - 1))}"
python -m rlvr_v2.cli study2-jobs --config configs/study2.yaml ${EXTRA_ARGS:-} 2>/dev/null
failed=()
for i in $(seq "$FIRST" "$LAST"); do
  log "Study-2 job $i (rows 0..$((N - 1)))"
  if run_logged "study2_job${i}" python -m rlvr_v2.cli curriculum --config configs/study2.yaml --job-index "$i" ${EXTRA_ARGS:-}; then
    log "job $i done"
  else
    log "job $i FAILED, continuing with the next one"
    failed+=("$i")
  fi
done
log "Study-2 failed in this pass: ${failed[*]:-none}"
[[ ${#failed[@]} -eq 0 ]]
