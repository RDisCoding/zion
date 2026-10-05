#!/bin/bash
# Study 1 on one GPU (equivalent of scripts/slurm/study1_array.sbatch): runs the (candidate, seed) rows of
# manifests/study1_jobs.csv one after another, each in its own Python process (fresh base model + fresh LoRA,
# exactly like one SLURM array task).
#   scripts/local/study1.sh            # all rows
#   scripts/local/study1.sh 0 19       # rows 0..19 only
# Re-running is safe: finished jobs exit at once, interrupted ones resume from their last checkpoint. A failing job
# is recorded and the loop continues (array tasks are independent); the script exits 1 at the end if any failed.
# This takes days: run it inside tmux/screen, or `nohup scripts/local/study1.sh > results/local/study1_nohup.log 2>&1 &`.
# To also score the held-out train slice (pre-registered sign check):
#   EXTRA_ARGS="--heldout" scripts/local/study1.sh        (frozen gate overrides are always applied too)
source "$(dirname "$0")/common.sh"
test -f manifests/study1_jobs.csv || { echo "manifests/study1_jobs.csv missing: run scripts/local/sieve.sh first"; exit 2; }
require_gates
load_frozen_args
N=$(($(wc -l < manifests/study1_jobs.csv) - 1))
FIRST="${1:-0}"; LAST="${2:-$((N - 1))}"
failed=()
for i in $(seq "$FIRST" "$LAST"); do
  log "Study-1 job $i (rows 0..$((N - 1)))"
  if run_logged "study1_job${i}" python -m rlvr_v2.cli study1 --config configs/study1.yaml --job-index "$i" ${EXTRA_ARGS:-}; then
    log "job $i done"
  else
    log "job $i FAILED, continuing with the next one"
    failed+=("$i")
  fi
done
done_n=$(grep -l '"state": "done"' results/study1/*/*/status.json 2>/dev/null | wc -l)
log "Study-1 jobs finished so far: $done_n of $N ; failed in this pass: ${failed[*]:-none}"
[[ ${#failed[@]} -eq 0 ]]
