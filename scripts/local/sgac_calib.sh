#!/bin/bash
# E0-calibrated SGAC selector track on the single-GPU workstation (paper/sgac_calib_protocol.md section 9).
# Run inside tmux with the env active, ONE STAGE AT A TIME, in this order; each stage starts only after the previous
# gate has been decided (the gate commands below stop the queue with exit 3 if a gate fails):
#   tmux new -s calib ; conda activate rlvr_v2
#   STAGES=t0 scripts/local/sgac_calib.sh          # spec/manifest/feature checks + GPU smoke + timing projection
#   STAGES=audit scripts/local/sgac_calib.sh       # Stage A: original Phase-1 candidates (E0 + historical settings)
#   STAGES=pilot scripts/local/sgac_calib.sh       # base dev eval, 16 x 2 C1 runs + pi1 x 2, gate G-P
#   STAGES=expansion scripts/local/sgac_calib.sh   # only if G-P passed: 32 x 2 C1 runs, labels
#   STAGES=selector scripts/local/sgac_calib.sh    # CPU: Stage D, gate G-S, freeze + pin the selector
#   STAGES=curriculum scripts/local/sgac_calib.sh  # only if G-S passed: base eval, all arms, analysis
#   STAGES=tm scripts/local/sgac_calib.sh          # exploratory time-matched Random
# CURRENT PLAN (amendment A6): the equal-compute comparison only, about 1-1.5 GPU-days:
#   STAGES=compute_budget scripts/local/sgac_calib.sh
# WITHDRAWN by A6 (never run): the selector-calibration queue below.
# After G-P failed on the C1 pilot (protocol section 13), the queue was (one command, ~17 GPU-days):
#   STAGES=os_pilot,os_expansion,os_selector,os_curriculum,c2 scripts/local/sgac_calib.sh
#   os_pilot      EXPLORATORY one-shot regime (prereg Study-1 trainer) on the 16 pilot candidates x 2 seeds + pi1 x 2
#   os_expansion  only if the OS pilot meets the G-P thresholds: 32 more candidates x 2 seeds
#   os_selector   CPU: Stage D on the OS labels; the selector is frozen whatever G-S says (recorded)
#   os_curriculum the Stage-E design with that selector (8 seeds learned/random/SGAC, 3 seeds heuristics)
#   c2            the protocol's pre-declared escalation rung: C1 with 80 bursts, 16 x 2 + pi1 x 2, gate G-P
# If require-os-gate stops the queue, continue with STAGES=os_selector,os_curriculum,c2 (pre-declared fallback:
# the equation is then fitted on the 16 OS pilot candidates).
# Every job is resumable and finished jobs return immediately: re-launch the same command after any interruption.
# The prereg K=32 sieve must be at results/pool/{rollouts,signals}.jsonl (verified by `check-features`).
source "$(dirname "$0")/common.sh"
LOG_DIR="$REPO_ROOT/results_sgac_calib/logs"
mkdir -p "$LOG_DIR"

STAGES="${STAGES:-t0}"
if [[ "${FORCE:-0}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
  if [[ "$busy" -gt 0 ]]; then
    log "another process is using the GPU (nvidia-smi lists $busy compute apps); refusing to start (FORCE=1 overrides)"
    exit 2
  fi
fi

# Commands whose failure must stop the queue (gates and pipeline-validity checks).
is_gate() {
  case "$1" in
    show-spec*|"build-manifests --check"*|check-features*|base-dev-eval*|gate-p*|require-gate-p*|stage-d*|\
freeze-selector*|require-gate-s*|curriculum-base-eval*|require-os-gate*|explore-curriculum-base-eval*|smoke-rungs*|cb-schedules*|cb-smoke*|cb-base-eval*) return 0 ;;
    *) return 1 ;;
  esac
}

failed=()
IFS=',' read -ra stage_list <<< "$STAGES"
for stage in "${stage_list[@]}"; do
  log "calibration stage $stage"
  mapfile -t jobs < <(python -m rlvr_v2.sgac_calib jobs --stage "$stage")
  for job in "${jobs[@]}"; do
    name="calib_${stage}_$(echo "$job" | tr ' -=/' '____' | tr -s '_' | cut -c1-80)"
    if [[ "$job" == SGAC\ * ]]; then
      cmd=(python -m rlvr_v2.sgac ${job#SGAC })
    else
      # shellcheck disable=SC2206
      cmd=(python -m rlvr_v2.sgac_calib $job)
    fi
    if run_logged "$name" "${cmd[@]}"; then
      log "done: $job"
    else
      rc=$?
      log "FAILED (exit $rc): $job"
      failed+=("$stage: $job")
      if is_gate "$job"; then
        log "gate or validity check failed: stopping the queue (decide per paper/sgac_calib_protocol.md before re-running)"
        exit 3
      fi
    fi
  done
done
log "calibration failed jobs in this pass: ${failed[*]:-none}"
[[ ${#failed[@]} -eq 0 ]]
