#!/bin/bash
# SGAC reproduction queue on the single-GPU workstation (paper/sgac_repro_protocol.md section 5).
# Run inside tmux, with the conda env active:
#   tmux new -s sgac
#   conda activate rlvr_v2
#   TIERS=0 scripts/local/sgac_repro.sh            # probe + manifest check + GPU smoke (~45 min); send me the summary
#   TIERS=1,2,3,4,5 scripts/local/sgac_repro.sh    # the core queue (~30 h); safe to re-launch after any interruption
#   TIERS=6 scripts/local/sgac_repro.sh            # optional arms, only if time remains
#
# Tiers run in the pre-declared order and are never reordered on results. Every job is resumable at curriculum-step
# granularity and finished jobs return immediately, so re-launching the same command continues where it stopped.
# A failed job is logged and the queue continues, EXCEPT `base-eval --profile e0`: if the E0 G1 agreement gate fails
# (exit 3) the queue stops before any e0 curriculum runs.
# Does not use require_gates/load_frozen_args: the SGAC track has its own configs (configs/sgac/), and never touches
# the pre-registered study's configs, manifests or results.
source "$(dirname "$0")/common.sh"
LOG_DIR="$REPO_ROOT/results_sgac/logs"
mkdir -p "$LOG_DIR"

TIERS="${TIERS:-0}"
if [[ "${FORCE:-0}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
  if [[ "$busy" -gt 0 ]]; then
    log "another process is using the GPU (nvidia-smi lists $busy compute apps); refusing to start (FORCE=1 overrides)"
    exit 2
  fi
fi

failed=()
IFS=',' read -ra tier_list <<< "$TIERS"
for tier in "${tier_list[@]}"; do
  log "SGAC tier $tier"
  mapfile -t jobs < <(python -m rlvr_v2.sgac jobs --tier "$tier")
  for job in "${jobs[@]}"; do
    name="sgac_t${tier}_$(echo "$job" | tr ' -' '__' | tr -s '_' | cut -c1-80)"
    # shellcheck disable=SC2086
    if run_logged "$name" python -m rlvr_v2.sgac $job; then
      log "done: $job"
    else
      rc=$?
      log "FAILED (exit $rc): $job"
      failed+=("t${tier}: $job")
      if [[ "$job" == "base-eval --profile e0" && "$rc" -eq 3 ]]; then
        log "E0 G1 agreement gate failed: stopping before any e0 curriculum run (see results_sgac/e0/*/_shared/base_eval)"
        exit 3
      fi
    fi
  done
done
python -m rlvr_v2.sgac report --profiles e0,as_run || log "report failed (non-fatal)"
log "SGAC failed jobs in this pass: ${failed[*]:-none}"
[[ ${#failed[@]} -eq 0 ]]
