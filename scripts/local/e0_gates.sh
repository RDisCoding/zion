#!/bin/bash
# E0 gates on the workstation (equivalent of scripts/slurm/e0_gates.sbatch).
#   GATES=g1,g2,g3,g5 scripts/local/e0_gates.sh    # Day 1: cheap gates (base evals, grader, stop tokens, determinism)
#   GATES=g4 scripts/local/e0_gates.sh             # positive control on pi1, up to 3 budget rungs (hours)
#   scripts/local/e0_gates.sh                      # all five, pre-registered order
# Writes results/e0/gates.json (cumulative) and results/e0/frozen_args.txt (pinned prompt style, G4 budget).
# Pass EXTRA_ARGS for memory knobs only, e.g. EXTRA_ARGS="--override train.per_device_train_batch_size=4".
source "$(dirname "$0")/common.sh"
GATES="${GATES:-g1,g2,g3,g5,g4}"
# The committed manifests are the frozen splits: this only verifies them (it refuses to overwrite a different cut).
run_logged manifests python -m rlvr_v2.cli manifests --config configs/study1.yaml
run_logged "e0_${GATES//,/-}" python -m rlvr_v2.cli gates --config configs/e0.yaml --gates "$GATES" \
  --out results/e0/gates.json ${EXTRA_ARGS:-}
python -c "import json; r=json.load(open('results/e0/gates.json')); print({g: v.get('passed') for g, v in r['gates'].items()}, 'all_passed =', r.get('all_passed'))"
log "report: results/e0/gates.json ; frozen overrides: $(cat results/e0/frozen_args.txt 2>/dev/null || echo '<none yet>')"
