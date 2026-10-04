#!/bin/bash
# Pool sieve at K=32 (resumable), base MATH-500 eval, deterministic candidate selection, Study-1 job table.
# Equivalent of scripts/slurm/sieve.sbatch. Requires G1 to have pinned the prompt style (results/e0/frozen_args.txt),
# otherwise hours of sieving could run under a prompt the gates later reject.
#   scripts/local/sieve.sh
#   FORCE=1 scripts/local/sieve.sh      # sieve under the default prompt style without G1 (development only)
source "$(dirname "$0")/common.sh"
if [[ ! -f results/e0/frozen_args.txt && "${FORCE:-0}" != 1 ]]; then
  echo "results/e0/frozen_args.txt missing: run GATES=g1,g2,g3,g5 scripts/local/e0_gates.sh first (or FORCE=1)"; exit 2
fi
load_frozen_args
run_logged sieve python -m rlvr_v2.cli sieve --config configs/study1.yaml --out results/pool ${EXTRA_ARGS:-}
run_logged base_eval python -m rlvr_v2.cli eval --config configs/study1.yaml --tag base --out results/study1_base ${EXTRA_ARGS:-}
run_logged select_candidates python scripts/select_candidates.py --signals results/pool/signals.jsonl \
  --pool-manifest manifests/pool.json --config configs/study1.yaml --out manifests/study1_candidates.json
run_logged study1_jobs python -m rlvr_v2.cli study1-jobs --config configs/study1.yaml --out manifests/study1_jobs.csv
log "candidates: manifests/study1_candidates.json ; jobs: manifests/study1_jobs.csv ($(($(wc -l < manifests/study1_jobs.csv) - 1)) rows)"
