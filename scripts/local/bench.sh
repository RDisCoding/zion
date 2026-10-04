#!/bin/bash
# Day-1 benchmark on the workstation (equivalent of scripts/slurm/bench.sbatch, plus a training probe):
#   1. batched generation speed and memory at several batch sizes  -> results/bench/bench_<host>.json
#   2. three GRPO rounds on pi1 at the full Study-1 config          -> results/bench/train_probe_<stamp>/train_probe.json
# The probe answers "does G=64 x 3072 tokens at per_device_train_batch_size=8 fit in this GPU, and how many
# seconds is one round" before any multi-hour run is started.
#   scripts/local/bench.sh
#   BATCH_SIZES=1,8,32,64,128 scripts/local/bench.sh
#   PROBE_STEPS=0 scripts/local/bench.sh                      # generation benchmark only
#   EXTRA_ARGS="--override train.per_device_train_batch_size=4" scripts/local/bench.sh   # probe a smaller batch
source "$(dirname "$0")/common.sh"
run_logged bench python scripts/bench_throughput.py --config configs/base.yaml --n-prompts 64 \
  --batch-sizes "${BATCH_SIZES:-1,8,32,64}" --max-new-tokens 512 --out "results/bench/bench_$(hostname).json"
if [[ "${PROBE_STEPS:-3}" -gt 0 ]]; then
  run_logged train_probe python -m rlvr_v2.cli train-probe --config configs/study1.yaml --steps "${PROBE_STEPS:-3}" \
    --out "results/bench/train_probe_${STAMP}" ${EXTRA_ARGS:-}
fi
log "done: results/bench/bench_$(hostname).json  results/bench/train_probe_${STAMP}/train_probe.json"
