#!/bin/bash
#SBATCH --job-name=rlvr2-s2
#SBATCH --partition=gpu
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --array=0-8%2
#SBATCH --output=results/slurm/%x_%A_%a.out
#SBATCH --error=results/slurm/%x_%A_%a.err
# Study 2: one array task per (arm, seed) from configs/study2.yaml (arms x seeds, row-major). Resumable at
# curriculum-step granularity; re-submit on timeout. Add the learned arm only if the Study-1 gate passed:
#   sbatch scripts/slurm/study2_array.sbatch   with EXTRA_ARGS="--override study2.arms=[random,variance,disagreement,learned]"
source scripts/slurm/common.sh
test -f results/e0/gates.json || { echo "E0 gates not run; refusing to start"; exit 2; }
python -m rlvr_v2.cli curriculum --config configs/study2.yaml --job-index "$SLURM_ARRAY_TASK_ID" ${EXTRA_ARGS:-}
