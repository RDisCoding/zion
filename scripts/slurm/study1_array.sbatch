#!/bin/bash
#SBATCH --job-name=rlvr2-s1
#SBATCH --partition=gpu
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --array=0-39%2
#SBATCH --output=results/slurm/%x_%A_%a.out
#SBATCH --error=results/slurm/%x_%A_%a.err
# Study 1: one array task per (candidate, seed) row of manifests/study1_jobs.csv. Resumable: re-submitting
# the same array is safe (completed jobs exit immediately; partial ones resume from the last checkpoint).
# Split across two nodes:  sbatch --nodelist=node1 --array=0-19%1 scripts/slurm/study1_array.sbatch
#                          sbatch --nodelist=node2 --array=20-39%1 scripts/slurm/study1_array.sbatch
source scripts/slurm/common.sh
test -f results/e0/gates.json || { echo "E0 gates not run; refusing to start"; exit 2; }
python -m rlvr_v2.cli study1 --config configs/study1.yaml --job-index "$SLURM_ARRAY_TASK_ID" ${EXTRA_ARGS:-}
