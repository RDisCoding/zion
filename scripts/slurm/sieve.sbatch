#!/bin/bash
#SBATCH --job-name=rlvr2-sieve
#SBATCH --partition=gpu
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err
# Base-policy sieve of the candidate pool at K=32 (resumable), then base evals and candidate selection.
source scripts/slurm/common.sh
# Needs the pinned prompt style from G1:  export EXTRA_ARGS="$(cat results/e0/frozen_args.txt)"  before sbatch.
python -m rlvr_v2.cli pool-ineligible --config configs/study1.yaml --check
python -m rlvr_v2.cli sieve --config configs/study1.yaml --out results/pool ${EXTRA_ARGS:-}
python -m rlvr_v2.cli eval --config configs/study1.yaml --tag base --out results/study1_base ${EXTRA_ARGS:-}
python scripts/select_candidates.py --signals results/pool/signals.jsonl --pool-manifest manifests/pool.json \
  --config configs/study1.yaml --out manifests/study1_candidates.json
python -m rlvr_v2.cli study1-jobs --config configs/study1.yaml --out manifests/study1_jobs.csv
