#!/bin/bash
#SBATCH --job-name=rlvr2-bench
#SBATCH --partition=gpu
#SBATCH --gres=shard:10
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err
# Day-1 throughput benchmark: batched generation speed, memory, extrapolated eval/round times.
# Submit with:  sbatch --nodelist=node1 scripts/slurm/bench.sbatch   (or node2)
source scripts/slurm/common.sh
python scripts/bench_throughput.py --config configs/base.yaml --n-prompts 64 --batch-sizes 1,8,32,64 \
  --max-new-tokens 512 --out results/bench/bench_$(hostname).json
