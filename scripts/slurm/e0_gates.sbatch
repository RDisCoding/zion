#!/bin/bash
#SBATCH --job-name=rlvr2-e0
#SBATCH --partition=gpu
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err
# E0 gates. Run G1-G3 and G5 first (cheap), then G4 (positive control on pi1, ladder of 3 rungs).
#   sbatch scripts/slurm/e0_gates.sbatch                 -> all gates
#   GATES=g1,g2,g3,g5 sbatch scripts/slurm/e0_gates.sbatch
source scripts/slurm/common.sh
GATES="${GATES:-g1,g2,g3,g5,g4}"
python -m rlvr_v2.cli manifests --config configs/study1.yaml
python -m rlvr_v2.cli gates --config configs/e0.yaml --gates "$GATES" --out results/e0/gates.json ${EXTRA_ARGS:-}
