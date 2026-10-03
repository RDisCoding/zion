# Runbook: cluster steps for 4–10 October 2026

Audience: whoever submits jobs on the institute SLURM cluster (Rudray or Vedang). Every step is idempotent; re-running a
command or re-submitting an array is safe. Send `results/` back (rsync or zip) after each milestone so the analysis
can run locally. Nothing below needs editing except node names and array ranges.

## Day 1 (4 Oct) — environment, benchmark, cheap gates
```bash
git clone <repo-url> ~/rlvr_v2 && cd ~/rlvr_v2          # or copy the folder; keep the path free of spaces
bash env/setup_cluster.sh                                  # ~15 min: venv ~/envs/rlvr_v2 + model/dataset prefetch
sbatch --nodelist=node1 scripts/slurm/bench.sbatch         # ~20 min: results/bench/bench_node1.json
sbatch --nodelist=node2 scripts/slurm/bench.sbatch
GATES=g2,g1,g3,g5 sbatch scripts/slurm/e0_gates.sbatch     # G2 grader tests, G1 base evals (2 prompt styles), G3 stop tokens, G5 determinism
```
What to send back: `results/bench/*.json`, `results/e0/gates.json`, `results/slurm/*.out|err`.
Decision taken from the benchmark: `gen.hf_batch_size`, `train.per_device_train_batch_size` (keep `num_generations` 64
unless free memory < 20 GB, then `--override train.num_generations=32 gen.max_new_tokens=2048`), and whether N stays 32.

## Day 2 (5 Oct) — positive control, sieve, launch Study 1
```bash
GATES=g4 sbatch scripts/slurm/e0_gates.sbatch              # pi1 ladder; several hours; stops at the first passing rung
sbatch scripts/slurm/sieve.sbatch                          # pool sieve K=32 + base eval + candidate selection + jobs CSV
# after gates.json shows all_passed: true and manifests/study1_jobs.csv exists:
sbatch --nodelist=node1 --array=0-19%1 scripts/slurm/study1_array.sbatch
sbatch --nodelist=node2 --array=20-39%1 scripts/slurm/study1_array.sbatch
```
If G4 passed on rung 2 or 3, add `EXTRA_ARGS="--override train.rounds=<R> train.learning_rate=<lr>"` to every later
sbatch line (the values are printed in `results/e0/gates.json` under `g4.frozen_budget`).

## Days 3–5 (6–8 Oct) — monitor, resubmit, analyse
```bash
squeue -u $USER
grep -l '"state": "done"' results/study1/*/*/status.json | wc -l     # finished jobs
sbatch --nodelist=node1 --array=0-19%1 scripts/slurm/study1_array.sbatch   # resubmit after a timeout; finished jobs exit at once
```
Send back `results/` (excluding `*/trainer/checkpoint-*` if large; adapters are ~70 MB each and can be excluded too).
Locally: `python -m rlvr_v2.cli aggregate && python analysis/study1_analysis.py`.

## Days 6–7 (9–10 Oct) — Study 2 only if the Study-1 gate allows
```bash
sbatch scripts/slurm/study2_array.sbatch                   # 3 arms x 3 seeds; add learned arm via EXTRA_ARGS if gated in
```

## Red flags (stop and report)
- `completions/clipped_ratio` > 0.2 in any `train_metrics.jsonl` (truncation guard should have aborted the job).
- `frac_reward_zero_std` ≈ 1 for most steps of a run (no learning signal; check the candidate's p_s).
- Base MATH-500 accuracy in G1 outside 15–95% or truncation > 10% (prompt/stop-token problem).
- Two G5 evals disagreeing on > 2% of items (nondeterministic generation; lower batch size or disable SDPA fast paths).
