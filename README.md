# rlvr_v2 — pre-registered re-examination of example selection for 1-shot RLVR

Clean re-implementation of the "which single example should 1-shot RLVR train on?" study
(Qwen2.5-Math-1.5B, GRPO via TRL, LoRA, MATH). The protocol lives in `paper/prereg.md`;
the diagnosis of the previous codebase is in the project plan (see the top-level `ghoul/`,
`loop/`, `baseline_*/` folders, kept untouched as the audit record).

## Design rules (each one closes a failure of the old code)
- One `GenerationCfg` owns lengths/decoding for sieve, training and eval; `TrainCfg` has no length fields.
- Prompts are rendered in one place (`rlvr_v2/prompts.py`) and passed to TRL as strings; stop tokens include `<|im_end|>`.
- Correctness = `math-verify` on the last balanced `\boxed{}`; unboxed/truncated ⇒ incorrect.
- One curriculum implementation with an `--arm` switch; selectors carry named coefficients (JSON/LaTeX export).
- Manifests of `unique_id` + text hash for every split; disjointness asserted at every entry point.
- Batched generation only; every rollout, signal, selection, GRPO step metric and per-item eval outcome is logged.
- Study jobs refuse to start until `results/e0/gates.json` reports `all_passed: true`.

## Local (CPU) workflow
```bash
cd "D:/RLVR Project/rlvr_v2"
./.venv/Scripts/python -m pytest -q                       # unit tests (no model downloads)
./.venv/Scripts/python -m rlvr_v2.cli smoke --config configs/smoke_cpu.yaml   # tiny-model end-to-end (~minutes)
./.venv/Scripts/python -m rlvr_v2.cli show-config --config configs/study1.yaml
```

## Cluster workflow (institute SLURM shards)
```bash
bash env/setup_cluster.sh                         # once, on the login node (venv ~/envs/rlvr_v2 + prefetch)
sbatch --nodelist=node1 scripts/slurm/bench.sbatch   # Day 1: throughput benchmark -> results/bench/
sbatch scripts/slurm/e0_gates.sbatch                 # E0 gates (G1,G2,G3,G5 then G4 positive control)
sbatch scripts/slurm/sieve.sbatch                    # pool sieve at K=32, base eval, candidate selection, job CSV
sbatch --nodelist=node1 --array=0-19%1 scripts/slurm/study1_array.sbatch
sbatch --nodelist=node2 --array=20-39%1 scripts/slurm/study1_array.sbatch
python -m rlvr_v2.cli aggregate && python analysis/study1_analysis.py   # after rsync of results/
sbatch scripts/slurm/study2_array.sbatch             # only if the Study-1 gate allows
```
Re-submitting any array is safe: finished jobs exit immediately, partial ones resume.

## Layout
`rlvr_v2/` package (config, data, prompts, grader, signals, selectors, sampling, sieve, evaluate, train_grpo,
curriculum, study1, gates, stats, aggregate, cli) · `configs/` · `manifests/` (committed) · `scripts/slurm/` ·
`analysis/` · `paper/` (prereg, tables, figures) · `results/` (git-ignored except summaries) · `tests/`.
