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

## GPU workflow (see RUNBOOK.md for the full sequence and decision points)
Single-GPU workstation, no SLURM (`scripts/local/`, same outputs and resume behaviour as the cluster scripts):
```bash
scripts/local/bench.sh                               # generation benchmark + 3-round GRPO memory/time probe
GATES=g1,g2,g3,g5 scripts/local/e0_gates.sh          # cheap gates, pins the prompt style
GATES=g4 scripts/local/e0_gates.sh                   # positive control on pi1 (stop/go)
scripts/local/sieve.sh                               # pool sieve K=32, base eval, candidates, job table
scripts/local/study1.sh                              # 40 jobs sequentially, resumable
scripts/local/study2.sh                              # only if decided
```
Institute SLURM cluster (`scripts/slurm/`): `bash env/setup_cluster.sh`, then `sbatch` the same steps
(`bench.sbatch`, `e0_gates.sbatch`, `sieve.sbatch`, `study1_array.sbatch`, `study2_array.sbatch`).
Re-running any step or re-submitting any array is safe: finished jobs exit immediately, partial ones resume.

## Layout
`rlvr_v2/` package (config, data, prompts, grader, signals, selectors, sampling, sieve, evaluate, train_grpo,
curriculum, study1, gates, stats, aggregate, cli) · `configs/` · `manifests/` (committed) · `scripts/slurm/` ·
`analysis/` · `paper/` (prereg, tables, figures) · `results/` (git-ignored except summaries) · `tests/`.
