# Runbook: what to run, where, and how

Two launch layers exist over the same code, configs, manifests and protocol:

- **A. Workstation with one GPU, no SLURM** (RTX PRO 4500, 32 GB): `scripts/local/*.sh`. This is the current setup.
- **B. Institute SLURM cluster**: `scripts/slurm/*.sbatch` (kept unchanged).

Both write the same `results/` tree and are idempotent: re-running any step skips finished work and resumes partial work.
Analysis always happens on the laptop from a zip of `results/` (section C).

---

## A. Workstation (single GPU, no SLURM)

Setup is already done on this machine (torch 2.11 + cu128, transformers 5.18, TRL 1.14.1, PEFT 0.21.2, repo installed).
All commands are typed from the repo root with the venv active (or `export RLVR_ENV=/path/to/venv`). Make the scripts
executable once: `chmod +x scripts/local/*.sh`. Long steps belong in `tmux` (or `nohup ... &`), so an SSH drop does not kill
them. Run **one step at a time**: the GPU is the whole budget and the scripts assume they own it.

### A1. Benchmark (about 30 min)
```bash
scripts/local/bench.sh
```
Outputs `results/bench/bench_<host>.json` (generation tok/s and memory per batch size) and
`results/bench/train_probe_<stamp>/train_probe.json` (three real GRPO rounds on π1 at the Study-1 config: peak GPU memory,
seconds per round, projected hours per 100 rounds). Send me both files before going further; they fix two knobs for
every later run:
- `gen.hf_batch_size` = the largest batch size that ran without out-of-memory in the benchmark;
- `train.per_device_train_batch_size` = 8 if the probe fit (peak memory comfortably under 32 GB), else 4.

If the probe hits out-of-memory, re-run it with the smaller batch: `EXTRA_ARGS="--override train.per_device_train_batch_size=4" PROBE_STEPS=3 scripts/local/bench.sh`.
Whatever we pick goes into **every** later command through `EXTRA_ARGS` and never changes mid-study (it is part of the
config hash that groups runs).

### A2. Cheap gates (1–3 h)
```bash
FRESH=1 GATES=g1,g2,g3,g5 scripts/local/e0_gates.sh     # FRESH=1 moves any earlier results/e0 aside (never deletes)
```
Use `FRESH=1` whenever earlier gate outputs must not be reused (the 2026-10-04 run 1 is superseded: see the prereg
deviations log). If G2 fails, read `results/e0/g2_pytest.log`; if the grader cannot verify known equivalences, the
run now stops at once with `GraderUnavailableError` and a diagnosis instead of grading silently by string match.
G1 scores the base model on MATH-500 under both prompt styles and pins the better one; G2 runs the grader tests; G3 checks
truncation and stop tokens; G5 checks that two identical evals agree. The script prints each gate's pass/fail. Expected:
all four `True`, `all_passed = False` (G4 has not run yet; that is correct). `results/e0/frozen_args.txt` now holds the
pinned prompt style.

### A2b. G2 grader cross-check against Qwen2.5-Math (CPU, ~10 min, needs internet once)
```bash
scripts/local/grader_crosscheck.sh
```
Grades 200 randomly sampled G1 base outputs (pinned style, seed 20261005) with our grader and with Qwen2.5-Math's
own evaluation code (commit a45202b), run in an isolated environment at `~/envs/qwen_grader`, because Qwen's code pins
sympy 1.12 / antlr4 4.11.1, which must not enter the rlvr_v2 environment. Send `results/e0/g2_crosscheck/summary.json`
and `disagreements.jsonl`; every disagreement is adjudicated before G2 counts as complete and before G4 runs.

### A3. Positive control (1–8 h, stop/go point)
```bash
GATES=g4 scripts/local/e0_gates.sh
```
1-shot GRPO on Wang et al.'s π1 through the budget ladder {100 rounds, 2e-5} → {100, 5e-5} → {200, 5e-5}; it stops at the
first rung that gains ≥ 3 points on MATH-500. Pass ⇒ `all_passed = True` and `frozen_args.txt` gains the budget.
**Fail on all three rungs ⇒ stop and send me `results/`; Study 1 must not start** (the code refuses anyway).

### A4. Sieve and candidate selection (3–5 h)
```bash
scripts/local/sieve.sh
```
Pool sieve at K = 32 (resumable), base eval, deterministic matched-pair candidate selection, job table. Produces
`manifests/study1_candidates.json` and `manifests/study1_jobs.csv` (40 rows: 32 candidates + 8 replicate seeds).
A3 and A4 are independent; either order. Send me `results/pool/signals.jsonl` and the two manifests.

### A5. Study 1 (days)
```bash
tmux new -s s1
scripts/local/study1.sh                 # all 40 jobs, sequentially; Ctrl-b d to detach, `tmux attach -t s1` to return
```
Each job is its own process: fresh base model, fresh LoRA, R rounds of G = 64, MATH-500 eval. Progress:
`grep -l '"state": "done"' results/study1/*/*/status.json | wc -l`. After any interruption just run the same command
again. Optional held-out sign check (one extra eval per job):
`EXTRA_ARGS="$(cat results/e0/frozen_args.txt) --heldout" scripts/local/study1.sh`.

**Time check.** One job ≈ R × (seconds per round from the probe) + two evals (from the benchmark). With 40 jobs on one
GPU this is the critical path for the 12 Oct deadline. If the projection does not finish by 9 Oct, tell me *before*
launching: the pre-registered fallbacks are, in order, dropping the 8 replicate seeds (rows 32–39), then N = 24, and no
Study 2.

### A6. Study 2 (only if we decide to run it)
```bash
scripts/local/study2.sh
```

### A7. Send results back
```bash
zip -qr results_$(date +%m%d_%H%M).zip results manifests -x "*/trainer/*" -x "*.safetensors" -x "*.bin" -x "*.pt"
```

### What `--gres=shard:10/20` meant, and what replaces it
SLURM's `shard` resource splits one physical GPU into N schedulable slices so several jobs can share it; `shard:10` or
`shard:20` reserved that many slices (a quarter to a half of the institute card). It is a scheduling token only, not a
memory partition. On this workstation the whole GPU is yours, so the translation is: one job at a time, the full 32 GB,
and batch sizes chosen by the A1 benchmark rather than by a shard share.

### Red flags (stop and report)
- `completions/clipped_ratio` above 0.2 in any `train_metrics.jsonl` (the truncation guard should already have aborted).
- `frac_reward_zero_std` near 1 for most steps of a run (no learning signal; check that candidate's P_s).
- G1 base accuracy outside 15–95 % or truncation above 10 %; G3 secondary-stop share above 5 %.
- Two G5 evals disagreeing on more than 2 % of items.

---

## B. Institute SLURM cluster (unchanged)

```bash
git clone https://github.com/RDisCoding/zion.git rlvr_v2 && cd rlvr_v2     # or unzip the git-archive zip
bash env/setup_cluster.sh && mkdir -p results/slurm
sbatch --nodelist=node1 scripts/slurm/bench.sbatch
GATES=g1,g2,g3,g5 sbatch scripts/slurm/e0_gates.sbatch
GATES=g4 sbatch scripts/slurm/e0_gates.sbatch                      # never two e0 jobs at once
export EXTRA_ARGS="$(cat results/e0/frozen_args.txt)"
sbatch scripts/slurm/sieve.sbatch
python -c "import json; print(json.load(open('results/e0/gates.json'))['all_passed'])"   # must be True
export EXTRA_ARGS="$(cat results/e0/frozen_args.txt)"              # re-read: now includes the G4 budget
sbatch --nodelist=node1 --array=0-19%1  scripts/slurm/study1_array.sbatch
sbatch --nodelist=node2 --array=20-39%1 scripts/slurm/study1_array.sbatch
sbatch scripts/slurm/study2_array.sbatch                           # only if decided
```
If a job stays pending with `Reason=Resources`, lower the request: `sbatch --gres=shard:15 ...`.

---

## C. Analysis (laptop)
Unzip over the local `results/`, then `python -m rlvr_v2.cli aggregate` and `python analysis/study1_analysis.py`
(`study2_analysis.py` if Study 2 ran). Every table and figure under `paper/` is regenerated from `results/tables/*.parquet`.
