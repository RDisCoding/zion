# Runbook: what to run, where, and how (cluster, 4–10 Oct 2026)

Everything below is typed **on the cluster login node** in `~/rlvr_v2` (the cluster account is Vedang's per the old logs,
or whoever has access). The laptop is only used to package the code (step 0) and to analyse `results/` afterwards.
One person owns the cluster copy; never run two `e0_gates` jobs at once (both write `results/e0/gates.json`).
Every step is idempotent: re-submitting a job or array is safe, finished work is skipped and partial work resumes.

## 0. Package on the laptop (once, after the last commit)
```bash
cd "D:/RLVR Project/rlvr_v2"
git status                                   # must be clean
git archive --format=zip --prefix=rlvr_v2/ -o ../rlvr_v2_cluster.zip HEAD     # ~0.7 MB, tracked files only
scp ../rlvr_v2_cluster.zip <user>@<login-host>:~/
```
On the login node: `cd ~ && unzip -q rlvr_v2_cluster.zip && cd rlvr_v2`

## 1. One-time setup (login node, ~15 min, needs internet)
```bash
bash env/setup_cluster.sh        # creates ~/envs/rlvr_v2 (NOT the old selector_env), installs pinned libs, prefetches model + datasets
mkdir -p results/slurm           # REQUIRED before any sbatch: the #SBATCH output paths are relative
```
Success = last line `Environment ready at ~/envs/rlvr_v2`. Partition/shards in the scripts (`--partition=gpu`, `--gres=shard:...`) match
the old `train.sh` files. If a job sits pending with `Reason=Resources`, override on the command line, e.g. `sbatch --gres=shard:15 ...`.

## 2. Day 1 (4 Oct): benchmark + cheap gates
```bash
sbatch --nodelist=node1 scripts/slurm/bench.sbatch
sbatch --nodelist=node2 scripts/slurm/bench.sbatch
GATES=g1,g2,g3,g5 sbatch scripts/slurm/e0_gates.sbatch
```
Watch: `squeue -u $USER`, `tail -f results/slurm/rlvr2-e0_<jobid>.out`.
Outputs: `results/bench/bench_<node>.json`, `results/e0/gates.json`, `results/e0/frozen_args.txt`, `results/slurm/*.out|err`.
Send these back (command in section 6). The benchmark decides the HF batch size and whether N stays 32.

## 3. Day 2 (5 Oct): positive control, sieve, candidates
```bash
GATES=g4 sbatch scripts/slurm/e0_gates.sbatch              # pi1 positive control, up to 3 rungs, several hours
export EXTRA_ARGS="$(cat results/e0/frozen_args.txt)"      # after G1: pins the prompt style
sbatch scripts/slurm/sieve.sbatch                          # pool sieve K=32 -> base eval -> candidate selection -> jobs CSV
```
Both can run at the same time on different nodes. When both are done:
```bash
python -c "import json; r=json.load(open('results/e0/gates.json')); print(r['all_passed'], {g: v['passed'] for g, v in r['gates'].items()})"   # must be True
ls manifests/study1_candidates.json manifests/study1_jobs.csv
cat results/e0/frozen_args.txt                              # now also carries the G4-frozen budget
```
If G4 failed on all three rungs: stop and send everything back; do not start Study 1.

## 4. Study 1 (launch 5 Oct, finish ~8 Oct)
```bash
export EXTRA_ARGS="$(cat results/e0/frozen_args.txt)"       # re-read: includes prompt style + frozen budget
sbatch --nodelist=node1 --array=0-19%1  scripts/slurm/study1_array.sbatch
sbatch --nodelist=node2 --array=20-39%1 scripts/slurm/study1_array.sbatch
```
Optional: add `--heldout` to EXTRA_ARGS to also score the 500 held-out train problems (the pre-registered sign check; costs one
extra eval per job). Study jobs refuse to start unless all five gates (G1–G5) are present and passed.
Monitor / resume after a timeout (same commands again):
```bash
squeue -u $USER
grep -l '"state": "done"' results/study1/*/*/status.json | wc -l      # finished jobs out of 40
```

## 5. Study 2 (only after Study 1 is done and we decide to run it)
```bash
sbatch scripts/slurm/study2_array.sbatch                    # 3 arms x 3 seeds; learned arm only if told to add it
```

## 6. What to send back (after each milestone)
```bash
cd ~/rlvr_v2
zip -qr results_$(date +%m%d_%H%M).zip results manifests -x "*/trainer/*" -x "*.safetensors" -x "*.bin" -x "*.pt"
```
then `scp` the zip to the laptop. Analysis runs locally: `python -m rlvr_v2.cli aggregate` then `python analysis/study1_analysis.py`.

## Red flags (stop and report)
- `completions/clipped_ratio` above 0.2 in any `train_metrics.jsonl` (the truncation guard should already have aborted the job).
- `frac_reward_zero_std` near 1 for most steps of a run (no learning signal; check that candidate's P_s).
- G1 base accuracy outside 15–95% or truncation above 10%; G3 secondary-stop share above 5% (prompt or stop-token problem).
- Two G5 evals disagreeing on more than 2% of items (nondeterministic generation).
