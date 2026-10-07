# SGAC reproduction protocol (frozen as tag `sgac-protocol-v1` before any GPU run)

This is a separate track from the pre-registered signal-validation study (`paper/prereg.md`, tag `prereg-v1`). It
adds code and configuration only; it changes none of that study's code paths, configs, manifests, eligibility rules
or results. Guarded by `tests/test_sgac_shared_frozen.py`: the Study-1 and E0 config hashes still read `84e100e2` and
`924c03f2`, and `git diff prereg-v1 -- rlvr_v2/config.py configs/{base,e0,study1,study2}.yaml manifests paper/prereg.md`
is empty.

**Question.** Does the original Selector-Guided Autonomous Curriculum (old paper: base 64.0 %, 1-shot RLVR checkpoint
66.0 %, SGAC 68.0 % on 50 held-out MATH problems) reproduce?
- Primary: on the E0-validated substrate, so that it also serves as the SGAC baseline for the new study.
- Secondary: as a fidelity check of the historical configuration.

Spec hashes: e0 `577f3319`, as_run `64ba360f` (`python -m rlvr_v2.sgac show-spec --profile <p>`).

## 1. What the published run did (verified from the notebooks)

**Source.** `ghoul/rl-llm-phase-5-the-loop-modified-results.ipynb` (sha256 `08818a2f…1be41b`).
- Cell 7 ends with `Historical Accuracies: [0.66, 0.62, 0.6, 0.68]`.
- The loop ran 10.96 h on a Kaggle T4.
- The paper's quoted step-1 losses match it exactly.
- Base 64 % and pi1 66 % appear only in `phase-5-results/autonomous_rlvr_model/outputs.png`. Base was measured after training, with the adapter disabled.

**Data.**
- `nlile/hendrycks-MATH-benchmark`, train split, revision `465bcdb3…`, `.shuffle(seed=42)`.
- Pool = rows 0–999. Test = rows 1000–1049, which come from the train split; 20 are MATH `test/` ids.
- Gold = the first `\boxed{}` of the solution.
- Rebuilt exactly in `configs/sgac/manifests/data.json`: pool levels 87/152/203/259/299, test levels 5/9/9/11/16.

**Prompts.**
- Sieve and eval: the raw instruction `"Solve the following math problem step by step and give the final answer in \boxed{}:\n\n{problem}"`.
- GRPO: the same text as a one-message conversation, so TRL applied Qwen's chat template and its default system prompt.

**Model.**
- Qwen2.5-Math-1.5B in bitsandbytes NF4 (double quant, fp16 compute).
- LoRA r=16, α=32 on q/k/v/o, dropout 0, plain `get_peft_model`.
- One PeftModel persists across all bursts. pad = eos = `<|endoftext|>`.

**Sieve.**
- Each step: `random.sample(pool, 4)`, unseeded.
- **All four candidates are removed** each step, so 80 problems are used in total.
- K=4 rollouts per candidate, T=1.0, top_k=50 (transformers default), 1024 new tokens, batch size 1.

**Signals.**
- Ps = mean of the binary rewards.
- Var = `np.var(binary + 0.5·[\boxed{ in text])`, population variance.
- D = |{str(extract_answer)}| / K. An answer-less completion counts as the string `"None"`.
- L = dataset level.

**Selector.** `0.0050·Ps + 0.1832·Var − 0.0751·D + 0.2188·L`, hardcoded, `np.argmax` (first maximum). The pickle is never loaded.

**Burst.**
- `GRPOConfig(lr=2e-5, per_device_train_batch_size=1, generation_batch_size=4, num_generations=4, max_steps=5, logging_steps=1, save_strategy="no", report_to="none")`, with TRL 1.0.0 defaults for everything else. In particular `max_completion_length` = **256**, beta 0, dapo, group scaling, linear LR, seed 42.
- Net effect: **5 single-completion AdamW updates from 2 generation rounds of 4**, with a fresh optimizer and schedule every burst.

**Eval.** Greedy, 1024 tokens, raw prompt, legacy grader, at steps 5, 10, 15, 20.

**Log.**
- Picks: L5 13/20, L4 7/20.
- Ps = 0 in 10/20 picks.
- 7/20 bursts had all-zero loss.
- Problem ids were never printed and step-1 sampling was unseeded, so the original selections cannot be recovered.

## 2. Selector coefficients actually used

**The rule.** Every SGAC arm uses the hardcoded rule that ran, `Score = 0.0050·Ps + 0.1832·Var − 0.0751·D + 0.2188·L`, with no intercept and first-max argmax (paper Eq. 10).

**Provenance.**
- The rule is the N=20 phase-4 regression (`rl-llm-phase-4-modified.ipynb`, cell 8). It was fitted on `X = [L, Ps, Var, D]` but its coefficients were printed with labels shifted by one place. Ten of its 20 rows were that notebook's test items.
- The saved regression (`learned_selector.pkl`) means `0.00504·L + 0.18324·Ps − 0.07506·Var + 0.21881·D + 0.279`. This is the optional `sgac_label_corrected` arm.
- Paper Table 2 (−0.0574 / −0.2511 / +0.0393 / +0.1095) is the N=4 fit with the same label shift. It is non-identifiable (4 rows, 5 parameters) and no executed run used it. It is logged only as a counterfactual pick.

**Consequence.** At K=4 the non-level terms span 0.1619, less than 0.2188, so the rule always picks a maximum-level candidate, breaking ties by higher Var and then lower D. This is proved exhaustively in `tests/test_sgac_selection.py`. Expected agreement with Max-difficulty on the same batches is about 75 %.

## 3. Library versions and the TRL 1.0.0 → 1.14.1 audit

The original ran TRL ~1.0.0 (transformers ~5.5, PEFT 0.18.1). This reproduction runs TRL 1.14.1, transformers 5.18 and PEFT 0.21.2, and `tests/test_sgac_trl_pins.py` asserts that version. The trl-1.0.0 wheel was diffed against 1.14.1.

**Pinned explicitly**, because TRL 1.0.0's defaults are the original's:
- `max_completion_length` (1.0.0: 256; 1.14.1: 512)
- temperature 1.0, top_p 1.0, top_k 0
- beta 0, num_iterations 1, ε 0.2, loss dapo, scale_rewards group, `sum_then_normalize`
- `mask_truncated_completions` False, `shuffle_dataset` True, `disable_dropout` False
- gradient checkpointing True, bf16 True, linear schedule, warmup 0, `adamw_torch_fused`, weight decay 0, `max_grad_norm` 1.0

Any unknown key is refused rather than silently dropped.

**Identical in both versions:**
- Advantage = (r − mean)/(std + 1e-4), per group.
- LoRA weights of a 4-bit base are cast to bf16 at every trainer init. Kept: it is faithful.
- `set_seed(args.seed)` runs at every trainer init.
- A "ref" adapter is created only when beta ≠ 0 (not the case here; asserted).
- `enable_input_require_grads` is called at every init.

**One compensation.** For this batch shape, TRL 1.14.1's dapo normaliser is TRL 1.0.0's × gradient_accumulation / steps_per_generation = ¼. Every loss and gradient is therefore exactly 4× larger.
- AdamW is invariant to a constant gradient scale (up to ε), but clipping is not. So `max_grad_norm` is set to 4.0 (= 1.0 × 4), which makes the clipped updates identical to TRL 1.0.0's.
- Logged losses are 4× the original's. `loss_trl100` divides them back for comparison with the NB-M log.

**Hygiene, with no effect on updates.**
- The input-require-grads hooks that every trainer registers are removed after each burst.
- Sieve and eval run through `modeling.generation_mode`.

## 4. Design

### Profiles

Both profiles run the same algorithm on the same batches. **e0** (PRIMARY) is the E0-validated substrate running the SGAC recipe; **as_run** (secondary) is the historical configuration.

| | e0 (PRIMARY) | as_run (secondary) |
|---|---|---|
| Precision | bf16 | NF4 + fp16 compute; bf16 if bitsandbytes cannot load (recorded) |
| Prompt | `oneshot_rlvr_chat` for sieve, train and eval (E0 G1 pin) | Legacy raw prompt for sieve and eval; conversational prompt for training |
| Correctness / gold | math-verify E0 grader (last box required, truncated = wrong) / `answer` column | Verbatim legacy grader / first box of the solution |
| D | math-verify answer classes / K (`u_ratio`) | Legacy string classes / K |
| Caps (sieve / GRPO / eval) | **3072 / 3072 / 3072** (E0 frozen) | 1024 / 256 / 1024 |
| Stops / pad | `<\|endoftext\|>` + `<\|im_end\|>` / `<\|fim_pad\|>` | `<\|endoftext\|>` only / pad = eos |
| Sieve top_k | 0 | 50 |
| Prompt-length guard | 2048 (see deviation D1) | 2048 |
| Recipe (both) | B=4, K=4, T=1.0, 20 steps, all 4 candidates removed per step, the rule above. LoRA r16/α32 q/k/v/o. Burst = 5 single-completion updates (pdbs 1, generation batch 4, G 4), lr 2e-5 linear, reward = correct + 0.5·boxed. Fresh trainer per burst; the adapter persists. | same |

Every rollout and evaluated item records both graders' verdicts and both D variants.

### Arms

**Core arms.**
- `sgac`: the rule above.
- `random`: dedicated RNG `default_rng([seed, 0x5A17, t])`.
- `max_var`, `max_d`, `max_level`: single-signal rules; ties go to the first maximum, NB-M's convention.

Every step logs all four candidates' signals and every rule's counterfactual pick.

**Optional arms** (tier 6, only if time remains):
- `sgac_label_corrected` (e0)
- `fixed_pi1`: Wang et al.'s π1 trained for the same 20×5 updates with no sieve (e0)
- Phase-1 replication (as_run)

### Seeds

Seeds are 42 (primary), 43 and 44. A seed fixes:
- the candidate schedule: `random.Random(seed).sample(remaining, 4)`, all four removed, frozen in `configs/sgac/manifests/batches_seed{S}.json`. It is identical for every arm and both profiles, so the design is paired.
- the sieve seeds (`seed·1000 + t`);
- the GRPO seed (the run seed at every burst, as NB-M used 42 at every burst);
- the LoRA initialisation.

### Evaluations

**test50 (the original 50 items).**
- Run at steps 5, 10, 15 and 20 for every run.
- Step 0 is the shared base evaluation; a fresh LoRA has B = 0, so it is identical to the base model.

**MATH-500** (`manifests/math500.json`, read-only).
- Shared base at step 0, and every run at step 20.
- Steps 5, 10 and 15 are evaluated later from the saved adapters, in tier 6.

**Base and π1** are evaluated once per profile.

**as_run** also measures `base_after_run` (adapter disabled, after step 20) for seed-42 SGAC, exactly like NB-M cell 11.

**Adapters** are kept for the eval steps (5, 10, 15, 20) and for the latest step, which is the resume point.

### Pre-declared conditionals

These depend on the environment, not on results.
- **C1. Quantisation fallback.** If bitsandbytes cannot load NF4 on the GPU, as_run loads bf16. The fallback is recorded in every run's `load_records`.
- **C2. Base-eval agreement gate (e0).** The e0 base MATH-500 evaluation mirrors E0's batching (64 prompts per call, HF batch 32, length-sorted, seed 0). Its per-item verdicts must agree with E0 G1's (`results/e0/g1/oneshot_rlvr_chat/math500/per_item.jsonl`) on ≥ 98 % of items. Otherwise the queue stops before any e0 curriculum run.
- **C3. as_run test50 batch size.** The as_run base test50 is evaluated at batch 1 and at batch 16. If verdict agreement is < 98 %, every as_run test50 evaluation uses batch 1, which is NB-M's.
- **C4. Projection-based trimming.** If the tier-0 projection exceeds 36 h for tiers 1–5, the as_run seeds 43/44 heuristic arms (max_var, max_d, max_level) are dropped first.
- **C5. Hard stop.** Anything not finished by 2026-10-11 06:00 IST is reported as not run.

### Queue order

The order is pre-declared and never changed on results (`python -m rlvr_v2.sgac jobs --tier N`).

| Tier | Jobs |
|---|---|
| T0 | Probe, manifest check, GPU smoke for both profiles (2 steps, 5 eval items) with a forced resume and a time projection |
| T1 | e0 base and π1 evaluations |
| T2 | e0 seed 42: sgac, random, max_var, max_d, max_level |
| T3 | as_run base and π1; as_run seed 42 sgac (+ base_after_run) and random |
| T4 | e0 seeds 43 and 44: sgac and random first, then the three heuristics |
| T5 | as_run: the remaining seed-42 heuristics, then seeds 43 and 44 |
| T6 | The optional arms above |

No early stopping on accuracy, ever.

## 5. Endpoints, analysis, verdict

**Primary endpoints:** e0 profile, step 20, MATH-500. Per-item correctness is averaged over the completed seeds. Analysis is a paired item bootstrap: 10,000 resamples, seed 0, percentile 95 % CI (`stats.paired_item_bootstrap`).
- **P1 (training effect):** SGAC − Base.
- **P2 (selection effect):** SGAC − Random, on the seeds completed by both arms.

**Verdict** (computed by `rlvr_v2/sgac/report.py::verdict`, never by hand):
- **"Original SGAC reproduced"** if the 95 % CI lower bounds of both P1 and P2 are > 0.
- **"Original SGAC partially reproduced"** if exactly one of them is.
- **"Original SGAC could not be reproduced"** otherwise. The reasons come from the diagnostics below. If P1's upper bound is < 0, the report says SGAC degrades accuracy.

**Secondary and descriptive** (no multiplicity claims):
- SGAC vs Max-variance, Max-disagreement, Max-difficulty and π1, on MATH-500 and on test50.
- test50 trajectories at 0/5/10/15/20 against 64/66/62/60/68, read as "directionally consistent?".
- as_run fidelity:
  - the seed-42 SGAC trajectory and base-after-run against the paper, within item-bootstrap CIs;
  - as_run SGAC vs Random.
- Selection agreement of the rules on identical batches, including whether SGAC reduces to Max-difficulty, plus the level, Ps and D distributions of the picks against the NB-M log.
- Training diagnostics: zero-advantage groups, all-zero-loss bursts, clipped ratio, training reward, ‖ΔA‖ and ‖ΔB‖.
- Grader sensitivity: math-verify vs legacy on every output. Optionally, the symbolic legacy regrade (`scripts/sgac_legacy_regrade.py` in `~/envs/qwen_grader`).
- Runtime and peak GPU memory.

**Never done:**
- tuning coefficients or settings after seeing results;
- refitting the selector;
- early stopping;
- claiming anything from the N=4 or N=20 selector fits;
- reporting a "successful" reproduction because a single number equals 68 %.

## 6. Deviations from the original run

**e0 vs the original** (intentional; the user chose the validated substrate as primary on 2026-10-07):
- bf16 instead of NF4.
- One `oneshot_rlvr_chat` prompt for sieve, train and eval, instead of a raw sieve/eval prompt plus a chat-template training prompt.
- The math-verify grader with the `answer`-column gold, instead of first-box / last-number / sympy.
- D from math-verify classes.
- Caps 3072/3072/3072 instead of 1024/256/1024.
- Stops 151643 + 151645 with a fim_pad pad.
- Sieve top_k 0 instead of 50.
- Batched generation.
- A seeded, shared candidate schedule.
- TRL 1.14.1 with the pins and compensation of section 3.
- Base measured at step 0.
- MATH-500 added.
- RTX PRO 4500 instead of T4.

**as_run vs the original:**
- Batched sieve; test50 batch size per C3.
- Seeded schedule.
- Library versions (section 3).
- Possible bf16 fallback (C1).
- The legacy symbolic path is off under the pinned antlr4 4.13.2, so grading is string equality. Whether Kaggle had antlr4 4.11 is unknown; the regrade script measures the effect.

**D1. Prompt-length guard.** E0's 1024-token prompt guard refuses rather than truncates. One original pool item exceeds it: row 398, `test/counting_and_probability/670`, with 1379 tokens (e0) or 1383 (as_run). Seed 44 draws it. The guard is raised to 2048 in both SGAC profiles, so the original pool stays intact; no other item exceeds 938 tokens, so no other generation changes. For that one item in e0, prompt + completion can exceed the model's 4096-token context if a completion runs past 2717 tokens. Prompt and completion lengths are logged.

**Corrections to the brief:**
- All four candidates are removed per step, not only the selected one.
- Phase-1 rollouts used 2048 tokens (not 1024), with 20-step bursts.
- The "hand-picked" Phase-1 candidates were shuffled rows 0–3.
- Eq. 10 comes from the leaky N=20 fit, not the N=4 fit.
- The "5 GRPO steps" are 5 single-completion updates.
- Base 64 % was measured after training.

**Phase-1 replication (optional).** It mirrors `rl-llm-phase-4-selector-experiments-results.ipynb`:
- rows 0–3;
- K=8 rollouts at 2048 tokens (T=1.0, top_k 50);
- a fresh LoRA per candidate with a 20-step GRPO burst under the same pinned config;
- greedy eval at 2048 tokens on rows 4–13 with that notebook's own eval prompt, plus MATH-500.

The four candidates' Table-1 signals and accuracies are compared with the paper. The selector is not refitted for use; both label mappings are shown only to illustrate non-identifiability.

## 7. Amendments (dated, append-only)

_None yet._

## 8. Runbook

Run on the workstation, inside `tmux new -s sgac`, with `conda activate rlvr_v2`.

```bash
git fetch --tags && git checkout sgac-repro
TIERS=0 scripts/local/sgac_repro.sh          # ~45 min: env probe, manifest check, GPU smoke + projection
TIERS=1,2,3,4,5 scripts/local/sgac_repro.sh  # core queue, resumable: re-run the same command after any stop
TIERS=6 scripts/local/sgac_repro.sh          # optional arms, only if time remains
python -m rlvr_v2.sgac report                # reports/sgac_repro/REPORT.md (+ tables, figures, report_data.json)
```

Results go to `results_sgac/` (git-ignored); logs go to `results_sgac/logs/`. Send `reports/sgac_repro/` and a zip of `results_sgac/` back for analysis.
