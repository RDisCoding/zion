  # E0-calibrated SGAC selector: protocol (FROZEN as tag `sgac-calib-protocol-v1`, 2026-10-12; no calibration label existed at freeze)

  This is a third track. It is separate from the pre-registered signal study (`paper/prereg.md`, tag `prereg-v1`) and from the historical SGAC reproduction (`paper/sgac_repro_protocol.md`, tag `sgac-protocol-v1`). It reads their code, manifests and sieve outputs and writes none of them. It is frozen as tag `sgac-calib-protocol-v1` once the team signs off and before any calibration label exists. After that, changes go only in §13.

  **Question.** Does a selector trained on measured one-shot utility pick better curriculum problems than Random and than the historical Eq. 10, inside the unchanged SGAC loop?

  **Prior expectation, stated up front.**
  - In the historical reproduction (e0 profile, 3 seeds), every arm finished within 1.1 pp of Base on MATH-500. SGAC − Random was +0.7 pp [−0.8, +2.3].
  - A selector can only improve on Random if the training effect itself is non-trivial and differs between problems.
  - The pilot (§4) therefore measures whether the per-problem utility differences are large enough to learn. It does so before the bulk of the budget is spent.
  - An "inconclusive" or "no usable signal" outcome is a reportable result. The design ensures it is a clean one.

  **Evidence ladder: what each stage can and cannot establish.** Each gate is necessary but not sufficient.
  - G-P only shows that the utility labels are measurable.
  - G-S only shows that the signals predict utility for unseen problems, under the calibration protocol and on the dev set.
  - Only Stage E can support the claim that the learned selector *outperforms*. That claim requires **both** H1 (vs Random) and H2 (vs SGAC Eq. 10) to meet the §7 decision rule.

  | G-P | G-S | Stage E | What the paper may claim |
  |---|---|---|---|
  | fail | — | not run | Per-problem utility is not measurably different at the SGAC budget, so a selector cannot be calibrated in this regime. |
  | pass | fail | not run | Utility differs between problems, but Ps/Var/D/L do not predict it out of sample beyond the baselines. |
  | pass | pass | H1 and H2 met | The learned selector improves on Random and on Eq. 10 at a matched RL budget, reported with its calibration cost. |
  | pass | pass | equivalent (±2 pp) | Calibrated selection predicts one-shot utility, but this does not translate into curriculum gains larger than 2 pp. |
  | pass | pass | inconclusive | Same as the row above, except that the curriculum effect is below what the design can detect. The detection limit (§7) is reported. |
  | pass | pass | worse than X | The learned selector harms the curriculum relative to X. |

  ## 0. Separation of the three tracks

  | | Historical reproduction | Calibration (this protocol) | Final evaluation (this protocol) |
  |---|---|---|---|
  | Purpose | Did the published SGAC reproduce? | Learn and validate a selector | Compare curricula |
  | Status | Done; verdict "could not be reproduced" | Frozen; not yet run | Frozen; not yet run |
  | Code | `rlvr_v2/sgac/` (frozen) | `rlvr_v2/sgac_calib/` (new) | `rlvr_v2/sgac_calib/` + the frozen SGAC loop |
  | Config / manifests | `configs/sgac/` | `configs/sgac_calib/` | `configs/sgac_calib/` |
  | Evaluation data | MATH-500, test50 | Dev set only (§2) | MATH-500, test50 |
  | Results / report | `results_sgac/`, `report.md` | `results_sgac_calib/{audit,calib,selector}` | `results_sgac_calib/curriculum`, `reports/sgac_calib/REPORT.md` |

  **Guard test** (`tests/test_sgac_calib_frozen.py`). It checks that:
  - the historical spec hashes still read e0 `577f3319` and as_run `64ba360f`;
  - `git diff sgac-repro-frozen -- rlvr_v2/sgac configs/sgac paper/sgac_repro_protocol.md report.md paper/sgac_repro_freeze.md` is empty. This is the code that produced the frozen results, including the pre-run fixes committed after `sgac-protocol-v1`;
  - `git diff sgac-protocol-v1 -- paper/sgac_repro_protocol.md` is empty;
  - every calibration and curriculum run trains on the e0 substrate. The substrate spec hash must read `577f3319` (`require_e0_substrate`); only the results root differs;
  - importing `rlvr_v2.sgac_calib` patches no frozen callable;
  - the prereg hashes (Study 1 `84e100e2`, E0 `924c03f2`) are unchanged (`tests/test_sgac_shared_frozen.py`).

  **Freezing the historical reproduction** (decided and done 2026-10-12). It is frozen as it stands, with no further compute.
  - The record is `paper/sgac_repro_freeze.md`, committed with `report.md` to `sgac-repro` and tagged `sgac-repro-frozen`. It holds:
    - the report hash and the results-archive hash (`sgac_repro_frozen.tar`, sha256 `76c601aa…d636`, kept on the workstation);
    - the verdict;
    - the runs not completed, reported as not run under the pre-declared rule C5;
    - the post-hoc leakage finding.
  - `max_level` seeds 43/44 and the as_run T5 and T6 tiers are not finished.
  - This track never regenerates the historical report.

  ## 1. Stage A — audit of the original Phase-1 pipeline

  The historical files are never edited. Corrections live in this track's report.

  ### Already established (read-only analysis)

  **Selector provenance.**
  - The phase-4 notebooks fit `X = [L, Ps, Var, D]` and print the coefficients shifted by one label.
  - Eq. 10 (`0.0050·Ps + 0.1832·Var − 0.0751·D + 0.2188·L`, no intercept) is the N=20 fit read with the shifted labels. Ten of its 20 rows were that notebook's test items.
  - `ghoul/learned_selector.pkl` actually means `0.0050·L + 0.1832·Ps − 0.0751·Var + 0.2188·D + 0.279`.
  - Table 2 is the N=4 fit with the same shift. Four rows for five parameters make it non-identifiable.
  - At K=4, Eq. 10 always picks a maximum-level candidate. This is proved in `tests/test_sgac_selection.py`; in practice it picked one in 272/272 steps.

  **Leakage audit (2026-10-10, CPU).** Method: `rlvr_v2.data.near_duplicates` (exact normalised text, or difflib ratio ≥ threshold after a token-Jaccard ≥ 0.5 prefilter) over all relevant splits.

  | Pair | Exact-id overlap | Near-dup ≥ 0.9 | Near-dup ≥ 0.8 |
  |---|---|---|---|
  | SGAC pool (1000) vs MATH-500 | 0 | **10** | **46 pairs / 34 pool items** |
  | SGAC pool vs test50 | 0 | 1 | 3 |
  | test50 vs MATH-500 | 0 | 0 | 2 |
  | prereg pool (500) vs test50 | **2** | 4 (incl. the 2) | 5 |
  | prereg held-out (500) vs test50 | **3** | 6 (incl. the 3) | 6 |
  | prereg held-out vs MATH-500 | 0 | 0 | 11 |
  | prereg pool vs SGAC pool | 49 | 10 (other ids) | 18 |
  | π1 vs everything | 0 | 0 | 0 |

  **Consequence for the historical e0 runs.**
  - Some runs *trained on* an item that is a near-duplicate of a MATH-500 item:
    - ≥ 0.9: 5 picks in 3 runs: SGAC s44 (step 12); Random s42 (steps 1 and 14); Max-D s42 (steps 1 and 14).
    - ≥ 0.8: 10 picks in 7 runs.
  - At most a handful of the 500 eval items are affected. The bias is small, but it is real.
  - Those runs are therefore not reused as comparators. The new curriculum pool excludes these items, so every arm is rerun on clean batches (§5).

  ### Still to do in Stage A

  | Item | How | Compute |
  |---|---|---|
  | Feature order, coefficient mapping, intercept, serialized selector | Load the pickle and both notebooks' fits. A unit test asserts the mapping above. Document the 10 leaked rows of the N=20 fit. | CPU |
  | Recompute Ps, Var, D, L for the four original Phase-1 candidates (shuffled rows 0–3; none contaminated) | (a) Historical settings: K=8, 2048 tokens, legacy prompt and grader. (b) E0 pipeline: K=32, 3072 tokens, math-verify. Compare both with the paper's Table 1. Classify every discrepancy: K=8 sampling error (binomial CI), grader, truncation, prompt, gold source (first box vs `answer`), D definition (`"None"` string vs one unparsable class). | ≈ 0.3 GPU-h |
  | Phase-1 downstream replication (optional) | Existing tier-6 code (`rlvr_v2/sgac/phase1.py`). | ≈ 2.6 GPU-h |
  | Extraction / equivalence / truncation / format reward / D | Covered by E0 G2/G3 and the 337 SGAC tests. Add explicit tests for: truncated completion ⇒ correct = 0 and the format-bonus rule as implemented; last-box extraction; unparsable answers forming one D class; ties in D. | CPU |

  ## 2. Data roles and leakage prevention

  There are four roles. They are pairwise disjoint, by exact id *and* by near-duplicate at **≥ 0.8**. That is conservative: the prereg used 0.9. All 0.8–0.9 pairs are listed in the manifests with their scores.

  | Role | Source | Exclusions | Size |
  |---|---|---|---|
  | **Final evaluation** | MATH-500 (`manifests/math500.json`) + test50 (`configs/sgac/manifests/data.json`) | — | 500 + 50 |
  | **Dev set** (calibration labels only) | prereg `heldout.json` | test50 exact ids (3) and near-dups; MATH-500 near-dups | **486** |
  | **Calibration candidates** | prereg `pool.json` (K=32 base sieve already exists) | ineligible gold (1); test50 (5); MATH-500 (12); dev near-dups (3 items); SGAC-pool exact (49) and near-dups (13); the 22 Study-1 candidates, which keeps prereg Study 1 independent | **408** eligible |
  | **Curriculum pool** | historical SGAC pool (1000) | near-dups of MATH-500 or test50 (36) | **964** |

  **Candidate–dev exclusion (added 2026-10-12, before any label).** A candidate that near-duplicates a dev item would inflate its own label, because training on it improves that dev item. Three such candidates are excluded (similarity 0.83–0.94).

  All four roles, with every exclusion and its similarity score, are in `configs/sgac_calib/manifests/data_roles.json` (built by `python -m rlvr_v2.sgac_calib build-manifests`; `--check` rebuilds and compares).

  **Rules.**
  - MATH-500 and test50 are touched only by base evaluations and by the Stage E step-5/10/15/20 evaluations. They are never used for labels, model selection, gates or hyperparameters.
  - Selector train/validation splits are grouped by candidate. A candidate's seeds and feature subsamples never straddle folds.
  - Calibration candidates and the curriculum pool are disjoint, so the deployed selector only ever scores unseen problems.
  - `assert_disjoint` plus a near-duplicate scan run as a test and at job start. A job refuses to run if a manifest no longer reproduces.

  ## 3. Stage B — calibration features (no new sieve)

  **Feature source.**
  - The prereg sieve stores 32 base-policy rollouts per pool item, each with its text, `correct`, `boxed`, `answer_class` and `truncated`.
  - Its generation settings equal the SGAC e0 sieve's: the same prompt builder, T 1.0, top_p 1.0, top_k 0, 3072 tokens and the same stops.
  - **Checked 2026-10-12 for all 48 candidates.** Every stored prompt hash equals the prompt the SGAC e0 sieve renders. Regrading the 32 stored rollouts reproduces the stored sieve signals exactly. No re-sieve is needed.
  - The output is `configs/sgac_calib/manifests/features.json`, which pins the sieve files' sha256. On the workstation, `check-features` verifies the local sieve files against those hashes.

  **Deployment-matched features.** SGAC deploys on **K=4** rollouts, so the selector is trained on K=4 features:
  - Ps; Var = population variance of (correct + 0.5·boxed); D = math-verify classes / K; L.
  - They are computed with the deployed SGAC e0 signal code on 16 random K=4 subsamples per candidate (RNG `default_rng([20261011, index])`). Each subsample is regraded and re-clustered on its four answers alone, as at deployment. The definitions therefore match by construction.
  - K=32 features are a sensitivity analysis only.

  **Known limitation.** In the curriculum, signals come from the *current* policy, while calibration uses the base policy. The drift is logged (per-step signals of every candidate) but not corrected.

  ## 4. Stage C — calibration labels

  ### Protocol C1: one fixed one-shot RLVR protocol for every candidate

  **Training.**
  - Fresh LoRA (r16/α32, q/k/v/o), then **20 SGAC bursts on the single candidate**, persisting the adapter across bursts.
  - Each burst is exactly the e0 burst: fresh trainer, 5 single-completion updates from 2 rounds of G=4, lr 2e-5 linear, reward correct + 0.5·boxed, 3072 tokens, β 0, dapo, `max_grad_norm` 4.0.
  - That is 100 updates, the same budget as one full curriculum run.
  - It is the reproduction's `fixed_pi1` arm with the candidate substituted, so the code already exists.

  **Label.** ΔA = acc_dev(trained) − acc_dev(base) on the 486-item dev set (greedy, 3072 tokens, `oneshot_rlvr_chat`, math-verify). Per-item outcomes are stored.
  - **Base reference.** The base model is evaluated on the dev set twice. The first evaluation is the reference for every ΔA.
  - **Determinism check (E0 G5).** The two evaluations must agree on ≥ 98% of item verdicts, with accuracies within 0.5 pp. Otherwise the pilot queue stops.
  - **Adapters.** A calibration run keeps only its final adapter and its resume point. The previous adapter is deleted only after the next step is committed, so a crash always resumes from an existing adapter.
  - **Evaluation sets.** A calibration run is never given MATH-500 or test50, and it evaluates nothing but the dev set at its final step. The CPU smoke asserts this.

  **Seeds.** Two training seeds per candidate: 7000 + i and 17000 + i. The label is their mean.

  **Logged, not used as labels.** Training reward per burst, zero-advantage bursts, clipped ratio, ‖ΔW‖, and wall-clock per phase.

  **Why C1.**
  - It uses the deployment recipe exactly. The only difference is "20 bursts on x" instead of "1 burst on x among 19 others".
  - A single burst cannot be labelled by accuracy: a whole 20-burst curriculum moved MATH-500 by +0.9 pp [−1.4, +3.1].
  - Whether C1-utility ranks problems the way within-curriculum utility does is exactly what Stage E tests.

  **Alternative considered and not chosen (team decision, 2026-10-12).** The prereg Study-1 regime (G=64 × 100 rounds). It gives a measurable effect (π1 +5.2 pp) but is a different regime from deployment, and costs ≈ 3.9 h per run (≈ 375 GPU-h for N=48 × 2 seeds).

  ### Sampling (outcome-blind; fixed before any label exists)

  - **Strata.** The 408 eligible candidates are stratified by K=32 base Ps into four bins:

    | Ps bin | Eligible items | Levels present |
    |---|---|---|
    | {0} | 37 | 2–5 (29 at L5) |
    | (0, 0.4] | 127 | 2–5 |
    | (0.4, 0.8] | 194 | 1–5 |
    | (0.8, 1] | 50 | 1–5 |

    No eligible item has Ps = 1 at K=32.
  - **Draw.** RNG `default_rng(20261010)`. Within each bin, levels are cycled from 5 down to 1 among the levels present, with a uniform draw within each level.
    - Pilot = the first 4 per bin (16).
    - Expansion = the next 8 per bin (32).
    - Full N = **48** (12 per bin).
  - **Manifest.** The ordered list is committed before the pilot as `configs/sgac_calib/manifests/candidates.json`. It was drawn 2026-10-12 from base-policy signals and levels only. Candidate index i (0–15 pilot, 16–47 expansion) fixes the training seeds 7000 + i and 17000 + i. The pilot holds levels 5→2 in every bin; level-1 items enter in the expansion, as the cycle order implies.
  - **Positive control.** π1 (Wang et al.) under C1 × 2 seeds. It is a diagnostic only and never a selector training row.

  ### Pilot gate G-P (looks only at label reliability, never at features vs labels)

  - **G-P1.** ICC(1,1) of single-seed ΔA across the 16 pilot candidates ≥ 0.30 (one-way random effects; 95% CI reported).
  - **G-P2.** The SD across candidates of the two-seed mean ΔA ≥ 1.0 pp.

  **Both pass** ⇒ run the expansion. The pilot's 16 candidates stay in the final 48: the gate never looked at the feature–label relationship, so no selection bias enters.

  **Either fails** ⇒ stop and write the diagnosis:
  - ΔA distribution and ICC with CI;
  - the π1 control;
  - training diagnostics, e.g. zero-advantage bursts and reward saturation.

  The team then chooses one of two options:
  - (i) End with a negative calibration result: labels are not measurable in this regime.
  - (ii) Escalation rung **C2** = 80 bursts per candidate (400 updates, ≈ 1.9 GPU-h per run). It reruns the same 16 × 2 pilot (≈ 65 GPU-h). C2 is declared here so it is not invented after the fact.

  **Power at N = 48 (two-sided α = 0.05, 80% power).**
  - The minimum detectable observed |ρ| is ≈ 0.39.
  - With label reliability R, the true ρ must be ≥ 0.39/√R. For example, at R = 0.5 it must be ≥ 0.55.

  ## 5. Stage D — selector training, validation and model selection

  **Data.** 48 candidates × 16 K=4 feature subsamples. Each row's label is the candidate's mean ΔA, with weight 1/16 so each candidate counts once.

  **Models.**
  1. OLS on [Ps, Var, D, L] + intercept.
  2. Ridge on standardised features. α is chosen by inner grouped 5-fold CV over a fixed grid 10^{−3…3} (13 values).
  3. Gradient boosting (exploratory at this N), with fixed and untuned settings: depth 2, 200 trees, lr 0.05, subsample 0.8. Each leaf must hold at least 5 *candidates* (80 rows). Every candidate contributes 16 rows with one label, so a 5-row leaf could isolate a single candidate and memorise it.
  4. OLS with pairwise interactions (exploratory).

  **Baselines, scored on the same folds.**
  - Intercept-only.
  - L only (difficulty).
  - Ps + Ps² (success probability, quadratic).
  - Eq. 10 as run.
  - The max-Var, max-D and max-L rules.
  - Random.

  **Validation.** Repeated grouped 6-fold CV (8 candidates per fold, stratified by Ps bin, 100 repeats). LOO is a sensitivity analysis.

  **Metrics.** All are out-of-fold: each candidate's prediction comes from models that never saw it, averaged over the 100 repeats. CIs come from a candidate bootstrap (2,000).
  - **Spearman ρ (primary).** At deployment the selector sees *one* K=4 sample per candidate, so ρ is the mean over subsample draws m of Spearman(pred[:, m], ΔA). The ρ of subsample-averaged scores is reported as secondary.
  - **Permutation p-value.** One-sided. Candidate labels are shuffled 1,000 times and the whole pipeline (outer CV with inner α selection) is rerun with 10 CV repeats per shuffle. The observed statistic uses the same 10 repeats.
  - **R²_oos and MAE** against the intercept-only model's out-of-fold predictions.
  - **Top-1-of-4 gain.** E[ΔA of the pick] − E[ΔA of a uniform pick], over 20,000 Monte-Carlo batches of 4 candidates. Each candidate shows one random K=4 sample, batch order is random, and ties go to the first maximum (the loop's rule). This is the deployment decision. Batches are importance-weighted to the prereg pool's K=32 Ps-bin mix (0.088 / 0.314 / 0.480 / 0.118). The oracle gain is reported alongside.

  **Reported questions.**
  - Coefficients and intercept, with bootstrap CIs and the percentage of resamples with the same sign.
  - VIFs and the condition number. Var and D are near-functions of Ps at K=4, so the coefficients are expected to be unstable. Model selection does not depend on any coefficient's sign.
  - Signals beyond difficulty: full vs L-only and full vs Ps-only, using the paired bootstrap of the out-of-fold loss difference.
  - Nonlinearity and interactions are "material" only if they raise R²_oos by ≥ 0.05 with a CI excluding 0.

  **Model choice (pre-declared).**
  - Ridge by default.
  - GBM replaces it only if its out-of-fold ρ beats Ridge's by more than 1 SE *and* its R²_oos is higher. The SE is the candidate-bootstrap SD of the paired difference in ρ.
  - OLS is reported but not chosen, because Ridge's α grid contains the near-OLS limit.

  **Freeze.**
  - The chosen model is refit on all 48 candidates. Ridge's α comes from 10 repeats of inner grouped CV on all candidates.
  - It is serialised as `configs/sgac_calib/manifests/selector.json`: feature order and definitions, K, scaler mean/SD, standardised and raw coefficients, intercept, α, model class, sha256 of the training table, calib hash and code commit. A GBM is additionally pickled, with its sha256 and the sklearn version recorded.
  - `selector_pin.json` holds the JSON's sha256. The learned arm refuses to load a selector whose hash differs.
  - Deployment uses argmax over the 4 candidates, first maximum on ties, as in every other arm.

  **Selector gate G-S** (before any Stage-E run). Proceed only if both hold:
  - the chosen model's out-of-fold ρ ≥ 0.30 with permutation p < 0.05;
  - its top-1-of-4 gain over Random > 0 with a CI excluding 0.

  Otherwise no learned-selector curriculum runs; the null calibration result is reported and the team decides.

  ## 6. Stage E — the full 20-step curriculum

  **Substrate.**
  - The e0 spec is unchanged (hash `577f3319`, enforced); only its results root moves to `results_sgac_calib/curriculum`. Every arm runs the frozen loop `rlvr_v2.sgac.loop.SgacRun`. What differs:
    - the clean 964-item pool (rows kept, so the loop is unchanged);
    - new frozen schedules `configs/sgac_calib/manifests/batches_seed{42..49}.json` (`random.Random(seed).sample(remaining, 4)`, all four removed per step, identical across arms). They are written 40 steps long; every arm reads its own prefix, which is identical to a 20-step schedule;
    - the `learned` arm. It reuses the loop's selection step (sieve, signals, every rule's counterfactual pick) and replaces the pick with the frozen selector's argmax.
  - Everything else is the historical recipe: B=4, K=4, T=1.0, 20 steps, 5-update bursts, persistent LoRA, and the same reward, grader, caps and stops.

  **Arms and seeds.**

  | Arm | Role | Seeds |
  |---|---|---|
  | Learned selector | confirmatory | **42–49** (8 seeds, fixed by the §7 rule) |
  | Random | confirmatory comparator | 42–49 |
  | SGAC, Eq. 10 as run | confirmatory comparator | 42–49 |
  | Max variance, Max disagreement, Max difficulty | secondary | 42–44 |
  | Random, time-matched (`random_tm`: no sieve, more steps, §8; its first 20 picks equal Random's) | exploratory | 42–44 |
  | Base | reference (eval only) | — |

  **Evaluation.** Each candidate's signals and every rule's counterfactual pick are logged at every step.
  - MATH-500 at step 0 (shared base) and at step 20 for every run.
  - test50 at steps 0/5/10/15/20.
  - Greedy, 3072 tokens, same prompt and grader. Per-item outcomes are kept under both graders.

  **Queue (pre-declared; `python -m rlvr_v2.sgac_calib jobs --stage curriculum`).**
  1. The G-S guard, then the base evaluation (MATH-500 + test50, with the E0 G1 agreement check).
  2. Seeds 42–44, all six arms.
  3. Seeds 45–49, confirmatory arms.
  4. The confirmatory analysis.
  5. Time-matched Random (stage `tm`).

  There is no early stopping and no change to the selector or protocol mid-run.

  ## 7. Statistical analysis

  **Confirmatory endpoints** (MATH-500, step 20).
  - H1: Learned − Random.
  - H2: Learned − SGAC (Eq. 10).

  **Method (primary): crossed random-effects analysis.**
  - Seeds and MATH-500 items are both random factors, and they are *crossed*: every run answers the same 500 items.
  - For each seed s and item i, the paired difference is e_si = y_Learned − y_X. Pairing is by seed because batches are shared.
  - A two-way random-effects ANOVA (one observation per cell) gives Var̂(mean) = (MS_seed + MS_item − MS_res) / (S·N).
  - The test is a t-test with Satterthwaite degrees of freedom. Both 95% and 90% CIs are reported.
  - **Calibration of the method** (simulation, `tests/test_sgac_calib_crossed.py`). The SE matches the true sampling SD within 8%.
    - 95% CI coverage is 95–97%: nominal when a seed component is present, mildly conservative (≈ 97% at 6 seeds) when it is absent, and never below nominal.
    - Coverage holds for paired 0/1 outcomes as well as Gaussian ones.
  - Implementation: `rlvr_v2/sgac_calib/crossed.py`, `analysis.py`.

  **Why not a bootstrap.** Each simple bootstrap is wrong in one direction:
  - The reproduction's item bootstrap on seed-averaged correctness ignores seed-to-seed variance, so it is anticonservative.
  - Resampling seeds and then items counts the run-to-run noise about three times, so it is needlessly conservative.

  **Sensitivity analyses**, reported next to the primary:
  - (a) Pigeonhole bootstrap: seeds and items resampled independently, 10,000 resamples, seed 0.
  - (b) Item bootstrap on seed-averaged correctness, for continuity with the reproduction.

  A claim in the paper needs the primary analysis. Any disagreement between the methods is reported.

  **Multiplicity.** Holm correction across H1 and H2. "Outperforms both" requires both to be rejected under Holm.

  **Decision rule.**
  - **"Learned beats X"** requires all three:
    - Holm-adjusted p < 0.05;
    - point estimate ≥ +1.0 pp;
    - a positive difference in at least 5/6 of the seeds, rounded up (5 of 6, 7 of 8, 9 of 10).
  - **"Worse than X"** mirrors the same rule.
  - **"Equivalent within ±2 pp"** requires the 90% CI ⊂ [−2, +2] (TOST).
  - Otherwise the result is **"inconclusive"**.
  - No winner is declared from one seed or from a < 1 pp difference.

  **Power, stated honestly.**
  - In the reproduction, any two trained runs disagreed on 6–11% of MATH-500 items. That held for two seeds of one arm as much as for two arms at the same seed. This run-to-run noise sets the detectable difference.
  - All historical arms were within 1.1 pp of Base, so "inconclusive" is a realistic outcome. The ±2 pp TOST makes a null informative.

  **The seed count is fixed before freezing**, from historical data only. No outcome of this track exists yet.
  - **Input.** Step-20 MATH-500 per-item outcomes of the four e0 arms that completed seeds 42–44 (SGAC, Random, Max variance, Max disagreement), as six arm pairs on shared batches. The arm × seed variance is estimated jointly from all four arms (6 df rather than the 2 df from a single pair).
  - **Projection.** Project the minimum detectable difference (MDE) at S = 6, 8 and 10. Use the primary analysis, α = 0.025 two-sided (the Holm threshold the smaller p-value must meet for "beats both") and 80% power.
  - **Rule.** Use the smallest S ≥ 6 with MDE ≤ 2.0 pp, capped at 10. If 10 seeds still give MDE > 2.0 pp, use 10 and report the achieved MDE. Each extra seed costs ≈ 3.9 GPU-h across the three confirmatory arms.
  - **Result (2026-10-12): 8 seeds, 42–49.**
    - Command: `python -m rlvr_v2.sgac_calib seed-count --dirs sgac_step20_results,sgac_step20_extra`; output `configs/sgac_calib/manifests/seed_count.json` with input hashes.
    - The 12 input files match the report's per-seed accuracies exactly under both graders. Their hashes are in `paper/sgac_repro_freeze.md`.
    - Joint components from 4 arms × 3 seeds × 500 items:
      - MS_arm×seed = 0.0213 (6 df), below MS_residual = 0.0463, so σ_seed = 0. Its 80% upper bound is also 0.
      - For a paired difference: σ_item = 0.030 and σ_residual = 0.304 (proportion scale).
      - The arms do not swing differently from seed to seed; item-level noise dominates.

    | Seeds | 5 | 6 | 7 | 8 | 9 | 10 |
    |---|---|---|---|---|---|---|
    | MDE (pp), α 0.025, 80% power | 2.67 | 2.26 | 2.004 | **1.82** | 1.68 | 1.57 |

    - 7 seeds misses the 2.0 pp threshold by 0.004 pp, so the rule gives 8.
    - Earlier draft figures (2.22 / 1.83 / 1.61 pp, from SGAC and Random alone) are superseded.

  **Secondary and descriptive** (no multiplicity claims).
  - All other pairwise comparisons and vs Base.
  - test50, which is underpowered: 1 item = 2 pp.
  - Trajectories.
  - Pick distributions (level, Ps, D) and agreement with Eq. 10 and Max-difficulty on identical batches.
  - Zero-advantage bursts, grader sensitivity, and how far signals drift from base to current policy.

  ## 8. Compute accounting and efficiency

  **Logging.** Every job logs GPU-seconds per phase: sieve, training, evaluation, fitting.

  **Report tables.**
  - **One-time calibration cost.** Stage A + dev base evals + pilot + expansion + fitting. Reported separately from deployment.
  - **Per-deployment cost per arm.** Sieve, training and evaluation. Random's deployment cost excludes the sieve, which it only runs to log counterfactuals.
  - **Matched RL budget (primary).** Every arm makes exactly 100 updates.
  - **Matched deployment GPU-time (exploratory).** Random-TM runs round(20 × (t_sieve + t_burst) / t_burst) steps, capped at 40 (the schedule length). The step count is computed from T0-smoke timings and written to `results_sgac_calib/t0/tm_steps.json` before any curriculum outcome exists.
  - **Matched total GPU-time including calibration.** This is not achievable: calibration (≈ 90 GPU-h) is ≈ 70 curriculum runs. Instead, report the ratio, "GPU-h per pp gained", and the number of deployments needed to amortise calibration. If the gain is ≤ 0, break-even is undefined.

  ## 9. Budget — GPU-hours and independent RL runs (one RTX PRO 4500)

  Per-run rates:
  - Curriculum run ≈ 1.3 h, measured in the reproduction.
  - MATH-500 eval ≈ 0.53 h, measured.
  - Calibration run ≈ 0.9 h, estimated (≈ 0.35 h for 20 bursts + 0.52 h dev eval). The T0 smoke measures it; if it exceeds 1.1 h, the plan is re-costed *before* the pilot.

  | Stage | RL runs | GPU-h |
  |---|---|---|
  | T0 smoke + timing projection | — | 0.8 |
  | A: Phase-1 signal recomputation | — | 0.3 |
  | A: Phase-1 replication (optional) | 4 | 2.6 |
  | B: base dev eval ×2 (determinism, G5-style) | — | 1.1 |
  | C: pilot, 16 × 2 seeds + π1 × 2 | 34 | 31 |
  | C: expansion, 32 × 2 seeds (if G-P passes) | 64 | 58 |
  | D: fitting, CV, bootstrap | — | CPU |
  | E: confirmatory, 3 arms × 8 seeds | 24 | 31.2 |
  | E: secondary, 3 arms × 3 seeds | 9 | 11.7 |
  | E: base MATH-500 + test50 | — | 0.6 |
  | E: Random-TM, 3 seeds (exploratory) | 3 | ≈ 4.5 |
  | **All gates pass** | **131 core (138 with optional)** | **≈ 135 core (≈ 142 with optional) ≈ 5.6–5.9 days** |
  | **Stops at G-P** | **34 (38)** | **≈ 33 (36)** |

  **Order of work.** The order follows the gates, not the calendar. Each step starts only when the previous gate has been decided.

  Durations assume one dedicated GPU, with each job launched as soon as the previous one ends. Add a 20–30% buffer for reruns.

  | Step | Work | Approx. duration | Ends with |
  |---|---|---|---|
  | 0 | Historical freeze committed (done) | — | tag `sgac-repro-frozen` |
  | 1 | Implementation, CPU tests, seed count (done, branch `sgac-calib`); team review | — | tag `sgac-calib-protocol-v1` |
  | 2 | `STAGES=t0`, then `audit`, then `pilot` | ≈ 33 GPU-h | G-P |
  | 3 | `STAGES=expansion` | ≈ 58 GPU-h | — |
  | 4 | `STAGES=selector` (CPU) | hours | G-S |
  | 5 | `STAGES=curriculum`, then `tm` | ≈ 44 GPU-h + ≈ 4.5 | — |
  | 6 | Report | — | verdict |

  Every stage runs with `STAGES=<stage> scripts/local/sgac_calib.sh`. Its gates stop the queue with exit 3.

  ## 10. Never done

  - Labels, model selection or gates computed on MATH-500 or test50.
  - Refitting or retuning the selector after any Stage-E outcome.
  - Changing C1, the gates, the arms, the seeds or the schedules after seeing labels.
  - Dropping candidates post hoc. A crash or NaN is rerun with the same seed; clipped ratio > 20% is flagged, never dropped.
  - Early stopping on accuracy.
  - Editing the historical reproduction's code, configs, results or verdict.

  ## 11. Decisions

  **Decided 2026-10-12.**
  - Pilot first: G-P gates the expansion.
  - Near-duplicate threshold 0.8.
  - At least 6 seeds for the confirmatory arms. The final count comes from the §7 simulation.
  - The historical reproduction is frozen as it stands. Incomplete runs are documented, not finished.
  - **Seed count, by the §7 rule:** 8 (42–49).

  **Decided 2026-10-12, second round.**
  - Calibration regime C1: the deployment-matched 100-update protocol of §4.
  - Prereg Study 1 stays separate from this track. Whether it runs is decided after the calibration results. If it is not run, that is recorded as a dated deviation in `paper/prereg.md` §10.

  **Before freezing.**
  - Done: the seed count (8) and the historical freeze, with archive hashes.
  - Done 2026-10-12: team review of this document and the implementation (branch `sgac-calib`); approved, including the choices below and the 8-seed plan, subject to the gates and decision rules. Tagged `sgac-calib-protocol-v1`. No GPU job of this track ran before the tag.

  **Choices made while implementing (2026-10-12, before any label).** Each one only makes precise something the draft left open:
  - The candidate–dev near-duplicate exclusion (§2).
  - ρ computed on single K=4 draws (§5).
  - GBM's minimum leaf counted in candidates (§5).
  - 10 CV repeats per permutation shuffle (§5).
  - Monte-Carlo top-1 batches with the loop's tie rule (§5).
  - The crossed-design primary analysis (§7).
  - 40-step schedules read by prefix (§6).
  - The base dev reference (the first of the two evaluations), the determinism check, and the calibration runs' adapter keeping (§4).
  - The guard baseline for the historical code is `sgac-repro-frozen`, not `sgac-protocol-v1`, because pre-run fixes landed after v1 (§0).

  ## 12. Implementation (branch `sgac-calib`)

  | Piece | Where |
  |---|---|
  | Config | `configs/sgac_calib/calib.yaml` (calib hash via `show-spec`) |
  | Committed manifests | `configs/sgac_calib/manifests/`: `data_roles.json`, `candidates.json`, `features.json`, `seed_count.json`, `batches_seed{42..49}.json`; later `selector.json` + `selector_pin.json` |
  | Code | `rlvr_v2/sgac_calib/`: `spec`, `manifests`, `features`, `calib` (C1, labels, G-P), `selector` (Stage D, G-S, freeze), `curriculum` (learned arm, `random_tm`), `crossed` + `analysis` (§7), `seedcount`, `audit`, `jobs`, `smoke`, `cli` |
  | Launcher | `scripts/local/sgac_calib.sh` (`STAGES=t0,audit,pilot,expansion,selector,curriculum,tm`) |
  | Tests | `tests/test_sgac_calib_*.py`: frozen guards, manifests, features, crossed estimator, selector, gates and analysis, CPU smoke (`-m smoke`) |
  | Results | `results_sgac_calib/` (git-ignored) |

  ## 13. Amendments (dated, append-only)

  _None yet._
