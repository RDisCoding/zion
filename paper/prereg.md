# Pre-registration: Do cheap rollout signals predict which single example to train on in 1-shot RLVR?

**Status:** DRAFT (to be frozen and git-tagged `prereg-v1` after the Day-1 throughput benchmark, before the first Study-1 job is queued). Any later change is recorded in the Deviations log at the end, never edited in place.

**Frozen on:** _pending_ · **Git tag:** _pending_ · **Config hashes:** _pending (results/e0/gates.json, configs/study1.yaml)_

## 1. Question and scope
Model: Qwen2.5-Math-1.5B (bf16 base weights + LoRA r=32, α=64, all linear modules). Algorithm: GRPO (TRL), one prompt per optimizer step with G = 64 completions, correctness-only reward. Data: candidate problems from the genuine MATH train split only (`nlile/hendrycks-MATH-benchmark` rows whose `unique_id` starts with `train/`; that split also packages 4,501 non-MATH-500 test problems, which are excluded; 2 rows without an `answer` value are dropped, leaving 7,497). Shuffle seed 20260101; near-duplicates of MATH-500 (difflib ratio ≥ 0.9 after a token-Jaccard prefilter; 29 found) are removed; pool = first 500, held-out = next 500 (`manifests/pool.json`, `manifests/heldout.json`, committed with text hashes). Evaluation on `HuggingFaceH4/MATH-500` only (`manifests/math500.json`). Question: which pre-training statistics of a candidate problem (from K = 32 base-model rollouts) predict its downstream transfer Δ after 1-shot GRPO, and does any selection rule beat random in a short sequential curriculum. The study is designed to return a clean answer either way; a null is a reportable result.

## 2. Identifiability statements (reported in the paper)
- For binary rewards, within-group reward variance is V = P̂_s(1−P̂_s), a deterministic function of P̂_s; "variance vs success probability" is not a separable hypothesis, and max-variance selection ≡ "P̂_s nearest 0.5".
- Wang et al. (2025) rank examples by the *historical variance of per-epoch training accuracy during full-dataset RLVR*; that criterion is not tested here. We test within-group rollout statistics of the current policy only.
- Unconditional "disagreement" (unique answers / K) is strongly collinear with 1−P_s (pilot Spearman −0.84 at K = 8). The identifiable disagreement hypothesis is wrong-answer diversity conditional on P_s.

## 3. Hypotheses, endpoints, thresholds (two-sided, α = 0.05; H1 and H2 are primary and Holm-corrected)
Endpoint per training run: Δ = acc(MATH-500, greedy, after training) − acc(base), per-item paired bootstrap CI.
- **H1 (learnable zone / variance):** E[Δ | P_s] is hump-shaped. Tests: Spearman ρ(V, Δ) with bootstrap CI; coefficient of centred P_s² in Δ ~ P_s + P_s²; mid-bin (0.25, 0.75) vs extreme bins (Mann–Whitney). Outcome map: ρ(V, Δ) > 0 with CI excluding 0 ⇒ variance IS a useful signal (the old paper's claim is falsified); CI including 0 and |ρ| < 0.2 ⇒ no usable signal; ρ < 0 with CI excluding 0 ⇒ old direction supported.
- **H2 (disagreement beyond P_s):** matched pairs (same level, |ΔP̂_s| ≤ 2/32, high vs low wrong-answer diversity D_wrong): mean(Δ_high − Δ_low) with paired Wilcoxon and bootstrap CI; partial Spearman of D_simpson, entropy, D_wrong with Δ controlling for P_s and P_s². Confirmation requires the pair-difference CI to exclude 0 AND |mean difference| ≥ 1.5 points.
- **H3 (learned selector):** leave-one-out predicted score correlates with Δ: Spearman ≥ 0.30 with permutation p < 0.05 (1,000 label shuffles); the LOO R² gain over a P_s-only model (features P_s and V_bin = P_s − P_s², which span the same quadratic) is reported alongside. Ridge is the primary model (α by LOO MSE over a fixed grid); Lasso is a sensitivity analysis. Selector features: P_s, V_bin, D_simpson, D_wrong (mean-imputed), entropy, level, mean length.
- **H4 (curriculum):** at T = 20 the best pre-specified arm beats random on MATH-500 by ≥ 1.5 points with item-bootstrap CI excluding 0 (also AUC over checkpoints).
- **Nulls:** reported with CIs. Equivalence (TOST, Fisher z, 90% interval): Study 1 uses bound **|ρ| ≤ 0.4**, because with n ≈ 32 the 90% half-width is ≈ 0.30, so a bound of 0.2 could never be established at this N (it would need n ≳ 70 even at ρ̂ = 0); |ρ| < 0.2 remains the *descriptive* label "no usable signal" in the H1 outcome map, not an equivalence claim. Study 2 uses ±2 points.
- **Binning convention:** every P̂_s bin is half-open on the left, (lo, hi]; with K = 32, 8/32 falls in the first bin and 24/32 in the second. Anchors count as extremes in the hump test.

## 4. Signal measurement (K = 32 for Study 1, K = 16 online in Study 2)
From K rollouts at T = 1.0, top_p = 1.0, max 3072 new tokens (2048 if E0 shows < 2% truncation), same prompt as training: P_s; V_bin = P_s(1−P_s); V_legacy (variance of correct + 0.5·format, reported once for continuity, never used for selection); U = unique classes/K (legacy, K-dependent, reported only); **D_simpson = K/(K−1)(1 − Σ p̂_c²)** (primary); Shannon entropy with Miller–Madow correction; **D_wrong** = Simpson disagreement among parsable incorrect rollouts (defined when ≥ 4); majority share/margin/correctness; format, truncation and unparsable rates; token-length statistics; level, subject, prompt tokens. Answer classes are `math-verify` equivalence classes; unparsable outputs form one class. Any feature used by a selector must show K16-vs-K32 Spearman agreement ≥ 0.9 on the pool.

## 5. Study 1 design
- Pool sieve: 500 problems × K = 32 (1,000 if the benchmark allows).
- **N = 32 = 2 null anchors (P̂_s = 1) + 2 zero anchors (P̂_s = 0, parsable) + 14 matched pairs** (4 / 6 / 4 pairs in bins (0, 0.25], (0.25, 0.75], (0.75, 0.9]; same level; |ΔP̂_s| ≤ 2/32; D_wrong gap ≥ 0.4, relaxed to 0.3 then 0.2 only if a bin is short, with the relaxation logged in the manifest). Selection is deterministic and input-order invariant (`scripts/select_candidates.py`; RNG only for ties; manifest committed). Replicate seeds go to 8 candidates drawn round-robin across bins alternating high/low D_wrong.
- Training per candidate: fresh LoRA; R generation rounds × G = 64 (one optimizer step per round); lr constant; β = 0; loss "dapo"; group-scaled rewards; ε = 0.2; `mask_truncated_completions = true`; correctness-only reward. **Budget ladder fixed by the positive control (E0 G4):** {R = 100, lr 2e-5} → {R = 100, lr 5e-5} → {R = 200, lr 5e-5}; the first passing rung is used for every candidate and for Study 2. If the shard has < 20 GB free: G = 32 and 2048 tokens (recorded).
- Seeds: training seed = 1000 + candidate index; 8 candidates (spread across bins, alternating high/low) get a second seed (+100000) to estimate σ_seed.
- Evaluation: MATH-500, greedy, same prompt/grader; per-item outcomes stored; secondary: 500 held-out train items (must agree in sign with MATH-500 for any claimed effect).
- Power (80%, two-sided): N = 32 detects |ρ| ≥ 0.48; 14 pairs detect ≥ 3-point pair differences. Smaller effects are reported as undetermined.
- Analysis code (`analysis/study1_analysis.py`) is committed before the first Study-1 job runs.

## 6. Study 2 design (run only after Study 1 analysis; see stop/go)
T = 20 steps; each step: sieve B = 16 unused pool problems with the current policy at K = 16, select one by the arm rule, train 10 rounds × G = 64 (adapter persisted). Arms: random; max within-group variance (= P̂_s nearest 0.5); max D_simpson; learned selector (only if H3 passes); repeat-one control (π1 from Wang et al., 200 rounds) if budget allows. Seeds 1234, 2345, 3456. MATH-500 at steps 0, 4, 8, 12, 16, 20. Primary: chosen arm vs random at T = 20, paired item bootstrap (10,000) over seed-averaged correctness; secondary: AUC; mechanism diagnostics (selected P̂_s distribution, zero-std-group fraction).

## 7. Infrastructure gates (all must pass before any study job)
G1 base MATH-500 accuracy under `qwen_math_chat` and `oneshot_rlvr_chat` (greedy, 3072 tokens) with truncation and format rates; the better style is pinned. G2 ≥ 40 grader unit tests pass; cross-check against the Qwen2.5-Math grader on 200 base outputs (≤ 2% disagreement, adjudicated). G3 stop-token/truncation check on 32 pool problems × K = 8: truncation < 10%, secondary-stop share < 5%, no `<|im_start|>` in completions. G4 positive control: 1-shot GRPO on π1 (label 12.8) through the ladder; PASS = MATH-500 gain ≥ +3.0 points with paired-bootstrap CI excluding 0 and training reward ≥ 0.8 by round 100; failure at all rungs ⇒ re-plan, no Study 1. G5 eval determinism: two identical evals agree within 0.5 points and ≥ 98% item agreement. Day-1 throughput benchmark results are pasted here before freezing N.

## 8. Stop/go and exclusion rules
- Study-1 → learned arm: GO only if any single signal or the LOO selector reaches |ρ| ≥ 0.30 with CI excluding 0. H2 confirmed ⇒ add arm "max D_wrong within P̂_s ∈ [0.2, 0.8]".
- σ_seed > 3 points ⇒ 5 seeds in Study 2 before interpreting gaps < 3 points.
- Crash/NaN ⇒ rerun the same seed; `completions/clipped_ratio` > 20% ⇒ flag, never drop; no candidate is excluded post hoc.
- Deadline rule: if Study 1 is < 75% complete on 9 Oct 2026, the ARR October cycle is skipped; nothing unsupported is submitted.

## 9. Confounds acknowledged in advance
P_s–disagreement collinearity (pairs match P̂_s and level, not content); MATH-500 used for Study-1 labels and the Study-2 endpoint (held-out train items as a sign check); small-budget regime (~6,400 rollouts per candidate, LoRA, β = 0, no entropy bonus; ~300× below Wang et al.), calibrated only by the π1 gate; online policy drift of sieve signals in Study 2; probable pre-training exposure to MATH train; batched-greedy nondeterminism bounded by G5; single model family.

## 10. Deviations log
_(append-only; date, what changed, why, which results it affects)_
