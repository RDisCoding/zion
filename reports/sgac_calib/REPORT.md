# SGAC calibration track — report

_Generated 2026-10-15T08:02:06Z by `python -m rlvr_v2.sgac_calib report` from `sgac_results_for_review/results_sgac_calib`. Protocol: `paper/sgac_calib_protocol.md` (tag `sgac-calib-protocol-v1`; amendments A1–A5 in §13)._

## 1. Conclusion so far

- Fitted equations: E1_c1_pilot; validated by the G-S thresholds: none.
- The deployment-dose (C1) target has no problem-specific component (§3), so the intended C1 equation is unidentifiable; the one-shot-regime equation (E3_os) is the closest defensible alternative (amendment A3).
- A fitted equation is not a validated selector, and a validated selector is not a demonstrated improvement; each claim below names which of the three it is.

## 2. The equations

- **Original SGAC (Eq. 10, as run):** `score = 0.005·Ps + 0.1832·Var − 0.0751·D + 0.2188·L` — no intercept, argmax of 4; the N = 20 phase-4 fit read with labels shifted by one feature (`paper/sgac_repro_protocol.md` §2).
- **Intended new selector:** `score = b0 + b1·Ps + b2·Var + b3·D + b4·L`, predicting dev ΔA (pp); argmax of 4, first maximum on ties. Ridge on standardised features, α by inner grouped CV over 10^−3…10^3 (13 values), squared-error loss, closed-form solution; GBM / OLS / interactions compared (protocol §5).
- **Features** (from K = 4 rollouts of the current policy, `rlvr_v2.sgac.sieve.selection_signals`): Ps = share correct (math-verify, last box, truncated = wrong), in {0, .25, .5, .75, 1}; Var = population variance of (correct + 0.5·boxed), in [0, 0.5625]; D = distinct math-verify answer classes / 4 (unparsable = one class), in {.25, .5, .75, 1}; L = MATH level 1–5. Training rows: 16 random K = 4 subsamples of each candidate's 32 stored base rollouts. The coefficients below are on these raw units (the standardisation is folded in).

### E1_c1_pilot — C1 pilot labels (16 candidates, deployment dose)

N = 16 candidates; label mean +0.270 pp, SD 0.752 pp. Chosen model: **ridge**.

**Fitted equation (Ridge, α = 1000.0):** `score = 0.350127 + 0.0643933*Ps + 0.0999404*Var - 0.0634632*D - 0.0233164*L   (predicted dev ΔA, pp)`

| Term | Ridge coefficient (raw units) | 95% CI (candidate bootstrap) | Sign agreement | OLS coefficient |
|---|---|---|---|---|
| intercept | +0.350127 | — | — | +0.241016 |
| Ps | +0.064393 | [-0.0685, +0.2019] | 0.82 | +0.484609 |
| Var | +0.099940 | [-0.1474, +0.3344] | 0.79 | +0.309175 |
| D | -0.063463 | [-0.2498, +0.1085] | 0.74 | +0.260120 |
| L | -0.023316 | [-0.0958, +0.0476] | 0.72 | -0.116935 |

**Out-of-fold validation** (repeated grouped CV; 95% CIs by candidate bootstrap):

| Model / rule | Spearman ρ | R²_oos | Top-1-of-4 gain over random (pp) | Oracle gain (pp) |
|---|---|---|---|---|
| ridge | -0.809 [-0.93, -0.51] | -0.085 | -0.646 [-0.85, -0.33] | +0.765 |
| ols | -0.474 [-0.81, +0.05] | -0.294 | -0.112 [-0.53, +0.24] | +0.761 |
| gbm | -0.561 [-0.82, -0.14] | -0.140 | -0.323 [-0.60, -0.05] | +0.761 |
| ols_int | -0.428 [-0.71, -0.00] | -0.618 | -0.403 [-0.61, -0.11] | +0.758 |
| mean | -0.964 [-0.99, -0.86] | +0.000 | -0.831 [-0.97, -0.48] | +0.756 |
| level | -0.522 [-0.84, +0.04] | -0.170 | -0.095 [-0.65, +0.31] | +0.762 |
| ps_quad | -0.566 [-0.75, -0.23] | -0.117 | -0.497 [-0.67, -0.21] | +0.760 |
| rule: eq10 | -0.109 | — | -0.089 | +0.760 |
| rule: max_var | +0.035 | — | +0.049 | +0.759 |
| rule: max_d | -0.098 | — | -0.029 | +0.759 |
| rule: max_level | -0.146 | — | -0.082 | +0.757 |
| rule: random | — | — | -0.097 | +0.760 |

Gate G-S thresholds: ρ ≥ 0.30 (not met), permutation p < 0.05 (p = 0.755, not met), top-1 gain CI above 0 (not met) → **NOT validated**.

### E2_c2 — C2 labels (16 candidates, 4x dose)

Pending: its labels do not exist yet.

### E3_os — one-shot-regime labels (prereg Study-1 trainer)

Pending: its labels do not exist yet.

## 3. Label reliability and why

- **C1 (G-P):** fail — ICC(1,1) 0.147 [-0.35, +0.58], between-candidate SD 0.75 pp, mean ΔA +0.27 pp, π1 ΔA +1.54 pp (thresholds: ICC ≥ 0.30 and SD ≥ 1.0 pp).
- **C2 (G-P):** pending.
- **OS (G-P thresholds):** pending.

- **C1:** 34 runs. Problem specificity: per-item Δ correlation within +0.473 vs between +0.466 candidates (permutation p 0.293). σ_between 0.38 pp, σ_seed 0.92 pp; best possible top-1-of-4 gain 0.39 pp. Fragile items (flip rate > 25%): 76.
  Dose: corr(effective updates, |ΔA|) -0.004; mean ΔA +0.09 pp with ≥ 50 effective updates vs +0.69 pp below; runs with zero updates: 3.
- **C2:** pending.
- **OS:** pending.

## 4. Stage A audit (original Phase-1 candidates)

| Candidate | Paper Table 1 (Ps / Var / D) | Historical settings K=8 | E0 K=32 | MATH-500 after Phase-1 burst |
|---|---|---|---|---|
| test/prealgebra/1820.json (L5) | 0.375 / 0.250 / 0.625 | 0.250 / 0.172 / 0.750 | 0.219 / 0.213 / 0.438 | 0.574 |
| test/geometry/554.json (L2) | 0.875 / 0.109 / 0.250 | 0.750 / 0.309 / 0.375 | 0.531 / 0.273 / 0.219 | 0.568 |
| test/counting_and_probability/559.json (L5) | 0.125 / 0.152 / 1.000 | 0.250 / 0.246 / 0.875 | 0.156 / 0.144 / 0.719 | 0.576 |
| test/number_theory/457.json (L4) | 0.250 / 0.188 / 0.750 | 0.375 / 0.309 / 0.750 | 0.000 / 0.027 / 0.812 | 0.576 |

Table 1's signals do not reproduce (K = 8 sampling noise, grader and truncation differences), and the four candidates' downstream MATH-500 accuracies are indistinguishable; the 4-row fit (5 parameters) is non-identifiable.

## 5. End-to-end curriculum (exploratory, amendment A4)

Pending: the end-to-end curriculum has not run yet.

## 6. Compute consumed (GPU-hours, from run timings)

| Stage | GPU-h |
|---|---|
| Stage A audit (logged phases) | 0.6 |
| C1 pilot | 23.4 |

## 7. Reproduce

```bash
python -m rlvr_v2.sgac_calib diagnose --source c1        # and c2, os
python -m rlvr_v2.sgac_calib fit-explore --source c1 --tag E1_c1_pilot
STAGES=os_pilot,os_expansion,os_selector,os_curriculum,c2 scripts/local/sgac_calib.sh   # GPU queue
python -m rlvr_v2.sgac_calib report
```
