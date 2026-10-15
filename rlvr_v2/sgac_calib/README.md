# SGAC calibration track: the whole picture

**The question.** Can we build a *better* SGAC selector by measuring how much each training problem actually helps,
and does that selector beat Random and the original SGAC (Eq. 10) inside the unchanged 20-step curriculum?

The rules are frozen in [`paper/sgac_calib_protocol.md`](../../paper/sgac_calib_protocol.md) (tag
`sgac-calib-protocol-v1`). This page is a plain-language map of that protocol; where they differ, the protocol wins.

> **Update 2026-10-15: the pilot failed gate G-P.** The reasons are in protocol §13, amendment A1, and in
> [`reports/sgac_calib/REPORT.md`](../../reports/sgac_calib/REPORT.md).
>
> - At SGAC's training dose, every problem perturbs the same ~76 fragile dev questions. Two seeds of one problem agree
>   no more than runs on different problems, so there is no problem-specific usefulness to learn.
> - The equation fitted on those labels (E1) has no predictive validity.
>
> The queue continues as follows:
>
> 1. The protocol's pre-declared rung C2 (4× training).
> 2. An **exploratory** calibration in the one-shot regime whose positive control is known to work: the prereg
>    Study-1 trainer, groups of 64.
> 3. A selector fitted on those labels and tested end to end in the curriculum below.
>
> One command runs it all:
> `STAGES=os_pilot,os_expansion,os_selector,os_curriculum,c2 scripts/local/sgac_calib.sh`.

## 1. The pipeline

```mermaid
flowchart TD
    subgraph INPUTS["Inputs (read-only)"]
        POOL["Prereg pool: 500 MATH train problems<br/>+ 32 stored attempts each (K=32 sieve)"]
        HELD["Prereg held-out: 500 MATH train problems"]
        SGP["Historical SGAC pool: 1000 problems"]
        FIN["Final exams: MATH-500 + test50"]
        CODE["Frozen SGAC code and e0 recipe<br/>tag sgac-repro-frozen, spec hash 577f3319"]
    end

    POOL --> ROLES
    HELD --> ROLES
    SGP --> ROLES
    FIN -. only to exclude lookalikes .-> ROLES
    ROLES["Data roles, pairwise disjoint (near-duplicates at 0.8 removed)<br/>dev exam 486 · calibration candidates 408 · curriculum pool 964"]
    ROLES --> DRAW["Outcome-blind draw: 48 candidates<br/>12 per success-rate bin (16 pilot + 32 expansion)"]
    DRAW --> FEAT["Features per candidate: Ps, Var, D, L<br/>from 16 random 4-attempt samples (what the curriculum sees)"]

    SGP -.-> AUDIT["Stage A · audit<br/>re-measure the paper's 4 Phase-1 problems"]

    DRAW --> PILOT["Stage C · pilot<br/>16 candidates × 2 seeds + π1 × 2<br/>train on ONE problem, retake the dev exam: ΔA"]
    CODE -. same training recipe .-> PILOT
    PILOT --> GP{"Gate G-P<br/>is ΔA repeatable?<br/>ICC ≥ 0.30 and spread ≥ 1 pp"}
    GP -- no --> STOP1["Stop and report:<br/>usefulness not measurable at this budget"]
    GP -- yes --> EXP["Stage C · expansion<br/>32 more candidates × 2 seeds"]
    EXP --> LABELS["48 labels (ΔA per candidate)"]
    FEAT --> FIT
    LABELS --> FIT["Stage D · fit the selector<br/>Ridge by default; OLS, boosted trees and baselines compared<br/>cross-validated on candidates it never saw"]
    FIT --> GS{"Gate G-S<br/>does it predict unseen candidates?<br/>ρ ≥ 0.30 with p below 0.05,<br/>positive top-1-of-4 gain"}
    GS -- no --> STOP2["Stop and report:<br/>the signals don't predict usefulness"]
    GS -- yes --> FREEZE["Freeze selector.json + hash pin"]
    FREEZE --> CUR["Stage E · 20-step curriculum on the clean pool<br/>learned · random · SGAC Eq. 10: 8 seeds each<br/>max-var · max-D · max-level: 3 seeds each"]
    CODE -. same loop and recipe .-> CUR
    CUR --> EVAL["Final exams after step 20"]
    FIN -. first and only use .-> EVAL
    EVAL --> VERDICT["Verdict by the pre-declared rule<br/>beats · worse · equivalent · inconclusive"]
```

Three things to notice:
- **Two gates can stop the track early** (G-P, G-S). Stopping is a result, not a failure: it says what cannot be learned
  at this training budget, and it saves the GPU time of the later stages.
- **The final exams are touched only at the very end.** Labels come from a separate 486-question dev exam, so the
  selector is never tuned on MATH-500 or test50.
- **Nothing historical changes.** Every run uses the frozen SGAC loop and the e0 recipe (the code refuses to run if the
  spec hash is not `577f3319`); only the results folder is different.

## 2. Two kinds of training run

```mermaid
flowchart LR
    subgraph CAL["Calibration run (Stage C): produces ONE number"]
        direction LR
        c1["Fresh LoRA on<br/>Qwen2.5-Math-1.5B"] --> c2["20 bursts on the SAME problem<br/>5 updates each = 100 updates"]
        c2 --> c3["Dev exam<br/>486 questions"]
        c3 --> c4["ΔA = trained − base<br/>dev accuracy"]
        c4 --> c5["Model discarded"]
    end
```

```mermaid
flowchart LR
    subgraph STEP["Curriculum step (Stage E), repeated 20 times"]
        direction LR
        s1["4 candidate problems<br/>same batch for every arm"] --> s2["4 attempts each<br/>with the CURRENT model"]
        s2 --> s3["Signals per candidate<br/>Ps, Var, D, L"]
        s3 --> s4["The arm's rule picks 1<br/>learned: highest selector score"]
        s4 --> s5["1 burst = 5 updates<br/>on the pick; LoRA kept"]
    end
```

Both use exactly 100 updates in total. A calibration run spends them all on one problem (so its effect is big enough
to measure); a curriculum spends 5 on each of 20 picked problems. Whether "useful alone" predicts "useful inside the
curriculum" is precisely what Stage E tests.

## 3. The selector

Four signals per candidate, measured from 4 attempts by the current model:

| Signal | Meaning |
|---|---|
| **Ps** | share of the 4 attempts that are correct |
| **Var** | variance of the reward (correct + 0.5 × has a boxed answer) |
| **D** | number of distinct answers / 4 (disagreement) |
| **L** | MATH difficulty level, 1–5 |

The default model is a ridge regression:

&nbsp;&nbsp;&nbsp;&nbsp;`score = b0 + b1·Ps + b2·Var + b3·D + b4·L`, trained to predict ΔA.

- No coefficient has a required sign; any may come out near zero.
- It is compared with OLS, a small boosted-tree model, and baselines: difficulty only, success rate only, the
  original Eq. 10, the single-signal rules and random. A nonlinear model replaces Ridge only if clearly better.
- It must predict candidates it never saw (gate G-S) before it is ever deployed.
- Deployed rule: in each batch of 4, train on the candidate with the highest score.

For contrast, the original Eq. 10 (`0.0050·Ps + 0.1832·Var − 0.0751·D + 0.2188·L`) was fit on 20 problems with its
labels shifted by one feature, and at K=4 it always picks a maximum-level problem.

## 4. What each outcome would let us say

| G-P | G-S | Final curriculum | Claim |
|---|---|---|---|
| fail | — | not run | One problem's usefulness is not measurable at this budget. |
| pass | fail | not run | Problems differ in usefulness, but Ps/Var/D/L don't predict it. |
| pass | pass | beats Random **and** SGAC | The calibrated selector improves the curriculum (with its cost reported). |
| pass | pass | equivalent / inconclusive | It predicts usefulness, but no curriculum gain above ~2 pp. |

"Beats" needs a Holm-corrected p < 0.05, a gain of at least 1 pp, and the same direction in at least 7 of 8 seeds.

## 5. Stages, commands and cost

All GPU work runs on the workstation, one stage at a time: `STAGES=<stage> scripts/local/sgac_calib.sh`.
A failed gate stops the queue (exit code 3); re-running the same command resumes.

| Stage | What it does | GPU-h | Gate at the end |
|---|---|---|---|
| `t0` | spec, manifest and feature checks; GPU smoke; timing projection | ~1 | — |
| `audit` | Stage A: the paper's 4 Phase-1 problems, E0 and historical settings | ~1.5–3 | — |
| `pilot` | base dev exam ×2, 16 candidates × 2 seeds, π1 × 2 | ~27 | G-P |
| `expansion` | 32 more candidates × 2 seeds, labels | ~50 | — |
| `selector` | fit, cross-validate, permutation test, freeze (CPU) | 0 | G-S |
| `curriculum` | base exams, 6 arms (learned/random/SGAC × 8 seeds, heuristics × 3), analysis | ~44 | verdict |
| `tm` | exploratory: Random given the selector arms' extra GPU time | ~2–5 | — |

Costs use the T0 measurement (0.78 h per calibration run) and the reproduction's 1.3 h per curriculum run.

## 6. Where things are

| | Location |
|---|---|
| Protocol (frozen) | `paper/sgac_calib_protocol.md` |
| Config | `configs/sgac_calib/calib.yaml` |
| Committed manifests | `configs/sgac_calib/manifests/`: data roles, the 48-candidate draw, features, seed count, schedules (seeds 42–49); later the frozen selector |
| Code | `rlvr_v2/sgac_calib/` (this folder): `manifests`, `features`, `calib` (runs, labels, G-P), `selector` (fit, G-S, freeze), `curriculum` (learned arm), `crossed` + `analysis` (final statistics), `cli`, `jobs` |
| Launcher | `scripts/local/sgac_calib.sh` |
| Tests | `tests/test_sgac_calib_*.py` |
| Results (workstation, not in git) | `results_sgac_calib/{t0,audit,calib,selector,curriculum}/` |
| Historical reproduction (frozen) | `rlvr_v2/sgac/`, `report.md`, `paper/sgac_repro_freeze.md`, tag `sgac-repro-frozen` |
