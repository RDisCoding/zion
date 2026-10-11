## H. Verdict (pre-declared rule, e0 profile, MATH-500, step 20, seeds pooled)

**Original SGAC could not be reproduced** — no training effect: SGAC - Base 95% CI includes 0; no selection effect: SGAC - Random 95% CI includes 0

| Endpoint | Δ accuracy (pp) [95% CI] | Seeds | Items |
|---|---|---|---|
| P1: SGAC − Base | +0.9 [-1.4, +3.1] | 42,43,44 | 500 |
| P2: SGAC − Random | +0.7 [-0.8, +2.3] | 42,43,44 | 500 |

## B. Selector coefficients actually used

`Score = 0.0050·Ps + 0.1832·Var − 0.0751·D + 0.2188·L`, first-maximum argmax, no intercept: the hardcoded line of the published notebook (paper Eq. 10). It is the N=20 phase-4 regression printed with labels shifted by one feature; the regression itself (`learned_selector.pkl`) means `0.0050·L + 0.1832·Ps − 0.0751·Var + 0.2188·D + 0.279` (optional label-corrected arm). Paper Table 2 is the N=4 fit with the same shift and was never run.

## Profile `e0` (spec 577f3319)

### A. Configuration

| Setting | Value |
|---|---|
| Model | Qwen/Qwen2.5-Math-1.5B (loaded: bf16; commit 4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2) |
| Quantisation requested / fallback | none / none |
| Prompt (sieve, eval) / training | oneshot_rlvr_chat / rendered |
| Grader (primary) / D metric | math_verify / u_ratio |
| Token caps sieve / GRPO / eval | 3072 / 3072 / 3072 |
| Sieve top_k / stops / pad | 0 / style / fim_pad |
| Loop | 20 steps, B=4, K=4, T=1.0, eval every 5 |
| Burst | 5 single-completion updates (pdbs 1, gen batch 4, G 4), lr 2e-05 linear |
| LoRA | r=16, alpha=32, ['q_proj', 'k_proj', 'v_proj', 'o_proj'] |
| test50 batch decision | n/a |
| E0 G1 agreement (base MATH-500) | 1.0 |

### D. Accuracy trajectory (original 50-item test set; seeds averaged)

| Arm | Base | Step 5 | Step 10 | Step 15 | Step 20 |
|---|---|---|---|---|---|
| SGAC (Eq. 10 as run) | 70.0 | 70.0 (n=3) | 72.0 (n=3) | 69.3 (n=3) | 70.0 (n=3) |
| Random | 70.0 | 72.0 (n=3) | 70.7 (n=3) | 68.7 (n=3) | 70.0 (n=3) |
| Max variance | 70.0 | 73.3 (n=3) | 69.3 (n=3) | 70.7 (n=3) | 66.0 (n=3) |
| Max disagreement | 70.0 | 70.0 (n=3) | 70.0 (n=3) | 69.3 (n=3) | 69.3 (n=3) |
| Max difficulty | 70.0 | 71.0 (n=2) | 71.0 (n=2) | 68.0 (n=1) | 66.0 (n=1) |

Original NB-M run (one unseeded T4 run): 0: 64.0, 5: 66.0, 10: 62.0, 15: 60.0, 20: 68.0.

![trajectory](figures/trajectory_test50_e0.png)

### E. Comparison at step 20

**MATH-500 (500 items)**, primary grader `math_verify`:

| Model / arm | Accuracy % [95% CI] | Per seed | vs Base (pp) [95% CI] | SGAC − arm (pp) [95% CI] |
|---|---|---|---|---|
| Base | 63.0 | — | — | — |
| pi1 checkpoint (Wang et al.) | 12.4 | — | -50.6 [-55.4, -45.6] | — |
| SGAC (Eq. 10 as run) (seeds 42,43,44) | 63.9 [59.9, 67.9] | 62.4, 64.4, 64.8 | +0.9 [-1.4, +3.1] | — |
| Random (seeds 42,43,44) | 63.1 [59.3, 67.1] | 61.4, 63.8, 64.2 | +0.1 [-2.0, +2.3] | +0.7 [-0.8, +2.3] |
| Max variance (seeds 42,43,44) | 64.1 [60.2, 68.1] | 63.4, 63.8, 65.2 | +1.1 [-1.1, +3.3] | -0.3 [-1.9, +1.3] |
| Max disagreement (seeds 42,43,44) | 64.0 [60.1, 68.0] | 62.8, 65.2, 64.0 | +1.0 [-1.3, +3.3] | -0.1 [-1.7, +1.4] |
| Max difficulty (seeds 42) | 63.2 [59.0, 67.4] | 63.2 | +0.2 [-2.6, +3.0] | -0.8 [-3.6, +2.0] |

![final math500](figures/final_math500_e0.png)

**original 50-item test set**, primary grader `math_verify`:

| Model / arm | Accuracy % [95% CI] | Per seed | vs Base (pp) [95% CI] | SGAC − arm (pp) [95% CI] |
|---|---|---|---|---|
| Base | 70.0 | — | — | — |
| pi1 checkpoint (Wang et al.) | 12.0 | — | -58.0 [-72.0, -44.0] | — |
| SGAC (Eq. 10 as run) (seeds 42,43,44) | 70.0 [58.0, 81.3] | 68.0, 74.0, 68.0 | -0.0 [-5.3, +5.3] | — |
| Random (seeds 42,43,44) | 70.0 [58.0, 81.3] | 64.0, 74.0, 72.0 | -0.0 [-6.0, +6.7] | +0.0 [-3.3, +4.0] |
| Max variance (seeds 42,43,44) | 66.0 [53.3, 78.0] | 64.0, 68.0, 66.0 | -4.0 [-11.3, +2.0] | +4.0 [+0.7, +8.0] |
| Max disagreement (seeds 42,43,44) | 69.3 [56.7, 81.3] | 68.0, 74.0, 66.0 | -0.7 [-5.3, +4.0] | +0.7 [-2.0, +4.0] |
| Max difficulty (seeds 42) | 66.0 [52.0, 78.0] | 66.0 | -4.0 [-12.0, +4.0] | +2.0 [+0.0, +6.0] |

![final test50](figures/final_test50_e0.png)

Grader sensitivity at the final step (same outputs, both graders):

| Run | Set | Primary | math-verify | legacy | Truncated |
|---|---|---|---|---|---|
| max_d seed 42 | test50 | 68.0 | 68.0 | 64.0 | 6.0 |
| max_d seed 42 | math500 | 62.8 | 62.8 | 56.0 | 4.2 |
| max_d seed 43 | test50 | 74.0 | 74.0 | 70.0 | 2.0 |
| max_d seed 43 | math500 | 65.2 | 65.2 | 58.0 | 3.4 |
| max_d seed 44 | test50 | 66.0 | 66.0 | 64.0 | 4.0 |
| max_d seed 44 | math500 | 64.0 | 64.0 | 57.6 | 4.2 |
| max_level seed 42 | test50 | 66.0 | 66.0 | 64.0 | 4.0 |
| max_level seed 42 | math500 | 63.2 | 63.2 | 55.4 | 3.6 |
| max_var seed 42 | test50 | 64.0 | 64.0 | 62.0 | 4.0 |
| max_var seed 42 | math500 | 63.4 | 63.4 | 56.2 | 3.0 |
| max_var seed 43 | test50 | 68.0 | 68.0 | 66.0 | 6.0 |
| max_var seed 43 | math500 | 63.8 | 63.8 | 57.2 | 4.8 |
| max_var seed 44 | test50 | 66.0 | 66.0 | 64.0 | 4.0 |
| max_var seed 44 | math500 | 65.2 | 65.2 | 58.6 | 3.6 |
| random seed 42 | test50 | 64.0 | 64.0 | 60.0 | 6.0 |
| random seed 42 | math500 | 61.4 | 61.4 | 55.0 | 3.8 |
| random seed 43 | test50 | 74.0 | 74.0 | 72.0 | 4.0 |
| random seed 43 | math500 | 63.8 | 63.8 | 57.0 | 4.2 |
| random seed 44 | test50 | 72.0 | 72.0 | 68.0 | 2.0 |
| random seed 44 | math500 | 64.2 | 64.2 | 56.6 | 4.8 |
| sgac seed 42 | test50 | 68.0 | 68.0 | 64.0 | 2.0 |
| sgac seed 42 | math500 | 62.4 | 62.4 | 55.8 | 4.6 |
| sgac seed 43 | test50 | 74.0 | 74.0 | 70.0 | 2.0 |
| sgac seed 43 | math500 | 64.4 | 64.4 | 57.4 | 4.8 |
| sgac seed 44 | test50 | 68.0 | 68.0 | 64.0 | 4.0 |
| sgac seed 44 | math500 | 64.8 | 64.8 | 57.6 | 3.8 |

### C. Selection trajectories (20 steps per run)

<details><summary>SGAC (Eq. 10 as run), seed 42 (20 steps done, status done)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | train/geometry/984.json | 0.500 | 0.562 | 0.500 | 5 | 1.1620 | yes | no | 0.875 |  |
| 2 | train/intermediate_algebra/541.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | no | 0.438 |  |
| 3 | train/algebra/778.json | 0.750 | 0.188 | 0.500 | 5 | 1.0945 | yes | no | 0.875 |  |
| 4 | test/intermediate_algebra/1475.json | 0.000 | 0.000 | 0.750 | 5 | 1.0377 | yes | yes | 0.500 |  |
| 5 | train/prealgebra/601.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | yes | no | 1.375 | 72.0 |
| 6 | train/algebra/2115.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | yes | no | 1.375 |  |
| 7 | test/intermediate_algebra/1974.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | yes | no | 0.375 |  |
| 8 | test/geometry/454.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | no | 0.562 |  |
| 9 | train/algebra/179.json | 0.250 | 0.188 | 1.000 | 5 | 1.0545 | yes | no | 0.750 |  |
| 10 | train/intermediate_algebra/369.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | yes | no | 1.000 | 72.0 |
| 11 | train/geometry/240.json | 0.250 | 0.188 | 1.000 | 4 | 0.8357 | yes | yes | 0.500 |  |
| 12 | train/prealgebra/911.json | 0.750 | 0.188 | 0.500 | 3 | 0.6569 | yes | no | 1.250 |  |
| 13 | train/algebra/1654.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | yes | no | 0.438 |  |
| 14 | train/counting_and_probability/1111.json | 0.250 | 0.188 | 0.750 | 4 | 0.8545 | yes | no | 0.562 |  |
| 15 | test/algebra/1623.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | yes | no | 1.250 | 72.0 |
| 16 | train/intermediate_algebra/1546.json | 0.000 | 0.000 | 0.750 | 5 | 1.0377 | yes | yes | 0.500 |  |
| 17 | test/counting_and_probability/789.json | 0.250 | 0.188 | 1.000 | 5 | 1.0545 | yes | no | 0.562 |  |
| 18 | test/geometry/780.json | 0.500 | 0.250 | 0.750 | 5 | 1.0860 | yes | no | 1.125 |  |
| 19 | train/number_theory/402.json | 0.500 | 0.250 | 0.750 | 5 | 1.0860 | yes | no | 1.375 |  |
| 20 | train/counting_and_probability/707.json | 0.750 | 0.188 | 0.500 | 5 | 1.0945 | yes | yes | 1.500 | 68.0 |

</details>

<details><summary>SGAC (Eq. 10 as run), seed 43 (20 steps done, status done)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | train/geometry/550.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | yes | no | 1.000 |  |
| 2 | train/number_theory/167.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | no | 1.000 |  |
| 3 | train/geometry/572.json | 1.000 | 0.000 | 0.250 | 5 | 1.0802 | yes | no | 1.375 |  |
| 4 | train/counting_and_probability/5119.json | 0.250 | 0.297 | 1.000 | 5 | 1.0745 | yes | no | 0.625 |  |
| 5 | train/intermediate_algebra/9011.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | yes | 0.500 | 70.0 |
| 6 | test/intermediate_algebra/1374.json | 0.750 | 0.422 | 0.500 | 3 | 0.6999 | yes | no | 1.250 |  |
| 7 | train/precalculus/1195.json | 0.000 | 0.062 | 0.750 | 5 | 1.0491 | yes | yes | 0.500 |  |
| 8 | train/algebra/1637.json | 0.500 | 0.250 | 0.750 | 5 | 1.0860 | yes | no | 0.875 |  |
| 9 | train/intermediate_algebra/1494.json | 0.500 | 0.422 | 0.750 | 4 | 0.8987 | yes | no | 0.750 |  |
| 10 | train/intermediate_algebra/943.json | 0.250 | 0.297 | 0.750 | 5 | 1.0933 | yes | no | 0.625 | 74.0 |
| 11 | test/algebra/578.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | no | 1.125 |  |
| 12 | train/geometry/570.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | yes | no | 0.625 |  |
| 13 | test/counting_and_probability/3.json | 0.500 | 0.250 | 0.750 | 4 | 0.8672 | yes | no | 0.438 |  |
| 14 | train/geometry/6017.json | 0.000 | 0.047 | 0.750 | 5 | 1.0463 | yes | yes | 0.500 |  |
| 15 | test/algebra/2557.json | 0.500 | 0.250 | 0.750 | 5 | 1.0860 | yes | no | 1.250 | 68.0 |
| 16 | test/geometry/66.json | 0.250 | 0.297 | 1.000 | 5 | 1.0745 | yes | yes | 0.500 |  |
| 17 | test/intermediate_algebra/551.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | yes | 0.500 |  |
| 18 | test/geometry/1044.json | 0.000 | 0.062 | 0.500 | 5 | 1.0679 | yes | no | 0.312 |  |
| 19 | train/number_theory/7072.json | 0.500 | 0.422 | 0.750 | 5 | 1.1175 | yes | no | 1.375 |  |
| 20 | train/counting_and_probability/961.json | 0.750 | 0.188 | 0.500 | 3 | 0.6569 | yes | no | 1.125 | 74.0 |

</details>

<details><summary>SGAC (Eq. 10 as run), seed 44 (20 steps done, status done)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | test/algebra/2531.json | 0.750 | 0.422 | 0.500 | 5 | 1.1375 | yes | no | 1.250 |  |
| 2 | test/prealgebra/1234.json | 0.750 | 0.422 | 0.500 | 3 | 0.6999 | yes | no | 1.188 |  |
| 3 | test/algebra/2544.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | no | 0.688 |  |
| 4 | train/precalculus/47.json | 0.000 | 0.000 | 0.750 | 5 | 1.0377 | yes | no | 0.250 |  |
| 5 | train/counting_and_probability/567.json | 0.500 | 0.250 | 0.500 | 3 | 0.6672 | yes | no | 0.938 | 68.0 |
| 6 | train/intermediate_algebra/1935.json | 0.000 | 0.000 | 0.500 | 5 | 1.0564 | yes | no | 0.438 |  |
| 7 | train/intermediate_algebra/1062.json | 0.250 | 0.297 | 1.000 | 5 | 1.0745 | yes | yes | 0.500 |  |
| 8 | train/number_theory/7016.json | 1.000 | 0.000 | 0.250 | 5 | 1.0802 | yes | no | 1.125 |  |
| 9 | test/algebra/2398.json | 1.000 | 0.000 | 0.250 | 5 | 1.0802 | yes | no | 1.000 |  |
| 10 | train/precalculus/351.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | no | 0.750 | 70.0 |
| 11 | train/intermediate_algebra/1159.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | no | 0.625 |  |
| 12 | test/algebra/491.json | 0.250 | 0.188 | 1.000 | 5 | 1.0545 | yes | no | 0.750 |  |
| 13 | test/precalculus/804.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | yes | no | 0.875 |  |
| 14 | train/precalculus/1296.json | 0.250 | 0.188 | 1.000 | 5 | 1.0545 | yes | no | 0.688 |  |
| 15 | test/precalculus/27.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | yes | 0.500 | 68.0 |
| 16 | train/prealgebra/686.json | 0.250 | 0.188 | 0.750 | 5 | 1.0733 | yes | no | 0.562 |  |
| 17 | test/precalculus/274.json | 0.250 | 0.188 | 1.000 | 5 | 1.0545 | yes | no | 0.750 |  |
| 18 | test/precalculus/992.json | 0.500 | 0.250 | 0.750 | 5 | 1.0860 | yes | no | 0.750 |  |
| 19 | train/counting_and_probability/5014.json | 0.000 | 0.000 | 0.500 | 5 | 1.0564 | yes | no | 0.438 |  |
| 20 | test/prealgebra/1084.json | 1.000 | 0.000 | 0.250 | 4 | 0.8614 | yes | yes | 1.500 | 68.0 |

</details>

<details><summary>Random, seed 42 (20 steps done, status done)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | train/geometry/721.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | no | no | 0.438 |  |
| 2 | test/intermediate_algebra/984.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | no | no | 1.250 |  |
| 3 | test/counting_and_probability/247.json | 1.000 | 0.000 | 0.250 | 1 | 0.2050 | no | no | 1.375 |  |
| 4 | train/prealgebra/597.json | 0.000 | 0.000 | 0.750 | 5 | 1.0377 | yes | no | 0.562 |  |
| 5 | train/prealgebra/601.json | 1.000 | 0.000 | 0.250 | 4 | 0.8614 | yes | no | 1.062 | 70.0 |
| 6 | train/intermediate_algebra/1954.json | 0.500 | 0.250 | 0.750 | 1 | 0.2108 | no | no | 1.000 |  |
| 7 | test/algebra/1345.json | 0.750 | 0.422 | 0.500 | 3 | 0.6999 | no | no | 1.375 |  |
| 8 | test/geometry/454.json | 0.000 | 0.000 | 0.750 | 5 | 1.0377 | yes | no | 0.438 |  |
| 9 | test/prealgebra/1195.json | 0.250 | 0.297 | 1.000 | 3 | 0.6369 | no | no | 0.812 |  |
| 10 | test/number_theory/14.json | 0.000 | 0.000 | 1.000 | 4 | 0.8001 | no | no | 0.750 | 74.0 |
| 11 | train/counting_and_probability/133.json | 1.000 | 0.000 | 0.250 | 1 | 0.2050 | no | no | 1.188 |  |
| 12 | train/algebra/2083.json | 0.500 | 0.250 | 0.750 | 2 | 0.4296 | no | no | 1.250 |  |
| 13 | train/algebra/1654.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | no | yes | 0.500 |  |
| 14 | test/intermediate_algebra/182.json | 0.000 | 0.000 | 1.000 | 2 | 0.3625 | no | no | 0.625 |  |
| 15 | test/algebra/1623.json | 1.000 | 0.000 | 0.250 | 4 | 0.8614 | yes | no | 1.250 | 66.0 |
| 16 | train/counting_and_probability/961.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | no | yes | 1.500 |  |
| 17 | test/counting_and_probability/498.json | 0.500 | 0.422 | 0.750 | 4 | 0.8987 | no | yes | 1.500 |  |
| 18 | test/intermediate_algebra/1626.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | no | no | 0.688 |  |
| 19 | train/precalculus/727.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | no | no | 0.875 |  |
| 20 | train/algebra/1853.json | 0.000 | 0.047 | 0.750 | 5 | 1.0463 | no | no | 0.375 | 64.0 |

</details>

<details><summary>Random, seed 43 (20 steps done, status done)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | train/intermediate_algebra/1971.json | 0.500 | 0.250 | 0.750 | 4 | 0.8672 | no | no | 1.250 |  |
| 2 | train/prealgebra/531.json | 0.250 | 0.188 | 1.000 | 2 | 0.3981 | no | no | 0.562 |  |
| 3 | train/precalculus/674.json | 0.250 | 0.297 | 1.000 | 3 | 0.6369 | no | no | 0.688 |  |
| 4 | test/counting_and_probability/846.json | 1.000 | 0.000 | 0.250 | 3 | 0.6426 | no | yes | 1.500 |  |
| 5 | test/prealgebra/1697.json | 0.000 | 0.000 | 0.500 | 4 | 0.8377 | no | no | 0.438 | 74.0 |
| 6 | test/counting_and_probability/659.json | 1.000 | 0.000 | 0.250 | 3 | 0.6426 | no | yes | 1.500 |  |
| 7 | train/algebra/2638.json | 0.500 | 0.422 | 0.750 | 1 | 0.2423 | no | no | 0.938 |  |
| 8 | train/intermediate_algebra/1251.json | 0.000 | 0.000 | 0.750 | 5 | 1.0377 | no | no | 0.438 |  |
| 9 | train/prealgebra/1731.json | 0.250 | 0.188 | 1.000 | 4 | 0.8357 | no | no | 1.312 |  |
| 10 | train/prealgebra/1259.json | 0.000 | 0.000 | 0.750 | 4 | 0.8189 | no | no | 1.125 | 74.0 |
| 11 | train/geometry/631.json | 0.250 | 0.188 | 1.000 | 4 | 0.8357 | no | no | 1.375 |  |
| 12 | train/counting_and_probability/658.json | 0.500 | 0.250 | 0.750 | 4 | 0.8672 | no | no | 0.688 |  |
| 13 | test/counting_and_probability/289.json | 0.500 | 0.422 | 0.500 | 1 | 0.2610 | no | no | 1.375 |  |
| 14 | test/algebra/2235.json | 1.000 | 0.000 | 0.250 | 1 | 0.2050 | no | no | 1.188 |  |
| 15 | test/algebra/2557.json | 0.750 | 0.188 | 0.500 | 5 | 1.0945 | yes | no | 1.125 | 70.0 |
| 16 | train/precalculus/152.json | 0.500 | 0.250 | 0.750 | 2 | 0.4296 | no | no | 1.375 |  |
| 17 | test/intermediate_algebra/1741.json | 0.500 | 0.250 | 0.750 | 4 | 0.8672 | no | no | 1.188 |  |
| 18 | test/algebra/862.json | 0.500 | 0.250 | 0.750 | 4 | 0.8672 | no | no | 1.125 |  |
| 19 | train/algebra/879.json | 0.250 | 0.188 | 1.000 | 3 | 0.6169 | no | no | 0.688 |  |
| 20 | train/prealgebra/292.json | 0.250 | 0.188 | 1.000 | 3 | 0.6169 | no | no | 1.375 | 74.0 |

</details>

<details><summary>Random, seed 44 (20 steps done, status done)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | test/algebra/1836.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | no | no | 1.250 |  |
| 2 | train/counting_and_probability/1097.json | 1.000 | 0.000 | 0.250 | 2 | 0.4238 | no | no | 1.375 |  |
| 3 | train/precalculus/856.json | 0.750 | 0.188 | 0.500 | 1 | 0.2193 | no | no | 0.562 |  |
| 4 | test/algebra/1800.json | 1.000 | 0.000 | 0.250 | 1 | 0.2050 | no | no | 0.938 |  |
| 5 | train/algebra/2250.json | 0.500 | 0.250 | 0.500 | 2 | 0.4483 | no | no | 1.062 | 72.0 |
| 6 | train/counting_and_probability/995.json | 0.750 | 0.188 | 0.500 | 3 | 0.6569 | no | no | 0.812 |  |
| 7 | train/intermediate_algebra/45.json | 0.000 | 0.047 | 0.750 | 2 | 0.3899 | no | no | 0.688 |  |
| 8 | train/number_theory/7016.json | 1.000 | 0.000 | 0.250 | 5 | 1.0802 | yes | no | 0.875 |  |
| 9 | train/prealgebra/911.json | 0.750 | 0.188 | 0.500 | 3 | 0.6569 | no | no | 1.000 |  |
| 10 | train/algebra/2581.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | no | no | 1.125 | 64.0 |
| 11 | test/prealgebra/1803.json | 0.750 | 0.188 | 0.500 | 2 | 0.4381 | no | no | 1.188 |  |
| 12 | test/algebra/427.json | 0.000 | 0.047 | 1.000 | 4 | 0.8087 | no | no | 0.812 |  |
| 13 | test/counting_and_probability/247.json | 0.500 | 0.250 | 0.750 | 1 | 0.2108 | no | yes | 1.500 |  |
| 14 | test/precalculus/812.json | 0.000 | 0.047 | 0.500 | 5 | 1.0650 | no | no | 0.562 |  |
| 15 | test/algebra/2649.json | 1.000 | 0.000 | 0.250 | 4 | 0.8614 | no | no | 1.375 | 70.0 |
| 16 | test/prealgebra/1977.json | 0.500 | 0.422 | 0.750 | 3 | 0.6799 | no | no | 0.625 |  |
| 17 | train/prealgebra/1731.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | no | no | 1.125 |  |
| 18 | test/precalculus/992.json | 0.500 | 0.250 | 0.750 | 5 | 1.0860 | yes | no | 0.875 |  |
| 19 | test/counting_and_probability/495.json | 0.000 | 0.047 | 1.000 | 3 | 0.5899 | no | no | 0.750 |  |
| 20 | test/prealgebra/1084.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | yes | no | 1.000 | 72.0 |

</details>

<details><summary>Max variance, seed 42 (20 steps done, status done)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | train/geometry/984.json | 0.500 | 0.562 | 0.500 | 5 | 1.1620 | yes | no | 0.938 |  |
| 2 | test/geometry/0.json | 0.500 | 0.422 | 0.750 | 2 | 0.4611 | no | no | 0.625 |  |
| 3 | train/algebra/778.json | 0.500 | 0.250 | 0.750 | 5 | 1.0860 | yes | no | 1.000 |  |
| 4 | train/prealgebra/597.json | 0.000 | 0.062 | 0.750 | 5 | 1.0491 | yes | no | 0.625 |  |
| 5 | test/precalculus/1090.json | 0.500 | 0.250 | 0.750 | 2 | 0.4296 | no | no | 0.750 | 72.0 |
| 6 | train/algebra/2115.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | yes | no | 1.125 |  |
| 7 | test/algebra/2559.json | 0.500 | 0.250 | 0.750 | 2 | 0.4296 | no | no | 0.875 |  |
| 8 | test/algebra/1043.json | 0.250 | 0.375 | 0.750 | 4 | 0.8888 | no | no | 0.625 |  |
| 9 | test/prealgebra/1195.json | 0.250 | 0.375 | 0.750 | 3 | 0.6700 | no | no | 0.625 |  |
| 10 | train/algebra/2269.json | 0.500 | 0.422 | 0.750 | 4 | 0.8987 | yes | no | 1.250 | 66.0 |
| 11 | train/algebra/316.json | 0.500 | 0.422 | 0.750 | 3 | 0.6799 | no | no | 0.938 |  |
| 12 | train/intermediate_algebra/995.json | 0.500 | 0.250 | 0.750 | 2 | 0.4296 | no | no | 1.000 |  |
| 13 | test/intermediate_algebra/1952.json | 0.000 | 0.047 | 1.000 | 4 | 0.8087 | no | no | 0.750 |  |
| 14 | train/counting_and_probability/1111.json | 0.250 | 0.188 | 0.750 | 4 | 0.8545 | yes | no | 0.750 |  |
| 15 | test/algebra/1623.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | yes | no | 1.250 | 70.0 |
| 16 | train/counting_and_probability/961.json | 0.250 | 0.297 | 1.000 | 3 | 0.6369 | no | no | 1.375 |  |
| 17 | train/prealgebra/1358.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | no | no | 1.125 |  |
| 18 | train/algebra/1167.json | 0.750 | 0.422 | 0.500 | 1 | 0.2623 | no | no | 1.125 |  |
| 19 | train/geometry/807.json | 0.500 | 0.250 | 0.750 | 5 | 1.0860 | yes | yes | 1.500 |  |
| 20 | train/counting_and_probability/941.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | no | no | 1.250 | 64.0 |

</details>

<details><summary>Max variance, seed 43 (20 steps done, status done)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | test/prealgebra/903.json | 0.250 | 0.297 | 1.000 | 3 | 0.6369 | no | no | 0.875 |  |
| 2 | test/counting_and_probability/535.json | 0.500 | 0.250 | 0.500 | 2 | 0.4483 | no | no | 0.875 |  |
| 3 | train/precalculus/501.json | 0.750 | 0.188 | 0.500 | 3 | 0.6569 | no | no | 1.375 |  |
| 4 | train/counting_and_probability/5119.json | 0.500 | 0.250 | 0.750 | 5 | 1.0860 | yes | no | 0.875 |  |
| 5 | train/precalculus/236.json | 0.750 | 0.188 | 0.500 | 1 | 0.2193 | no | no | 1.375 | 74.0 |
| 6 | train/precalculus/448.json | 0.500 | 0.250 | 0.750 | 2 | 0.4296 | no | no | 0.750 |  |
| 7 | train/precalculus/900.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | no | no | 0.938 |  |
| 8 | train/counting_and_probability/133.json | 0.750 | 0.188 | 0.500 | 1 | 0.2193 | no | no | 1.125 |  |
| 9 | train/geometry/944.json | 0.500 | 0.422 | 0.750 | 4 | 0.8987 | yes | no | 0.812 |  |
| 10 | test/precalculus/19.json | 0.750 | 0.188 | 0.500 | 1 | 0.2193 | no | no | 1.250 | 70.0 |
| 11 | train/number_theory/766.json | 0.750 | 0.422 | 0.500 | 4 | 0.9187 | no | no | 1.188 |  |
| 12 | train/counting_and_probability/658.json | 0.500 | 0.250 | 0.750 | 4 | 0.8672 | no | no | 0.812 |  |
| 13 | test/counting_and_probability/289.json | 0.750 | 0.422 | 0.500 | 1 | 0.2623 | no | yes | 1.500 |  |
| 14 | train/algebra/2727.json | 1.000 | 0.000 | 0.250 | 4 | 0.8614 | no | no | 1.062 |  |
| 15 | test/algebra/2557.json | 0.500 | 0.562 | 0.500 | 5 | 1.1620 | yes | no | 1.188 | 72.0 |
| 16 | train/precalculus/152.json | 0.250 | 0.188 | 1.000 | 2 | 0.3981 | no | no | 1.250 |  |
| 17 | train/prealgebra/2041.json | 0.500 | 0.422 | 0.750 | 4 | 0.8987 | no | no | 0.562 |  |
| 18 | test/algebra/862.json | 0.500 | 0.250 | 0.750 | 4 | 0.8672 | no | no | 1.125 |  |
| 19 | train/number_theory/7072.json | 0.750 | 0.422 | 0.500 | 5 | 1.1375 | yes | no | 1.125 |  |
| 20 | train/prealgebra/790.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | no | no | 1.000 | 68.0 |

</details>

<details><summary>Max variance, seed 44 (20 steps done, status done)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | test/algebra/2531.json | 0.750 | 0.422 | 0.500 | 5 | 1.1375 | yes | no | 1.125 |  |
| 2 | test/prealgebra/1234.json | 0.750 | 0.422 | 0.500 | 3 | 0.6999 | yes | no | 1.062 |  |
| 3 | train/algebra/2115.json | 0.250 | 0.188 | 1.000 | 3 | 0.6169 | no | no | 0.688 |  |
| 4 | train/geometry/6034.json | 0.000 | 0.062 | 0.750 | 5 | 1.0491 | yes | no | 0.312 |  |
| 5 | train/counting_and_probability/567.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | yes | no | 0.750 | 74.0 |
| 6 | train/number_theory/258.json | 0.250 | 0.188 | 1.000 | 5 | 1.0545 | yes | no | 0.812 |  |
| 7 | test/intermediate_algebra/1456.json | 0.250 | 0.297 | 1.000 | 2 | 0.4181 | no | no | 1.188 |  |
| 8 | train/counting_and_probability/941.json | 0.500 | 0.250 | 0.750 | 4 | 0.8672 | no | no | 1.000 |  |
| 9 | test/algebra/2398.json | 0.750 | 0.188 | 0.500 | 5 | 1.0945 | yes | no | 0.812 |  |
| 10 | test/precalculus/963.json | 0.250 | 0.297 | 1.000 | 3 | 0.6369 | no | no | 0.938 | 72.0 |
| 11 | train/algebra/1874.json | 0.500 | 0.250 | 0.500 | 2 | 0.4483 | no | no | 1.375 |  |
| 12 | test/prealgebra/1606.json | 0.500 | 0.422 | 0.750 | 3 | 0.6799 | no | no | 0.875 |  |
| 13 | test/precalculus/804.json | 0.500 | 0.250 | 0.750 | 4 | 0.8672 | no | no | 0.750 |  |
| 14 | train/precalculus/1296.json | 0.250 | 0.188 | 1.000 | 5 | 1.0545 | yes | no | 0.750 |  |
| 15 | test/algebra/2649.json | 0.500 | 0.422 | 0.750 | 4 | 0.8987 | no | no | 1.188 | 70.0 |
| 16 | train/counting_and_probability/122.json | 0.500 | 0.250 | 0.750 | 4 | 0.8672 | no | no | 0.875 |  |
| 17 | train/prealgebra/1731.json | 0.500 | 0.250 | 0.750 | 4 | 0.8672 | no | no | 1.000 |  |
| 18 | train/number_theory/1027.json | 0.750 | 0.188 | 0.500 | 1 | 0.2193 | no | no | 1.062 |  |
| 19 | test/algebra/1490.json | 0.500 | 0.250 | 0.750 | 1 | 0.2108 | no | no | 1.188 |  |
| 20 | test/prealgebra/1084.json | 0.500 | 0.422 | 0.500 | 4 | 0.9174 | yes | no | 1.312 | 66.0 |

</details>

<details><summary>Max disagreement, seed 42 (20 steps done, status done)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | train/geometry/721.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | no | no | 0.438 |  |
| 2 | train/precalculus/175.json | 0.000 | 0.000 | 1.000 | 3 | 0.5813 | no | no | 0.375 |  |
| 3 | train/counting_and_probability/949.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | no | no | 0.625 |  |
| 4 | test/intermediate_algebra/1475.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | no | yes | 0.500 |  |
| 5 | test/prealgebra/1299.json | 0.000 | 0.062 | 0.750 | 4 | 0.8303 | no | no | 0.625 | 72.0 |
| 6 | train/algebra/2115.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | yes | no | 0.938 |  |
| 7 | test/algebra/1345.json | 0.750 | 0.422 | 0.500 | 3 | 0.6999 | no | no | 1.250 |  |
| 8 | train/number_theory/94.json | 0.000 | 0.000 | 1.000 | 4 | 0.8001 | no | no | 0.938 |  |
| 9 | train/algebra/179.json | 0.250 | 0.188 | 1.000 | 5 | 1.0545 | yes | no | 0.938 |  |
| 10 | train/algebra/2269.json | 0.250 | 0.188 | 1.000 | 4 | 0.8357 | no | no | 1.062 | 70.0 |
| 11 | train/geometry/240.json | 0.250 | 0.188 | 1.000 | 4 | 0.8357 | yes | yes | 0.500 |  |
| 12 | train/intermediate_algebra/995.json | 0.500 | 0.422 | 0.750 | 2 | 0.4611 | no | no | 0.562 |  |
| 13 | train/intermediate_algebra/1236.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | no | yes | 0.500 |  |
| 14 | test/intermediate_algebra/182.json | 0.000 | 0.000 | 1.000 | 2 | 0.3625 | no | yes | 0.500 |  |
| 15 | train/intermediate_algebra/1919.json | 0.000 | 0.062 | 0.750 | 4 | 0.8303 | no | no | 0.625 | 70.0 |
| 16 | train/geometry/348.json | 0.250 | 0.188 | 1.000 | 4 | 0.8357 | no | no | 0.375 |  |
| 17 | test/counting_and_probability/789.json | 0.250 | 0.297 | 1.000 | 5 | 1.0745 | yes | no | 0.438 |  |
| 18 | test/intermediate_algebra/1626.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | no | no | 0.500 |  |
| 19 | train/precalculus/727.json | 0.250 | 0.188 | 1.000 | 3 | 0.6169 | no | no | 0.625 |  |
| 20 | train/counting_and_probability/941.json | 0.500 | 0.250 | 0.750 | 4 | 0.8672 | no | no | 1.188 | 68.0 |

</details>

<details><summary>Max disagreement, seed 43 (20 steps done, status done)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | test/prealgebra/903.json | 0.250 | 0.297 | 1.000 | 3 | 0.6369 | no | no | 1.000 |  |
| 2 | train/prealgebra/531.json | 0.250 | 0.188 | 1.000 | 2 | 0.3981 | no | no | 0.562 |  |
| 3 | train/precalculus/674.json | 0.000 | 0.000 | 1.000 | 3 | 0.5813 | no | no | 0.625 |  |
| 4 | test/geometry/435.json | 0.250 | 0.188 | 1.000 | 4 | 0.8357 | no | no | 0.750 |  |
| 5 | test/prealgebra/1947.json | 0.000 | 0.000 | 1.000 | 3 | 0.5813 | no | no | 0.875 | 68.0 |
| 6 | train/precalculus/448.json | 0.000 | 0.047 | 1.000 | 2 | 0.3711 | no | no | 0.688 |  |
| 7 | train/precalculus/900.json | 0.250 | 0.188 | 1.000 | 3 | 0.6169 | no | no | 0.688 |  |
| 8 | train/intermediate_algebra/1251.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | no | no | 0.625 |  |
| 9 | train/algebra/25070.json | 0.250 | 0.188 | 1.000 | 2 | 0.3981 | no | no | 1.125 |  |
| 10 | train/intermediate_algebra/621.json | 0.000 | 0.000 | 1.000 | 4 | 0.8001 | no | no | 0.688 | 72.0 |
| 11 | test/number_theory/961.json | 0.500 | 0.422 | 0.750 | 1 | 0.2423 | no | no | 0.875 |  |
| 12 | train/intermediate_algebra/1058.json | 0.250 | 0.188 | 1.000 | 5 | 1.0545 | yes | no | 1.125 |  |
| 13 | train/algebra/2249.json | 0.000 | 0.000 | 1.000 | 3 | 0.5813 | no | no | 0.625 |  |
| 14 | train/geometry/6017.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | no | no | 0.375 |  |
| 15 | test/precalculus/1129.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | no | yes | 0.500 | 72.0 |
| 16 | test/geometry/66.json | 0.250 | 0.297 | 1.000 | 5 | 1.0745 | yes | no | 0.625 |  |
| 17 | test/intermediate_algebra/551.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | no | 0.375 |  |
| 18 | test/geometry/25.json | 0.250 | 0.188 | 0.750 | 2 | 0.4169 | no | no | 0.750 |  |
| 19 | train/counting_and_probability/971.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | no | no | 0.938 |  |
| 20 | train/prealgebra/790.json | 0.250 | 0.188 | 1.000 | 3 | 0.6169 | no | no | 0.812 | 74.0 |

</details>

<details><summary>Max disagreement, seed 44 (20 steps done, status done)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | test/algebra/1836.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | no | no | 1.125 |  |
| 2 | test/precalculus/513.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | no | no | 0.438 |  |
| 3 | train/algebra/2115.json | 0.250 | 0.188 | 1.000 | 3 | 0.6169 | no | no | 0.812 |  |
| 4 | train/geometry/6034.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | no | no | 0.250 |  |
| 5 | train/counting_and_probability/567.json | 0.750 | 0.188 | 0.500 | 3 | 0.6569 | yes | no | 0.812 | 70.0 |
| 6 | train/intermediate_algebra/1935.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | no | yes | 0.500 |  |
| 7 | train/intermediate_algebra/45.json | 0.000 | 0.047 | 1.000 | 2 | 0.3711 | no | no | 0.562 |  |
| 8 | train/intermediate_algebra/1345.json | 0.000 | 0.000 | 0.750 | 4 | 0.8189 | no | no | 0.438 |  |
| 9 | test/prealgebra/1195.json | 0.000 | 0.047 | 0.750 | 3 | 0.6087 | no | yes | 0.500 |  |
| 10 | train/precalculus/351.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | no | 0.562 | 68.0 |
| 11 | test/number_theory/826.json | 0.000 | 0.047 | 1.000 | 3 | 0.5899 | no | yes | 0.500 |  |
| 12 | test/algebra/427.json | 0.250 | 0.188 | 1.000 | 4 | 0.8357 | no | no | 0.750 |  |
| 13 | test/intermediate_algebra/1811.json | 0.000 | 0.047 | 0.750 | 2 | 0.3899 | no | no | 0.438 |  |
| 14 | train/precalculus/1296.json | 0.250 | 0.297 | 1.000 | 5 | 1.0745 | yes | no | 0.500 |  |
| 15 | test/precalculus/27.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | yes | yes | 0.500 | 66.0 |
| 16 | train/prealgebra/686.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | no | 0.500 |  |
| 17 | test/precalculus/274.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | no | 0.562 |  |
| 18 | train/number_theory/1027.json | 0.500 | 0.250 | 0.750 | 1 | 0.2108 | no | no | 1.000 |  |
| 19 | test/intermediate_algebra/1442.json | 0.000 | 0.000 | 1.000 | 3 | 0.5813 | no | no | 0.875 |  |
| 20 | test/intermediate_algebra/246.json | 0.000 | 0.000 | 1.000 | 1 | 0.1437 | no | no | 0.750 | 66.0 |

</details>

<details><summary>Max difficulty, seed 42 (20 steps done, status done)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | train/geometry/984.json | 0.500 | 0.562 | 0.500 | 5 | 1.1620 | yes | no | 1.125 |  |
| 2 | train/intermediate_algebra/541.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | no | 0.375 |  |
| 3 | train/counting_and_probability/949.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | no | no | 0.688 |  |
| 4 | test/intermediate_algebra/1475.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | no | yes | 0.500 |  |
| 5 | train/prealgebra/601.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | yes | no | 1.375 | 68.0 |
| 6 | train/algebra/2115.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | yes | no | 1.188 |  |
| 7 | test/intermediate_algebra/1974.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | yes | no | 0.375 |  |
| 8 | test/geometry/454.json | 0.000 | 0.000 | 0.750 | 5 | 1.0377 | yes | yes | 0.500 |  |
| 9 | train/algebra/179.json | 0.250 | 0.188 | 1.000 | 5 | 1.0545 | yes | no | 0.938 |  |
| 10 | train/algebra/2269.json | 0.500 | 0.250 | 0.750 | 4 | 0.8672 | no | no | 1.062 | 72.0 |
| 11 | train/geometry/240.json | 0.000 | 0.000 | 1.000 | 4 | 0.8001 | yes | yes | 0.500 |  |
| 12 | train/prealgebra/911.json | 0.750 | 0.188 | 0.500 | 3 | 0.6569 | yes | no | 1.000 |  |
| 13 | train/algebra/1654.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | no | yes | 0.500 |  |
| 14 | test/prealgebra/447.json | 0.000 | 0.000 | 0.250 | 4 | 0.8564 | no | yes | 0.500 |  |
| 15 | test/algebra/1623.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | yes | no | 1.375 | 68.0 |
| 16 | train/intermediate_algebra/1546.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | no | 0.375 |  |
| 17 | test/counting_and_probability/789.json | 0.000 | 0.000 | 0.750 | 5 | 1.0377 | yes | no | 0.625 |  |
| 18 | test/intermediate_algebra/1626.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | no | no | 0.625 |  |
| 19 | train/number_theory/402.json | 1.000 | 0.000 | 0.250 | 5 | 1.0802 | no | no | 1.250 |  |
| 20 | train/counting_and_probability/707.json | 0.500 | 0.250 | 0.750 | 5 | 1.0860 | yes | no | 1.125 | 66.0 |

</details>

<details><summary>Max difficulty, seed 43 (12 steps done, status running)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | train/geometry/550.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | yes | no | 1.125 |  |
| 2 | train/number_theory/167.json | 0.250 | 0.188 | 0.750 | 5 | 1.0733 | yes | no | 0.938 |  |
| 3 | train/geometry/572.json | 1.000 | 0.000 | 0.250 | 5 | 1.0802 | yes | no | 1.250 |  |
| 4 | train/counting_and_probability/5119.json | 0.500 | 0.250 | 0.750 | 5 | 1.0860 | yes | no | 0.875 |  |
| 5 | train/intermediate_algebra/9011.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | yes | yes | 0.500 | 74.0 |
| 6 | test/algebra/2066.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | no | no | 0.812 |  |
| 7 | train/precalculus/1195.json | 0.000 | 0.062 | 0.750 | 5 | 1.0491 | yes | no | 0.438 |  |
| 8 | train/intermediate_algebra/1251.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | no | yes | 0.500 |  |
| 9 | train/prealgebra/1731.json | 0.750 | 0.422 | 0.500 | 4 | 0.9187 | yes | no | 1.188 |  |
| 10 | train/intermediate_algebra/943.json | 0.000 | 0.000 | 0.750 | 5 | 1.0377 | yes | yes | 0.500 | 70.0 |
| 11 | test/algebra/578.json | 0.250 | 0.188 | 1.000 | 5 | 1.0545 | yes | no | 0.688 |  |
| 12 | train/intermediate_algebra/1058.json | 0.000 | 0.000 | 0.750 | 5 | 1.0377 | no | no | 0.875 |  |

</details>

### Selection diagnostics (counterfactual picks on identical batches, all runs)

| Rule pair | Steps | Same pick |
|---|---|---|
| sgac_eq10==max_level | 272 | 79.8 |
| sgac_eq10==max_var | 272 | 30.5 |
| sgac_eq10==max_d | 272 | 23.5 |
| sgac_eq10==random | 272 | 15.4 |
| sgac_eq10==pickle_true_mapping | 272 | 32.4 |
| sgac_eq10==table2_as_printed | 272 | 51.8 |

Eq. 10 picked a maximum-level candidate in 100.0% of 272 steps (analytically 100% at K=4).

| Arm | Picks | L5 | L4 | L<=3 | Ps=0 | 0<Ps<=.5 | Ps>.5 | D>=.75 | zero-loss bursts | zero-std groups | clipped | train reward |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Max disagreement | 60 | 35.0 | 18.3 | 46.7 | 55.0 | 41.7 | 3.3 | 96.7 | 15.0 | 32.5 | 7.9 | 0.675 |
| Max difficulty | 32 | 68.8 | 21.9 | 9.4 | 50.0 | 28.1 | 21.9 | 71.9 | 25.0 | 35.9 | 6.6 | 0.803 |
| Max variance | 60 | 20.0 | 30.0 | 50.0 | 5.0 | 70.0 | 25.0 | 66.7 | 3.3 | 20.8 | 6.2 | 0.994 |
| Random | 60 | 18.3 | 28.3 | 53.3 | 25.0 | 41.7 | 33.3 | 60.0 | 10.0 | 27.5 | 5.6 | 0.993 |
| SGAC (Eq. 10 as run) | 60 | 73.3 | 16.7 | 10.0 | 35.0 | 38.3 | 26.7 | 65.0 | 20.0 | 39.2 | 10.2 | 0.817 |
| Original NB-M log | 20 | 65.0 | 35.0 | 0.0 | 50.0 | 25.0 | 25.0 | 65.0 | 35.0 | — | — | — |

### F. Runtime and memory

| Run | Steps | Curriculum (h) | Evals (h) | Total (h) | Peak GPU mem (GB) | Status |
|---|---|---|---|---|---|---|
| max_d__seed42 | 20 | 0.60 | 0.68 | 1.28 | 7.83 | done |
| max_d__seed43 | 20 | 0.62 | 0.64 | 1.25 | 7.83 | done |
| max_d__seed44 | 20 | 0.68 | 0.66 | 1.34 | 7.83 | done |
| max_level__seed42 | 20 | 0.56 | 0.67 | 1.23 | 7.83 | done |
| max_level__seed43 | 12 | 0.39 | 0.09 | 0.47 | 7.83 | running |
| max_var__seed42 | 20 | 0.52 | 0.58 | 1.10 | 7.83 | done |
| max_var__seed43 | 20 | 0.65 | 0.75 | 1.40 | 7.83 | done |
| max_var__seed44 | 20 | 0.63 | 0.68 | 1.31 | 7.83 | done |
| random__seed42 | 20 | 0.59 | 0.66 | 1.25 | 7.83 | done |
| random__seed43 | 20 | 0.56 | 0.67 | 1.23 | 7.83 | done |
| random__seed44 | 20 | 0.62 | 0.65 | 1.26 | 7.83 | done |
| sgac__seed42 | 20 | 0.65 | 0.69 | 1.33 | 7.83 | done |
| sgac__seed43 | 20 | 0.66 | 0.69 | 1.35 | 7.83 | done |
| sgac__seed44 | 20 | 0.60 | 0.70 | 1.30 | 7.83 | done |

Original NB-M loop: 10.96 h on a Kaggle T4 (one run).

## Profile `as_run` (spec 64ba360f)

### A. Configuration

| Setting | Value |
|---|---|
| Model | Qwen/Qwen2.5-Math-1.5B (loaded: bf16; commit 4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2) |
| Quantisation requested / fallback | nf4 / nf4 unavailable (ModuleNotFoundError: No module named 'bitsandbytes'); loaded bf16 |
| Prompt (sieve, eval) / training | legacy / conversational |
| Grader (primary) / D metric | legacy / legacy_str |
| Token caps sieve / GRPO / eval | 1024 / 256 / 1024 |
| Sieve top_k / stops / pad | 50 / eos_only / eos |
| Loop | 20 steps, B=4, K=4, T=1.0, eval every 5 |
| Burst | 5 single-completion updates (pdbs 1, gen batch 4, G 4), lr 2e-05 linear |
| LoRA | r=16, alpha=32, ['q_proj', 'k_proj', 'v_proj', 'o_proj'] |
| test50 batch decision | 1 |
| E0 G1 agreement (base MATH-500) | n/a |

### D. Accuracy trajectory (original 50-item test set; seeds averaged)

| Arm | Base | Step 5 | Step 10 | Step 15 | Step 20 |
|---|---|---|---|---|---|
| SGAC (Eq. 10 as run) | 60.0 | 62.0 (n=1) | 64.0 (n=1) | 64.0 (n=1) | 64.0 (n=1) |
| Random | 60.0 | 64.0 (n=1) | 64.0 (n=1) | 66.0 (n=1) | 62.0 (n=1) |

Original NB-M run (one unseeded T4 run): 0: 64.0, 5: 66.0, 10: 62.0, 15: 60.0, 20: 68.0.

![trajectory](figures/trajectory_test50_as_run.png)

### E. Comparison at step 20

**MATH-500 (500 items)**, primary grader `legacy`:

| Model / arm | Accuracy % [95% CI] | Per seed | vs Base (pp) [95% CI] | SGAC − arm (pp) [95% CI] |
|---|---|---|---|---|
| Base | 56.0 | — | — | — |
| pi1 checkpoint (Wang et al.) | 62.8 | — | +6.8 [+3.6, +10.0] | — |
| SGAC (Eq. 10 as run) (seeds 42) | 57.6 [53.2, 62.0] | 57.6 | +1.6 [-0.8, +4.0] | — |
| Random (seeds 42) | 56.6 [52.2, 61.0] | 56.6 | +0.6 [-1.4, +2.8] | +1.0 [-1.2, +3.2] |

![final math500](figures/final_math500_as_run.png)

**original 50-item test set**, primary grader `legacy`:

| Model / arm | Accuracy % [95% CI] | Per seed | vs Base (pp) [95% CI] | SGAC − arm (pp) [95% CI] |
|---|---|---|---|---|
| Base | 60.0 | — | — | — |
| pi1 checkpoint (Wang et al.) | 66.0 | — | +6.0 [+0.0, +14.0] | — |
| SGAC (Eq. 10 as run) (seeds 42) | 64.0 [50.0, 76.0] | 64.0 | +4.0 [+0.0, +10.0] | — |
| Random (seeds 42) | 62.0 [48.0, 74.0] | 62.0 | +2.0 [+0.0, +6.0] | +2.0 [+0.0, +6.0] |

![final test50](figures/final_test50_as_run.png)

Grader sensitivity at the final step (same outputs, both graders):

| Run | Set | Primary | math-verify | legacy | Truncated |
|---|---|---|---|---|---|
| random seed 42 | test50 | 62.0 | 66.0 | 62.0 | 16.0 |
| random seed 42 | math500 | 56.6 | 62.6 | 56.6 | 13.2 |
| sgac seed 42 | test50 | 64.0 | 68.0 | 64.0 | 18.0 |
| sgac seed 42 | math500 | 57.6 | 64.0 | 57.6 | 12.2 |

### C. Selection trajectories (20 steps per run)

<details><summary>SGAC (Eq. 10 as run), seed 42 (20 steps done, status done)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | train/geometry/984.json | 1.000 | 0.000 | 0.250 | 5 | 1.0802 | yes | no | 0.062 |  |
| 2 | train/intermediate_algebra/541.json | 0.000 | 0.062 | 0.750 | 5 | 1.0491 | yes | no | 0.125 |  |
| 3 | train/algebra/778.json | 0.750 | 0.188 | 0.500 | 5 | 1.0945 | yes | no | 0.125 |  |
| 4 | train/prealgebra/597.json | 0.500 | 0.250 | 0.750 | 5 | 1.0860 | yes | no | 0.312 |  |
| 5 | train/prealgebra/601.json | 0.250 | 0.375 | 0.750 | 4 | 0.8888 | yes | no | 0.250 | 62.0 |
| 6 | train/algebra/2115.json | 0.750 | 0.188 | 0.500 | 3 | 0.6569 | yes | no | 0.625 |  |
| 7 | test/intermediate_algebra/1974.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | yes | yes | 0.000 |  |
| 8 | test/geometry/454.json | 0.000 | 0.000 | 0.750 | 5 | 1.0377 | yes | no | 0.250 |  |
| 9 | train/algebra/179.json | 0.000 | 0.062 | 1.000 | 5 | 1.0303 | yes | no | 0.188 |  |
| 10 | train/algebra/2269.json | 1.000 | 0.047 | 0.250 | 4 | 0.8700 | yes | no | 0.312 | 64.0 |
| 11 | train/geometry/240.json | 0.000 | 0.062 | 1.000 | 4 | 0.8115 | yes | no | 0.188 |  |
| 12 | train/prealgebra/911.json | 1.000 | 0.000 | 0.250 | 3 | 0.6426 | yes | no | 0.250 |  |
| 13 | train/algebra/1654.json | 0.000 | 0.000 | 0.750 | 5 | 1.0377 | yes | no | 0.188 |  |
| 14 | test/prealgebra/447.json | 0.000 | 0.000 | 0.500 | 4 | 0.8377 | yes | no | 0.312 |  |
| 15 | test/algebra/1623.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | yes | no | 0.375 | 64.0 |
| 16 | train/intermediate_algebra/1546.json | 0.000 | 0.000 | 0.500 | 5 | 1.0564 | yes | no | 0.062 |  |
| 17 | test/counting_and_probability/789.json | 0.000 | 0.000 | 0.500 | 5 | 1.0564 | yes | no | 0.125 |  |
| 18 | test/geometry/780.json | 0.000 | 0.000 | 0.750 | 5 | 1.0377 | yes | no | 0.062 |  |
| 19 | train/number_theory/402.json | 0.750 | 0.188 | 0.500 | 5 | 1.0945 | yes | no | 0.062 |  |
| 20 | train/counting_and_probability/707.json | 0.750 | 0.188 | 0.500 | 5 | 1.0945 | yes | no | 0.375 | 64.0 |

</details>

<details><summary>Random, seed 42 (20 steps done, status done)</summary>

| Step | Problem | Ps | Var | D | Level | Score (Eq.10) | Arm pick = Eq.10? | Zero-loss burst | Train reward | Eval acc (test50) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | train/geometry/721.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | no | no | 0.125 |  |
| 2 | test/intermediate_algebra/984.json | 0.500 | 0.250 | 0.750 | 3 | 0.6484 | no | no | 0.125 |  |
| 3 | test/counting_and_probability/247.json | 0.750 | 0.188 | 0.500 | 1 | 0.2193 | no | no | 0.250 |  |
| 4 | train/prealgebra/597.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | no | no | 0.375 |  |
| 5 | train/prealgebra/601.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | yes | no | 0.250 | 64.0 |
| 6 | train/intermediate_algebra/1954.json | 0.750 | 0.188 | 0.500 | 1 | 0.2193 | no | no | 0.438 |  |
| 7 | test/algebra/1345.json | 1.000 | 0.047 | 0.250 | 3 | 0.6512 | no | no | 0.375 |  |
| 8 | test/geometry/454.json | 0.000 | 0.000 | 0.750 | 5 | 1.0377 | yes | no | 0.188 |  |
| 9 | test/prealgebra/1195.json | 0.250 | 0.188 | 0.750 | 3 | 0.6357 | no | no | 0.188 |  |
| 10 | test/number_theory/14.json | 0.000 | 0.000 | 1.000 | 4 | 0.8001 | no | no | 0.125 | 64.0 |
| 11 | train/counting_and_probability/133.json | 0.500 | 0.250 | 0.750 | 1 | 0.2108 | no | no | 0.188 |  |
| 12 | train/algebra/2083.json | 1.000 | 0.000 | 0.250 | 2 | 0.4238 | no | no | 0.625 |  |
| 13 | train/algebra/1654.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | no | no | 0.250 |  |
| 14 | test/intermediate_algebra/182.json | 0.000 | 0.000 | 0.250 | 2 | 0.4188 | no | no | 0.062 |  |
| 15 | test/algebra/1623.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | yes | no | 0.312 | 66.0 |
| 16 | train/counting_and_probability/961.json | 0.750 | 0.188 | 0.500 | 3 | 0.6569 | no | no | 0.688 |  |
| 17 | test/counting_and_probability/498.json | 0.750 | 0.188 | 0.500 | 4 | 0.8758 | no | no | 0.250 |  |
| 18 | test/intermediate_algebra/1626.json | 0.000 | 0.047 | 1.000 | 5 | 1.0275 | no | no | 0.062 |  |
| 19 | train/precalculus/727.json | 0.250 | 0.422 | 0.500 | 3 | 0.6974 | no | no | 0.125 |  |
| 20 | train/algebra/1853.json | 0.000 | 0.000 | 1.000 | 5 | 1.0189 | no | no | 0.125 | 62.0 |

</details>

### Selection diagnostics (counterfactual picks on identical batches, all runs)

| Rule pair | Steps | Same pick |
|---|---|---|
| sgac_eq10==max_level | 40 | 82.5 |
| sgac_eq10==max_var | 40 | 20.0 |
| sgac_eq10==max_d | 40 | 22.5 |
| sgac_eq10==random | 40 | 20.0 |
| sgac_eq10==pickle_true_mapping | 40 | 27.5 |
| sgac_eq10==table2_as_printed | 40 | 50.0 |

Eq. 10 picked a maximum-level candidate in 100.0% of 40 steps (analytically 100% at K=4).

| Arm | Picks | L5 | L4 | L<=3 | Ps=0 | 0<Ps<=.5 | Ps>.5 | D>=.75 | zero-loss bursts | zero-std groups | clipped | train reward |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random | 20 | 30.0 | 20.0 | 50.0 | 40.0 | 20.0 | 40.0 | 50.0 | 0.0 | 12.5 | 82.5 | 0.256 |
| SGAC (Eq. 10 as run) | 20 | 65.0 | 25.0 | 10.0 | 50.0 | 10.0 | 40.0 | 45.0 | 5.0 | 15.0 | 82.5 | 0.212 |
| Original NB-M log | 20 | 65.0 | 35.0 | 0.0 | 50.0 | 25.0 | 25.0 | 65.0 | 35.0 | — | — | — |

### F. Runtime and memory

| Run | Steps | Curriculum (h) | Evals (h) | Total (h) | Peak GPU mem (GB) | Status |
|---|---|---|---|---|---|---|
| random__seed42 | 20 | 0.14 | 0.47 | 0.61 | 4.37 | done |
| sgac__seed42 | 20 | 0.14 | 0.53 | 0.66 | 4.37 | done |

Original NB-M loop: 10.96 h on a Kaggle T4 (one run).