# SGAC reproduction: freeze record

The historical SGAC reproduction is frozen **as it stands** (team decision, 2026-10-12). No further compute is spent on it. This file records exactly what the frozen track contains.

It sits next to the protocol rather than inside it. `paper/sgac_repro_protocol.md` stays byte-identical to tag `sgac-protocol-v1`, which the calibration track's guard test asserts.

## Frozen artefacts

| Artefact | Identifier |
|---|---|
| Protocol | `paper/sgac_repro_protocol.md` at tag `sgac-protocol-v1` (7b108ba) |
| Code at report time | branch `sgac-repro`, commit ca4431c |
| Spec hashes | e0 `577f3319`, as_run `64ba360f` |
| Report | `report.md`, generated 2026-10-09 00:42 IST, sha256 `aebc0533a0fbe326c85901feb14348c70713140b397a4f511a6f39813ff23328` |
| Raw results and report bundle | `sgac_repro_frozen.tar` = `results_sgac/` + `reports/sgac_repro/`, made on the workstation after the queue was confirmed stopped; kept there and not in git. sha256 `76c601aa99bf8d885b0e6cfc796259251666f74944c995295ca403a84e07d636` |
| Step-20 MATH-500 per-item copies used by the calibration track | `sgac_step20_results/` and `sgac_step20_extra/` (table below). Accuracies match the report exactly under both graders. |

## Step-20 MATH-500 per-item copies

Each file is a copy of `results_sgac/e0/577f3319/<arm>__seed<S>/evals/step-020/math500/per_item.jsonl`, renamed to `<arm>_seed<S>.jsonl`. The SGAC and Random files are in `sgac_step20_results/`; the Max-variance and Max-disagreement files are in `sgac_step20_extra/`.

| File | sha256 |
|---|---|
| `random_seed42.jsonl` | `8245bfbe2f0ff436b7676467b36755fb3ee92933d5efb6066593619bc5b22be7` |
| `random_seed43.jsonl` | `4c1a2595196ac045ee56d87d1406d77af758a5e9ae3313385079c7f2802cd456` |
| `random_seed44.jsonl` | `ec92d1f681399f9c60f2d46a8a02ea8fd14fe83ff8a9dcc5aed4711434273445` |
| `sgac_seed42.jsonl` | `6e5ca912a8318b3b54cab034e6121a2a8ce43681b89f1a9cbd64a3a27cfde7ac` |
| `sgac_seed43.jsonl` | `8e6befad7b5c0c94edd3aff7b7f8c24ff806709905f79832cff1a8eb5c1fc1d8` |
| `sgac_seed44.jsonl` | `359314ff9b4450a80784a19301556a511f67918d1f0fee213d7c806936c71c56` |
| `max_var_seed42.jsonl` | `e58bbe57770ef206dc3704146a73339604276dda32872410f9d3499012bed8b9` |
| `max_var_seed43.jsonl` | `3d0b3a7b6f656cfed48a6745f31dba46ba0c32c2cdfeccde61b6a0d79824e8bf` |
| `max_var_seed44.jsonl` | `dfce4ac776f451a107d988b9c7073ea2234d33b1031625181344cf882ca5a580` |
| `max_d_seed42.jsonl` | `ab492d8e799bd09434d2c86a204328ee209e0166651e128585b9598689d93509` |
| `max_d_seed43.jsonl` | `db3c11d0773aea72eb51ad3cc7bc931773959a981ed781331d7ae7b2ed60f788` |
| `max_d_seed44.jsonl` | `dd40bc1075aad6f66f4bdea1039c8acbb752d58e47e519d84be373102ad7fabc` |

## Verdict (unchanged)

**"Original SGAC could not be reproduced."** This is the e0 profile, MATH-500 at step 20, seeds 42–44:

| Endpoint | Δ accuracy [95% CI] |
|---|---|
| P1: SGAC − Base | +0.9 pp [−1.4, +3.1] |
| P2: SGAC − Random | +0.7 pp [−0.8, +2.3] |

## Runs not completed (reported as not run, under pre-declared rule C5)

**e0, Max difficulty.**
- Seed 43 is incomplete: the report shows 12/20 steps and no step-20 evaluation. The archive is the authority on how far it got before the queue stopped. Seed 44 never started.
- The report's step-20 Max-difficulty comparisons therefore rest on seed 42 alone.
- Its test50 trajectory mixes n = 2 (steps 5, 10) with n = 1 (steps 15, 20).
- Max difficulty is a secondary arm, so the verdict does not depend on it.

**as_run.** Tier T5 did not run: the remaining seed-42 heuristic arms, and seeds 43 and 44. The secondary fidelity check rests on seed 42 SGAC and Random only.

**Tier T6.** None of the optional work ran:
- the label-corrected selector;
- `fixed_pi1`;
- the Phase-1 replication;
- the MATH-500 evaluations at steps 5, 10 and 15.

## Known issue found after the report (does not alter the frozen verdict)

The leakage audit of 2026-10-10 is documented in `paper/sgac_calib_protocol.md` §1. It found that the historical 1000-item pool contains near-duplicates of MATH-500 items: 10 items at difflib similarity ≥ 0.9 and 34 at ≥ 0.8.

Three e0 runs trained on one of the ≥ 0.9 items, five picks in total:

| Run | Steps | Trained item | Matching MATH-500 item |
|---|---|---|---|
| SGAC, seed 44 | 12 | `test/algebra/491` | `test/algebra/1282` (0.96) |
| Random, seed 42 | 1 | `train/geometry/721` | `test/geometry/65` (0.92) |
| Random, seed 42 | 14 | `test/intermediate_algebra/182` | `test/intermediate_algebra/1102` (0.91) |
| Max disagreement, seed 42 | 1 and 14 | the same two items | — |

**Effect.** Each exposure can move a 500-item score by at most a few items. The frozen protocol is not re-analysed.

**Remedy.** The calibration track excludes these items from its curriculum pool and reruns every arm, rather than reusing these runs as comparators.
