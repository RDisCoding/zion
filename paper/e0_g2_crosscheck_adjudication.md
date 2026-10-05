# E0 G2 cross-check against the Qwen2.5-Math grader: adjudication

Prereg §7, G2: "cross-check against the Qwen2.5-Math grader on 200 base outputs (≤ 2% disagreement, adjudicated)".

**Run:** workstation, 2026-10-05. Sample = 200 of the 500 G1 base outputs under the pinned style `oneshot_rlvr_chat`
(`results/e0/g1/oneshot_rlvr_chat/math500/per_item.jsonl`, sampling seed 20261005). Reference grader = Qwen2.5-Math
`evaluation/` at commit `a45202bd16f1ec06f433442dc1152d0074773465` (`parse_ground_truth`, `extract_answer`,
`math_equal_process` with a 3 s timeout), run in an isolated environment. Tooling: `scripts/local/grader_crosscheck.sh`.

| | count |
|---|---|
| items | 200 |
| correct (ours / Qwen) | 117 / 118 |
| agreement | 197 (98.5%) |
| disagreements | 3 (1.5%): 2 `equivalence`, 1 `rule_truncated` |
| stored vs re-graded verdict mismatches | 0 |
| Qwen timeouts or errors | 0 |
| math-verify exceptions swallowed by our grader | 0 |

## Per-item adjudication

| # | item | gold | output | ours | Qwen | correct grader | grader error? |
|---|---|---|---|---|---|---|---|
| 1 | `test/prealgebra/1114` | `15\mbox{ cm}^2` | `\boxed{15}` "square centimeters" | correct | incorrect | **ours** | Qwen (gold normalisation) |
| 2 | `test/precalculus/625` | `\begin{pmatrix} 16/49 \\ 48/49 \\ 24/49 \end{pmatrix}` | `\boxed{\begin{pmatrix} \frac{16}{49} \\ \frac{48}{49} \\ \frac{24}{49} \end{pmatrix}}` | incorrect | correct | **Qwen** | **ours: false negative** |
| 3 | `test/intermediate_algebra/1510` | `0` | truncated at 3,072 tokens in a repetition loop, no `\boxed{}` | incorrect | correct | **ours** (pre-registered rule) | Qwen (spurious last-number credit) |

1. **Units in the gold.** The answer is 15 and the units are cm²; MATH convention does not require units in the
   final answer. Qwen's `strip_string` removes `\mbox{ cm}` but keeps `^2`, turning the gold into `15^2` (= 225), so
   it rejects a correct answer. Ours accepts `15` and, checked separately, rejects `225` and `15^2`.
   Not our error.
2. **Matrix answer.** The model's matrix is exactly the gold (16/49 = \frac{16}{49}, etc.); our verdict is wrong.
   Root cause, reproduced locally: `rlvr_v2/grader.py::normalize_answer` deletes every `\\` (pattern `\\\\` → ""),
   which removes the matrix row separators. Both normalised strings still parse, into meaningless 1×1 matrices
   (`(16/49)*(48/49)*(24/49)` vs a merged `16/4948/4924/49`), so `verify` returns False and `equivalent` never falls
   back to the raw strings because both parses were non-empty. On the raw strings math-verify parses both sides to
   the identical 3×1 matrix. **This is a genuine grader error.** It is systematic for matrix answers whose entries are
   written differently from the gold (slash vs `\frac`, decimals); identical forms still pass through the string
   path. Exposure: all gold answers with a LaTeX row break are `pmatrix`: 7 of 500 MATH-500 items (1.4%), 6 of the
   500 pool items, 5 of the 500 held-out items.
3. **Truncated loop.** The model never committed to an answer: it repeats the same failed configuration until the
   token limit. Qwen finds no box, falls back to the last number in the text, which is the `0` in
   "\sqrt{2} - \sqrt{2} = 0" inside the loop, and credits it because the gold is 0. Ours marks it incorrect by the
   pre-registered rule (truncated or unboxed ⇒ incorrect). Our verdict is right; Qwen's credit is coincidental.

## Conclusion

- Raw disagreement 1.5% (3/200); disagreements where adjudication finds **our** grader wrong: 1/200 = **0.5%**.
  Both are within the pre-registered ≤ 2%, so the G2 cross-check criterion is met.
- **One actual grader error** (item 2): a systematic false negative on matrix answers, caused by `normalize_answer`
  stripping `\\`. Items 1 and 3 are Qwen-side errors.
- Not observed in the sample but noted while reproducing item 1: a prediction that spells out units
  (`15\text{ cm}^2`) is not accepted against `15\mbox{ cm}^2`. Untested exposure; flagged, not adjudicated.

## E0 run 3 (final, after the matrix and currency fixes, commit 4af0d70)

Same procedure, same seed, hence the same 200 items, on the fresh `FRESH=1` G1 outputs (pinned style
`oneshot_rlvr_chat`, MATH-500 accuracy 0.630).

| | count |
|---|---|
| correct (ours / Qwen) | 118 / 118 |
| agreement | 198 (99.0%) |
| disagreements | 2 (1.0%): 1 `equivalence`, 1 `rule_truncated` |
| stored vs re-graded verdict mismatches / Qwen timeouts / swallowed grader exceptions | 0 / 0 / 0 |

| item | ours | Qwen | correct grader | grader error? |
|---|---|---|---|---|
| `test/prealgebra/1114` (gold `15\mbox{ cm}^2`, output `\boxed{15}`) | correct | incorrect | **ours** | Qwen: its `strip_string` turns the gold into `15^2` |
| `test/intermediate_algebra/1510` (truncated repetition loop, no box) | incorrect | correct | **ours** (pre-registered truncated / no-box rule) | Qwen: spurious last-number credit |

`test/precalculus/625`, the run-2 matrix false negative, is in the same sample and now agrees (both correct), which
confirms the `\\` fix on a real model output.

**Result:** raw disagreement 1.0%, our-grader errors after adjudication **0/200**. The G2 cross-check criterion
(≤ 2%, adjudicated) is met. Adjudicated 2026-10-05.

## Status

Recorded 2026-10-05. Fix approved and applied the same day: `normalize_answer` keeps `\\`. The proposed second part
(raw-string fallback after a failed normalised comparison) was tested and **not** applied, because it introduced
new false positives (`2007 + \frac\pi 2` accepted for gold `2`; a bare `-1` accepted for the 2×2 gold of
`train/precalculus/1049`). 18 matrix regression tests in `tests/test_grader_matrix.py`; the `\phantom` gold of
`train/precalculus/1049` stays ungradeable (known limitation, expected-fail test). Logged in the prereg deviations
log. This cross-check is repeated on the fresh G1 outputs of E0 run 3; G4 waits for that.
