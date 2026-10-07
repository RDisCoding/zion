"""Both graders on every text: E0's math-verify grader and NB-M's legacy check.

The profile's `grader` decides `correct` (rewards, Ps and the primary eval verdict); the other verdict is always
recorded so every evaluation can be re-read under the other convention (grader-sensitivity diagnostics).
- math_verify: `grader.MathVerifyGrader.grade(text, answer, truncated)` -- last balanced box, box required,
  truncated = incorrect, gold = dataset `answer` column (E0 G2-validated).
- legacy: `is_correct(extract_answer(text), solution)` -- first box, else last number, whitespace-insensitive string
  match, sympy path only if antlr4 4.11 is installed; truncation is ignored (NB-M never knew about it).
"""
from __future__ import annotations

from ..grader import MathVerifyGrader
from . import legacy
from .data import SgacItem


class DualGrader:
    def __init__(self, primary: str, mv: MathVerifyGrader | None = None):
        if primary not in ("math_verify", "legacy"):
            raise ValueError(f"unknown grader {primary!r}")
        self.primary = primary
        self.mv = mv if mv is not None else MathVerifyGrader()  # fail-loud self-check at construction
        self.legacy_symbolic = legacy.symbolic_status()

    def grade(self, text: str, item: SgacItem, truncated: bool) -> dict:
        text = text or ""
        g = self.mv.grade(text, item.answer, truncated=bool(truncated))
        legacy_answer = legacy.extract_answer(text)
        legacy_ok = bool(legacy.is_correct(legacy_answer, item.legacy_solution))
        rec = {
            "mv_correct": bool(g.correct), "mv_format_ok": bool(g.format_ok), "mv_boxed": g.boxed, "mv_method": g.method,
            "legacy_correct": legacy_ok, "legacy_answer": legacy_answer, "legacy_box_substr": "\\boxed{" in text,
        }
        rec["correct"] = rec["mv_correct"] if self.primary == "math_verify" else rec["legacy_correct"]
        return rec

    def describe(self) -> dict:
        return {"primary": self.primary, "legacy_symbolic": self.legacy_symbolic, "mv_errors": dict(self.mv.errors)}
