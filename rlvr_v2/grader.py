"""Correctness grading and answer-equivalence clustering.

Rules
- Correctness requires a balanced `\\boxed{...}`; the LAST box in the text counts.
- Unboxed or truncated outputs are incorrect. "Last number in text" is only a diagnostic.
- Equivalence uses `math-verify` (symbolic), with a normalised string comparison as fast path
  and fallback. On Windows math-verify cannot use signal-based timeouts, so they are disabled.
"""
from __future__ import annotations

import inspect
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class GradeResult:
    correct: bool
    format_ok: bool
    boxed: str | None
    method: str
    truncated: bool = False
    number_present: bool = False  # diagnostic: an unboxed number exists in the text


def extract_last_boxed(text: str) -> str | None:
    """Content of the last balanced `\\boxed{...}` (or `\\fbox{...}`). None if absent or unbalanced."""
    if not text:
        return None
    starts = [m for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    for m in reversed(starts):
        i = m.end()
        depth = 1
        buf = []
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(buf).strip()
            buf.append(ch)
            i += 1
        # unbalanced: try an earlier box
    return None


_STRIP_PATTERNS = [
    (re.compile(r"\\left|\\right|\\!|\\,|\\;|\\:|\\quad|\\qquad"), ""),
    (re.compile(r"\\text\{\s*([^{}]*)\}"), r"\1"),
    (re.compile(r"\\mathrm\{\s*([^{}]*)\}"), r"\1"),
    (re.compile(r"\\(?:d|t)frac"), r"\\frac"),
    (re.compile(r"\^\{?\\circ\}?|°"), ""),
    (re.compile(r"\\%|%"), ""),
    (re.compile(r"\$"), ""),
    (re.compile(r"\\\\"), ""),
]


def normalize_answer(s: str | None) -> str:
    if s is None:
        return ""
    out = str(s).strip()
    for pat, rep in _STRIP_PATTERNS:
        out = pat.sub(rep, out)
    out = out.replace(" ", "").replace("\n", "")
    out = out.rstrip(".")
    out = re.sub(r"^([a-zA-Z])=", "", out)  # "x=5" -> "5"
    # drop thousands separators in plain integers like 1,000
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", out):
        out = out.replace(",", "")
    # 0.50 -> 0.5 ; 5.0 -> 5
    if re.fullmatch(r"-?\d+\.\d*0+", out):
        out = out.rstrip("0").rstrip(".")
    return out


def _filter_kwargs(fn, **kwargs):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in kwargs.items() if k in params}


class GraderUnavailableError(RuntimeError):
    """math-verify cannot establish known equivalences in this process; grading would silently degrade to
    exact string matching, so every caller must stop."""


# Known-equivalent pairs that need the symbolic (LaTeX) path; the grader refuses to exist if any fails.
SELF_CHECK_PAIRS = (("0.5", "\\frac{1}{2}"), ("\\frac{\\sqrt{2}}{2}", "\\frac{1}{\\sqrt{2}}"))
_log = logging.getLogger(__name__)


class MathVerifyGrader:
    """math-verify backed grader with normalised-string fast path. Exceptions inside math-verify still mean
    "not equivalent" for that answer, but they are counted (`self.errors`) and logged, and a failing self-check
    at construction raises `GraderUnavailableError`."""

    def __init__(self, timeout_s: int | None = 5, float_rounding: int = 6, self_check: bool = True):
        self.timeout_s = None if os.name == "nt" else timeout_s
        self.float_rounding = float_rounding
        self.errors: Counter = Counter()
        self.last_error: str | None = None
        from math_verify import parse, verify  # type: ignore  # hard dependency: no silent fallback

        if self.timeout_s is None:  # math-verify warns on every call when timeouts are disabled (Windows)
            for name in ("math_verify", "math_verify.parser", "math_verify.grader", "math_verify.utils"):
                logging.getLogger(name).setLevel(logging.ERROR)
        self._parse, self._verify = parse, verify
        if self_check:
            self.self_check()

    def _record(self, where: str, exc: BaseException) -> None:
        key = f"{where}:{type(exc).__name__}"
        self.errors[key] += 1
        self.last_error = f"{key}: {exc!r}"
        if self.errors[key] <= 3:
            _log.warning("math-verify %s raised %r (occurrence %d; treated as not equivalent)", where, exc,
                         self.errors[key])

    def self_check(self) -> None:
        for a, b in SELF_CHECK_PAIRS:
            if not self.equivalent(a, b):
                raise GraderUnavailableError(
                    f"math-verify self-check failed: {a!r} vs {b!r} not equivalent in this process. "
                    + self.diagnose(a, b))

    def diagnose(self, a: str, b: str) -> str:
        """Re-run parse/verify for one pair WITHOUT exception handling and report what happens."""
        import importlib.metadata as md
        import sys

        info = {"timeout_s": self.timeout_s, "os": os.name, "python": sys.executable, "cwd": os.getcwd(),
                "sys.path[0]": sys.path[0] if sys.path else None, "last_swallowed": self.last_error,
                "errors": dict(self.errors)}
        for p in ("math-verify", "latex2sympy2_extended", "antlr4-python3-runtime", "sympy"):
            try:
                info[p] = md.version(p)
            except Exception as e:  # pragma: no cover
                info[p] = f"missing ({e!r})"
        try:
            import math_verify

            info["math_verify_file"] = math_verify.__file__
            pkw = _filter_kwargs(self._parse, parsing_timeout=self.timeout_s)
            pa, pb = self._parse(f"${a}$", **pkw), self._parse(f"${b}$", **pkw)
            info["parsed"] = [repr(pa), repr(pb)]
            vkw = _filter_kwargs(self._verify, float_rounding=self.float_rounding, timeout_seconds=self.timeout_s)
            info["verify"] = repr(self._verify(pa, pb, **vkw))
        except BaseException as e:  # noqa: BLE001 - diagnosis must report anything, incl. timeouts
            import traceback

            info["exception"] = "".join(traceback.format_exception(e))[-2000:]
        return "Diagnosis: " + repr(info)

    # ------------------------------------------------------------- equivalence
    @lru_cache(maxsize=65536)
    def _parsed(self, s: str):
        kw = _filter_kwargs(self._parse, parsing_timeout=self.timeout_s)
        for wrapped in (f"${s}$", "\\boxed{" + s + "}", s):
            try:
                out = self._parse(wrapped, **kw)
            except Exception as e:
                self._record("parse", e)
                out = None
            if out:
                return out
        return None

    def equivalent(self, a: str | None, b: str | None) -> bool:
        if a is None or b is None:
            return False
        na, nb = normalize_answer(a), normalize_answer(b)
        if na == "" or nb == "":
            return False
        if na == nb:
            return True
        pa, pb = self._parsed(na), self._parsed(nb)
        if not pa or not pb:
            pa, pb = self._parsed(a), self._parsed(b)
        if not pa or not pb:
            return False
        kw = _filter_kwargs(self._verify, float_rounding=self.float_rounding, timeout_seconds=self.timeout_s)
        try:
            return bool(self._verify(pa, pb, **kw)) or bool(self._verify(pb, pa, **kw))
        except Exception as e:
            self._record("verify", e)
            return False

    # ------------------------------------------------------------- grading
    def grade(self, pred_text: str, gold_answer: str, truncated: bool = False) -> GradeResult:
        boxed = extract_last_boxed(pred_text or "")
        number_present = bool(_NUMBER.search(pred_text or ""))
        if boxed is None or boxed == "":
            return GradeResult(False, False, None, "no_box", truncated, number_present)
        if truncated:
            return GradeResult(False, True, boxed, "truncated", True, number_present)
        if normalize_answer(boxed) == normalize_answer(gold_answer):
            return GradeResult(True, True, boxed, "string", False, number_present)
        ok = self.equivalent(boxed, gold_answer)
        return GradeResult(ok, True, boxed, "math_verify" if ok else "mismatch", False, number_present)

    def is_correct(self, pred_text: str, gold_answer: str, truncated: bool = False) -> bool:
        return self.grade(pred_text, gold_answer, truncated).correct


NONE_CLASS = -1


def cluster_answers(preds: Iterable[str | None], grader: MathVerifyGrader) -> list[int]:
    """Equivalence-class label per prediction (union-find over pairwise equivalence).
    All unparsable/None predictions share the single label NONE_CLASS (-1)."""
    preds = list(preds)
    n = len(preds)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    valid = [i for i, p in enumerate(preds) if p is not None and normalize_answer(p) != ""]
    for ii, i in enumerate(valid):
        for j in valid[ii + 1 :]:
            if find(i) == find(j):
                continue
            if grader.equivalent(preds[i], preds[j]):
                union(i, j)
    labels = [NONE_CLASS] * n
    next_label = 0
    root_to_label: dict[int, int] = {}
    for i in valid:
        r = find(i)
        if r not in root_to_label:
            root_to_label[r] = next_label
            next_label += 1
        labels[i] = root_to_label[r]
    return labels
