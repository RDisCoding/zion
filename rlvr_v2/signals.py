"""Per-candidate rollout signals with named fields.

Definitions (K rollouts, answer equivalence classes from `grader.cluster_answers`):
- p_s          fraction correct
- v_bin        p_s * (1 - p_s)  (binary-reward variance; identical information to p_s)
- v_legacy     population variance of (correct + 0.5 * format_ok), the v1 project's "reward variance"
- u_ratio      number of distinct answer classes / K (legacy, K-dependent; None outputs count as one class)
- d_simpson    K/(K-1) * (1 - sum p_c^2), probability two rollouts disagree (approx. K-invariant)
- entropy_bits plug-in Shannon entropy of the class distribution; entropy_mm adds the Miller-Madow correction
- d_wrong      Simpson disagreement among parsable incorrect rollouts (NaN if fewer than `min_wrong`)
- maj_share    share of the modal class; maj_margin = (top1 - top2)/K; maj_correct = modal class is correct
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, fields
from typing import Sequence

from .data import Problem
from .grader import NONE_CLASS, MathVerifyGrader, cluster_answers, extract_last_boxed


@dataclass(frozen=True)
class Rollout:
    text: str
    n_tokens: int
    truncated: bool
    boxed: str | None = None
    correct: bool | None = None
    answer_class: int | None = None
    finish_reason: str = "stop"
    stop_id: int | None = None  # which stop token ended the sequence (None if truncated/unknown)


@dataclass(frozen=True)
class Signals:
    unique_id: str
    k: int
    policy_tag: str
    p_s: float
    n_correct: int
    v_bin: float
    v_legacy: float
    u_ratio: float
    n_classes: int
    d_simpson: float
    entropy_bits: float
    entropy_mm: float
    d_wrong: float  # NaN when undefined
    n_wrong_parsable: int
    maj_share: float
    maj_margin: float
    maj_correct: bool
    format_rate: float
    trunc_rate: float
    none_rate: float
    len_mean: float
    len_sd: float
    len_correct_mean: float  # NaN when no correct rollouts
    len_wrong_mean: float  # NaN when no wrong rollouts
    level: int | None
    subject: str | None
    prompt_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Signals":
        """Inverse of to_dict(); JSON nulls for the float fields that may be undefined become NaN."""
        names = {f.name for f in fields(cls)}
        out = {k: v for k, v in d.items() if k in names}
        for k in ("d_wrong", "len_correct_mean", "len_wrong_mean", "d_simpson", "entropy_bits", "entropy_mm"):
            if k in out and out[k] is None:
                out[k] = float("nan")
        return cls(**out)


NUMERIC_FEATURES: tuple[str, ...] = (
    "p_s", "v_bin", "v_legacy", "u_ratio", "d_simpson", "entropy_bits", "entropy_mm", "d_wrong",
    "maj_share", "maj_margin", "format_rate", "trunc_rate", "none_rate", "len_mean", "len_sd", "level",
)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _pvar(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _simpson(counts: Sequence[int]) -> float:
    n = sum(counts)
    if n < 2:
        return float("nan")
    return n / (n - 1) * (1.0 - sum((c / n) ** 2 for c in counts))


def _entropy_bits(counts: Sequence[int]) -> float:
    n = sum(counts)
    return -sum((c / n) * math.log2(c / n) for c in counts if c > 0) if n else float("nan")


def grade_rollouts(rollouts: Sequence[Rollout], gold: str, grader: MathVerifyGrader) -> list[Rollout]:
    """Fill boxed/correct/answer_class for each rollout."""
    graded = []
    for r in rollouts:
        g = grader.grade(r.text, gold, truncated=r.truncated)
        graded.append(Rollout(r.text, r.n_tokens, r.truncated, g.boxed, g.correct, None, r.finish_reason, r.stop_id))
    preds = [r.boxed if (r.boxed and not r.truncated) else None for r in graded]
    classes = cluster_answers(preds, grader)
    return [Rollout(r.text, r.n_tokens, r.truncated, r.boxed, r.correct, c, r.finish_reason, r.stop_id)
            for r, c in zip(graded, classes)]


def compute_signals(
    rollouts: Sequence[Rollout],
    problem: Problem,
    policy_tag: str = "base",
    prompt_tokens: int = 0,
    format_bonus: float = 0.5,
    min_wrong: int = 4,
) -> Signals:
    """Rollouts must already be graded (see `grade_rollouts`)."""
    k = len(rollouts)
    if k < 2:
        raise ValueError("need at least 2 rollouts")
    if any(r.correct is None or r.answer_class is None for r in rollouts):
        raise ValueError("rollouts must be graded first")
    correct = [bool(r.correct) for r in rollouts]
    n_correct = sum(correct)
    p_s = n_correct / k
    fmt = [1.0 if r.boxed else 0.0 for r in rollouts]
    v_legacy = _pvar([float(c) + format_bonus * f for c, f in zip(correct, fmt)])

    classes = [r.answer_class for r in rollouts]
    counts = Counter(classes)  # NONE_CLASS counts as one class (legacy behaviour)
    n_classes = len(counts)
    ordered = counts.most_common()
    top1 = ordered[0][1]
    top2 = ordered[1][1] if len(ordered) > 1 else 0
    modal_class = ordered[0][0]
    maj_correct = any(r.correct for r in rollouts if r.answer_class == modal_class) and modal_class != NONE_CLASS

    wrong_counts = Counter(r.answer_class for r in rollouts if not r.correct and r.answer_class != NONE_CLASS)
    n_wrong_parsable = sum(wrong_counts.values())
    d_wrong = _simpson(list(wrong_counts.values())) if n_wrong_parsable >= min_wrong else float("nan")

    lens = [float(r.n_tokens) for r in rollouts]
    len_c = [float(r.n_tokens) for r in rollouts if r.correct]
    len_w = [float(r.n_tokens) for r in rollouts if not r.correct]
    ent = _entropy_bits(list(counts.values()))
    return Signals(
        unique_id=problem.unique_id,
        k=k,
        policy_tag=policy_tag,
        p_s=p_s,
        n_correct=n_correct,
        v_bin=p_s * (1.0 - p_s),
        v_legacy=v_legacy,
        u_ratio=n_classes / k,
        n_classes=n_classes,
        d_simpson=_simpson(list(counts.values())),
        entropy_bits=ent,
        entropy_mm=ent + (n_classes - 1) / (2.0 * k * math.log(2)),
        d_wrong=d_wrong,
        n_wrong_parsable=n_wrong_parsable,
        maj_share=top1 / k,
        maj_margin=(top1 - top2) / k,
        maj_correct=bool(maj_correct),
        format_rate=_mean(fmt),
        trunc_rate=_mean([1.0 if r.truncated else 0.0 for r in rollouts]),
        none_rate=counts.get(NONE_CLASS, 0) / k,
        len_mean=_mean(lens),
        len_sd=math.sqrt(_pvar(lens)),
        len_correct_mean=_mean(len_c),
        len_wrong_mean=_mean(len_w),
        level=problem.level,
        subject=problem.subject,
        prompt_tokens=prompt_tokens,
    )


def rollouts_from_texts(texts: Sequence[str], gold: str, grader: MathVerifyGrader, n_tokens: Sequence[int] | None = None,
                        truncated: Sequence[bool] | None = None) -> list[Rollout]:
    """Convenience for tests/scripts: build and grade rollouts from raw completion texts."""
    n = len(texts)
    n_tokens = list(n_tokens) if n_tokens is not None else [len(t.split()) for t in texts]
    truncated = list(truncated) if truncated is not None else [False] * n
    raw = [Rollout(t, nt, tr, extract_last_boxed(t), None, None, "length" if tr else "stop")
           for t, nt, tr in zip(texts, n_tokens, truncated)]
    return grade_rollouts(raw, gold, grader)
