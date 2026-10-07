"""Selection rules over one batch of candidates, with NB-M's tie-break (first maximum, `np.argmax`).

Every candidate is described by the four SGAC signals {"Ps", "Var", "D", "L"}. Rules are written with NAMED
coefficients so the label shift that broke the original fits cannot recur.

- sgac_eq10            the rule NB-M actually ran: 0.0050*Ps + 0.1832*Var - 0.0751*D + 0.2188*L (paper Eq. 10 as
                       printed; the 20-row regression's weights attached to the wrong features). With K=4 the
                       non-level terms span < 0.2188, so it always picks a maximum-level candidate, then prefers
                       higher Var, then lower D (tests/test_sgac_selection.py proves this exhaustively).
- pickle_true_mapping  the same 20-row regression applied to the features it was fit on
                       (learned_selector.pkl: 0.00504*L + 0.18324*Ps - 0.07506*Var + 0.21881*D + 0.279).
- table2_as_printed    paper Table 2 applied as labelled (never run by any original experiment; logged
                       counterfactually only).
- max_var / max_d / max_level   single-signal heuristics.
- random               uniform, from a dedicated RNG stream (schedule.random_arm_rng).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from . import nbm_original as nbm

SIGNALS = ("Ps", "Var", "D", "L")

LINEAR_RULES: dict[str, dict] = {
    "sgac_eq10": {"coef": dict(nbm.EQ10_AS_RUN), "intercept": 0.0},
    "pickle_true_mapping": {"coef": dict(zip(nbm.PICKLE_FEATURE_ORDER, nbm.PICKLE_COEF)),
                            "intercept": nbm.PICKLE_INTERCEPT},
    "table2_as_printed": {"coef": dict(nbm.TABLE2_AS_PRINTED), "intercept": 0.0},
}
SINGLE_RULES: dict[str, str] = {"max_var": "Var", "max_d": "D", "max_level": "L"}
RULES: tuple[str, ...] = tuple(LINEAR_RULES) + tuple(SINGLE_RULES) + ("random",)

# arm -> the rule that picks its training problem (fixed_pi1 has no selection)
ARM_RULE: dict[str, str | None] = {
    "sgac": "sgac_eq10",
    "random": "random",
    "max_var": "max_var",
    "max_d": "max_d",
    "max_level": "max_level",
    "sgac_label_corrected": "pickle_true_mapping",
    "fixed_pi1": None,
}
ARMS: tuple[str, ...] = tuple(ARM_RULE)


def _val(sig: Mapping[str, float], key: str) -> float:
    v = sig.get(key)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def score(rule: str, sig: Mapping[str, float]) -> float:
    """Score of one candidate under a deterministic rule (NaN inputs give NaN)."""
    if rule in LINEAR_RULES:
        spec = LINEAR_RULES[rule]
        return float(spec["intercept"] + sum(c * _val(sig, k) for k, c in spec["coef"].items()))
    if rule in SINGLE_RULES:
        return _val(sig, SINGLE_RULES[rule])
    raise ValueError(f"rule {rule!r} has no deterministic score (known: {RULES})")


def first_max(scores: Sequence[float]) -> int:
    """`np.argmax` semantics (first maximum) with NaN treated as -inf."""
    arr = np.array([(-math.inf if (s is None or math.isnan(float(s))) else float(s)) for s in scores], dtype=np.float64)
    if arr.size == 0:
        raise ValueError("no candidates")
    return int(np.argmax(arr))


def pick(rule: str, signals: Sequence[Mapping[str, float]], random_rng: np.random.Generator | None = None) -> tuple[int, list[float]]:
    """(index of the chosen candidate, per-candidate scores). `random` needs `random_rng` and scores all 0."""
    if not signals:
        raise ValueError("no candidates")
    if rule == "random":
        if random_rng is None:
            raise ValueError("the random rule needs a dedicated RNG (schedule.random_arm_rng)")
        return int(random_rng.integers(len(signals))), [0.0] * len(signals)
    scores = [score(rule, s) for s in signals]
    return first_max(scores), scores


def all_picks(signals: Sequence[Mapping[str, float]], random_rng: np.random.Generator) -> dict[str, dict]:
    """Every rule's choice on the same batch: {rule: {"idx", "scores"}} (counterfactual log for the report)."""
    out = {}
    for rule in RULES:
        idx, scores = pick(rule, signals, random_rng if rule == "random" else None)
        out[rule] = {"idx": idx, "scores": scores}
    return out


def picked_max_level(signals: Sequence[Mapping[str, float]], idx: int) -> bool:
    """Whether candidate `idx` has the batch's maximum level (always true for sgac_eq10 at K=4)."""
    levels = [_val(s, "L") for s in signals]
    finite = [v for v in levels if not math.isnan(v)]
    return bool(finite) and not math.isnan(levels[idx]) and levels[idx] == max(finite)
