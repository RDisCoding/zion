"""Candidate selection rules. Every rule exposes `scores()` and `select()`; the learned selector
stores NAMED coefficients and exports them to JSON/LaTeX so positional mix-ups are impossible."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import SelectorCfg
from .signals import NUMERIC_FEATURES, Signals

ARM_NAMES = ("random", "variance", "disagreement", "ps_band", "learned")


class Selector:
    name: str = "base"

    def scores(self, signals: Sequence[Signals], rng: np.random.Generator) -> list[float]:
        raise NotImplementedError

    def select(self, signals: Sequence[Signals], rng: np.random.Generator) -> tuple[int, list[float]]:
        if not signals:
            raise ValueError("no candidates")
        scores = self.scores(signals, rng)
        arr = np.asarray(scores, dtype=float)
        arr = np.where(np.isnan(arr), -np.inf, arr)
        best = float(arr.max())
        ties = np.flatnonzero(arr == best)
        idx = int(ties[0]) if len(ties) == 1 else int(rng.choice(ties))
        return idx, [float(s) for s in scores]


class RandomSelector(Selector):
    name = "random"

    def scores(self, signals, rng):
        return list(rng.random(len(signals)))


class VarianceSelector(Selector):
    """Max within-group binary-reward variance == p_s nearest 0.5."""

    name = "variance"

    def scores(self, signals, rng):
        return [float(s.v_bin) for s in signals]


class DisagreementSelector(Selector):
    name = "disagreement"

    def __init__(self, metric: str = "d_simpson"):
        if metric not in NUMERIC_FEATURES:
            raise ValueError(f"unknown disagreement metric {metric!r}")
        self.metric = metric

    def scores(self, signals, rng):
        return [float(getattr(s, self.metric)) for s in signals]


class PsBandSelector(Selector):
    """Prefer candidates inside [lo, hi] closest to `target`; outside the band is heavily penalised,
    and within the band higher d_wrong breaks ties (NaN d_wrong -> 0)."""

    name = "ps_band"

    def __init__(self, lo: float = 0.2, hi: float = 0.8, target: float = 0.5):
        self.lo, self.hi, self.target = lo, hi, target

    def scores(self, signals, rng):
        out = []
        for s in signals:
            inside = self.lo <= s.p_s <= self.hi
            dw = 0.0 if (s.d_wrong is None or math.isnan(s.d_wrong)) else float(s.d_wrong)
            out.append((0.0 if inside else -10.0) - abs(s.p_s - self.target) + 0.1 * dw)
        return out


def _tex_escape(name: str) -> str:
    return name.replace("_", "\\_")


@dataclass
class LearnedLinearSelector(Selector):
    """score = intercept + sum_f coef[f] * z(f), z = (x - mean)/std when `standardize` is given.
    NaN features are imputed with the training mean (i.e. z = 0)."""

    features: tuple[str, ...]
    coef: dict[str, float]
    intercept: float = 0.0
    standardize: dict[str, dict[str, float]] | None = None
    meta: dict | None = None
    name: str = "learned"

    def __post_init__(self):
        unknown = [f for f in self.features if f not in NUMERIC_FEATURES]
        if unknown:
            raise ValueError(f"unknown features {unknown}; allowed: {NUMERIC_FEATURES}")
        if set(self.coef) != set(self.features):
            raise ValueError(f"coef keys {sorted(self.coef)} must match features {sorted(self.features)}")

    def _z(self, name: str, value) -> float:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return 0.0
        if self.standardize and name in self.standardize:
            st = self.standardize[name]
            sd = st.get("std", 1.0) or 1.0
            return (float(value) - st.get("mean", 0.0)) / sd
        return float(value)

    def scores(self, signals, rng):
        return [self.intercept + sum(self.coef[f] * self._z(f, getattr(s, f)) for f in self.features) for s in signals]

    # --------------------------------------------------------------- io
    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "features": list(self.features),
            "coef": self.coef,
            "intercept": self.intercept,
            "standardize": self.standardize,
            "meta": self.meta or {},
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)

    @classmethod
    def from_json(cls, path: str | Path) -> "LearnedLinearSelector":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(
            features=tuple(d["features"]),
            coef={k: float(v) for k, v in d["coef"].items()},
            intercept=float(d.get("intercept", 0.0)),
            standardize=d.get("standardize"),
            meta=d.get("meta"),
        )

    def to_latex(self, path: str | Path, ci: dict[str, tuple[float, float]] | None = None, caption: str = "") -> str:
        """Booktabs table of named coefficients (optionally with CIs). Returns the LaTeX source."""
        header = "Feature & Coefficient" + (" [95\\% CI]" if ci else "") + " \\\\"
        rows = []
        for f in self.features:
            c = self.coef[f]
            ci_s = f" [{ci[f][0]:+.3f}, {ci[f][1]:+.3f}]" if ci and f in ci else ""
            rows.append(f"{_tex_escape(f)} & {c:+.4f}{ci_s} \\\\")
        rows.append(f"intercept & {self.intercept:+.4f} \\\\")
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\begin{tabular}{lr}",
            "\\toprule",
            header,
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
        if caption:
            lines.append("\\caption{" + caption + "}")
        lines.append("\\end{table}")
        tex = "\n".join(lines) + "\n"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(tex, encoding="utf-8")
        return tex


def build_selector(cfg: SelectorCfg, name: str | None = None) -> Selector:
    name = name or cfg.name
    if name == "random":
        return RandomSelector()
    if name == "variance":
        return VarianceSelector()
    if name == "disagreement":
        return DisagreementSelector(cfg.disagreement_metric)
    if name == "ps_band":
        return PsBandSelector(cfg.ps_lo, cfg.ps_hi, cfg.ps_target)
    if name == "learned":
        if not cfg.learned_path:
            raise ValueError("learned selector requires selector.learned_path")
        return LearnedLinearSelector.from_json(cfg.learned_path)
    raise ValueError(f"unknown selector {name!r}; expected one of {ARM_NAMES}")
