"""Statistics for crossed seed x item designs (protocol section 7).

Every curriculum run answers the same MATH-500 items, and arms are paired by seed (shared candidate batches), so
seeds and items are crossed random factors. For a paired difference e[s, i] = y_A[s, i] - y_B[s, i] the two-way
random-effects model (one observation per cell) is

    e_si = delta + a_s + b_i + eps_si,   Var(mean) = sa2 / S + sb2 / N + se2 / (S N),

estimated from the mean squares as Var^ = (MS_seed + MS_item - MS_res) / (S N), with Satterthwaite degrees of freedom.
Simpler bootstraps are wrong in one direction each: resampling items of seed-averaged differences ignores sa2
(anticonservative); resampling seeds and items ("pigeonhole") counts se2 about three times (conservative). Both are
kept as sensitivity analyses.

`arms_components` estimates the same variance components jointly from several arms (arm x seed x item, one
observation per cell), which is what the seed-count projection uses.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy import stats


def _ms_two_way(e: np.ndarray) -> dict:
    S, N = e.shape
    g = float(e.mean())
    ds, ei = e.mean(axis=1), e.mean(axis=0)
    ms_seed = N * float(((ds - g) ** 2).sum()) / (S - 1)
    ms_item = S * float(((ei - g) ** 2).sum()) / (N - 1)
    resid = e - ds[:, None] - ei[None, :] + g
    ms_res = float((resid ** 2).sum()) / ((S - 1) * (N - 1))
    return {"S": S, "N": N, "mean": g, "per_seed": ds.tolist(), "ms_seed": ms_seed, "ms_item": ms_item, "ms_res": ms_res}


def _satterthwaite(ms_seed: float, ms_item: float, ms_res: float, S: int, N: int) -> tuple[float, float]:
    """Variance of the grand mean and its Satterthwaite df (floored at the residual-only variance)."""
    num = ms_seed + ms_item - ms_res
    floor = ms_res
    num = max(num, floor)
    var = num / (S * N)
    den = ms_seed ** 2 / (S - 1) + ms_item ** 2 / (N - 1) + ms_res ** 2 / ((S - 1) * (N - 1))
    df = num ** 2 / den if den > 0 else float("inf")
    return var, df


def crossed_test(e, alpha: float = 0.05) -> dict:
    """Primary analysis of a paired arm difference. `e` has shape (seeds, items) and holds per-item correctness
    differences on the proportion scale (values in {-1, 0, 1}). Returns the mean, SE, df, t, two-sided p, and the
    (1 - alpha) and 90% CIs (the latter for TOST)."""
    e = np.asarray(e, dtype=float)
    if e.ndim != 2 or e.shape[0] < 2 or e.shape[1] < 2:
        raise ValueError("need at least 2 seeds and 2 items")
    m = _ms_two_way(e)
    var, df = _satterthwaite(m["ms_seed"], m["ms_item"], m["ms_res"], m["S"], m["N"])
    se = math.sqrt(var)
    t = m["mean"] / se if se > 0 else float("inf") * np.sign(m["mean"])
    p = float(2 * stats.t.sf(abs(t), df)) if se > 0 else (0.0 if m["mean"] != 0 else 1.0)
    q = stats.t.ppf(1 - alpha / 2, df)
    q90 = stats.t.ppf(0.95, df)
    return {**m, "se": se, "df": df, "t": t, "p": p, "alpha": alpha,
            "ci": [m["mean"] - q * se, m["mean"] + q * se], "ci90": [m["mean"] - q90 * se, m["mean"] + q90 * se],
            "components": components_from_ms(m)}


def components_from_ms(m: Mapping) -> dict:
    S, N = m["S"], m["N"]
    se2 = m["ms_res"]
    return {"sigma2_seed": max(0.0, (m["ms_seed"] - se2) / N), "sigma2_item": max(0.0, (m["ms_item"] - se2) / S),
            "sigma2_res": se2}


def pigeonhole_bootstrap(e, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Sensitivity (a): resample seeds and items independently (conservative for crossed designs)."""
    e = np.asarray(e, dtype=float)
    S, N = e.shape
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        si = rng.integers(0, S, S)
        ii = rng.integers(0, N, N)
        out[b] = e[np.ix_(si, ii)].mean()
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(e.mean()), "ci": [float(lo), float(hi)], "se": float(out.std(ddof=1)), "n_boot": n_boot,
            "seed": seed}


def item_bootstrap(e, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Sensitivity (b): the reproduction's method -- resample items of the seed-averaged difference."""
    e = np.asarray(e, dtype=float)
    d = e.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, (n_boot, d.size))
    out = d[idx].mean(axis=1)
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(d.mean()), "ci": [float(lo), float(hi)], "se": float(out.std(ddof=1)), "n_boot": n_boot,
            "seed": seed}


# ---------------------------------------------------------------------- seed-count projection
def arms_components(Y: Mapping[str, np.ndarray]) -> dict:
    """Joint variance components from A arms x S seeds x N items (one observation per cell).

    `Y[arm]` has shape (S, N), rows in the same seed order for every arm (paired by seed). The variance of a paired
    difference between two arms decomposes into 2*s2_as (seed level), 2*s2_ai (item level) and 2*s2_e (residual),
    with E[MS_AS] = s2_e + N s2_as, E[MS_AI] = s2_e + S s2_ai, E[MS_ASI] = s2_e."""
    arms = sorted(Y)
    X = np.stack([np.asarray(Y[a], dtype=float) for a in arms])  # (A, S, N)
    A, S, N = X.shape
    if A < 2 or S < 2:
        raise ValueError("need at least 2 arms and 2 seeds")
    g = X.mean()
    m_a = X.mean(axis=(1, 2))
    m_s = X.mean(axis=(0, 2))
    m_i = X.mean(axis=(0, 1))
    m_as = X.mean(axis=2)
    m_ai = X.mean(axis=1)
    m_si = X.mean(axis=0)
    as_dev = m_as - m_a[:, None] - m_s[None, :] + g
    ai_dev = m_ai - m_a[:, None] - m_i[None, :] + g
    resid = (X - m_as[:, :, None] - m_ai[:, None, :] - m_si[None, :, :]
             + m_a[:, None, None] + m_s[None, :, None] + m_i[None, None, :] - g)
    df_as, df_ai, df_asi = (A - 1) * (S - 1), (A - 1) * (N - 1), (A - 1) * (S - 1) * (N - 1)
    ms_as = N * float((as_dev ** 2).sum()) / df_as
    ms_ai = S * float((ai_dev ** 2).sum()) / df_ai
    ms_asi = float((resid ** 2).sum()) / df_asi
    s2_e = ms_asi
    return {"arms": arms, "A": A, "S": S, "N": N, "ms_as": ms_as, "ms_ai": ms_ai, "ms_asi": ms_asi,
            "df_as": df_as, "df_ai": df_ai, "df_asi": df_asi,
            "s2_as": max(0.0, (ms_as - s2_e) / N), "s2_ai": max(0.0, (ms_ai - s2_e) / S), "s2_e": s2_e}


def projected(s2_seed: float, s2_item: float, s2_res: float, S: int, N: int) -> tuple[float, float]:
    """(SE, df) of the crossed estimator for a paired difference whose components are given (already doubled)."""
    ms_seed = N * s2_seed + s2_res
    ms_item = S * s2_item + s2_res
    var, df = _satterthwaite(ms_seed, ms_item, s2_res, S, N)
    return math.sqrt(var), df


def mde(se: float, df: float, alpha: float, power: float) -> float:
    return float((stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * se)


def seed_count(Y: Mapping[str, np.ndarray], seeds_min: int, seeds_max: int, alpha: float, power: float,
               mde_max: float, upper_q: float = 0.80) -> dict:
    """Pre-declared rule (protocol section 7): the smallest S >= seeds_min whose MDE <= mde_max, capped at seeds_max.

    The decision uses the point estimates of the joint components. A sensitivity row uses the `upper_q` upper
    confidence bound of the arm x seed mean square (chi-square with df_as degrees of freedom)."""
    c = arms_components(Y)
    N = c["N"]
    ms_as_hi = c["ms_as"] * c["df_as"] / stats.chi2.ppf(1 - upper_q, c["df_as"])
    s2_as_hi = max(0.0, (ms_as_hi - c["s2_e"]) / N)
    rows = []
    for S in range(2, seeds_max + 1):
        se, df = projected(2 * c["s2_as"], 2 * c["s2_ai"], 2 * c["s2_e"], S, N)
        se_hi, df_hi = projected(2 * s2_as_hi, 2 * c["s2_ai"], 2 * c["s2_e"], S, N)
        rows.append({"S": S, "se": se, "df": df, "mde": mde(se, df, alpha, power),
                     "se_upper": se_hi, "df_upper": df_hi, "mde_upper": mde(se_hi, df_hi, alpha, power)})
    chosen = next((r["S"] for r in rows if seeds_min <= r["S"] and r["mde"] <= mde_max), seeds_max)
    return {"components": c, "s2_as_upper": s2_as_hi, "upper_q": upper_q, "alpha": alpha, "power": power,
            "mde_max": mde_max, "seeds_min": seeds_min, "seeds_max": seeds_max, "table": rows, "chosen_seeds": chosen,
            "chosen_mde": next(r["mde"] for r in rows if r["S"] == chosen),
            "rule": "smallest S >= seeds_min with MDE <= mde_max (point estimates), capped at seeds_max"}
