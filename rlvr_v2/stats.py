"""Statistics for the pre-registered analyses (pure numpy/scipy, no statsmodels).

Conventions
- Every function is deterministic given its ``seed`` argument and returns plain Python floats,
  dicts and lists (JSON serialisable), never numpy scalars.
- Confidence intervals are percentile bootstrap intervals over observations unless stated otherwise.
- Observations with NaN in any variable used by a test are dropped (the returned ``n`` is the
  number actually used); shape mismatches raise ``ValueError``.
- p-values that cannot be computed (too few observations, degenerate data) are returned as ``None``.
"""
from __future__ import annotations

import logging
import math
import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import stats as sps

log = logging.getLogger(__name__)

__all__ = [
    "bootstrap_mean_ci",
    "paired_item_bootstrap",
    "spearman_ci",
    "partial_spearman",
    "hump_test",
    "matched_pairs_test",
    "loo_linear_selector",
    "tost_correlation",
    "holm",
    "icc_oneway",
    "noise_decomposition",
    "auc_over_steps",
]

_EPS = 1e-12


# ---------------------------------------------------------------------------------------- helpers
def _as_1d(x, name: str = "x") -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def _finite_pairs(x, y, xname: str = "x", yname: str = "y") -> tuple[np.ndarray, np.ndarray]:
    xa, ya = _as_1d(x, xname), _as_1d(y, yname)
    if xa.shape != ya.shape:
        raise ValueError(f"{xname} and {yname} must have the same length, got {xa.shape} vs {ya.shape}")
    m = np.isfinite(xa) & np.isfinite(ya)
    return xa[m], ya[m]


def _f(v) -> float:
    """numpy scalar -> plain float (NaN stays NaN)."""
    return float(v)


def _percentile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(s, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return _f(lo), _f(hi)


def _boot_index_chunks(rng: np.random.Generator, n: int, n_boot: int, chunk: int = 1000) -> Iterable[np.ndarray]:
    """Yield (m, n) integer index matrices, m <= chunk, totalling n_boot rows."""
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        yield rng.integers(0, n, size=(m, n))
        done += m


def _rowwise_spearman(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row of X with the same row of Y (average ranks for ties).
    Rows where either side is constant give NaN."""
    rx = sps.rankdata(X, axis=1)
    ry = sps.rankdata(Y, axis=1)
    rx = rx - rx.mean(axis=1, keepdims=True)
    ry = ry - ry.mean(axis=1, keepdims=True)
    num = (rx * ry).sum(axis=1)
    den = np.sqrt((rx**2).sum(axis=1) * (ry**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out


def _spearman_scalar(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho of two 1-D arrays; NaN if either is constant or n < 2."""
    if x.size < 2:
        return float("nan")
    return _f(_rowwise_spearman(x[None, :], y[None, :])[0])


# ------------------------------------------------------------------------------- basic bootstraps
def bootstrap_mean_ci(x, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of ``x`` with a percentile-bootstrap ``1 - alpha`` confidence interval.

    NaN entries are dropped. Returns ``(mean, lo, hi)``; with a single observation lo == hi == mean.
    """
    xa = _as_1d(x)
    xa = xa[np.isfinite(xa)]
    if xa.size == 0:
        raise ValueError("bootstrap_mean_ci: no finite observations")
    mean = _f(xa.mean())
    if xa.size == 1 or n_boot <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boots = np.concatenate([xa[idx].mean(axis=1) for idx in _boot_index_chunks(rng, xa.size, n_boot)])
    lo, hi = _percentile_ci(boots, alpha)
    return mean, lo, hi


def paired_item_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0,
                          alpha: float = 0.05) -> dict:
    """Paired item bootstrap for two per-item correctness vectors of equal length.

    ``delta = mean(a) - mean(b)``; items are resampled with replacement (the same items in ``a``
    and ``b``, which is identical to resampling the per-item differences) and the percentile
    ``1 - alpha`` interval of the resampled mean difference is reported.
    ``p_two_sided = min(1, 2 * min(P*(delta* <= 0), P*(delta* >= 0)))`` is the bootstrap
    percentile p-value (the smallest level at which the percentile interval excludes 0).
    Items with NaN in either vector are dropped.
    """
    aa, bb = _as_1d(a, "a"), _as_1d(b, "b")
    if aa.shape != bb.shape:
        raise ValueError(f"a and b must have the same length, got {aa.shape} vs {bb.shape}")
    m = np.isfinite(aa) & np.isfinite(bb)
    d = aa[m] - bb[m]
    n = int(d.size)
    if n == 0:
        raise ValueError("paired_item_bootstrap: no finite item pairs")
    delta = _f(d.mean())
    if n == 1 or n_boot <= 0:
        return {"delta": delta, "ci_lo": delta, "ci_hi": delta, "p_two_sided": None, "n": n,
                "mean_a": _f(aa[m].mean()), "mean_b": _f(bb[m].mean()), "n_boot": int(n_boot)}
    rng = np.random.default_rng(seed)
    boots = np.concatenate([d[idx].mean(axis=1) for idx in _boot_index_chunks(rng, n, n_boot)])
    lo, hi = _percentile_ci(boots, alpha)
    p = min(1.0, 2.0 * min(_f(np.mean(boots <= 0.0)), _f(np.mean(boots >= 0.0))))
    return {"delta": delta, "ci_lo": lo, "ci_hi": hi, "p_two_sided": _f(p), "n": n,
            "mean_a": _f(aa[m].mean()), "mean_b": _f(bb[m].mean()), "n_boot": int(n_boot)}


# ----------------------------------------------------------------------------------- correlations
def spearman_ci(x, y, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Spearman rho with a percentile-bootstrap CI over observations.

    ``rho``/``p`` come from ``scipy.stats.spearmanr`` (two-sided, t approximation); the CI resamples
    (x_i, y_i) pairs with replacement and re-ranks inside every resample. Resamples in which either
    variable is constant are dropped from the bootstrap distribution. NaN pairs are dropped first.
    With fewer than 3 pairs rho is NaN and ``p`` is None.
    """
    xa, ya = _finite_pairs(x, y)
    n = int(xa.size)
    out = {"rho": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p": None, "n": n,
           "n_boot": int(n_boot)}
    if n < 3:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sps.spearmanr(xa, ya)
    rho = _f(res.statistic)
    out["rho"] = rho
    out["p"] = None if not np.isfinite(res.pvalue) else _f(res.pvalue)
    if not np.isfinite(rho) or n_boot <= 0:
        return out
    rng = np.random.default_rng(seed)
    boots = np.concatenate([_rowwise_spearman(xa[idx], ya[idx]) for idx in _boot_index_chunks(rng, n, n_boot)])
    out["ci_lo"], out["ci_hi"] = _percentile_ci(boots, alpha)
    out["n_boot_valid"] = int(np.isfinite(boots).sum())
    return out


def _control_design(C: np.ndarray, rank_controls: bool, degree: int) -> np.ndarray:
    """[1, controls] design. With ``rank_controls`` every control column is replaced by its scaled
    average rank in (0, 1) and the powers 1..degree of each ranked column are appended."""
    n = C.shape[0]
    if rank_controls:
        R = (sps.rankdata(C, axis=0) - 0.5) / n
        cols = [R**d for d in range(1, degree + 1)]
        return np.column_stack([np.ones(n)] + cols)
    return np.column_stack([np.ones(n), C])


def _partial_rank_corr(x: np.ndarray, y: np.ndarray, D: np.ndarray) -> float:
    rx, ry = sps.rankdata(x), sps.rankdata(y)
    bx, *_ = np.linalg.lstsq(D, rx, rcond=None)
    by, *_ = np.linalg.lstsq(D, ry, rcond=None)
    ex, ey = rx - D @ bx, ry - D @ by
    den = math.sqrt(float((ex**2).sum()) * float((ey**2).sum()))
    return _f((ex * ey).sum() / den) if den > _EPS else float("nan")


def partial_spearman(x, y, controls: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                     rank_controls: bool = True, control_degree: int = 2) -> dict:
    """Partial Spearman correlation of ``x`` and ``y`` given ``controls`` (n, k).

    ``x`` and ``y`` are rank-transformed (average ranks), each is residualised on ``[1, controls]``
    by (minimum-norm) least squares, and the Pearson correlation of the two residual vectors is
    returned. Everything lives in rank space: by default each control column is rank-transformed
    too (scaled to (0, 1)) and the powers ``1..control_degree`` of every ranked control are included,
    so a non-monotone (hump-shaped) dependence on a control such as ``p_s`` is removed as a
    polynomial in rank(p_s). Passing both ``p_s`` and ``p_s**2`` is harmless (identical ranks give
    duplicated columns, which do not change the least-squares fit). With ``rank_controls=False`` the
    control columns enter untransformed and no powers are added (supply polynomial terms yourself);
    note that ranks of x and y are then non-linear in the raw controls, which can leave shared
    structure in the residuals when a control is unbounded (e.g. normal).
    ``p`` is the usual partial-correlation t-test, t = r * sqrt(df / (1 - r^2)) with
    df = n - 2 - k_eff, k_eff = rank of the control design minus one (an approximation for ranks).
    The CI is a percentile bootstrap over rows (re-ranked per resample). Rows with NaN in x, y or any
    control are dropped.
    """
    xa, ya = _as_1d(x, "x"), _as_1d(y, "y")
    C = np.asarray(controls, dtype=float)
    if C.ndim == 1:
        C = C[:, None]
    if C.ndim != 2 or C.shape[0] != xa.size or ya.size != xa.size:
        raise ValueError(f"shape mismatch: x {xa.shape}, y {ya.shape}, controls {C.shape}")
    if control_degree < 1:
        raise ValueError("control_degree must be >= 1")
    m = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(C).all(axis=1)
    xa, ya, C = xa[m], ya[m], C[m]
    n = int(xa.size)
    out = {"rho": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p": None, "n": n,
           "k": int(C.shape[1]), "n_boot": int(n_boot), "rank_controls": bool(rank_controls),
           "control_degree": int(control_degree)}
    if n < C.shape[1] * control_degree + 3:
        return out
    D = _control_design(C, rank_controls, control_degree)
    k_eff = int(np.linalg.matrix_rank(D)) - 1
    out["k_eff"] = k_eff
    rho = _partial_rank_corr(xa, ya, D)
    out["rho"] = rho
    df = n - 2 - k_eff
    if np.isfinite(rho) and abs(rho) < 1.0 and df > 0:
        t = rho * math.sqrt(df / (1.0 - rho**2))
        out["p"] = _f(2.0 * sps.t.sf(abs(t), df))
    if not np.isfinite(rho) or n_boot <= 0:
        return out
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = _partial_rank_corr(xa[idx], ya[idx], _control_design(C[idx], rank_controls, control_degree))
    out["ci_lo"], out["ci_hi"] = _percentile_ci(boots, alpha)
    out["n_boot_valid"] = int(np.isfinite(boots).sum())
    return out


# ---------------------------------------------------------------------------------------- H1 / H2
def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def hump_test(p_s, delta, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
              mid: tuple[float, float] = (0.25, 0.75), centre: float = 0.5) -> dict:
    """Test for a hump-shaped dependence of ``delta`` on ``p_s``.

    (1) OLS ``delta ~ 1 + c + c^2`` with ``c = p_s - centre``. ``quad_coef`` is the coefficient of
        ``c^2``; its classical standard error is ``sqrt(s^2 [(X'X)^-1]_22)`` with
        ``s^2 = RSS / (n - 3)``, ``quad_p`` the two-sided t-test (df = n - 3) and
        ``quad_p_one_sided`` the pre-registered one-sided alternative ``quad_coef < 0``.
        ``quad_ci_lo/hi`` is a pairs-bootstrap percentile interval (resample observations, refit).
    (2) Mann-Whitney U (two-sided, scipy ``mannwhitneyu``) comparing ``delta`` for ``p_s`` in the
        middle bin against all other observations (the "extreme" bins including the anchors). The
        middle bin uses the design convention ``mid[0] < p_s <= mid[1]`` (lower bound exclusive,
        upper inclusive). ``mid_vs_extreme_p_greater`` is the one-sided alternative "middle > rest".
    """
    p, d = _finite_pairs(p_s, delta, "p_s", "delta")
    n = int(p.size)
    nan = float("nan")
    out = {"quad_coef": nan, "quad_se": nan, "quad_ci_lo": nan, "quad_ci_hi": nan, "quad_p": None,
           "quad_p_one_sided": None, "lin_coef": nan, "intercept": nan, "r2": nan, "n": n,
           "mid_vs_extreme_u": None, "mid_vs_extreme_p": None, "mid_vs_extreme_p_greater": None,
           "n_mid": 0, "n_extreme": 0, "mean_mid": nan, "mean_extreme": nan, "mid_bin": [float(mid[0]), float(mid[1])],
           "centre": float(centre)}
    if n == 0:
        return out
    mid_mask = (p > mid[0]) & (p <= mid[1])
    out["n_mid"], out["n_extreme"] = int(mid_mask.sum()), int((~mid_mask).sum())
    if mid_mask.any():
        out["mean_mid"] = _f(d[mid_mask].mean())
    if (~mid_mask).any():
        out["mean_extreme"] = _f(d[~mid_mask].mean())
    if mid_mask.any() and (~mid_mask).any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mw = sps.mannwhitneyu(d[mid_mask], d[~mid_mask], alternative="two-sided")
            mw_g = sps.mannwhitneyu(d[mid_mask], d[~mid_mask], alternative="greater")
        out["mid_vs_extreme_u"] = _f(mw.statistic)
        out["mid_vs_extreme_p"] = _f(mw.pvalue)
        out["mid_vs_extreme_p_greater"] = _f(mw_g.pvalue)

    c = p - centre
    X = np.column_stack([np.ones(n), c, c**2])
    if n < 4 or np.linalg.matrix_rank(X) < 3:
        return out
    beta = _ols(X, d)
    resid = d - X @ beta
    dof = n - 3
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tss = float(((d - d.mean()) ** 2).sum())
    out.update({"intercept": _f(beta[0]), "lin_coef": _f(beta[1]), "quad_coef": _f(beta[2]), "quad_se": _f(se[2]),
                "r2": _f(1.0 - float(resid @ resid) / tss) if tss > 0 else nan})
    if se[2] > 0:
        t = beta[2] / se[2]
        out["quad_p"] = _f(2.0 * sps.t.sf(abs(t), dof))
        out["quad_p_one_sided"] = _f(sps.t.cdf(t, dof))
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        boots = np.full(n_boot, nan)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            Xb = X[idx]
            if np.linalg.matrix_rank(Xb) < 3:
                continue
            boots[b] = _ols(Xb, d[idx])[2]
        out["quad_ci_lo"], out["quad_ci_hi"] = _percentile_ci(boots, alpha)
    return out


def matched_pairs_test(delta_high: np.ndarray, delta_low: np.ndarray, n_boot: int = 5000, seed: int = 0,
                       alpha: float = 0.05) -> dict:
    """Matched-pairs comparison of outcomes (high-d_wrong member minus low-d_wrong member).

    ``mean_diff = mean(delta_high - delta_low)`` with a percentile-bootstrap CI over pairs;
    ``wilcoxon_p`` is the two-sided Wilcoxon signed-rank p-value (scipy, zero differences discarded)
    and ``wilcoxon_p_greater`` the one-sided "high > low" alternative; both are None when fewer than
    6 pairs are available (the exact two-sided test cannot reach p < 0.05 below n = 6) or when all
    differences are zero. ``sign_test_p`` is the exact two-sided sign test (binomial) as a small-n
    fallback. Pairs with a NaN outcome are dropped.
    """
    h, lo_ = _finite_pairs(delta_high, delta_low, "delta_high", "delta_low")
    diff = h - lo_
    n = int(diff.size)
    if n == 0:
        raise ValueError("matched_pairs_test: no complete pairs")
    mean_diff = _f(diff.mean())
    out = {"mean_diff": mean_diff, "ci_lo": mean_diff, "ci_hi": mean_diff, "wilcoxon_p": None,
           "wilcoxon_stat": None, "wilcoxon_p_greater": None, "n_pairs": n, "median_diff": _f(np.median(diff)),
           "n_positive": int((diff > 0).sum()), "n_negative": int((diff < 0).sum()), "n_zero": int((diff == 0).sum()),
           "sign_test_p": None, "n_boot": int(n_boot)}
    if n > 1 and n_boot > 0:
        rng = np.random.default_rng(seed)
        boots = np.concatenate([diff[idx].mean(axis=1) for idx in _boot_index_chunks(rng, n, n_boot)])
        out["ci_lo"], out["ci_hi"] = _percentile_ci(boots, alpha)
    nz = out["n_positive"] + out["n_negative"]
    if nz > 0:
        out["sign_test_p"] = _f(sps.binomtest(out["n_positive"], nz, 0.5).pvalue)
    if n >= 6 and nz > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w2 = sps.wilcoxon(diff, alternative="two-sided")
            wg = sps.wilcoxon(diff, alternative="greater")
        out["wilcoxon_stat"] = _f(w2.statistic)
        out["wilcoxon_p"] = _f(w2.pvalue)
        out["wilcoxon_p_greater"] = _f(wg.pvalue)
    return out


# ------------------------------------------------------------------------------------------- H3
def _fold_designs(X: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each leave-one-out fold i: (train mask, standardised train matrix, standardised held-out row).
    Imputation means and scaler statistics (population std) are computed from the n - 1 training rows only;
    all-NaN training columns are imputed with 0 and zero-variance columns get std = 1."""
    n = X.shape[0]
    folds = []
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        Xtr = X[tr]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu_imp = np.nanmean(Xtr, axis=0)
        mu_imp = np.where(np.isfinite(mu_imp), mu_imp, 0.0)
        Xtr_f = np.where(np.isnan(Xtr), mu_imp, Xtr)
        mean = Xtr_f.mean(axis=0)
        std = Xtr_f.std(axis=0)
        std = np.where(std > _EPS, std, 1.0)
        Ztr = np.asfortranarray((Xtr_f - mean) / std)
        xi = np.where(np.isnan(X[i]), mu_imp, X[i])
        folds.append((tr, Ztr, (xi - mean) / std))
    return folds


def _make_model(model: str, alpha: float):
    from sklearn.linear_model import Lasso, Ridge

    if model == "ridge":
        return Ridge(alpha=float(alpha), fit_intercept=True)
    if model == "lasso":
        return Lasso(alpha=float(alpha), fit_intercept=True, max_iter=20000, tol=1e-6)
    raise ValueError(f"model must be 'ridge' or 'lasso', got {model!r}")


def _sklearn_loo_predictions(folds, y: np.ndarray, alpha: float, model: str) -> np.ndarray:
    """LOO predictions with sklearn; y is standardised with the training fold's mean/std and the
    prediction mapped back to the original units."""
    n = y.size
    pred = np.empty(n)
    for i, (tr, Ztr, zi) in enumerate(folds):
        ytr = y[tr]
        my, sy = float(ytr.mean()), float(ytr.std())
        if sy <= _EPS:
            pred[i] = my
            continue
        est = _make_model(model, alpha)
        est.fit(Ztr, (ytr - my) / sy)
        pred[i] = my + sy * float(est.predict(zi[None, :])[0])
    return pred


def _lasso_path_loo(folds, y: np.ndarray, alphas: Sequence[float]) -> dict[float, np.ndarray]:
    """LOO predictions of sklearn's Lasso for every alpha at once, via ``enet_path`` (the same coordinate
    descent solver as ``Lasso.fit``; with centred inputs the intercept-free path equals ``Lasso`` with an
    intercept). y is standardised per training fold exactly as in ``_sklearn_loo_predictions``."""
    from sklearn.linear_model import enet_path

    alphas_desc = sorted({float(a) for a in alphas}, reverse=True)
    n = y.size
    preds = {a: np.empty(n) for a in alphas_desc}
    for i, (tr, Ztr, zi) in enumerate(folds):
        ytr = y[tr]
        my, sy = float(ytr.mean()), float(ytr.std())
        if sy <= _EPS:
            for a in alphas_desc:
                preds[a][i] = my
            continue
        ys = np.ascontiguousarray((ytr - my) / sy)
        _, coefs, _ = enet_path(Ztr, ys, l1_ratio=1.0, alphas=alphas_desc, check_input=False, copy_X=False,
                                tol=1e-6, max_iter=20000)
        pred_std = zi @ coefs  # (n_alphas,)
        for k, a in enumerate(alphas_desc):
            preds[a][i] = my + sy * float(pred_std[k])
    return preds


def _ridge_loo_smoother(folds, n: int, p: int, alpha: float) -> np.ndarray:
    """The LOO prediction of ridge (with intercept, standardised train columns) is linear in y:
    yhat_i = mean(y_tr) + z_i' (Z'Z + alpha I)^-1 Z' y_tr. Returns the (n, n) matrix H with yhat = H y."""
    H = np.zeros((n, n))
    eye = np.eye(p)
    for i, (tr, Ztr, zi) in enumerate(folds):
        w_map = np.linalg.solve(Ztr.T @ Ztr + alpha * eye, Ztr.T)  # (p, n-1): y_tr -> coefficients
        H[i, tr] = zi @ w_map + 1.0 / (n - 1)
    return H


def _loo_stats(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """(mse, r2, spearman) of LOO predictions; a constant prediction vector gets spearman 0."""
    err = y - pred
    mse = _f(np.mean(err**2))
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = _f(1.0 - float(err @ err) / tss) if tss > 0 else float("nan")
    rho = _spearman_scalar(y, pred)
    return mse, r2, (0.0 if not np.isfinite(rho) else rho)


def loo_linear_selector(X: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
                        alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0), model: str = "ridge",
                        n_perm: int = 1000, seed: int = 0) -> dict:
    """Leave-one-out evaluation of a penalised linear selector with a label-permutation test.

    Procedure
    - Inside every LOO fold the n - 1 training rows provide the NaN-imputation means and the
      standardisation (mean / population std) applied to both the training rows and the held-out row;
      y is standardised with the training fold's mean/std as well (a no-op for ridge, but it puts the
      Lasso penalty on a scale-free footing: alpha_max <= 1 for standardised data). Models are
      sklearn ``Ridge`` / ``Lasso`` with an intercept.
    - alpha is chosen as the grid value minimising the LOO mean squared error on the full data; the
      reported LOO statistics use that same alpha, which is slightly optimistic (no nested CV).
    - ``loo_r2 = 1 - sum((y - yhat_loo)^2) / sum((y - mean(y))^2)`` and ``loo_spearman`` is the Spearman
      correlation of ``y`` with the LOO predictions (a constant prediction vector counts as 0).
    - Permutation test: the labels are shuffled ``n_perm`` times and the WHOLE procedure (LOO
      predictions over the alpha grid, alpha re-selected by LOO MSE) is re-run on each shuffle;
      ``perm_p = (1 + #{perm: loo_spearman_perm >= loo_spearman}) / (n_perm + 1)`` (Phipson & Smyth
      correction). For ridge the LOO prediction is linear in y, so the permutation null is computed
      exactly with the LOO smoother matrix (checked against the sklearn predictions); for Lasso every
      fold is refit with sklearn's ``enet_path`` (whole alpha grid per call).
    - The exported model is refit on all rows (imputation / standardisation from the full data);
      ``coef[name]`` is the change in y (original units) per one standard deviation of the feature and
      ``intercept`` the prediction at the feature means. ``standardize`` holds the full-data mean/std.
    """
    Xa = np.asarray(X, dtype=float)
    ya = _as_1d(y, "y")
    names = [str(f) for f in feature_names]
    if Xa.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {Xa.shape}")
    n, p = Xa.shape
    if ya.size != n:
        raise ValueError(f"X has {n} rows but y has {ya.size} entries")
    if len(names) != p:
        raise ValueError(f"X has {p} columns but {len(names)} feature names were given")
    if len(set(names)) != p:
        raise ValueError("feature names must be unique")
    if n < 4:
        raise ValueError("loo_linear_selector needs at least 4 observations")
    if not np.isfinite(ya).all():
        raise ValueError("y must be finite")
    if ya.std() <= _EPS:
        raise ValueError("y is constant")
    if model not in ("ridge", "lasso"):
        raise ValueError(f"model must be 'ridge' or 'lasso', got {model!r}")
    alphas = [float(a) for a in alphas]
    if not alphas:
        raise ValueError("alphas must be non-empty")

    folds = _fold_designs(Xa)
    if model == "ridge":
        preds = {a: _sklearn_loo_predictions(folds, ya, a, model) for a in alphas}
    else:
        preds = _lasso_path_loo(folds, ya, alphas)
    grid = {}
    for a in alphas:
        mse, r2, rho = _loo_stats(ya, preds[a])
        grid[a] = {"loo_mse": mse, "loo_r2": r2, "loo_spearman": rho}
    best_alpha = min(alphas, key=lambda a: (grid[a]["loo_mse"], a))
    loo_pred = preds[best_alpha]
    loo_mse, loo_r2, loo_rho = _loo_stats(ya, loo_pred)

    smoothers = None
    if model == "ridge":
        smoothers = {a: _ridge_loo_smoother(folds, n, p, a) for a in alphas}
        check = smoothers[best_alpha] @ ya
        if not np.allclose(check, loo_pred, atol=1e-6, rtol=1e-6):
            raise RuntimeError("ridge LOO smoother disagrees with sklearn predictions")

    perm_p = None
    null = np.empty(0)
    if n_perm > 0:
        rng = np.random.default_rng(seed)
        null = np.empty(n_perm)
        if model == "ridge":
            Yp = np.column_stack([ya[rng.permutation(n)] for _ in range(n_perm)])  # (n, n_perm)
            mses = np.empty((len(alphas), n_perm))
            P = {}
            for ai, a in enumerate(alphas):
                P[a] = smoothers[a] @ Yp
                mses[ai] = np.mean((Yp - P[a]) ** 2, axis=0)
            best = np.argmin(mses, axis=0)
            chosen = np.empty((n, n_perm))
            for ai, a in enumerate(alphas):
                cols = best == ai
                chosen[:, cols] = P[a][:, cols]
            rho_null = _rowwise_spearman(Yp.T, chosen.T)
            null = np.where(np.isfinite(rho_null), rho_null, 0.0)
        else:
            for b in range(n_perm):
                yp = ya[rng.permutation(n)]
                best_mse, best_rho = math.inf, 0.0
                for a, pr in _lasso_path_loo(folds, yp, alphas).items():
                    mse, _, rho = _loo_stats(yp, pr)
                    if mse < best_mse:
                        best_mse, best_rho = mse, rho
                null[b] = best_rho
                if (b + 1) % 100 == 0:
                    log.info("lasso permutation %d/%d", b + 1, n_perm)
        perm_p = _f((1.0 + float(np.sum(null >= loo_rho))) / (n_perm + 1.0))

    # final model on all rows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu_imp = np.nanmean(Xa, axis=0)
    mu_imp = np.where(np.isfinite(mu_imp), mu_imp, 0.0)
    Xf = np.where(np.isnan(Xa), mu_imp, Xa)
    mean, std = Xf.mean(axis=0), Xf.std(axis=0)
    std = np.where(std > _EPS, std, 1.0)
    Z = (Xf - mean) / std
    my, sy = float(ya.mean()), float(ya.std())
    est = _make_model(model, best_alpha)
    est.fit(Z, (ya - my) / sy)
    coef = {nm: _f(c * sy) for nm, c in zip(names, np.ravel(est.coef_))}
    intercept = my + sy * float(est.intercept_)
    return {
        "model": model,
        "alpha": float(best_alpha),
        "loo_mse": loo_mse,
        "loo_r2": loo_r2,
        "loo_spearman": loo_rho,
        "perm_p": perm_p,
        "n_perm": int(n_perm),
        "perm_null_mean": _f(null.mean()) if null.size else None,
        "perm_null_q95": _f(np.quantile(null, 0.95)) if null.size else None,
        "coef": coef,
        "intercept": _f(intercept),
        "standardize": {nm: {"mean": _f(m), "std": _f(s)} for nm, m, s in zip(names, mean, std)},
        "features": names,
        "loo_pred": [_f(v) for v in loo_pred],
        "alpha_grid": {str(a): grid[a] for a in alphas},
        "n": int(n),
        "y_mean": my,
        "y_std": sy,
    }


# ---------------------------------------------------------------------------------- equivalence
def tost_correlation(rho: float, n: int, bound: float = 0.2, alpha: float = 0.05) -> dict:
    """Two one-sided tests for the equivalence |rho| < bound using Fisher's z.

    z = atanh(rho), se = 1 / sqrt(n - 3), z_b = atanh(bound):
    ``p_lower = P(Z > (z + z_b) / se)`` tests H0: rho <= -bound and ``p_upper = P(Z < (z - z_b) / se)``
    tests H0: rho >= bound; ``equivalent`` iff both are below ``alpha``. The ``1 - 2 alpha`` CI
    ``[ci_lo, ci_hi]`` lies inside (-bound, bound) exactly when the TOST passes.
    Note: with bound 0.2 the test cannot pass for n < ~69 even at rho = 0 (se too large).
    """
    if n is None or n <= 3:
        raise ValueError("tost_correlation needs n > 3")
    if not (0 < bound < 1):
        raise ValueError("bound must be in (0, 1)")
    out = {"equivalent": False, "p_lower": None, "p_upper": None, "p_tost": None, "ci_lo": float("nan"),
           "ci_hi": float("nan"), "bound": float(bound), "alpha": float(alpha), "n": int(n), "rho": float(rho)}
    if rho is None or not np.isfinite(rho) or abs(rho) >= 1.0:
        return out
    z, se, zb = math.atanh(rho), 1.0 / math.sqrt(n - 3), math.atanh(bound)
    p_lower = _f(sps.norm.sf((z + zb) / se))
    p_upper = _f(sps.norm.cdf((z - zb) / se))
    crit = float(sps.norm.ppf(1.0 - alpha))
    out.update({"p_lower": p_lower, "p_upper": p_upper, "p_tost": max(p_lower, p_upper),
                "equivalent": bool(max(p_lower, p_upper) < alpha),
                "ci_lo": math.tanh(z - crit * se), "ci_hi": math.tanh(z + crit * se)})
    return out


def holm(pvals: Mapping[str, float | None]) -> dict[str, float | None]:
    """Holm step-down adjusted p-values: with the m finite p-values sorted ascending,
    ``adj_(i) = max_{j <= i} min(1, (m - j + 1) p_(j))``. Missing/NaN entries are returned as None
    and do not count towards m."""
    items = [(k, float(v)) for k, v in pvals.items() if v is not None and np.isfinite(v)]
    out: dict[str, float | None] = {k: None for k in pvals}
    m = len(items)
    running = 0.0
    for j, (k, pv) in enumerate(sorted(items, key=lambda kv: kv[1])):
        running = max(running, min(1.0, (m - j) * pv))
        out[k] = _f(running)
    return out


# ------------------------------------------------------------------------------------- replicates
def _anova_components(groups: Sequence[np.ndarray]) -> dict:
    sizes = np.array([g.size for g in groups], dtype=float)
    G, N = len(groups), float(sizes.sum())
    allv = np.concatenate(groups)
    grand = allv.mean()
    means = np.array([g.mean() for g in groups])
    ssb = float((sizes * (means - grand) ** 2).sum())
    ssw = float(sum(((g - mu) ** 2).sum() for g, mu in zip(groups, means)))
    msb = ssb / (G - 1) if G > 1 else float("nan")
    msw = ssw / (N - G) if N - G > 0 else float("nan")
    k0 = (N - float((sizes**2).sum()) / N) / (G - 1) if G > 1 else float("nan")
    return {"msb": msb, "msw": msw, "k0": k0, "G": G, "N": int(N)}


def icc_oneway(groups: Sequence[Sequence[float]]) -> float:
    """ICC(1), one-way random effects (Shrout & Fleiss): with between/within mean squares MSB, MSW and
    the unbalanced-design group size k0 = (N - sum n_i^2 / N) / (G - 1),
    ``ICC = (MSB - MSW) / (MSB + (k0 - 1) MSW)``. NaN when fewer than 2 groups or no within-group df."""
    gs = [np.asarray(g, dtype=float) for g in groups]
    gs = [g[np.isfinite(g)] for g in gs]
    gs = [g for g in gs if g.size >= 1]
    if len(gs) < 2:
        return float("nan")
    c = _anova_components(gs)
    if not np.isfinite(c["msw"]):
        return float("nan")
    den = c["msb"] + (c["k0"] - 1.0) * c["msw"]
    if den <= 0:
        return float("nan")
    return _f((c["msb"] - c["msw"]) / den)


def noise_decomposition(per_candidate_replicates) -> dict:
    """Seed-noise decomposition from replicated candidates (one-way random-effects ANOVA).

    Accepts ``{uid: [delta_seed1, delta_seed2, ...]}`` or a list of lists; only groups with >= 2
    finite values are used. ``sigma_seed = sqrt(MSW)`` (SD of the outcome across seeds for a fixed
    candidate), ``sigma_between = sqrt(max(0, (MSB - MSW) / k0))`` (SD of the true candidate effects)
    and ``icc`` = ICC(1) = share of outcome variance attributable to the candidate.
    """
    if isinstance(per_candidate_replicates, Mapping):
        raw = list(per_candidate_replicates.values())
    else:
        raw = list(per_candidate_replicates)
    gs = [np.asarray(g, dtype=float) for g in raw]
    gs = [g[np.isfinite(g)] for g in gs]
    gs = [g for g in gs if g.size >= 2]
    nan = float("nan")
    out = {"sigma_seed": nan, "sigma_between": nan, "sigma_total": nan, "icc": nan, "n_groups": len(gs),
           "n_obs": int(sum(g.size for g in gs))}
    if not gs:
        return out
    c = _anova_components(gs)
    out["sigma_seed"] = math.sqrt(c["msw"]) if np.isfinite(c["msw"]) else nan
    if c["G"] > 1 and np.isfinite(c["msw"]):
        var_between = max(0.0, (c["msb"] - c["msw"]) / c["k0"])
        out["sigma_between"] = math.sqrt(var_between)
        out["sigma_total"] = math.sqrt(var_between + c["msw"])
        out["icc"] = icc_oneway(gs)
    return out


# ---------------------------------------------------------------------------------------- study 2
def auc_over_steps(steps: Sequence[int], accs: Sequence[float]) -> float:
    """Trapezoid area under the accuracy-vs-step curve divided by the step span
    (``int_{s_0}^{s_T} acc(s) ds / (s_T - s_0)``), i.e. the time-averaged accuracy; a flat curve
    returns its level and a single checkpoint returns its accuracy. Steps are sorted first and must be
    distinct; checkpoints with NaN accuracy are dropped."""
    s = _as_1d(steps, "steps")
    a = _as_1d(accs, "accs")
    if s.shape != a.shape:
        raise ValueError(f"steps and accs must have the same length, got {s.shape} vs {a.shape}")
    m = np.isfinite(a) & np.isfinite(s)
    s, a = s[m], a[m]
    if s.size == 0:
        raise ValueError("auc_over_steps: no finite checkpoints")
    if s.size == 1:
        return _f(a[0])
    order = np.argsort(s, kind="stable")
    s, a = s[order], a[order]
    if np.any(np.diff(s) <= 0):
        raise ValueError("steps must be distinct")
    return _f(np.trapezoid(a, s) / (s[-1] - s[0]))
