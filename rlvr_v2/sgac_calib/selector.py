"""Stage D (protocol section 5): train, validate and choose the selector; gate G-S; freeze the deployed model.

Data: per calibration candidate c, a label y_c = its seed-mean ΔA (pp) and `n_subsamples` K=4 feature vectors
(features.json). Rows = candidates x subsamples, all with the candidate's label; every split is grouped by candidate.

Metrics are out-of-fold (OOF) over repeated grouped K-fold CV stratified by Ps bin, with predictions averaged over
repeats per (candidate, subsample):
- rho (primary): mean over subsample draws m of Spearman(pred[:, m], y) -- the rank correlation the selector achieves
  when, as in deployment, it sees ONE K=4 sample per candidate; rho_avg (secondary) uses the subsample-averaged score;
- R2_oos and MAE against the intercept-only model's OOF predictions;
- top-1-of-4 gain: E[y of the pick] - E[y of a uniform pick] over random 4-candidate batches, one random K=4 sample per
  candidate, random batch order, first-maximum ties (the loop's rule), importance-weighted to the pool's Ps-bin mix.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from ..artifacts import atomic_write_json, git_hash, read_json, utc_now
from ..sgac import selection as sgac_sel
from .manifests import ps_bin
from .spec import CalibSpec

log = logging.getLogger(__name__)

FIT_MODELS = ("ridge", "ols", "gbm", "ols_int", "mean", "level", "ps_quad")
RULE_RANKERS = ("eq10", "max_var", "max_d", "max_level", "random")
SELECTOR_JSON = "selector.json"
SELECTOR_PKL = "selector_gbm.pkl"


# ---------------------------------------------------------------------- data
@dataclass
class Table:
    uids: list
    bins: np.ndarray  # (n,)
    y: np.ndarray  # (n,) label, pp
    X: np.ndarray  # (n, M, p) K=4 subsample features
    X32: np.ndarray  # (n, p) K=32 features (sensitivity)
    features: tuple

    @property
    def n(self) -> int:
        return len(self.uids)

    @property
    def M(self) -> int:
        return self.X.shape[1]


def build_table(features_doc: dict, labels_doc: dict, feature_names, require_seeds: int | None = None) -> Table:
    lab = {c["unique_id"]: c for c in labels_doc["candidates"]}
    uids, bins, y, X, X32 = [], [], [], [], []
    for c in sorted(features_doc["candidates"], key=lambda r: r["index"]):
        L = lab.get(c["unique_id"])
        if L is None or L["delta_pp_mean"] is None:
            raise RuntimeError(f"no label for {c['unique_id']}")
        if require_seeds is not None and L["n_seeds_done"] < require_seeds:
            raise RuntimeError(f"{c['unique_id']}: {L['n_seeds_done']} of {require_seeds} seeds done")
        uids.append(c["unique_id"]); bins.append(int(c["bin"])); y.append(float(L["delta_pp_mean"]))
        X.append([[float(s[f]) for f in feature_names] for s in c["k4"]])
        X32.append([float(c["k32"][f]) for f in feature_names])
    return Table(uids, np.array(bins), np.array(y), np.array(X), np.array(X32), tuple(feature_names))


def bin_weights(bins: np.ndarray, pool_bin_freq: np.ndarray) -> np.ndarray:
    """Per-candidate importance weight that reweights the stratified sample to the pool's Ps-bin mix."""
    sample = np.bincount(bins, minlength=len(pool_bin_freq)) / len(bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(sample > 0, pool_bin_freq / sample, 0.0)
    return w[bins]


def pool_bin_frequencies(cs: CalibSpec, signals: dict[str, dict]) -> np.ndarray:
    bins = [tuple(b) for b in cs.sampling.ps_bins]
    counts = np.zeros(len(bins))
    for r in signals.values():
        counts[ps_bin(float(r["p_s"]), bins)] += 1
    return counts / counts.sum()


# ---------------------------------------------------------------------- models
def _std(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return mu, np.where(sd > 0, sd, 1.0)


def _design(name: str, X: np.ndarray, names) -> np.ndarray:
    col = {f: X[:, i] for i, f in enumerate(names)}
    if name == "level":
        return col["L"][:, None]
    if name == "ps_quad":
        return np.stack([col["Ps"], col["Ps"] ** 2], axis=1)
    if name == "ols_int":
        p = X.shape[1]
        prods = [X[:, i] * X[:, j] for i in range(p) for j in range(i + 1, p)]
        return np.concatenate([X, np.stack(prods, axis=1)], axis=1)
    return X


def _ridge_solve(Z, y, alpha):
    p = Z.shape[1]
    ym = y.mean()
    b = np.linalg.solve(Z.T @ Z + alpha * np.eye(p), Z.T @ (y - ym))
    return b, ym


@dataclass
class Linear:
    """y = b0 + Z_std b, with Z = design(X) standardised on the training rows."""

    name: str
    names: tuple
    mu: np.ndarray
    sd: np.ndarray
    b: np.ndarray
    b0: float
    alpha: float | None = None

    def predict(self, X):
        Z = (_design(self.name, X, self.names) - self.mu) / self.sd
        return self.b0 + Z @ self.b

    def raw_coefficients(self) -> tuple[np.ndarray, float]:
        """Coefficients on the unstandardised design columns and the matching intercept."""
        braw = self.b / self.sd
        return braw, float(self.b0 - (self.mu * braw).sum())


@dataclass
class Constant:
    value: float

    def predict(self, X):
        return np.full(X.shape[0], self.value)


def _group_folds(groups_unique: np.ndarray, k: int, rng) -> list[np.ndarray]:
    perm = rng.permutation(groups_unique)
    return [perm[i::k] for i in range(k)]


def _fit_linear(name, X, y, names, alpha=0.0) -> Linear:
    Z = _design(name, X, names)
    mu, sd = _std(Z)
    Zs = (Z - mu) / sd
    if alpha > 0:
        b, b0 = _ridge_solve(Zs, y, alpha)
    else:
        b, *_ = np.linalg.lstsq(Zs - Zs.mean(axis=0), y - y.mean(), rcond=None)
        b0 = y.mean() - (Zs.mean(axis=0) @ b)
    return Linear(name, tuple(names), mu, sd, b, float(b0), alpha if alpha > 0 else None)


def choose_alpha(X, y, groups, names, alphas, inner_folds, rng, repeats: int = 1) -> float:
    gu = np.unique(groups)
    err = np.zeros(len(alphas))
    for _ in range(repeats):
        for fold in _group_folds(gu, inner_folds, rng):
            te = np.isin(groups, fold)
            for j, a in enumerate(alphas):
                m = _fit_linear("ridge", X[~te], y[~te], names, a)
                err[j] += float(((m.predict(X[te]) - y[te]) ** 2).sum())
    return float(alphas[int(np.argmin(err))])


def fit_model(name: str, X, y, groups, cs: CalibSpec, rng, n_sub: int):
    sel = cs.selector
    names = sel.features
    if name == "mean":
        return Constant(float(y.mean()))
    if name == "ridge":
        a = choose_alpha(X, y, groups, names, sel.ridge_alphas, sel.inner_folds, rng)
        return _fit_linear("ridge", X, y, names, a)
    if name in ("ols", "ols_int", "level", "ps_quad"):
        return _fit_linear(name, X, y, names)
    if name == "gbm":
        from sklearn.ensemble import GradientBoostingRegressor

        g = dict(sel.gbm)
        leaf = int(g.pop("min_leaf_candidates")) * int(n_sub)
        m = GradientBoostingRegressor(min_samples_leaf=leaf, random_state=int(rng.integers(2 ** 31 - 1)), **g)
        return m.fit(X, y)
    raise ValueError(f"unknown model {name!r}")


def rule_scores(name: str, X: np.ndarray, names, rng) -> np.ndarray:
    """Scores of a fixed rule for every (candidate, subsample): X has shape (n, M, p)."""
    idx = {f: i for i, f in enumerate(names)}
    if name == "random":
        return rng.random(X.shape[:2])
    if name == "eq10":
        flat = X.reshape(-1, X.shape[2])
        s = [sgac_sel.score("sgac_eq10", {f: row[idx[f]] for f in names}) for row in flat]
        return np.array(s).reshape(X.shape[:2])
    col = {"max_var": "Var", "max_d": "D", "max_level": "L"}[name]
    return X[:, :, idx[col]].copy()


# ---------------------------------------------------------------------- CV
def outer_folds(bins: np.ndarray, k: int, rng) -> np.ndarray:
    """Fold id per candidate, stratified by Ps bin (each bin dealt round-robin after a shuffle)."""
    fold = np.empty(len(bins), dtype=int)
    offset = int(rng.integers(k))
    for b in np.unique(bins):
        members = rng.permutation(np.flatnonzero(bins == b))
        fold[members] = (np.arange(len(members)) + offset) % k
        offset = (offset + len(members)) % k
    return fold


def cv_oof(t: Table, name: str, cs: CalibSpec, repeats: int, seed: int, y=None) -> dict:
    """OOF predictions (n, M) averaged over `repeats` repeated stratified grouped K-fold splits."""
    y = t.y if y is None else y
    n, M, p = t.X.shape
    acc = np.zeros((n, M))
    rhos = []
    for r in range(repeats):
        rng = np.random.default_rng([seed, r])
        fold = outer_folds(t.bins, cs.selector.cv_folds, rng)
        pred_r = np.empty((n, M))
        for f in range(cs.selector.cv_folds):
            te = fold == f
            tr = ~te
            Xtr = t.X[tr].reshape(-1, p)
            ytr = np.repeat(y[tr], M)
            gtr = np.repeat(np.flatnonzero(tr), M)
            model = fit_model(name, Xtr, ytr, gtr, cs, rng, M)
            pred_r[te] = model.predict(t.X[te].reshape(-1, p)).reshape(int(te.sum()), M)
        acc += pred_r
        rhos.append(rho_deploy(pred_r, y))
    return {"pred": acc / repeats, "rho_per_repeat": rhos}


def _spearman(a, b) -> float:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def rho_deploy(pred: np.ndarray, y: np.ndarray) -> float:
    vals = [_spearman(pred[:, m], y) for m in range(pred.shape[1])]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def top1_gain(pred: np.ndarray, y: np.ndarray, w: np.ndarray, n_batches: int, rng, b: int = 4) -> dict:
    """Importance-weighted Monte-Carlo top-1-of-b gain (pp) with one random subsample per candidate per batch."""
    n, M = pred.shape
    order = np.argsort(rng.random((n_batches, n)), axis=1)[:, :b]  # uniform b-subsets in random order
    msel = rng.integers(0, M, (n_batches, b))
    s = pred[order, msel]
    pick = np.argmax(s, axis=1)  # first maximum, as the loop
    yb = y[order]
    gains = yb[np.arange(n_batches), pick] - yb.mean(axis=1)
    oracle = yb.max(axis=1) - yb.mean(axis=1)
    ww = np.prod(w[order], axis=1)
    if ww.sum() <= 0:
        ww = np.ones(n_batches)
    return {"gain_pp": float((ww * gains).sum() / ww.sum()), "oracle_pp": float((ww * oracle).sum() / ww.sum())}


def point_metrics(pred, y, mean_pred, w, cs: CalibSpec, rng) -> dict:
    mse = float(((y[:, None] - pred) ** 2).mean())
    mse0 = float(((y[:, None] - mean_pred) ** 2).mean())
    out = {"rho": rho_deploy(pred, y), "rho_avg": _spearman(pred.mean(axis=1), y), "mse": mse,
           "mae": float(np.abs(y[:, None] - pred).mean()), "r2_oos": 1 - mse / mse0 if mse0 > 0 else float("nan")}
    out.update(top1_gain(pred, y, w, cs.selector.n_mc_batches, rng))
    return out


def bootstrap_metrics(pred, y, mean_pred, w, cs: CalibSpec, seed: int, n_boot: int | None = None) -> dict:
    """Candidate-bootstrap 95% CIs of rho, R2_oos and the top-1 gain (OOF predictions fixed)."""
    n_boot = n_boot or cs.selector.n_bootstrap
    rng = np.random.default_rng(seed)
    acc = {"rho": [], "r2_oos": [], "gain_pp": []}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        mse = ((y[i, None] - pred[i]) ** 2).mean()
        mse0 = ((y[i, None] - mean_pred[i]) ** 2).mean()
        acc["rho"].append(rho_deploy(pred[i], y[i]))
        acc["r2_oos"].append(1 - mse / mse0 if mse0 > 0 else np.nan)
        acc["gain_pp"].append(top1_gain(pred[i], y[i], w[i], 2000, rng)["gain_pp"])
    return {k: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))] for k, v in acc.items()}


def paired_loss_diff(pred_a, pred_b, y, mean_pred, seed: int, n_boot: int) -> dict:
    """R2_oos(a) - R2_oos(b) with a candidate-bootstrap CI (a better => positive)."""
    rng = np.random.default_rng(seed)
    ea = ((y[:, None] - pred_a) ** 2).mean(axis=1)
    eb = ((y[:, None] - pred_b) ** 2).mean(axis=1)
    e0 = ((y[:, None] - mean_pred) ** 2).mean(axis=1)
    point = float((eb.mean() - ea.mean()) / e0.mean())
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        boots.append((eb[i].mean() - ea[i].mean()) / e0[i].mean())
    return {"delta_r2": point, "ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


def rho_diff_se(pred_a, pred_b, y, seed: int, n_boot: int) -> float:
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        d.append(rho_deploy(pred_a[i], y[i]) - rho_deploy(pred_b[i], y[i]))
    return float(np.nanstd(d, ddof=1))


def permutation_p(t: Table, name: str, cs: CalibSpec) -> dict:
    """One-sided permutation p of rho: candidate labels shuffled, the whole CV pipeline rerun each time with
    `perm_cv_repeats` repeats; the observed statistic uses the same repeats."""
    sel = cs.selector
    obs = rho_deploy(cv_oof(t, name, cs, sel.perm_cv_repeats, sel.seed)["pred"], t.y)
    rng = np.random.default_rng([sel.seed, 0xBEEF])
    null = []
    for _ in range(sel.n_permutations):
        yp = rng.permutation(t.y)
        null.append(rho_deploy(cv_oof(t, name, cs, sel.perm_cv_repeats, sel.seed, y=yp)["pred"], yp))
    null = np.array(null)
    return {"observed": obs, "p_one_sided": float((1 + np.sum(null >= obs)) / (1 + len(null))),
            "n_permutations": len(null), "cv_repeats": sel.perm_cv_repeats}


# ---------------------------------------------------------------------- coefficients
def vif(X: np.ndarray) -> list[float]:
    out = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(len(X)), others])
        coef, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        resid = X[:, j] - A @ coef
        r2 = 1 - resid.var() / X[:, j].var() if X[:, j].var() > 0 else float("nan")
        out.append(float(1 / (1 - r2)) if r2 < 1 else float("inf"))
    return out


def coefficient_report(t: Table, cs: CalibSpec, alpha: float) -> dict:
    names = list(t.features)
    Xr = t.X.reshape(-1, len(names))
    yr = np.repeat(t.y, t.M)
    rng = np.random.default_rng([cs.selector.seed, 0xC0EF])
    out: dict = {"vif": dict(zip(names, vif(Xr))),
                 "condition_number_std": float(np.linalg.cond((Xr - Xr.mean(0)) / np.where(Xr.std(0) > 0, Xr.std(0), 1)))}
    for name, a in (("ols", 0.0), ("ridge", alpha)):
        full = _fit_linear("ridge" if a > 0 else "ols", Xr, yr, names, a)
        braw, b0 = full.raw_coefficients()
        boots_std, boots_raw = [], []
        for _ in range(cs.selector.n_bootstrap):
            i = rng.integers(0, t.n, t.n)
            m = _fit_linear("ridge" if a > 0 else "ols", t.X[i].reshape(-1, len(names)), np.repeat(t.y[i], t.M), names, a)
            boots_std.append(m.b)
            boots_raw.append(m.raw_coefficients()[0])
        bs, br = np.array(boots_std), np.array(boots_raw)
        out[name] = {"alpha": a or None, "intercept_raw": b0,
                     "coef_raw": dict(zip(names, braw.tolist())), "coef_std": dict(zip(names, full.b.tolist())),
                     "coef_std_ci95": {f: [float(np.quantile(bs[:, j], .025)), float(np.quantile(bs[:, j], .975))]
                                       for j, f in enumerate(names)},
                     "coef_raw_ci95": {f: [float(np.quantile(br[:, j], .025)), float(np.quantile(br[:, j], .975))]
                                       for j, f in enumerate(names)},
                     "sign_agreement": {f: float(np.mean(np.sign(bs[:, j]) == np.sign(full.b[j])))
                                        for j, f in enumerate(names)}}
    return out


# ---------------------------------------------------------------------- the whole Stage D
def run_stage_d(cs: CalibSpec, t: Table, pool_freq: np.ndarray, with_permutation: bool = True) -> dict:
    sel = cs.selector
    w = bin_weights(t.bins, pool_freq)
    rng = np.random.default_rng([sel.seed, 0x5E1])
    oof = {name: cv_oof(t, name, cs, sel.cv_repeats, sel.seed) for name in FIT_MODELS}
    mean_pred = oof["mean"]["pred"]
    res: dict = {"created": utc_now(), "n_candidates": t.n, "n_subsamples": t.M, "features": list(t.features),
                 "label_mean_pp": float(t.y.mean()), "label_sd_pp": float(t.y.std(ddof=1)),
                 "pool_bin_freq": pool_freq.tolist(), "models": {}, "rules": {}}
    for name in FIT_MODELS:
        pm = point_metrics(oof[name]["pred"], t.y, mean_pred, w, cs, rng)
        pm["ci95"] = bootstrap_metrics(oof[name]["pred"], t.y, mean_pred, w, cs, seed=sel.seed)
        pm["rho_repeat_sd"] = float(np.std(oof[name]["rho_per_repeat"], ddof=1))
        res["models"][name] = pm
    for name in RULE_RANKERS:
        sc = rule_scores(name, t.X, t.features, np.random.default_rng([sel.seed, 0xA11]))
        g = top1_gain(sc, t.y, w, sel.n_mc_batches, rng)
        res["rules"][name] = {"rho": rho_deploy(sc, t.y) if name != "random" else None, **g}
    # LOO sensitivity for Ridge (deterministic folds = one candidate each)
    res["ridge_loo_rho"] = _loo_ridge_rho(t, cs)
    # K=32 sensitivity: Ridge fit/scored on K=32 features (one "subsample")
    t32 = Table(t.uids, t.bins, t.y, t.X32[:, None, :], t.X32, t.features)
    res["ridge_k32"] = point_metrics(cv_oof(t32, "ridge", cs, sel.cv_repeats, sel.seed)["pred"], t.y,
                                     cv_oof(t32, "mean", cs, sel.cv_repeats, sel.seed)["pred"], w, cs, rng)
    nb = sel.n_bootstrap
    res["beyond_difficulty"] = {
        "ridge_vs_level": paired_loss_diff(oof["ridge"]["pred"], oof["level"]["pred"], t.y, mean_pred, sel.seed, nb),
        "ridge_vs_ps_quad": paired_loss_diff(oof["ridge"]["pred"], oof["ps_quad"]["pred"], t.y, mean_pred, sel.seed, nb)}
    res["nonlinearity"] = {
        "gbm_vs_ridge": paired_loss_diff(oof["gbm"]["pred"], oof["ridge"]["pred"], t.y, mean_pred, sel.seed, nb),
        "ols_int_vs_ridge": paired_loss_diff(oof["ols_int"]["pred"], oof["ridge"]["pred"], t.y, mean_pred, sel.seed, nb),
        "material_if": "delta_r2 >= 0.05 and CI excludes 0"}
    for v in res["nonlinearity"].values():
        if isinstance(v, dict):
            v["material"] = bool(v["delta_r2"] >= 0.05 and v["ci"][0] > 0)
    # pre-declared model choice: Ridge unless GBM beats it on rho by > 1 SE AND has higher R2_oos
    se = rho_diff_se(oof["gbm"]["pred"], oof["ridge"]["pred"], t.y, sel.seed, nb)
    d_rho = res["models"]["gbm"]["rho"] - res["models"]["ridge"]["rho"]
    gbm_wins = bool(np.isfinite(se) and d_rho > se and res["models"]["gbm"]["r2_oos"] > res["models"]["ridge"]["r2_oos"])
    chosen = "gbm" if gbm_wins else "ridge"
    res["model_choice"] = {"chosen": chosen, "rho_gbm_minus_ridge": d_rho, "se": se,
                           "rule": "ridge unless gbm's rho exceeds ridge's by > 1 SE (candidate bootstrap) AND "
                                   "gbm's R2_oos is higher"}
    full_alpha = choose_alpha(t.X.reshape(-1, len(t.features)), np.repeat(t.y, t.M), np.repeat(np.arange(t.n), t.M),
                              t.features, sel.ridge_alphas, sel.inner_folds, np.random.default_rng([sel.seed, 0xA1]),
                              repeats=sel.alpha_cv_repeats)
    res["coefficients"] = coefficient_report(t, cs, full_alpha)
    if with_permutation:
        res["permutation"] = permutation_p(t, chosen, cs)
    res["gate_s"] = gate_s(cs, res, chosen)
    res["final_ridge_alpha"] = full_alpha
    return res


def _loo_ridge_rho(t: Table, cs: CalibSpec) -> float:
    n, M, p = t.X.shape
    pred = np.empty((n, M))
    rng = np.random.default_rng([cs.selector.seed, 0x100])
    for c in range(n):
        tr = np.arange(n) != c
        m = fit_model("ridge", t.X[tr].reshape(-1, p), np.repeat(t.y[tr], M), np.repeat(np.flatnonzero(tr), M), cs,
                      rng, M)
        pred[c] = m.predict(t.X[c])
    return rho_deploy(pred, t.y)


def gate_s(cs: CalibSpec, res: dict, chosen: str) -> dict:
    g = cs.selector.gate_s
    m = res["models"][chosen]
    perm = res.get("permutation")
    rho_ok = bool(np.isfinite(m["rho"]) and m["rho"] >= float(g["rho_min"]))
    perm_ok = None if perm is None else bool(perm["p_one_sided"] < float(g["perm_p_max"]))
    gain_ok = bool(m["ci95"]["gain_pp"][0] > 0)
    passed = bool(rho_ok and perm_ok and gain_ok) if perm_ok is not None else None
    return {"model": chosen, "rho": m["rho"], "rho_ok": rho_ok, "perm_p": None if perm is None else perm["p_one_sided"],
            "perm_ok": perm_ok, "top1_gain_pp": m["gain_pp"], "top1_gain_ci95": m["ci95"]["gain_pp"], "gain_ok": gain_ok,
            "passed": passed, "rule": g,
            "consequence": None if passed is None else ("run Stage E with the learned arm" if passed else
                                                        "no learned-selector curriculum; report the null; team decides")}


# ---------------------------------------------------------------------- freeze and deploy
def table_sha256(t: Table) -> str:
    blob = json.dumps({"uids": t.uids, "y": t.y.round(12).tolist(), "X": t.X.round(12).tolist()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def freeze_selector(cs: CalibSpec, t: Table, stage_d: dict, out_dir: Path) -> dict:
    """Refit the chosen model on all candidates and write selector.json (+ the GBM pickle if chosen)."""
    names = list(t.features)
    Xr = t.X.reshape(-1, len(names))
    yr = np.repeat(t.y, t.M)
    chosen = stage_d["model_choice"]["chosen"]
    doc = {"created": utc_now(), "model": chosen, "features": names, "k_deploy": int(cs.features.k_deploy),
           "feature_definitions": "Ps = p_s, Var = v_legacy, D = u_ratio, L = level (rlvr_v2.sgac.sieve.selection_signals)",
           "units": "predicted ΔA on the dev set, pp", "training_table_sha256": table_sha256(t),
           "n_candidates": t.n, "calib_hash": cs.calib_hash(), "code_commit": git_hash(),
           "deploy_rule": "argmax over the batch, first maximum on ties"}
    if chosen == "ridge":
        m = _fit_linear("ridge", Xr, yr, names, stage_d["final_ridge_alpha"])
        braw, b0 = m.raw_coefficients()
        doc.update(alpha=stage_d["final_ridge_alpha"], scaler_mean=m.mu.tolist(), scaler_scale=m.sd.tolist(),
                   coef_std=m.b.tolist(), coef_raw=dict(zip(names, braw.tolist())), intercept_raw=b0)
    else:
        import sklearn

        model = fit_model("gbm", Xr, yr, np.repeat(np.arange(t.n), t.M), cs,
                          np.random.default_rng([cs.selector.seed, 0xF1]), t.M)
        blob = pickle.dumps(model)
        (out_dir / SELECTOR_PKL).write_bytes(blob)
        doc.update(pickle=SELECTOR_PKL, pickle_sha256=hashlib.sha256(blob).hexdigest(), sklearn=sklearn.__version__)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / SELECTOR_JSON, doc)
    doc["sha256"] = hashlib.sha256((out_dir / SELECTOR_JSON).read_bytes()).hexdigest()
    return doc


class LearnedSelector:
    """The frozen selector as the curriculum uses it. Refuses to load if the file's sha256 differs from the one the
    curriculum config pins."""

    def __init__(self, path: str | Path, expected_sha256: str):
        path = Path(path)
        raw = path.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if got != expected_sha256:
            raise RuntimeError(f"{path} sha256 {got} != pinned {expected_sha256}; refusing to deploy")
        self.path, self.sha256 = path, got
        self.doc = json.loads(raw)
        self.features = self.doc["features"]
        self._gbm = None
        if self.doc["model"] == "gbm":
            blob = (path.parent / self.doc["pickle"]).read_bytes()
            if hashlib.sha256(blob).hexdigest() != self.doc["pickle_sha256"]:
                raise RuntimeError("selector pickle does not match its pinned sha256")
            self._gbm = pickle.loads(blob)  # our own artifact, hash-verified above

    def score(self, sig: dict) -> float:
        x = np.array([float(sig[f]) for f in self.features], dtype=float)
        if not np.all(np.isfinite(x)):
            return float("nan")
        if self._gbm is not None:
            return float(self._gbm.predict(x[None, :])[0])
        coef = self.doc["coef_raw"]
        return float(self.doc["intercept_raw"] + sum(coef[f] * v for f, v in zip(self.features, x)))


def load_stage_inputs(cs: CalibSpec) -> tuple[dict, dict]:
    feats = read_json(cs.manifest_dir / "features.json")
    labels = read_json(cs.results_dir / "calib" / "labels.json")
    if feats is None or labels is None:
        raise FileNotFoundError("features.json or calib/labels.json missing")
    return feats, labels
