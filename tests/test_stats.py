import math

import numpy as np
import pytest

from rlvr_v2 import stats as st


def test_bootstrap_mean_ci_contains_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 0.5, size=50)
    mean, lo, hi = st.bootstrap_mean_ci(x, n_boot=500, seed=1)
    assert lo <= mean <= hi and mean == pytest.approx(x.mean())
    assert hi - lo < 0.6
    # NaNs are dropped, a single value gives a degenerate interval
    assert st.bootstrap_mean_ci([1.0, float("nan"), 3.0], n_boot=10) == pytest.approx((2.0, 1.0, 3.0), abs=1.0)
    assert st.bootstrap_mean_ci([2.0]) == (2.0, 2.0, 2.0)
    with pytest.raises(ValueError):
        st.bootstrap_mean_ci([float("nan")])
    with pytest.raises(ValueError):
        st.bootstrap_mean_ci(np.zeros((2, 2)))


def test_paired_item_bootstrap_identical_and_shift():
    rng = np.random.default_rng(0)
    a = (rng.random(200) < 0.6).astype(float)
    same = st.paired_item_bootstrap(a, a.copy(), n_boot=2000, seed=0)
    assert same["delta"] == 0.0 and same["ci_lo"] <= 0.0 <= same["ci_hi"] and same["p_two_sided"] == 1.0
    b = a.copy()
    b[:60] = 0.0  # b strictly worse on 60 items that a got right (some of them)
    res = st.paired_item_bootstrap(a, b, n_boot=2000, seed=0)
    assert res["delta"] == pytest.approx(a.mean() - b.mean()) and res["delta"] > 0
    assert res["ci_lo"] > 0 and res["p_two_sided"] < 0.01 and res["n"] == 200
    assert st.paired_item_bootstrap(a, a, n_boot=2000, seed=3) == same  # deterministic
    with pytest.raises(ValueError):
        st.paired_item_bootstrap(a, a[:-1])


def test_spearman_ci_monotone_and_independent():
    rng = np.random.default_rng(1)
    x = rng.normal(size=80)
    mono = st.spearman_ci(x, np.exp(x), n_boot=300, seed=0)
    assert mono["rho"] == pytest.approx(1.0) and mono["ci_lo"] == pytest.approx(1.0) and mono["p"] < 1e-6
    indep = st.spearman_ci(x, rng.normal(size=80), n_boot=300, seed=0)
    assert abs(indep["rho"]) < 0.25 and indep["ci_lo"] < 0.0 < indep["ci_hi"] and indep["n"] == 80
    # NaN pairs dropped; too few pairs handled
    with_nan = st.spearman_ci([1, 2, 3, float("nan"), 5], [1, 2, 3, 4, float("nan")], n_boot=50)
    assert with_nan["n"] == 3 and with_nan["rho"] == pytest.approx(1.0)
    tiny = st.spearman_ci([1, 2], [2, 1])
    assert math.isnan(tiny["rho"]) and tiny["p"] is None
    with pytest.raises(ValueError):
        st.spearman_ci([1, 2, 3], [1, 2])


def test_partial_spearman_removes_shared_confound():
    rng = np.random.default_rng(2)
    c = rng.normal(size=150)
    x = c + 0.3 * rng.normal(size=150)
    y = c + 0.3 * rng.normal(size=150)
    raw = st.spearman_ci(x, y, n_boot=100)["rho"]
    part = st.partial_spearman(x, y, np.column_stack([c, c**2]), n_boot=200, seed=0)
    assert raw > 0.8 and abs(part["rho"]) < 0.15 and part["ci_lo"] < 0.0 < part["ci_hi"]
    # c is signed, so c**2 has its own ranks: 2 ranked controls x degree 2 = 4 effective columns
    assert part["n"] == 150 and part["k"] == 2 and part["k_eff"] == 4 and 0.0 <= part["p"] <= 1.0
    assert abs(st.partial_spearman(x, y, c, n_boot=0)["rho"]) < 0.15
    # a hump-shaped confound in a non-negative control (like p_s) is removed as well, and passing
    # [u, u**2] is the same as passing u (identical ranks -> duplicated columns)
    u = rng.uniform(0, 1, size=200)
    xh = -((u - 0.5) ** 2) + 0.05 * rng.normal(size=200)
    yh = -((u - 0.5) ** 2) + 0.05 * rng.normal(size=200)
    assert st.spearman_ci(xh, yh, n_boot=0)["rho"] > 0.6
    single = st.partial_spearman(xh, yh, u, n_boot=0)
    both = st.partial_spearman(xh, yh, np.column_stack([u, u**2]), n_boot=0)
    assert abs(single["rho"]) < 0.15 and both["rho"] == pytest.approx(single["rho"]) and both["k_eff"] == 2
    # a genuine partial association survives
    y2 = c + x + 0.1 * rng.normal(size=150)
    assert st.partial_spearman(x, y2, c, n_boot=50)["rho"] > 0.5
    with pytest.raises(ValueError):
        st.partial_spearman(x, y, c[:-1])


def test_hump_test_negative_quadratic():
    rng = np.random.default_rng(3)
    p_s = rng.uniform(0, 1, size=60)
    delta = -(p_s - 0.5) ** 2 + 0.02 * rng.normal(size=60)
    res = st.hump_test(p_s, delta, n_boot=300, seed=0)
    assert res["quad_coef"] == pytest.approx(-1.0, abs=0.15)
    assert res["quad_ci_hi"] < 0 and res["quad_p"] < 1e-6 and res["quad_p_one_sided"] < 1e-6
    assert res["mid_vs_extreme_p"] < 1e-3 and res["mean_mid"] > res["mean_extreme"]
    assert res["n_mid"] + res["n_extreme"] == 60 and res["n"] == 60
    # bin convention: lo exclusive, hi inclusive
    r2 = st.hump_test([0.25, 0.75, 0.5, 0.9, 0.1, 0.3], [0, 0, 0, 0, 0, 0], n_boot=0)
    assert r2["n_mid"] == 3 and r2["n_extreme"] == 3
    with pytest.raises(ValueError):
        st.hump_test([0.1, 0.2], [1.0])


def test_matched_pairs_detects_shift():
    rng = np.random.default_rng(4)
    low = rng.normal(0.0, 1.0, size=20)
    high = low + 2.0 + 0.5 * rng.normal(size=20)
    res = st.matched_pairs_test(high, low, n_boot=1000, seed=0)
    assert res["mean_diff"] == pytest.approx(2.0, abs=0.5) and res["ci_lo"] > 1.0
    assert res["wilcoxon_p"] < 1e-3 and res["wilcoxon_p_greater"] < 1e-3 and res["n_pairs"] == 20
    small = st.matched_pairs_test([1, 2, 3, 4, 5], [0, 1, 2, 3, 4], n_boot=100)
    assert small["wilcoxon_p"] is None and small["n_pairs"] == 5 and small["sign_test_p"] == pytest.approx(0.0625)
    zeros = st.matched_pairs_test(np.ones(8), np.ones(8), n_boot=100)
    assert zeros["wilcoxon_p"] is None and zeros["mean_diff"] == 0.0
    with pytest.raises(ValueError):
        st.matched_pairs_test([1, 2], [1])


def _toy(n=40, seed=5, signal=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    X[::7, 2] = np.nan  # a column with missing values
    y = 2.0 * X[:, 0] + 0.3 * rng.normal(size=n) if signal else rng.normal(size=n)
    return X, y, ["x1", "x2", "x3", "x4"]


def test_loo_linear_selector_ridge_signal_and_noise():
    X, y, names = _toy(signal=True)
    res = st.loo_linear_selector(X, y, names, model="ridge", n_perm=300, seed=0)
    assert res["loo_spearman"] > 0.8 and res["perm_p"] < 0.05 and res["loo_r2"] > 0.7
    assert set(res["coef"]) == set(names) and res["coef"]["x1"] > 1.0 and abs(res["coef"]["x2"]) < 0.5
    assert set(res["standardize"]) == set(names) and len(res["loo_pred"]) == 40
    assert res["alpha"] in (0.01, 0.1, 1.0, 10.0, 100.0) and "0.1" in res["alpha_grid"]
    again = st.loo_linear_selector(X, y, names, model="ridge", n_perm=300, seed=0)
    assert again["perm_p"] == res["perm_p"] and again["loo_pred"] == res["loo_pred"]
    Xn, yn, _ = _toy(seed=11, signal=False)
    noise = st.loo_linear_selector(Xn, yn, names, model="ridge", n_perm=300, seed=0)
    assert noise["perm_p"] > 0.05 and noise["loo_spearman"] < 0.3


def test_loo_linear_selector_lasso_and_validation():
    X, y, names = _toy(signal=True)
    res = st.loo_linear_selector(X, y, names, model="lasso", n_perm=20, seed=0)
    assert res["loo_spearman"] > 0.8 and res["perm_p"] < 0.1 and res["coef"]["x1"] > 1.0
    with pytest.raises(ValueError):
        st.loo_linear_selector(X, y[:-1], names)
    with pytest.raises(ValueError):
        st.loo_linear_selector(X, y, names[:-1])
    with pytest.raises(ValueError):
        st.loo_linear_selector(X, np.ones(len(y)), names)
    with pytest.raises(ValueError):
        st.loo_linear_selector(X, y, names, model="forest")


def test_tost_correlation():
    ok = st.tost_correlation(0.05, n=200)
    assert ok["equivalent"] is True and ok["p_lower"] < 0.05 and ok["p_upper"] < 0.05
    assert -0.2 < ok["ci_lo"] < ok["ci_hi"] < 0.2
    bad = st.tost_correlation(0.5, n=64)
    assert bad["equivalent"] is False and bad["p_upper"] > 0.5
    # the bound is unreachable for small n even at rho == 0
    assert st.tost_correlation(0.0, n=64)["equivalent"] is False
    with pytest.raises(ValueError):
        st.tost_correlation(0.1, n=3)


def test_holm_hand_example():
    adj = st.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    # sorted: a=0.01*3=0.03, c=0.03*2=0.06, b=max(0.06, 0.04*1)=0.06
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert adj["d"] is None
    assert st.holm({"x": 0.5}) == {"x": 0.5}
    assert st.holm({}) == {}


def test_icc_and_noise_decomposition():
    assert st.icc_oneway([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]) == pytest.approx(1.0)
    assert math.isnan(st.icc_oneway([[1.0, 2.0]]))
    rng = np.random.default_rng(6)
    groups = {f"c{i}": list(rng.normal(i * 0.5, 0.1, size=2)) for i in range(8)}
    nd = st.noise_decomposition(groups)
    assert nd["icc"] > 0.9 and nd["sigma_seed"] == pytest.approx(0.1, abs=0.07) and nd["n_groups"] == 8
    assert nd["sigma_between"] > nd["sigma_seed"]
    pure_noise = {f"c{i}": list(rng.normal(0.0, 1.0, size=3)) for i in range(30)}
    assert st.noise_decomposition(pure_noise)["icc"] < 0.3
    empty = st.noise_decomposition({"a": [1.0]})
    assert empty["n_groups"] == 0 and math.isnan(empty["icc"])


def test_auc_over_steps():
    assert st.auc_over_steps([0, 4, 8, 12], [0.3, 0.3, 0.3, 0.3]) == pytest.approx(0.3)
    assert st.auc_over_steps([0, 10], [0.0, 1.0]) == pytest.approx(0.5)
    assert st.auc_over_steps([10, 0], [1.0, 0.0]) == pytest.approx(0.5)  # unsorted input
    assert st.auc_over_steps([5], [0.7]) == 0.7
    with pytest.raises(ValueError):
        st.auc_over_steps([0, 1], [0.1])
    with pytest.raises(ValueError):
        st.auc_over_steps([0, 0], [0.1, 0.2])
