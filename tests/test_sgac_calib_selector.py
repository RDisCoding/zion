"""Stage D on synthetic tables: finds real signal, rejects noise, and the frozen selector round-trips with its hash."""
import hashlib

import numpy as np
import pytest

from rlvr_v2.sgac import selection as sgac_sel
from rlvr_v2.sgac_calib.selector import (
    FIT_MODELS,
    LearnedSelector,
    Table,
    _fit_linear,
    bin_weights,
    cv_oof,
    freeze_selector,
    rule_scores,
    run_stage_d,
    top1_gain,
)
from rlvr_v2.sgac_calib.spec import load_calib

FAST = ["selector.cv_repeats=4", "selector.n_bootstrap=100", "selector.n_permutations=40",
        "selector.perm_cv_repeats=2", "selector.n_mc_batches=3000", "selector.alpha_cv_repeats=2"]
POOL_FREQ = np.array([0.088, 0.314, 0.48, 0.118])


def synthetic(signal: float, seed: int = 0, n_per_bin: int = 12, M: int = 16) -> Table:
    rng = np.random.default_rng(seed)
    bins = np.repeat(np.arange(4), n_per_bin)
    ps_true = np.array([0.0, 0.2, 0.6, 0.9])[bins] + rng.uniform(0, 0.1, bins.size) * (bins > 0)
    level = rng.integers(1, 6, bins.size).astype(float)
    k = rng.binomial(4, np.clip(ps_true, 0, 1)[:, None], (bins.size, M)) / 4.0
    var = k * (1 - k) + rng.uniform(0, 0.05, k.shape)
    d = np.clip(1 - k + rng.choice([0, 0.25], k.shape), 0.25, 1.0)
    X = np.stack([k, var, d, np.repeat(level[:, None], M, axis=1)], axis=2)
    y = signal * (level - 3) + rng.normal(0, 1.0, bins.size)
    return Table([f"c{i}" for i in range(bins.size)], bins, y, X, X.mean(axis=1), ("Ps", "Var", "D", "L"))


def test_strong_signal_passes_gate_s():
    cs = load_calib(overrides=FAST)
    res = run_stage_d(cs, synthetic(2.0), POOL_FREQ)
    assert res["models"]["ridge"]["rho"] > 0.7
    assert res["models"]["ridge"]["gain_pp"] > 0
    assert res["gate_s"]["passed"] is True
    assert res["rules"]["max_level"]["gain_pp"] > res["rules"]["random"]["gain_pp"] - 0.5
    assert set(res["models"]) == set(FIT_MODELS)


def test_pure_noise_fails_gate_s():
    cs = load_calib(overrides=FAST)
    res = run_stage_d(cs, synthetic(0.0, seed=3), POOL_FREQ)
    assert res["gate_s"]["passed"] is False


def test_top1_gain_of_a_perfect_predictor_equals_the_oracle():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 2, 40)
    pred = np.repeat(y[:, None], 8, axis=1)
    g = top1_gain(pred, y, np.ones(40), 5000, rng)
    assert g["gain_pp"] == pytest.approx(g["oracle_pp"])
    rnd = top1_gain(rng.random((40, 8)), y, np.ones(40), 20000, rng)
    assert abs(rnd["gain_pp"]) < 0.1


def test_cv_never_trains_on_the_held_out_candidate():
    t = synthetic(2.0)
    cs = load_calib(overrides=FAST)
    a = cv_oof(t, "ridge", cs, 2, 0)["pred"]
    y2 = t.y.copy()
    y2[0] += 1000.0  # a leak would move candidate 0's own prediction
    b = cv_oof(t, "ridge", cs, 2, 0, y=y2)["pred"]
    assert np.allclose(a[0], b[0])


def test_eq10_rule_matches_the_frozen_rule():
    t = synthetic(1.0)
    s = rule_scores("eq10", t.X, t.features, np.random.default_rng(0))
    sig = dict(zip(t.features, t.X[3, 5]))
    assert s[3, 5] == pytest.approx(sgac_sel.score("sgac_eq10", sig))


def test_bin_weights_reweight_to_pool_mix():
    w = bin_weights(np.repeat(np.arange(4), 12), POOL_FREQ)
    assert w[:12].mean() == pytest.approx(POOL_FREQ[0] / 0.25)


def test_freeze_and_deploy_round_trip(tmp_path):
    cs = load_calib(overrides=FAST)
    t = synthetic(2.0)
    stage_d = {"model_choice": {"chosen": "ridge"}, "final_ridge_alpha": 1.0}
    doc = freeze_selector(cs, t, stage_d, tmp_path)
    sha = hashlib.sha256((tmp_path / "selector.json").read_bytes()).hexdigest()
    sel = LearnedSelector(tmp_path / "selector.json", sha)
    ref = _fit_linear("ridge", t.X.reshape(-1, 4), np.repeat(t.y, t.M), t.features, 1.0)
    x = t.X[7, 2]
    assert sel.score(dict(zip(t.features, x))) == pytest.approx(float(ref.predict(x[None, :])[0]))
    assert doc["features"] == ["Ps", "Var", "D", "L"]
    with pytest.raises(RuntimeError, match="refusing"):
        LearnedSelector(tmp_path / "selector.json", "0" * 64)
