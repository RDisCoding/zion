"""The crossed seed x item estimator (protocol section 7): calibrated SE, nominal coverage, joint components, and the
pre-declared seed-count rule."""
import numpy as np
import pytest

from rlvr_v2.sgac_calib.crossed import (
    _ms_two_way,
    arms_components,
    components_from_ms,
    crossed_test,
    item_bootstrap,
    pigeonhole_bootstrap,
    projected,
    seed_count,
)


def _gauss(rng, S, N, delta, sa, sb, se):
    return delta + rng.normal(0, sa, (S, 1)) + rng.normal(0, sb, (1, N)) + rng.normal(0, se, (S, N))


def test_se_matches_sampling_spread_and_ci_covers():
    rng = np.random.default_rng(1)
    S, N, sa, sb, se = 6, 400, 0.01, 0.05, 0.3
    means, ses, cover = [], [], 0
    for _ in range(1500):
        r = crossed_test(_gauss(rng, S, N, 0.0, sa, sb, se))
        means.append(r["mean"]); ses.append(r["se"])
        cover += r["ci"][0] <= 0.0 <= r["ci"][1]
    true_sd = np.sqrt(sa ** 2 / S + sb ** 2 / N + se ** 2 / (S * N))
    assert np.std(means) == pytest.approx(true_sd, rel=0.08)
    assert np.sqrt(np.mean(np.square(ses))) == pytest.approx(true_sd, rel=0.08)
    # Satterthwaite is mildly conservative when the seed component is small (~97% at S=6, sigma_seed ~ 0; nominal
    # once the seed component is present) -- never anticonservative in these settings.
    assert 0.93 <= cover / 1500 <= 0.98


def test_binary_paired_arms_coverage():
    """Paired 0/1 outcomes from two arms with item difficulty, a shared seed effect and an arm x seed effect."""
    rng = np.random.default_rng(2)
    S, N, cover, reps = 6, 500, 0, 600
    theta = rng.normal(0.8, 1.6, N)  # item logits (~63% base accuracy)
    for _ in range(reps):
        u = rng.normal(0, 0.15, (S, 1))  # shared seed effect (cancels in pairs)
        v = rng.normal(0, 0.05, (2, S, 1))  # arm x seed
        p = 1 / (1 + np.exp(-(theta[None, None, :] + u[None] + v)))
        y = rng.random((2, S, N)) < p
        r = crossed_test(y[0].astype(float) - y[1].astype(float))
        cover += r["ci"][0] <= 0.0 <= r["ci"][1]
    assert cover / reps >= 0.92


def test_bootstraps_err_in_the_stated_directions():
    rng = np.random.default_rng(3)
    e = _gauss(rng, 6, 500, 0.0, 0.02, 0.03, 0.3)
    prim = crossed_test(e)["se"]
    assert item_bootstrap(e, 4000)["se"] < prim < pigeonhole_bootstrap(e, 4000)["se"]


def test_joint_components_with_two_arms_equal_half_the_pair_components():
    rng = np.random.default_rng(4)
    Y = {"a": rng.random((3, 200)) < 0.6, "b": rng.random((3, 200)) < 0.6}
    Y = {k: v.astype(float) for k, v in Y.items()}
    c = arms_components(Y)
    m = _ms_two_way(Y["a"] - Y["b"])
    assert c["ms_as"] == pytest.approx(m["ms_seed"] / 2)
    assert c["ms_ai"] == pytest.approx(m["ms_item"] / 2)
    assert c["ms_asi"] == pytest.approx(m["ms_res"] / 2)
    pc = components_from_ms(m)
    assert 2 * c["s2_e"] == pytest.approx(pc["sigma2_res"])


def test_joint_components_recover_truth():
    rng = np.random.default_rng(5)
    A, S, N, s_as, s_ai, s_e = 4, 8, 2000, 0.02, 0.04, 0.3
    X = (rng.normal(0, 1, (1, 1, N)) + rng.normal(0, s_as, (A, S, 1)) + rng.normal(0, s_ai, (A, 1, N))
         + rng.normal(0, s_e, (A, S, N)))
    c = arms_components({f"arm{a}": X[a] for a in range(A)})
    assert np.sqrt(c["s2_e"]) == pytest.approx(s_e, rel=0.02)
    assert np.sqrt(c["s2_ai"]) == pytest.approx(s_ai, rel=0.15)
    assert np.sqrt(c["s2_as"]) == pytest.approx(s_as, rel=0.5)


def test_projection_shrinks_with_seeds_and_rule_respects_bounds():
    se6, _ = projected(0.0, 0.002, 0.09, 6, 500)
    se10, _ = projected(0.0, 0.002, 0.09, 10, 500)
    assert se10 < se6
    rng = np.random.default_rng(6)
    Y = {a: (rng.random((3, 500)) < 0.63).astype(float) for a in ("a", "b", "c")}
    out = seed_count(Y, seeds_min=6, seeds_max=10, alpha=0.025, power=0.8, mde_max=0.02)
    assert 6 <= out["chosen_seeds"] <= 10
    ok = [r for r in out["table"] if r["S"] >= 6 and r["mde"] <= 0.02]
    assert out["chosen_seeds"] == (ok[0]["S"] if ok else 10)
    lax = seed_count(Y, seeds_min=6, seeds_max=10, alpha=0.025, power=0.8, mde_max=1.0)
    assert lax["chosen_seeds"] == 6
