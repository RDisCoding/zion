"""Gate G-P (label reliability) and the Stage-E decision rule on synthetic inputs."""
import numpy as np
import pytest

from rlvr_v2.sgac_calib.analysis import confirmatory, decide
from rlvr_v2.sgac_calib.calib import gate_p, icc_ci
from rlvr_v2.sgac_calib.spec import load_calib

CS = load_calib()


def _labels(deltas: dict[int, list[float]]) -> dict:
    return {"candidates": [{"index": i, "unique_id": f"c{i}", "n_seeds_done": len(v),
                            "runs": [{"delta_pp": d} for d in v]} for i, v in deltas.items()], "pi1": None}


def test_gate_p_passes_reliable_labels_and_fails_noise():
    rng = np.random.default_rng(0)
    true = rng.normal(0, 2.0, 16)
    reliable = {i: [true[i] + rng.normal(0, 0.5), true[i] + rng.normal(0, 0.5)] for i in range(16)}
    res = gate_p(CS, _labels(reliable), list(range(16)))
    assert res["passed"] is True and res["icc"] > 0.8 and res["icc_ci95"][0] <= res["icc"] <= res["icc_ci95"][1]
    noise = {i: [rng.normal(0, 2.0), rng.normal(0, 2.0)] for i in range(16)}
    assert gate_p(CS, _labels(noise), list(range(16)))["icc"] < 0.5


def test_gate_p_waits_for_every_seed():
    lab = _labels({i: [1.0, 2.0] for i in range(15)} | {15: [1.0]})
    res = gate_p(CS, lab, list(range(16)))
    assert res["status"] == "incomplete" and res["passed"] is None


def test_icc_ci_brackets_known_value():
    lo, hi = icc_ci([[1.0, 1.1], [2.0, 2.2], [3.0, 2.9], [0.0, 0.2]])
    assert 0 < lo < hi <= 1


def _test(mean_pp, se_pp, df=20):
    from scipy import stats

    q90 = stats.t.ppf(0.95, df)
    return {"mean": mean_pp / 100, "ci90": [(mean_pp - q90 * se_pp) / 100, (mean_pp + q90 * se_pp) / 100]}


@pytest.mark.parametrize("mean_pp,se_pp,p,seeds,expected", [
    (2.5, 0.5, 0.001, [3, 2, 2, 3, 1, 2, 3, 4], "beats"),
    (2.5, 0.5, 0.001, [3, 2, -2, -3, 1, 2, 3, 4], "inconclusive"),  # only 6/8 seeds positive (< 7)
    (0.8, 0.3, 0.01, [1] * 8, "equivalent"),  # significant but < 1 pp: no winner; CI inside ±2
    (-2.5, 0.5, 0.001, [-3] * 8, "worse"),
    (1.5, 1.2, 0.2, [1] * 8, "inconclusive"),
])
def test_decision_rule(mean_pp, se_pp, p, seeds, expected):
    assert decide(_test(mean_pp, se_pp), p, seeds, CS) == expected


def test_confirmatory_on_synthetic_arms():
    rng = np.random.default_rng(5)
    theta = rng.normal(0.6, 1.5, 500)
    def run(shift):
        return (rng.random(500) < 1 / (1 + np.exp(-(theta + shift)))).astype(float)
    arms = {"random": {s: run(0.0) for s in range(42, 50)}, "sgac": {s: run(0.0) for s in range(42, 50)},
            "learned": {s: run(0.6) for s in range(42, 50)}}
    res = confirmatory(CS, arms)
    assert res["results"]["H1"]["verdict"] == "beats" and res["results"]["H2"]["verdict"] == "beats"
    assert res["outperforms_both"] is True
    assert res["results"]["H1"]["p_holm"] >= res["results"]["H1"]["p"]
