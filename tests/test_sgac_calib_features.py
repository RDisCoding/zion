"""Deployment-matched K=4 features: the deployed definitions, valid value grids, and agreement with the manifests."""
import numpy as np
import pytest

from rlvr_v2.artifacts import read_json
from rlvr_v2.data import Problem
from rlvr_v2.grader import MathVerifyGrader
from rlvr_v2.sgac_calib.features import deploy_signals
from rlvr_v2.sgac_calib.spec import load_calib, substrate_spec

CS = load_calib()
FEATS = read_json(CS.manifest_dir / "features.json")
CANDS = read_json(CS.manifest_dir / "candidates.json")


def test_features_cover_the_draw_with_deployed_shapes():
    assert FEATS["k_deploy"] == 4 and FEATS["n_subsamples"] == 16
    assert FEATS["substrate_spec_hash"] == "577f3319"
    assert FEATS["sieve_signals_sha256"] == CANDS["sieve_signals_sha256"]
    by_idx = {r["index"]: r for r in CANDS["pilot"] + CANDS["expansion"]}
    assert sorted(c["index"] for c in FEATS["candidates"]) == list(range(48))
    for c in FEATS["candidates"]:
        assert c["unique_id"] == by_idx[c["index"]]["unique_id"] and c["level"] == by_idx[c["index"]]["level"]
        assert len(c["k4"]) == 16 and all(len(s) == 4 for s in c["subsample_rollout_idx"])
        for s in c["k4"]:
            assert s["Ps"] in (0.0, 0.25, 0.5, 0.75, 1.0)
            assert s["D"] in (0.25, 0.5, 0.75, 1.0)
            assert 0.0 <= s["Var"] <= 0.5625 + 1e-12  # pvar of 4 values in {0, .5, 1, 1.5}
            assert s["L"] == c["level"]
        assert c["k32"]["Ps"] == pytest.approx(by_idx[c["index"]]["ps32"])


def test_k4_means_track_k32():
    ps4 = np.array([np.mean([s["Ps"] for s in c["k4"]]) for c in FEATS["candidates"]])
    ps32 = np.array([c["k32"]["Ps"] for c in FEATS["candidates"]])
    assert np.corrcoef(ps4, ps32)[0, 1] > 0.9


def test_deploy_signals_use_the_sgac_definitions():
    spec = substrate_spec(CS, "calib")
    prob = Problem("train/x/1.json", "What is 1+1?", "2", None, 3, "Algebra", "math_train")
    recs = [{"text": t, "n_tokens": 10, "truncated": "False"} for t in
            ("so \\boxed{2}", "thus \\boxed{2}", "I get \\boxed{3}", "no box here")]
    sig, _ = deploy_signals(spec, recs, prob, MathVerifyGrader())
    assert sig["Ps"] == pytest.approx(0.5) and sig["L"] == 3
    rewards = [1.5, 1.5, 0.5, 0.0]  # correct + 0.5 * boxed
    assert sig["Var"] == pytest.approx(np.var(rewards))
    assert sig["D"] == pytest.approx(3 / 4)  # classes {2}, {3}, {none}
