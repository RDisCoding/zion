import json

import numpy as np
import pytest

from rlvr_v2.config import SelectorCfg
from rlvr_v2.selectors import (DisagreementSelector, LearnedLinearSelector, PsBandSelector, RandomSelector,
                               VarianceSelector, build_selector)


def _cands(mk):
    return [
        mk("a", p_s=0.0, d_simpson=0.9, d_wrong=0.9, level=5),
        mk("b", p_s=0.5, d_simpson=0.4, d_wrong=0.2, level=3),
        mk("c", p_s=0.875, d_simpson=0.2, level=1),
        mk("d", p_s=0.25, d_simpson=0.7, d_wrong=0.8, level=4),
    ]


def test_variance_picks_ps_nearest_half(mk_signals):
    idx, _ = VarianceSelector().select(_cands(mk_signals), np.random.default_rng(0))
    assert idx == 1


def test_disagreement_picks_max_metric(mk_signals):
    idx, _ = DisagreementSelector("d_simpson").select(_cands(mk_signals), np.random.default_rng(0))
    assert idx == 0
    with pytest.raises(ValueError):
        DisagreementSelector("not_a_feature")


def test_ps_band(mk_signals):
    idx, _ = PsBandSelector(0.2, 0.8, 0.5).select(_cands(mk_signals), np.random.default_rng(0))
    assert idx == 1
    only_out = [mk_signals("x", p_s=0.0), mk_signals("y", p_s=1.0), mk_signals("z", p_s=0.9)]
    idx, _ = PsBandSelector(0.2, 0.8, 0.5).select(only_out, np.random.default_rng(0))
    assert idx == 2  # least far from the target when nothing is in band


def test_learned_named_coefficients(mk_signals, tmp_path):
    sel = LearnedLinearSelector(features=("p_s",), coef={"p_s": -1.0})
    idx, scores = sel.select(_cands(mk_signals), np.random.default_rng(0))
    assert idx == 0 and scores[0] == 0.0
    sel2 = LearnedLinearSelector(features=("p_s", "d_simpson", "level"),
                                 coef={"p_s": 0.1, "d_simpson": 1.0, "level": 0.01},
                                 intercept=0.5, standardize={"level": {"mean": 3.0, "std": 1.0}})
    path = tmp_path / "sel.json"
    sel2.to_json(path)
    back = LearnedLinearSelector.from_json(path)
    rng = np.random.default_rng(1)
    assert back.select(_cands(mk_signals), rng)[1] == pytest.approx(sel2.select(_cands(mk_signals), rng)[1])
    # permuting key order in the JSON must not change scores
    d = json.loads(path.read_text())
    d["coef"] = dict(reversed(list(d["coef"].items())))
    d["features"] = list(reversed(d["features"]))
    (tmp_path / "perm.json").write_text(json.dumps(d))
    perm = LearnedLinearSelector.from_json(tmp_path / "perm.json")
    assert perm.scores(_cands(mk_signals), rng) == pytest.approx(sel2.scores(_cands(mk_signals), rng))
    tex = back.to_latex(tmp_path / "t.tex", ci={"p_s": (-0.1, 0.3)})
    assert "p\\_s" in tex and "intercept" in tex


def test_learned_validation():
    with pytest.raises(ValueError):
        LearnedLinearSelector(features=("nope",), coef={"nope": 1.0})
    with pytest.raises(ValueError):
        LearnedLinearSelector(features=("p_s", "d_simpson"), coef={"p_s": 1.0})


def test_nan_feature_is_imputed(mk_signals):
    sel = LearnedLinearSelector(features=("d_wrong",), coef={"d_wrong": 1.0})
    scores = sel.scores([mk_signals("a", d_wrong=float("nan")), mk_signals("b", d_wrong=0.5)],
                        np.random.default_rng(0))
    assert scores == [0.0, 0.5]


def test_random_is_seeded_and_roughly_uniform(mk_signals):
    cands = _cands(mk_signals)
    a = RandomSelector().select(cands, np.random.default_rng(7))[0]
    b = RandomSelector().select(cands, np.random.default_rng(7))[0]
    assert a == b
    rng = np.random.default_rng(0)
    counts = np.bincount([RandomSelector().select(cands, rng)[0] for _ in range(2000)], minlength=4)
    assert counts.min() > 300


def test_ties_break_randomly_but_validly(mk_signals):
    cands = [mk_signals("a", p_s=0.5), mk_signals("b", p_s=0.5)]
    picks = {VarianceSelector().select(cands, np.random.default_rng(s))[0] for s in range(20)}
    assert picks <= {0, 1} and len(picks) == 2


def test_build_selector_each_name(mk_signals, tmp_path):
    LearnedLinearSelector(features=("p_s",), coef={"p_s": 1.0}).to_json(tmp_path / "l.json")
    for name in ("random", "variance", "disagreement", "ps_band", "learned"):
        cfg = SelectorCfg(name=name, learned_path=str(tmp_path / "l.json"))
        sel = build_selector(cfg)
        assert sel.name == name
        idx, scores = sel.select(_cands(mk_signals), np.random.default_rng(0))
        assert 0 <= idx < 4 and len(scores) == 4
    with pytest.raises(ValueError):
        build_selector(SelectorCfg(name="learned"))
