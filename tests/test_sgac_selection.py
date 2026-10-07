"""Selection rules: NB-M's logged scores, the max-level property of Eq. 10 at K=4, tie-breaks, counterfactual picks."""
import itertools
import math

import numpy as np
import pytest

from rlvr_v2.sgac import nbm_original as nbm
from rlvr_v2.sgac import selection as sel
from rlvr_v2.sgac.schedule import random_arm_rng


def sig(ps, var, d, lv):
    return {"Ps": ps, "Var": var, "D": d, "L": lv}


def test_eq10_reproduces_every_logged_nbm_score():
    for s in nbm.STEPS:
        assert sel.score("sgac_eq10", sig(s["Ps"], s["Var"], s["D"], s["L"])) == pytest.approx(s["score"], abs=6e-5)


def _achievable_ps_var(k=4):
    """All (Ps, Var) pairs reachable with K rewards R = correct + 0.5*box (legacy allows R=1.0 without a box)."""
    out = set()
    for rewards in itertools.combinations_with_replacement((0.0, 0.5, 1.0, 1.5), k):
        ps = sum(r >= 1.0 for r in rewards) / k
        out.add((ps, float(np.var(rewards))))
    return sorted(out)


def test_eq10_always_picks_a_maximum_level_candidate_at_k4():
    nonlevel = [sel.score("sgac_eq10", sig(ps, var, d, 0)) for ps, var in _achievable_ps_var()
                for d in (0.25, 0.5, 0.75, 1.0)]
    span = max(nonlevel) - min(nonlevel)
    assert span < nbm.EQ10_AS_RUN["L"]  # one level outweighs every other term combined
    # max at Ps=0.5, Var=0.5625 (two 1.5s, two 0s), D=0.25; min at Ps=0, Var=0, D=1
    assert span == pytest.approx((0.0050 * 0.5 + 0.1832 * 0.5625 - 0.0751 * 0.25) + 0.0751, abs=1e-12)


def test_eq10_tie_breaks_within_a_level():
    cands = [sig(0.5, 0.25, 0.5, 4), sig(0.0, 0.0, 1.0, 5), sig(0.25, 0.1875, 0.75, 5), sig(0.25, 0.1875, 0.5, 5)]
    idx, _ = sel.pick("sgac_eq10", cands)
    assert idx == 3  # level 5, higher Var than #1, lower D than #2
    assert sel.picked_max_level(cands, idx)


def test_first_max_tie_break_and_nan():
    assert sel.first_max([1.0, 3.0, 3.0]) == 1
    assert sel.first_max([float("nan"), 0.0]) == 1
    idx, _ = sel.pick("max_level", [sig(0, 0, 0, 5), sig(1, 1, 1, 5)])
    assert idx == 0


def test_pickle_true_mapping_uses_the_features_it_was_fit_on():
    base = sel.score("pickle_true_mapping", sig(0, 0, 0, 0))
    assert base == pytest.approx(nbm.PICKLE_INTERCEPT)
    assert sel.score("pickle_true_mapping", sig(0, 0, 1, 0)) - base == pytest.approx(nbm.PICKLE_COEF[3])  # D
    assert sel.score("pickle_true_mapping", sig(0, 0, 0, 1)) - base == pytest.approx(nbm.PICKLE_COEF[0])  # level


def test_random_rule_is_reproducible_and_needs_its_rng():
    cands = [sig(0, 0, 0, 1)] * 4
    a = [sel.pick("random", cands, random_arm_rng(42, t))[0] for t in range(1, 21)]
    b = [sel.pick("random", cands, random_arm_rng(42, t))[0] for t in range(1, 21)]
    assert a == b and len(set(a)) > 1
    with pytest.raises(ValueError):
        sel.pick("random", cands)


def test_all_picks_covers_every_rule_and_arm_mapping():
    cands = [sig(1.0, 0.0, 0.25, 3), sig(0.5, 0.5625, 0.5, 4), sig(0.0, 0.0, 1.0, 5), sig(0.25, 0.1875, 0.75, 2)]
    picks = sel.all_picks(cands, random_arm_rng(1, 1))
    assert set(picks) == set(sel.RULES)
    assert picks["sgac_eq10"]["idx"] == 2 and picks["max_level"]["idx"] == 2
    assert picks["max_var"]["idx"] == 1 and picks["max_d"]["idx"] == 2
    assert sel.ARM_RULE["sgac"] == "sgac_eq10" and sel.ARM_RULE["sgac_label_corrected"] == "pickle_true_mapping"
    assert sel.ARM_RULE["fixed_pi1"] is None and all(r in sel.RULES for r in sel.ARM_RULE.values() if r)


def test_nbm_log_statistics():
    d = nbm.pick_distribution()
    assert (d["level5"], d["level4"], d["level_le3"]) == (0.65, 0.35, 0.0)
    assert d["ps0"] == 0.5 and nbm.zero_loss_bursts() == 7
    assert math.isclose(sum(v["eval"] for v in nbm.STEPS if v["eval"] is not None), 0.66 + 0.62 + 0.60 + 0.68)
