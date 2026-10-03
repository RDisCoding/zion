import math

from rlvr_v2.signals import NUMERIC_FEATURES, Signals, compute_signals, rollouts_from_texts
from tests.conftest import approx


def _box(x):
    return f"Solution ... \\boxed{{{x}}}"


def test_reference_case(grader, problem):
    texts = [_box(7)] * 4 + [_box(3), _box(3), _box(5), _box(9)]
    ro = rollouts_from_texts(texts, "7", grader, n_tokens=[100] * 8)
    s = compute_signals(ro, problem, "base", prompt_tokens=20)
    assert approx(s.p_s, 0.5) and s.n_correct == 4
    assert approx(s.v_bin, 0.25)
    assert approx(s.v_legacy, 0.25)
    assert s.n_classes == 4 and approx(s.u_ratio, 0.5)
    assert approx(s.entropy_bits, 1.75)
    assert approx(s.d_simpson, 0.75)
    assert approx(s.d_wrong, 4 / 3 * (1 - (0.25 + 0.0625 + 0.0625)))
    assert approx(s.maj_share, 0.5) and approx(s.maj_margin, 0.25) and s.maj_correct
    assert approx(s.format_rate, 1.0) and approx(s.none_rate, 0.0) and approx(s.trunc_rate, 0.0)
    assert s.level == 1 and s.subject == "Prealgebra" and s.k == 8


def test_all_identical_correct(grader, problem):
    ro = rollouts_from_texts([_box(7)] * 8, "7", grader)
    s = compute_signals(ro, problem)
    assert approx(s.p_s, 1.0) and approx(s.v_bin, 0.0) and approx(s.u_ratio, 1 / 8)
    assert approx(s.entropy_bits, 0.0) and approx(s.d_simpson, 0.0)
    assert math.isnan(s.d_wrong) and math.isnan(s.len_wrong_mean) and s.maj_correct


def test_all_unboxed(grader, problem):
    ro = rollouts_from_texts(["no box here 42"] * 8, "7", grader)
    s = compute_signals(ro, problem)
    assert approx(s.p_s, 0.0) and approx(s.format_rate, 0.0) and approx(s.none_rate, 1.0)
    assert s.n_classes == 1 and not s.maj_correct and math.isnan(s.d_wrong)


def test_reviewer_matched_pair(grader, problem):
    a = compute_signals(rollouts_from_texts([_box(7)] * 4 + [_box(3)] * 4, "7", grader), problem)
    b = compute_signals(rollouts_from_texts([_box(7)] * 4 + [_box(3), _box(5), _box(9), _box(11)], "7", grader), problem)
    assert approx(a.v_bin, b.v_bin) and approx(a.v_bin, 0.25)
    assert approx(a.entropy_bits, 1.0) and approx(b.entropy_bits, 2.0)
    assert approx(a.d_wrong, 0.0) and approx(b.d_wrong, 1.0)
    assert b.d_simpson > a.d_simpson


def test_truncation_counts_and_disqualifies(grader, problem):
    texts = [_box(7)] * 8
    ro = rollouts_from_texts(texts, "7", grader, truncated=[True, True] + [False] * 6)
    s = compute_signals(ro, problem)
    assert approx(s.trunc_rate, 0.25) and approx(s.p_s, 0.75)


def test_equivalent_answers_cluster(grader, problem):
    texts = [_box("\\frac{1}{2}"), _box("0.5"), _box("\\dfrac{1}{2}"), _box("2")]
    ro = rollouts_from_texts(texts, "\\frac{1}{2}", grader)
    s = compute_signals(ro, problem)
    assert s.n_classes == 2 and approx(s.p_s, 0.75)


def test_roundtrip_and_feature_names(grader, problem):
    s = compute_signals(rollouts_from_texts([_box(7)] * 3 + [_box(1)], "7", grader), problem)
    d = s.to_dict()
    assert Signals.from_dict(d) == s
    for f in NUMERIC_FEATURES:
        assert f in d
