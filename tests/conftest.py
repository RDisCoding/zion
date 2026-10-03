import math

import pytest

from rlvr_v2.data import Problem
from rlvr_v2.grader import MathVerifyGrader
from rlvr_v2.signals import Signals


@pytest.fixture(scope="session")
def grader():
    return MathVerifyGrader()


@pytest.fixture
def problem():
    return Problem("math_train/test-1", "What is 3+4?", "7", None, 1, "Prealgebra", "math_train")


def make_signals(uid="c", p_s=0.5, d_simpson=0.5, d_wrong=float("nan"), level=3, u_ratio=0.5, entropy_bits=1.0, k=8):
    n_correct = round(p_s * k)
    return Signals(
        unique_id=uid, k=k, policy_tag="base", p_s=p_s, n_correct=n_correct, v_bin=p_s * (1 - p_s),
        v_legacy=p_s * (1 - p_s), u_ratio=u_ratio, n_classes=max(1, round(u_ratio * k)), d_simpson=d_simpson,
        entropy_bits=entropy_bits, entropy_mm=entropy_bits, d_wrong=d_wrong, n_wrong_parsable=k - n_correct,
        maj_share=max(p_s, 1 - p_s), maj_margin=abs(2 * p_s - 1), maj_correct=p_s >= 0.5, format_rate=1.0,
        trunc_rate=0.0, none_rate=0.0, len_mean=300.0, len_sd=50.0, len_correct_mean=280.0, len_wrong_mean=320.0,
        level=level, subject="Algebra", prompt_tokens=120,
    )


@pytest.fixture
def mk_signals():
    return make_signals


def approx(a, b, tol=1e-6):
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(a - b) <= tol
