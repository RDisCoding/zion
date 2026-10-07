"""The legacy port IS the published notebook's code, and its signals reproduce the paper's Table 1 values."""
import ast
from pathlib import Path

import numpy as np
import pytest

from rlvr_v2.sgac import legacy

FIXTURE = Path(__file__).parent / "data" / "sgac_nbm_cell3.py.txt"


def _functions(src: str) -> dict[str, str]:
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}


def test_port_is_ast_identical_to_nbm_cell3():
    original = _functions(FIXTURE.read_text(encoding="utf-8"))
    port = _functions(Path(legacy.__file__).read_text(encoding="utf-8"))
    for name in legacy.LEGACY_FUNCTIONS:
        assert name in original, name
        assert port[name] == original[name], f"{name} differs from NB-M cell 3"


def test_prompt_matches_nbm_cell4():
    assert legacy.legacy_prompt("What is {x}?") == (
        "Solve the following math problem step by step and give the final answer in \\boxed{}:\n\nWhat is {x}?")


@pytest.mark.parametrize("text, expected", [
    ("so \\boxed{1} and later \\boxed{2}", "1"),  # FIRST box (the E0 grader takes the last)
    ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),  # nested braces
    ("\\boxed{12", "12"),  # unterminated: the rest of the text
    ("\\boxed{}", ""),
])
def test_extract_box(text, expected):
    assert legacy.extract_box(text) == expected


def test_extract_answer_fallbacks():
    assert legacy.extract_answer("no box, the answer is 42.") == "42."  # regex keeps the trailing period
    assert legacy.extract_answer("values -3 then 7.5") == "7.5"
    assert legacy.extract_answer("no digits at all") is None
    assert legacy.extract_answer("\\boxed{ 5 }") == "5"


def test_rewards_and_correctness_quirks():
    comps = [[{"content": "\\boxed{12}"}], "the result is 12", "\\boxed{13}", "nothing", "\\boxed{1 2}"]
    sol = "We get \\boxed{12}. Also \\boxed{99}."
    assert legacy.binary_match_reward(comps, solution=[sol] * len(comps)) == [1.0, 1.0, 0.0, 0.0, 1.0]
    assert legacy.format_reward(comps) == [0.5, 0.0, 0.5, 0.0, 0.5]
    assert legacy.format_reward(["broken \\boxed{ never closed"]) == [0.5]  # substring check, not balanced
    assert legacy.legacy_gold(sol) == "12"
    assert legacy.legacy_eval_correct("answer: 12", sol) is True  # last-number fallback earns credit
    if not legacy.symbolic_status()["available"]:  # pinned antlr4 4.13: string equality only
        assert legacy.legacy_eval_correct("\\boxed{\\dfrac{1}{2}}", "\\boxed{\\frac{1}{2}}") is False


def test_table1_candidate_1_and_2():
    sol = "\\boxed{12}"
    one = legacy.legacy_signals(["\\boxed{12}"] * 7 + ["\\boxed{13}"], sol)
    assert (one["Ps"], one["Var"], one["D"]) == (0.875, 0.109375, 0.25)
    two = legacy.legacy_signals(["\\boxed{12}"] + [f"\\boxed{{{v}}}" for v in range(20, 26)] + ["I think 7"], sol)
    assert two["Ps"] == 0.125 and two["Var"] == pytest.approx(0.15234375) and two["D"] == 1.0
    assert two["Var"] == pytest.approx(float(np.var(two["total"])))  # population variance (ddof=0)


def test_disagreement_counts_none_as_one_answer():
    sig = legacy.legacy_signals(["\\boxed{1}", "no answer", "still none", "\\boxed{1}"], "\\boxed{1}")
    assert sig["answers"] == ["1", None, None, "1"]
    assert sig["D"] == 0.5


def test_symbolic_status_is_recorded():
    st = legacy.symbolic_status()
    assert set(st) >= {"available", "error", "sympy", "antlr4"}
