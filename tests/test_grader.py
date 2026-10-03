import pytest

from rlvr_v2.grader import NONE_CLASS, cluster_answers, extract_last_boxed, normalize_answer


@pytest.mark.parametrize(
    "text,expected",
    [
        ("so \\boxed{\\frac{1}{2}} done", "\\frac{1}{2}"),
        ("\\boxed{3} then \\boxed{5}", "5"),
        ("\\boxed{\\sqrt{2}+\\sqrt{3}}", "\\sqrt{2}+\\sqrt{3}"),
        ("\\boxed{\\frac{a}{\\frac{b}{c}}}", "\\frac{a}{\\frac{b}{c}}"),
        ("\\boxed {7}", "7"),
        ("\\fbox{12}", "12"),
        ("\\boxed{\\frac{1}{2", None),
        ("no box at all 42", None),
        ("", None),
        ("\\boxed{}", ""),
    ],
)
def test_extract_last_boxed(text, expected):
    assert extract_last_boxed(text) == expected


@pytest.mark.parametrize(
    "raw,norm",
    [
        ("\\dfrac{3}{4}", "\\frac{3}{4}"),
        ("\\text{Evelyn}", "Evelyn"),
        ("90^\\circ", "90"),
        ("10\\%", "10"),
        ("x=5", "5"),
        ("1,000", "1000"),
        ("5.0", "5"),
        ("0.50", "0.5"),
        ("\\left( 1, 2 \\right)", "(1,2)"),
    ],
)
def test_normalize_answer(raw, norm):
    assert normalize_answer(raw) == norm


CORRECT = [
    ("The answer is \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}"),
    ("\\boxed{0.5}", "\\frac{1}{2}"),
    ("\\boxed{\\dfrac{3}{4}}", "\\frac{3}{4}"),
    ("\\boxed{-\\frac{1}{3}}", "-\\frac{1}{3}"),
    ("\\boxed{[-2, 7]}", "[-2,7]"),
    ("\\boxed{\\text{Evelyn}}", "\\text{Evelyn}"),
    ("\\boxed{Evelyn}", "\\text{Evelyn}"),
    ("\\boxed{3} ... \\boxed{5}", "5"),
    ("\\boxed{5}", "5.0"),
    ("\\boxed{90^\\circ}", "90"),
    ("\\boxed{1, 2}", "1,2"),
    ("\\boxed{x=5}", "5"),
    ("\\boxed{10\\%}", "10"),
    ("\\boxed{1,000}", "1000"),
    ("\\boxed{\\frac{2}{4}}", "\\frac{1}{2}"),
    ("\\boxed{2\\sqrt{3}}", "\\sqrt{12}"),
    ("\\boxed{\\frac{\\sqrt{2}}{2}}", "\\frac{1}{\\sqrt{2}}"),
    ("\\boxed{12.8}", "12.8"),
    ("\\boxed{\\pi}", "\\pi"),
    ("\\boxed{(1,2)}", "(1, 2)"),
]

INCORRECT = [
    ("\\boxed{\\frac{1}{3}}", "-\\frac{1}{3}"),
    ("\\boxed{(-2, 7)}", "[-2,7]"),
    ("\\boxed{3} ... \\boxed{5}", "3"),
    ("\\boxed{12.7}", "12.8"),
    ("\\boxed{6}", "7"),
    ("\\boxed{\\frac{1}{2}}", "\\frac{1}{3}"),
    ("\\boxed{}", "5"),
]


@pytest.mark.parametrize("text,gold", CORRECT)
def test_correct_cases(grader, text, gold):
    g = grader.grade(text, gold)
    assert g.correct and g.format_ok, g


@pytest.mark.parametrize("text,gold", INCORRECT)
def test_incorrect_cases(grader, text, gold):
    g = grader.grade(text, gold)
    assert not g.correct, g


def test_unboxed_number_is_incorrect_but_flagged(grader):
    g = grader.grade("The answer is 5.", "5")
    assert not g.correct and not g.format_ok and g.number_present and g.method == "no_box"


def test_truncated_boxed_is_incorrect(grader):
    g = grader.grade("\\boxed{5}", "5", truncated=True)
    assert not g.correct and g.format_ok and g.method == "truncated"


@pytest.mark.parametrize(
    "text,gold",
    [("\\boxed{0.3333}", "\\frac{1}{3}"), ("\\boxed{\\frac{1}{3}}", "0.333"), ("\\boxed{1/2}", "0.5")],
)
def test_documented_behaviour(grader, text, gold):
    """Recorded, not asserted: edge cases whose policy we document in the paper."""
    g = grader.grade(text, gold)
    print(f"DOCUMENTED {text!r} vs {gold!r}: correct={g.correct} method={g.method}")


def test_cluster_answers(grader):
    labels = cluster_answers(["0.5", "\\frac{1}{2}", "1/2", "2", None, "\\dfrac{1}{2}", ""], grader)
    assert labels[0] == labels[1] == labels[2] == labels[5]
    assert labels[3] != labels[0]
    assert labels[4] == NONE_CLASS and labels[6] == NONE_CLASS
    assert len({lab for lab in labels if lab != NONE_CLASS}) == 2
