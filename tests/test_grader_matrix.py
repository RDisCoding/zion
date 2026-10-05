"""Regression tests for matrix answers (E0 G2 cross-check, 2026-10-05: `test/precalculus/625` was a false negative
because normalisation deleted the `\\` row separators). One test per gold answer with a LaTeX row break in the
pool, held-out and MATH-500 manifests (18 in total, tests/data/matrix_golds.json).

For every gold: an equivalent rewrite in the form a model typically produces must be accepted, and a copy with one
entry changed must be rejected (the fix must not trade precision for recall)."""
import json
import re
from pathlib import Path

import pytest

GOLDS = json.loads((Path(__file__).parent / "data" / "matrix_golds.json").read_text(encoding="utf-8"))
BEGIN, END = "\\begin{pmatrix}", "\\end{pmatrix}"


def _rows(gold: str) -> list[list[str]]:
    body = gold.strip()
    assert body.startswith(BEGIN) and body.endswith(END), gold
    body = body[len(BEGIN):-len(END)]
    return [[e.strip() for e in row.split("&")] for row in body.split("\\\\")]


def _render(rows: list[list[str]]) -> str:
    return BEGIN + " " + " \\\\ ".join(" & ".join(r) for r in rows) + " " + END


def _rewrite(entry: str) -> str:
    """Equivalent entry in the form models usually write. `\\phantom X` typesets X invisibly, so it is dropped
    together with its argument (`\\phantom -1` has the value 1)."""
    e = re.sub(r"\\phantom\s*(\{[^{}]*\}|\S)", "", entry).strip()
    m = re.fullmatch(r"(-?)(\d+)/(\d+)", e)
    if m:
        return f"{m.group(1)}\\frac{{{m.group(2)}}}{{{m.group(3)}}}"
    m = re.fullmatch(r"(-?)(\\sqrt\{\d+\})/(\d+)", e)
    if m:
        return f"{m.group(1)}\\frac{{{m.group(2)}}}{{{m.group(3)}}}"
    if re.fullmatch(r"-?\d+", e):
        return f"{e}.0"
    return e


def equivalent_variant(gold: str) -> str:
    return _render([[_rewrite(e) for e in row] for row in _rows(gold)])


def perturbed_variant(gold: str) -> str:
    rows = _rows(gold)
    rows[0][0] = f"{_rewrite(rows[0][0])} + 1"
    return _render(rows)


PHANTOM_LIMITATION = pytest.mark.xfail(
    strict=True,
    reason="known limitation, flagged 2026-10-05, not fixed: math-verify cannot parse \\phantom inside a pmatrix "
           "gold (the parse degenerates to a scalar), so even the true answer is rejected",
)
ACCEPT_PARAMS = [pytest.param(g, id=g["unique_id"], marks=PHANTOM_LIMITATION if "\\phantom" in g["gold"] else ())
                 for g in GOLDS]


@pytest.mark.parametrize("item", ACCEPT_PARAMS)
def test_matrix_gold_equivalent_rewrite_is_accepted(grader, item):
    variant = equivalent_variant(item["gold"])
    assert variant.replace(" ", "") != item["gold"].replace(" ", ""), "variant must differ textually from the gold"
    text = f"Therefore the answer is $\\boxed{{{variant}}}$."
    result = grader.grade(text, item["gold"])
    assert result.correct, (variant, result)
    assert grader.equivalent(item["gold"], variant)  # symmetric


@pytest.mark.parametrize("item", GOLDS, ids=[g["unique_id"] for g in GOLDS])
def test_matrix_gold_with_one_wrong_entry_is_rejected(grader, item):
    wrong = perturbed_variant(item["gold"])
    assert not grader.grade(f"\\boxed{{{wrong}}}", item["gold"]).correct, wrong


@pytest.mark.parametrize("item", GOLDS, ids=[g["unique_id"] for g in GOLDS])
def test_scalar_never_matches_a_matrix_gold(grader, item):
    """Guard against degenerate parses: no single entry of the gold, written as a bare scalar, may match it."""
    for entry in {e for row in _rows(item["gold"]) for e in row}:
        scalar = _rewrite(entry)
        assert not grader.grade(f"\\boxed{{{scalar}}}", item["gold"]).correct, scalar


def test_phantom_gold_rejects_bare_scalar(grader):
    gold = next(g["gold"] for g in GOLDS if "\\phantom" in g["gold"])
    assert not grader.grade("\\boxed{-1}", gold).correct


def test_matrix_golds_cover_all_three_splits():
    assert len(GOLDS) == 18
    assert {g["split"] for g in GOLDS} == {"pool", "heldout", "math500"}


def test_cross_check_item_625_exact_output(grader):
    gold = "\\begin{pmatrix} 16/49 \\\\ 48/49 \\\\ 24/49 \\end{pmatrix}"
    text = ("So the final answer is:\n\n\\[\n\\boxed{\\begin{pmatrix} \\frac{16}{49} \\\\ \\frac{48}{49} \\\\ "
            "\\frac{24}{49} \\end{pmatrix}}\n\\]")
    assert grader.grade(text, gold).correct
