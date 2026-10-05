"""Prereg §5 pool-eligibility rule: gold answers that the grader cannot parse are excluded before the sieve."""
import json
from pathlib import Path

import pytest

from rlvr_v2.candidates import unparseable_golds
from rlvr_v2.data import Problem

DATA = Path(__file__).parent / "data"
MATRIX = json.loads((DATA / "matrix_golds.json").read_text(encoding="utf-8"))
CURRENCY = json.loads((DATA / "currency_golds.json").read_text(encoding="utf-8"))
REPO = Path(__file__).parents[1]


@pytest.mark.parametrize("gold", ["5", "-\\frac{1}{3}", "\\sqrt{12}", "(1,2)", "[-2,7]", "0.5"])
def test_ordinary_golds_are_parseable(grader, gold):
    assert grader.gold_parseable(gold)


@pytest.mark.parametrize("item", CURRENCY, ids=[c["unique_id"] for c in CURRENCY])
def test_currency_golds_are_parseable_after_the_fix(grader, item):
    assert grader.gold_parseable(item["gold"])


@pytest.mark.parametrize("item", MATRIX, ids=[m["unique_id"] for m in MATRIX])
def test_matrix_golds_parseable_except_phantom(grader, item):
    assert grader.gold_parseable(item["gold"]) == ("\\phantom" not in item["gold"])


def test_unparseable_golds_lists_only_the_failing_items(grader):
    probs = [Problem(m["unique_id"], "q", m["gold"], None, 4, "Precalculus", "math_train") for m in MATRIX]
    assert [i["unique_id"] for i in unparseable_golds(probs, grader)] == ["train/precalculus/1049.json"]


def test_committed_pool_ineligibility_list():
    """The list committed before the sieve: exactly 1049 (the 2 currency pool items are parseable after the fix)."""
    committed = json.loads((REPO / "manifests" / "pool_ineligible.json").read_text(encoding="utf-8"))
    assert [i["unique_id"] for i in committed["items"]] == ["train/precalculus/1049.json"]
    assert committed["n_pool"] == 500
