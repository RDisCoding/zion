"""Regression tests for currency answers (gold-parseability screen, 2026-10-05): normalisation stripped `$` but
left the backslash of `\\$`, so `\\$36` became `\\36` and only the exact string `\\$36` was accepted. One test per gold
with `\\$` in the pool, held-out and MATH-500 manifests (7, tests/data/currency_golds.json)."""
import json
from decimal import Decimal
from pathlib import Path

import pytest

from rlvr_v2.grader import normalize_answer

GOLDS = json.loads((Path(__file__).parent / "data" / "currency_golds.json").read_text(encoding="utf-8"))
IDS = [g["unique_id"] for g in GOLDS]


def value_of(gold: str) -> str:
    return gold.replace("\\$", "").replace("\\!", "").replace(",", "").strip()


def accepted_forms(gold: str) -> list[str]:
    v = value_of(gold)
    forms = [v, "\\$" + v, "\\$ " + v]
    if "." in v:
        forms.append(str(Decimal(v).normalize()))  # 18.90 -> 18.9, 5.50 -> 5.5
    if len(v.split(".")[0]) > 3:
        forms.append(f"{Decimal(v):,}")  # 32348 -> 32,348
    return forms


def wrong_value(gold: str) -> str:
    v = Decimal(value_of(gold))
    return str(v + 1)


@pytest.mark.parametrize("item", GOLDS, ids=IDS)
def test_currency_gold_accepts_value_with_or_without_dollar(grader, item):
    for form in accepted_forms(item["gold"]):
        assert grader.grade(f"The answer is $\\boxed{{{form}}}$.", item["gold"]).correct, form


@pytest.mark.parametrize("item", GOLDS, ids=IDS)
def test_currency_gold_rejects_wrong_value(grader, item):
    wrong = wrong_value(item["gold"])
    for form in (wrong, "\\$" + wrong):
        assert not grader.grade(f"\\boxed{{{form}}}", item["gold"]).correct, form


@pytest.mark.parametrize("item", GOLDS, ids=IDS)
def test_currency_gold_normalises_without_stray_backslash(item):
    assert "\\" not in normalize_answer(item["gold"]), normalize_answer(item["gold"])


def test_currency_golds_cover_all_three_splits():
    assert len(GOLDS) == 7 and {g["split"] for g in GOLDS} == {"pool", "heldout", "math500"}


def test_plain_dollar_math_delimiters_still_stripped():
    assert normalize_answer("$\\frac{1}{2}$") == "\\frac{1}{2}"
