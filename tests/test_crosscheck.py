import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("grader_crosscheck", Path(__file__).parents[1] / "scripts" / "grader_crosscheck.py")
gc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gc)


def row(ours, qwen, truncated=False, boxed="5", on_box=None):
    return {"ours_correct": ours, "qwen_correct": qwen, "truncated": truncated, "our_boxed": boxed,
            "qwen_on_our_box": on_box, "stored_correct": ours, "qwen_status": "ok"}


def test_categories():
    assert gc.categorize(row(True, True)) is None
    assert gc.categorize(row(False, True, truncated=True)) == "rule_truncated"
    assert gc.categorize(row(False, True, boxed=None)) == "rule_unboxed"
    assert gc.categorize(row(True, False, on_box=True)) == "extraction"
    assert gc.categorize(row(True, False, on_box=False)) == "equivalence"


def test_summary_rates():
    rows = [row(True, True), row(False, True, truncated=True), row(True, False, on_box=False), row(False, False)]
    for r in rows:
        r["category"] = gc.categorize(r)
    s = gc.summarize(rows, {})
    assert s["n"] == 4 and s["disagree"] == 2 and s["raw_disagreement_rate"] == 0.5
    assert s["by_category"] == {"rule_truncated": 1, "equivalence": 1} and s["equivalence_disagreement_rate"] == 0.25
