"""Evaluation records both graders per item, batches like evaluate.evaluate, and resumes."""
from pathlib import Path

import pytest

from rlvr_v2.artifacts import read_json, read_jsonl
from rlvr_v2.sgac.data import SgacItem
from rlvr_v2.sgac.evaluation import evaluate_items, item_agreement
from rlvr_v2.sgac.grading import DualGrader
from rlvr_v2.sgac.spec import load_profile
from rlvr_v2.signals import Rollout


class FakeTok:
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [0] * len(text.split())}


class FakeSampler:
    """Answers item i with its gold if i is even; odd items get an unboxed correct number (legacy-only credit)."""

    def __init__(self, items):
        self.answers = {it.problem: (i, it.answer) for i, it in enumerate(items)}
        self.calls = []
        self.stats = {}

    def describe(self):
        return {"max_new_tokens": 64, "batch_size": 2, "stop_token_ids": [1]}

    def generate(self, prompts, n, params, seed):
        assert n == 1 and params.temperature == 0.0 and seed == 0
        self.calls.append(len(prompts))
        out = []
        for p in prompts:
            i, ans = next(v for k, v in self.answers.items() if k in p)
            text = f"so \\boxed{{{ans}}}" if i % 2 == 0 else f"the answer is {ans}"
            out.append([Rollout(text=text, n_tokens=5, truncated=False, finish_reason="stop", stop_id=1)])
        return out


def items(n=5):
    return [SgacItem(row=1000 + i, orig_index=i, unique_id=f"u{i}", problem=f"Question number {i} zz{i}zz",
                     solution=f"\\boxed{{{i + 3}}}", answer=str(i + 3), level=1 + i % 5, subject="Algebra") for i in range(n)]


@pytest.fixture(scope="module")
def mv():
    from rlvr_v2.grader import MathVerifyGrader

    return MathVerifyGrader()


@pytest.mark.parametrize("profile, primary_acc", [("e0", 3 / 5), ("as_run", 1.0)])
def test_both_verdicts_and_primary(tmp_path, mv, profile, primary_acc):
    spec = load_profile(profile, ["profile.eval_prompts_per_call=2"])
    its = items()
    s = evaluate_items(FakeSampler(its), FakeTok(), its, spec, DualGrader(spec.profile.grader, mv), tmp_path, "t", "p")
    assert s["acc"] == pytest.approx(primary_acc)
    assert s["acc_mv"] == pytest.approx(3 / 5) and s["acc_legacy"] == pytest.approx(1.0)
    rows = read_jsonl(Path(tmp_path) / "per_item.jsonl")
    assert [r["unique_id"] for r in rows] == [it.unique_id for it in its]
    assert {"mv_correct", "legacy_correct", "gold_legacy", "text", "correct"} <= set(rows[0])


def test_cached_and_partial_resume(tmp_path, mv):
    spec = load_profile("e0", ["profile.eval_prompts_per_call=2"])
    its = items()
    g = DualGrader("math_verify", mv)
    first = FakeSampler(its)
    s1 = evaluate_items(first, FakeTok(), its, spec, g, tmp_path, "t", "p")
    assert first.calls == [2, 2, 1]  # prompts_per_call chunks, like evaluate.evaluate
    again = FakeSampler(its)
    assert evaluate_items(again, FakeTok(), its, spec, g, tmp_path, "t", "p")["acc"] == s1["acc"] and not again.calls
    (Path(tmp_path) / "summary.json").unlink()  # interrupted after all items were written
    resumed = FakeSampler(its)
    evaluate_items(resumed, FakeTok(), its, spec, g, tmp_path, "t", "p")
    assert not resumed.calls
    assert read_json(Path(tmp_path) / "summary.json")["n"] == 5


def test_item_agreement(tmp_path, mv):
    spec = load_profile("e0", ["profile.eval_prompts_per_call=4"])
    its = items()
    g = DualGrader("math_verify", mv)
    evaluate_items(FakeSampler(its), FakeTok(), its, spec, g, tmp_path / "a", "a", "p")
    evaluate_items(FakeSampler(its), FakeTok(), its, spec, g, tmp_path / "b", "b", "p")
    agree = item_agreement(tmp_path / "a" / "per_item.jsonl", tmp_path / "b" / "per_item.jsonl")
    assert agree["verdict_agreement"] == 1.0 and agree["text_agreement"] == 1.0 and agree["n_shared"] == 5
