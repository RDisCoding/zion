import dataclasses
import json
import math
import re

import pytest

from rlvr_v2.artifacts import read_json, read_jsonl
from rlvr_v2.config import Config, EvalCfg
from rlvr_v2.data import Problem
from rlvr_v2.evaluate import EvalSummary, bootstrap_ci, evaluate, load_per_item, paired_delta
from rlvr_v2.sieve import measure_signals, preflight_truncation_guard, prompt_token_count
from rlvr_v2.signals import Rollout

_IDX = re.compile(r"Problem (\d+):")


def _cfg(**eval_kwargs) -> Config:
    cfg = Config()
    cfg = dataclasses.replace(cfg, study2=dataclasses.replace(cfg.study2, k=cfg.gen.sieve.n))
    return dataclasses.replace(cfg, eval=EvalCfg(bootstrap_samples=300, **eval_kwargs))


def _problems(n=10):
    subjects = ("Algebra", "Geometry")
    return [Problem(f"math_train/{i}", f"Problem {i}: what is 3+4?", "7", None, 1 + i % 5, subjects[i % 2], "math_train")
            for i in range(n)]


class FakeSampler:
    """`\\boxed{7}` for problems whose index satisfies `is_correct(i)`, "nope 7" otherwise; n copies each."""

    def __init__(self, is_correct=lambda i: i % 2 == 0, texts_for=None, truncated_last=False):
        self.is_correct = is_correct
        self.texts_for = texts_for
        self.truncated_last = truncated_last
        self.calls = 0
        self.prompts_seen: list[str] = []
        self.stats = {"tokens_per_s": 1.0}

    def generate(self, prompts, n, params, seed):
        self.calls += 1
        self.prompts_seen.extend(prompts)
        out = []
        for p in prompts:
            i = int(_IDX.search(p).group(1))
            if self.texts_for is not None:
                texts = self.texts_for(i, n)
            else:
                texts = ["\\boxed{7}" if self.is_correct(i) else "nope 7"] * n
            rollouts = []
            for j, t in enumerate(texts):
                trunc = self.truncated_last and j == n - 1
                rollouts.append(Rollout(t, 10 + j, trunc, finish_reason="length" if trunc else "stop"))
            out.append(rollouts)
        return out


# ---------------------------------------------------------------------- evaluate
def test_evaluate_end_to_end(tmp_path, grader):
    cfg = _cfg()
    probs = _problems(10)
    sampler = FakeSampler()
    summ = evaluate(sampler, None, probs, cfg, grader, tmp_path / "eval", tag="t", policy="base", prompts_per_batch=4)
    assert isinstance(summ, EvalSummary)
    assert summ.n == 10 and summ.n_correct == 5 and summ.acc == pytest.approx(0.5)
    assert 0.0 <= summ.ci_lo <= summ.acc <= summ.ci_hi <= 1.0 and summ.ci_lo < summ.ci_hi
    assert summ.trunc_rate == 0.0 and summ.format_rate == pytest.approx(0.5)
    assert summ.prompt_style == cfg.prompt.style and len(summ.prompt_hash) == 12
    assert summ.backend == "hf" and summ.policy == "base" and summ.adapter_path is None
    assert summ.max_new_tokens == cfg.gen.max_new_tokens and summ.n_boot == 300
    assert sum(v["n"] for v in summ.by_level.values()) == 10 and set(summ.by_subject) == {"Algebra", "Geometry"}
    assert summ.by_subject["Algebra"]["acc"] == pytest.approx(1.0) and summ.by_subject["Geometry"]["acc"] == 0.0
    assert sampler.calls == 3  # 10 prompts in batches of 4

    rows = read_jsonl(tmp_path / "eval" / "per_item.jsonl")
    assert len(rows) == 10 and [r["unique_id"] for r in rows] == [p.unique_id for p in probs]
    for key in ("unique_id", "level", "subject", "correct", "format_ok", "method", "boxed", "finish_reason",
                "n_tokens", "answer", "text"):
        assert key in rows[0]
    assert rows[0]["correct"] is True and rows[0]["boxed"] == "7" and rows[0]["method"] == "string"
    assert rows[1]["correct"] is False and rows[1]["method"] == "no_box"

    saved = read_json(tmp_path / "eval" / "summary.json")
    assert saved["acc"] == pytest.approx(0.5) and saved["n"] == 10 and saved["tag"] == "t"
    assert EvalSummary.from_dict(saved) == summ

    # resume: identical summary without touching the sampler
    again = evaluate(sampler, None, probs, cfg, grader, tmp_path / "eval", tag="t")
    assert again == summ and sampler.calls == 3
    # force recomputes
    forced = evaluate(sampler, None, probs, cfg, grader, tmp_path / "eval", tag="t", force=True)
    assert forced.acc == pytest.approx(0.5) and sampler.calls > 3


def test_evaluate_store_text_false_and_max_items(tmp_path, grader):
    cfg = _cfg(store_text=False, max_items=4)
    summ = evaluate(FakeSampler(), None, _problems(10), cfg, grader, tmp_path, tag="t")
    rows = read_jsonl(tmp_path / "per_item.jsonl")
    assert summ.n == 4 and len(rows) == 4 and "text" not in rows[0]


def test_evaluate_partial_resume_from_per_item(tmp_path, grader):
    cfg = _cfg()
    probs = _problems(8)
    first = FakeSampler()
    evaluate(first, None, probs, cfg, grader, tmp_path, tag="t", prompts_per_batch=8)
    (tmp_path / "summary.json").unlink()  # simulate an interrupted run with graded items on disk
    second = FakeSampler()
    summ = evaluate(second, None, probs, cfg, grader, tmp_path, tag="t")
    assert summ.n == 8 and second.calls == 0 and len(read_jsonl(tmp_path / "per_item.jsonl")) == 8
    # a different policy must not reuse the rows
    third = FakeSampler()
    (tmp_path / "summary.json").unlink()
    evaluate(third, None, probs, cfg, grader, tmp_path, tag="t", policy="lora")
    assert third.calls == 1


def test_bootstrap_ci_edges():
    assert bootstrap_ci([1.0, 1.0, 1.0], 100) == (1.0, 1.0)
    lo, hi = bootstrap_ci([], 100)
    assert math.isnan(lo) and math.isnan(hi)
    assert bootstrap_ci([0.0, 1.0], 0) == (0.5, 0.5)
    lo, hi = bootstrap_ci([0.0] * 50 + [1.0] * 50, 500, seed=0)
    assert 0.35 < lo < 0.5 < hi < 0.65
    assert bootstrap_ci([0.0, 1.0, 1.0, 0.0], 50, seed=1) == bootstrap_ci([0.0, 1.0, 1.0, 0.0], 50, seed=1)


# ---------------------------------------------------------------------- paired delta
def test_paired_delta(tmp_path, grader):
    cfg = _cfg()
    probs = _problems(10)
    evaluate(FakeSampler(lambda i: i % 2 == 0), None, probs, cfg, grader, tmp_path / "a", tag="a")
    evaluate(FakeSampler(lambda i: i % 2 == 1 or i == 0), None, probs, cfg, grader, tmp_path / "b", tag="b")
    d = paired_delta(tmp_path / "a" / "per_item.jsonl", tmp_path / "b" / "per_item.jsonl", n_boot=300)
    assert d["n_shared"] == 10 and d["delta"] == pytest.approx(0.1)
    assert d["acc_a"] == pytest.approx(0.5) and d["acc_b"] == pytest.approx(0.6)
    assert d["n_a_only_correct"] == 4 and d["n_b_only_correct"] == 5
    assert d["ci_lo"] <= d["delta"] <= d["ci_hi"]

    # subset of ids on one side -> join on the shared ones only
    rows = load_per_item(tmp_path / "b" / "per_item.jsonl")[:4]
    sub = tmp_path / "b_sub.jsonl"
    sub.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    d2 = paired_delta(tmp_path / "a" / "per_item.jsonl", sub, n_boot=100)
    assert d2["n_shared"] == 4 and d2["acc_a"] == pytest.approx(0.5) and d2["acc_b"] == pytest.approx(0.75)
    assert d2["delta"] == pytest.approx(0.25)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    d3 = paired_delta(tmp_path / "a" / "per_item.jsonl", empty)
    assert d3["n_shared"] == 0 and math.isnan(d3["delta"])


# ---------------------------------------------------------------------- sieve
def _four(i, n):
    assert n == 4
    return ["\\boxed{7}", "\\boxed{7}", "\\boxed{3}", "no box 9"]


def _same_signals(a, b) -> bool:
    """Field-wise equality that treats NaN == NaN (d_wrong is NaN here and NaN != NaN after a JSON round trip)."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        dx, dy = x.to_dict(), y.to_dict()
        for k in dx:
            vx, vy = dx[k], dy[k]
            if isinstance(vx, float) and isinstance(vy, float) and math.isnan(vx) and math.isnan(vy):
                continue
            if vx != vy:
                return False
    return True


def test_measure_signals_writes_files_and_resumes(tmp_path, grader):
    cfg = _cfg()
    probs = _problems(6)
    sampler = FakeSampler(texts_for=_four, truncated_last=True)
    out = tmp_path / "sieve"
    sigs = measure_signals(sampler, None, probs, cfg, grader, policy_tag="base", k=4, seed=7, out_dir=out,
                           prompts_per_batch=4)
    assert [s.unique_id for s in sigs] == [p.unique_id for p in probs]
    for s in sigs:
        assert s.k == 4 and s.policy_tag == "base" and s.p_s == pytest.approx(0.5) and s.n_correct == 2
        assert s.format_rate == pytest.approx(0.75) and s.trunc_rate == pytest.approx(0.25)
        assert s.n_classes == 3 and s.prompt_tokens > 0
    assert sampler.calls == 2 and len(sampler.prompts_seen) == 6

    rollouts = read_jsonl(out / "rollouts.jsonl")
    assert len(rollouts) == 24
    r0 = rollouts[0]
    for key in ("unique_id", "policy_tag", "rollout_idx", "prompt_hash", "n_tokens", "finish_reason", "truncated",
                "boxed", "correct", "answer_class", "text"):
        assert key in r0
    assert r0["correct"] is True and r0["boxed"] == "7" and rollouts[3]["truncated"] is True
    assert rollouts[3]["finish_reason"] == "length"
    srows = read_jsonl(out / "signals.jsonl")
    assert len(srows) == 6 and srows[0]["seed"] == 7 and len(srows[0]["prompt_hash"]) == 12

    # full resume: nothing sampled, identical signals
    again = measure_signals(sampler, None, probs, cfg, grader, "base", 4, 7, out_dir=out)
    assert _same_signals(again, sigs) and sampler.calls == 2
    assert all(math.isnan(s.d_wrong) for s in again)  # NaN survives the JSON round trip
    # partial resume: only the two new problems are sampled, order follows the input list
    more = probs + _problems(8)[6:]
    sigs2 = measure_signals(sampler, None, more, cfg, grader, "base", 4, 7, out_dir=out, prompts_per_batch=4)
    assert sampler.calls == 3 and len(sampler.prompts_seen) == 8
    assert [s.unique_id for s in sigs2] == [p.unique_id for p in more] and _same_signals(sigs2[:6], sigs)
    assert len(read_jsonl(out / "signals.jsonl")) == 8 and len(read_jsonl(out / "rollouts.jsonl")) == 32
    # a different policy tag is not served from the cache
    other = FakeSampler(texts_for=_four)
    measure_signals(other, None, probs, cfg, grader, "lora", 4, 7, out_dir=out)
    assert other.calls == 1


def test_measure_signals_without_out_dir_and_validation(tmp_path, grader):
    cfg = _cfg()
    sampler = FakeSampler(texts_for=_four)
    sigs = measure_signals(sampler, None, _problems(3), cfg, grader, "base", 4, 0)
    assert len(sigs) == 3 and not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        measure_signals(sampler, None, _problems(3), cfg, grader, "base", 1, 0)
    bad = FakeSampler(texts_for=lambda i, n: ["\\boxed{7}"] * 3)  # wrong number of rollouts
    with pytest.raises(RuntimeError):
        measure_signals(bad, None, _problems(2), cfg, grader, "base", 4, 0)
    assert prompt_token_count(None, "a b  c") == 3


def test_preflight_truncation_guard(mk_signals):
    ok = [dataclasses.replace(mk_signals(uid=f"c{i}"), trunc_rate=0.05) for i in range(4)]
    preflight_truncation_guard(ok, 0.2)
    preflight_truncation_guard([], 0.2)
    bad = ok + [dataclasses.replace(mk_signals(uid="bad"), trunc_rate=1.0)]
    with pytest.raises(RuntimeError, match="bad"):
        preflight_truncation_guard(bad, 0.2)
