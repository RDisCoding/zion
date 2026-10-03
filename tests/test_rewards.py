from types import SimpleNamespace

import pytest

from rlvr_v2.artifacts import JsonlWriter, read_jsonl
from rlvr_v2.rewards import completion_text, is_truncated_ids, make_reward_funcs

STOP = [151645, 151643]  # <|im_end|>, <|endoftext|>
# everything TRL 1.14 passes besides the dataset columns; reward functions must tolerate all of it
TRAINER_KW = dict(trainer_state=SimpleNamespace(global_step=3), log_extra=lambda *a, **k: None,
                  log_metric=lambda *a, **k: None, environments=None)


def _ids(n, stop=True):
    return list(range(10, 10 + n)) + ([STOP[0]] if stop else [])


def test_reward_names_and_order(grader):
    fns = make_reward_funcs(grader, stop_token_ids=STOP)
    assert [f.__name__ for f in fns] == ["correctness", "format"]


def test_string_completions_and_no_last_number_fallback(grader):
    correctness, fmt = make_reward_funcs(grader, stop_token_ids=STOP)
    comps = ["So the answer is \\boxed{7}.", "I get \\boxed{8}.", "The answer is 7."]
    kw = dict(prompts=["p"] * 3, completions=comps, completion_ids=[_ids(4), _ids(4), _ids(3)],
              answer=["7", "7", "7"], unique_id=["u"] * 3, **TRAINER_KW)
    assert correctness(**kw) == [1.0, 0.0, 0.0]  # an unboxed "7" earns nothing
    assert fmt(**kw) == [1.0, 1.0, 0.0]


def test_truncation_is_derived_from_completion_ids(grader):
    correctness, _ = make_reward_funcs(grader, stop_token_ids=STOP)
    boxed = ["\\boxed{7}"]
    assert correctness(prompts=["p"], completions=boxed, completion_ids=[_ids(2, stop=True)], answer=["7"]) == [1.0]
    assert correctness(prompts=["p"], completions=boxed, completion_ids=[_ids(2, stop=False)], answer=["7"]) == [0.0]
    assert correctness(prompts=["p"], completions=boxed, completion_ids=[[5, STOP[1]]], answer=["7"]) == [1.0]
    assert correctness(prompts=["p"], completions=boxed, answer=["7"]) == [1.0]  # no ids -> not truncated
    no_stops, _ = make_reward_funcs(grader)
    assert no_stops(prompts=["p"], completions=boxed, completion_ids=[[5, 6]], answer=["7"]) == [1.0]
    assert is_truncated_ids([], STOP) is False and is_truncated_ids(None, STOP) is False
    assert is_truncated_ids([1, 2, 3], STOP) is True and is_truncated_ids([1, STOP[0]], STOP) is False


def test_message_list_completions(grader):
    correctness, fmt = make_reward_funcs(grader, stop_token_ids=STOP)
    comps = [
        [{"role": "assistant", "content": "x = \\boxed{\\frac{1}{2}}"}],
        [{"role": "assistant", "content": "\\boxed{0.3}"}],
        [{"role": "assistant", "content": [{"type": "text", "text": "\\boxed{3}"}]}],
    ]
    prompts = [[{"role": "user", "content": "q"}]] * 3
    assert correctness(prompts=prompts, completions=comps, answer=["\\frac{1}{2}", "1/2", "3"]) == [1.0, 0.0, 1.0]
    assert fmt(prompts=prompts, completions=comps) == [1.0, 1.0, 1.0]
    assert completion_text("abc") == "abc" and completion_text([]) == ""
    assert completion_text({"role": "assistant", "content": "z"}) == "z"


def test_answer_list_is_routed_per_completion(grader):
    correctness, _ = make_reward_funcs(grader, stop_token_ids=STOP)
    comps = ["\\boxed{1}", "\\boxed{2}"]
    assert correctness(prompts=["p", "p"], completions=comps, answer=["2", "2"]) == [0.0, 1.0]
    assert correctness(prompts=["p", "p"], completions=comps, answer=["1", "2"]) == [1.0, 1.0]
    assert correctness(prompts=["p", "p"], completions=comps, answer="1") == [1.0, 0.0]  # scalar broadcast
    with pytest.raises(ValueError):
        correctness(prompts=["p"], completions=["\\boxed{1}"])


def test_rollout_writer_gets_one_record_per_completion(grader, tmp_path):
    path = tmp_path / "rollouts.jsonl"
    writer = JsonlWriter(path)
    correctness, _ = make_reward_funcs(grader, stop_token_ids=STOP, rollout_writer=writer,
                                       context={"arm": "variance", "curriculum_step": 2})
    out = correctness(prompts=["p"] * 3, completions=["\\boxed{7}", "nothing here", "\\boxed{7} and more"],
                      completion_ids=[_ids(1), _ids(2, stop=False), _ids(5, stop=False)], answer=["7"] * 3,
                      unique_id=["a", "b", "c"], weird_future_kwarg=object(), **TRAINER_KW)
    writer.close()
    assert out == [1.0, 0.0, 0.0]
    recs = read_jsonl(path)
    assert len(recs) == 3
    assert recs[0] == {"step": 3, "optimizer_step": 4, "unique_id": "a", "correct": True, "format_ok": True,
                       "boxed": "7", "method": "string", "truncated": False, "n_tokens": 2, "text": "\\boxed{7}",
                       "arm": "variance", "curriculum_step": 2}
    assert recs[1]["correct"] is False and recs[1]["method"] == "no_box" and recs[1]["truncated"] is True
    assert recs[2]["correct"] is False and recs[2]["method"] == "truncated" and recs[2]["n_tokens"] == 5
    assert recs[2]["unique_id"] == "c"
