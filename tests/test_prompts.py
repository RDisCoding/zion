import pytest

from rlvr_v2.data import ONESHOT_SUFFIX
from rlvr_v2.prompts import IM_END, IM_START, build_prompt, prompt_hash, resolve_stop_token_ids, stop_strings


class FakeTok:
    eos_token_id = 151643
    unk_token_id = None
    chat_template = None
    vocab = {"<|im_end|>": 151645, "<|endoftext|>": 151643}

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, None)


def test_chat_render_is_deterministic_and_chatml():
    p1 = build_prompt("What is 1+1?", "qwen_math_chat")
    p2 = build_prompt("What is 1+1?", "qwen_math_chat")
    assert p1 == p2
    assert p1.startswith(f"{IM_START}system\n") and p1.endswith(f"{IM_START}assistant\n")
    assert f"{IM_START}user\nWhat is 1+1?{IM_END}" in p1


def test_oneshot_style_appends_suffix():
    p = build_prompt("Q", "oneshot_rlvr_chat")
    assert ("Q" + ONESHOT_SUFFIX) in p


def test_raw_has_no_chatml():
    p = build_prompt("Q", "raw")
    assert IM_START not in p and p.startswith("Q")


def test_unknown_style():
    with pytest.raises(ValueError):
        build_prompt("Q", "nope")


def test_hash_and_stops():
    assert prompt_hash("abc") == prompt_hash("abc") and len(prompt_hash("abc")) == 12
    assert stop_strings("raw") == ["<|endoftext|>"]
    assert resolve_stop_token_ids(FakeTok(), "qwen_math_chat") == [151645, 151643]
    assert resolve_stop_token_ids(FakeTok(), "raw") == [151643]
