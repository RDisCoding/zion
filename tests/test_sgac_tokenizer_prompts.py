"""Per-profile tokens and prompts with the real (cached) Qwen2.5-Math tokenizer."""
import pytest

from rlvr_v2 import prompts
from rlvr_v2.sgac import data as sdata
from rlvr_v2.sgac import legacy, model_io
from rlvr_v2.sgac.spec import load_profile


def _tok(profile):
    spec = load_profile(profile)
    try:
        return spec, model_io.load_tokenizer(spec)
    except Exception as e:
        pytest.skip(f"Qwen tokenizer unavailable: {e!r}")


ITEM = sdata.SgacItem(row=0, orig_index=0, unique_id="x", problem="What is $1+1$?", solution="\\boxed{2}", answer="2",
                      level=1, subject="Algebra")


def test_e0_tokens_and_prompts():
    spec, tok = _tok("e0")
    assert tok.eos_token_id == 151643 and tok.convert_ids_to_tokens(tok.pad_token_id) == "<|fim_pad|>"
    assert tok.padding_side == "left"
    assert model_io.stop_token_ids(spec, tok) == [151643, 151645]
    rendered = model_io.render_prompt(ITEM, spec, tok)
    assert rendered == prompts.build_prompt(ITEM.problem, "oneshot_rlvr_chat", tok)
    assert model_io.train_prompt(ITEM, spec, tok) == rendered  # e0 trains on the identical string


def test_as_run_tokens_and_prompts():
    spec, tok = _tok("as_run")
    assert tok.pad_token_id == tok.eos_token_id == 151643
    assert model_io.stop_token_ids(spec, tok) == [151643]
    assert model_io.render_prompt(ITEM, spec, tok) == legacy.legacy_prompt(ITEM.problem)  # raw, no template
    conv = model_io.train_prompt(ITEM, spec, tok)
    assert conv == [{"role": "user", "content": legacy.legacy_prompt(ITEM.problem)}]
    rendered = model_io.rendered_train_prompt(ITEM, spec, tok)
    assert "Please reason step by step, and put your final answer within \\boxed{}." in rendered  # Qwen default
    assert rendered.endswith("<|im_start|>assistant\n")
    ids = tok(legacy.legacy_prompt(ITEM.problem))["input_ids"]
    assert ids == tok(legacy.legacy_prompt(ITEM.problem), add_special_tokens=False)["input_ids"]  # no BOS added


def test_every_prompt_fits_the_prompt_budget():
    for profile in ("e0", "as_run"):
        spec, tok = _tok(profile)
        try:
            split = sdata.load_split(spec)
        except Exception as e:
            pytest.skip(f"dataset unavailable: {e!r}")
        items = sdata.items_from_rows(split, range(1050), spec.data.shuffle_seed, check_hf_shuffle=False)
        lens = {it.row: len(tok(model_io.rendered_train_prompt(it, spec, tok), add_special_tokens=False)["input_ids"])
                for it in items}
        assert max(lens.values()) < spec.profile.max_prompt_tokens, (profile, max(lens.values()))
        # the single item above E0's 1024-token guard (the reason the SGAC profiles use 2048)
        assert [r for r, n in lens.items() if n > 1024] == [398]
