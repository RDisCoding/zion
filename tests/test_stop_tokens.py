"""Stop-token / pad configuration against TRL 1.14's completion masking (E0 G3 incident, 2026-10-04).

`trl_view` copies the logic of trl/trainer/grpo_trainer.py (TRL 1.14.1): the completion mask ends at the first
`tokenizer.eos_token_id`, and a completion counts as truncated when its last kept id is neither eos nor pad."""
import pytest

from rlvr_v2.prompts import choose_pad_token, configure_tokenizer, resolve_stop_token_ids

IM_START, IM_END, EOT, FIM_PAD, VISION_PAD = 151644, 151645, 151643, 151662, 151654


class QwenLikeTok:
    """The relevant slice of the Qwen2.5-Math-1.5B tokenizer: eos = pad = <|endoftext|> out of the box."""

    unk_token_id = None
    chat_template = None

    def __init__(self, eos="<|endoftext|>", pad="<|endoftext|>"):
        self.vocab = {"<|im_start|>": IM_START, "<|im_end|>": IM_END, "<|endoftext|>": EOT,
                      "<|fim_pad|>": FIM_PAD, "<|vision_pad|>": VISION_PAD}
        self.inv = {v: k for k, v in self.vocab.items()}
        self.eos_token, self.pad_token, self.padding_side = eos, pad, "right"

    eos_token_id = property(lambda self: self.vocab.get(self.eos_token))
    pad_token_id = property(lambda self: self.vocab.get(self.pad_token) if self.pad_token else None)

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t)

    def convert_ids_to_tokens(self, i):
        return self.inv[i]

    def get_vocab(self):
        return dict(self.vocab)


def trl_view(row, eos_id, pad_id):
    """(kept completion ids, is_truncated) exactly as TRL 1.14.1 computes them for a generated row."""
    end = row.index(eos_id) if eos_id in row else len(row) - 1
    kept = row[: end + 1]
    return kept, kept[-1] not in (eos_id, pad_id)


def test_primary_stop_is_the_models_eos_and_pad_is_not_a_stop():
    tok = configure_tokenizer(QwenLikeTok(), "oneshot_rlvr_chat")
    stops = resolve_stop_token_ids(tok, "oneshot_rlvr_chat")
    assert stops == [EOT, IM_END]
    assert tok.eos_token_id == EOT and tok.pad_token_id == FIM_PAD and tok.padding_side == "left"
    assert tok.pad_token_id not in stops
    assert resolve_stop_token_ids(configure_tokenizer(tok, "oneshot_rlvr_chat"), "oneshot_rlvr_chat") == stops


def test_instruct_style_tokenizer_keeps_im_end_primary():
    tok = configure_tokenizer(QwenLikeTok(eos="<|im_end|>", pad="<|im_end|>"), "qwen_math_chat")
    assert resolve_stop_token_ids(tok, "qwen_math_chat") == [IM_END, EOT]
    assert tok.pad_token_id == FIM_PAD


def test_trl_mask_ends_at_the_real_stop_after_the_fix():
    tok = configure_tokenizer(QwenLikeTok(), "oneshot_rlvr_chat")
    row = [11, 12, 13, EOT] + [tok.pad_token_id] * 5  # what generate() returns for a finished row
    kept, truncated = trl_view(row, tok.eos_token_id, tok.pad_token_id)
    assert kept == [11, 12, 13, EOT] and not truncated
    capped = [11, 12, 13, 14, 15]  # ran into max_new_tokens
    assert trl_view(capped, tok.eos_token_id, tok.pad_token_id)[1]


def test_old_configuration_leaked_padding_into_the_mask():
    """Documents the bug G3 caught: eos <|im_end|>, pad <|endoftext|>, model stops on <|endoftext|>."""
    row = [11, 12, 13, EOT] + [EOT] * 5
    kept, truncated = trl_view(row, IM_END, EOT)
    assert len(kept) == len(row) and not truncated  # padding inside the loss, yet not flagged


def test_choose_pad_refuses_when_only_stop_tokens_exist():
    tok = QwenLikeTok()
    tok.vocab = {"<|im_end|>": IM_END, "<|endoftext|>": EOT}
    with pytest.raises(RuntimeError, match="no pad token candidate"):
        choose_pad_token(tok, [EOT, IM_END])
