"""The only place prompts are rendered. Sieve, training and evaluation all call `build_prompt`.

Styles
- qwen_math_chat:    ChatML with the Qwen2.5-Math system prompt; user turn = problem.
- oneshot_rlvr_chat: ChatML, no explicit system turn (the Qwen2.5-Math template inserts its default),
                     user turn = problem + Wang et al.'s suffix. Matches the pi1 training data.
- raw:               plain completion prompt, no chat template.

Rendered prompts are plain strings so TRL never re-applies a chat template.
"""
from __future__ import annotations

import hashlib

from .config import DEFAULT_SYSTEM_PROMPT
from .data import ONESHOT_SUFFIX

STYLES = ("qwen_math_chat", "oneshot_rlvr_chat", "raw")
IM_START, IM_END, ENDOFTEXT = "<|im_start|>", "<|im_end|>", "<|endoftext|>"


def _chatml(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Manual ChatML rendering identical to Qwen2.5's template for system/user turns.
    Used when no tokenizer is available (tests) and cross-checked against the tokenizer in E0."""
    out = []
    for m in messages:
        out.append(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n")
    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n")
    return "".join(out)


def messages_for(problem_text: str, style: str, system: str = DEFAULT_SYSTEM_PROMPT) -> list[dict]:
    if style == "qwen_math_chat":
        return [{"role": "system", "content": system}, {"role": "user", "content": problem_text}]
    if style == "oneshot_rlvr_chat":
        return [{"role": "system", "content": system}, {"role": "user", "content": problem_text + ONESHOT_SUFFIX}]
    raise ValueError(f"style {style!r} has no chat messages")


def build_prompt(problem_text: str, style: str, tokenizer=None, system: str = DEFAULT_SYSTEM_PROMPT) -> str:
    if style not in STYLES:
        raise ValueError(f"unknown prompt style {style!r}; expected one of {STYLES}")
    if style == "raw":
        return f"{problem_text}\n\n{system}\n\nSolution:"
    msgs = messages_for(problem_text, style, system)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return _chatml(msgs)


def prompt_hash(rendered: str) -> str:
    return hashlib.sha1(rendered.encode("utf-8")).hexdigest()[:12]


def stop_strings(style: str) -> list[str]:
    return [IM_END, ENDOFTEXT] if style != "raw" else [ENDOFTEXT]


def resolve_stop_token_ids(tokenizer, style: str) -> list[int]:
    """All stop ids for the style, primary first. Primary = the tokenizer's EOS (the model's declared end of
    sequence), which is also the single id TRL ends the completion mask at, so it must be the token the model
    actually emits; the style's other stop strings follow as secondary stops. Qwen2.5-Math-1.5B is a base model whose EOS is <|endoftext|>; it ends
    chat-formatted answers with <|endoftext|>, not <|im_end|> (E0 G3, 2026-10-04)."""
    ids: list[int] = []
    vocab = None
    for s in stop_strings(style):
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is None or tid in ids:
            continue
        # A token that is genuinely in the vocabulary may legitimately share its id with unk (SmolLM2's
        # <|endoftext|> is id 0 == unk), so check vocabulary membership instead of comparing with unk_token_id.
        if tid == getattr(tokenizer, "unk_token_id", None):
            if vocab is None:
                try:
                    vocab = tokenizer.get_vocab()
                except Exception:  # pragma: no cover - fake tokenizers in tests
                    vocab = {}
            if s not in vocab:
                continue
        ids.append(int(tid))
    eos = tokenizer.eos_token_id
    if eos is not None:
        eos = int(eos)
        if eos in ids:
            ids.remove(eos)
        ids.insert(0, eos)
    return ids


# Existing vocabulary tokens that are never generated in this setting (no embedding resize needed).
PAD_CANDIDATES = ("<|fim_pad|>", "<|vision_pad|>", "<empty_output>", "<pad>", "[PAD]")


def choose_pad_token(tokenizer, stop_ids: list[int]) -> str:
    """A pad token that is NOT a stop token. TRL treats a completion whose last id is eos OR pad as finished, so a
    pad that is also a stop token lets padding leak into the GRPO loss."""
    vocab = tokenizer.get_vocab()
    for tok in PAD_CANDIDATES:
        tid = vocab.get(tok)
        if tid is not None and int(tid) not in stop_ids:
            return tok
    raise RuntimeError(f"no pad token candidate {PAD_CANDIDATES} in the vocabulary is distinct from the stop ids "
                       f"{stop_ids}; add one to PAD_CANDIDATES (adding a new token would require an embedding resize)")


def configure_tokenizer(tokenizer, style: str):
    """Make the tokenizer consistent with the prompt style: eos = primary stop (the model's own EOS), pad = an
    existing token that is not any stop token, left padding."""
    stops = resolve_stop_token_ids(tokenizer, style)
    primary = tokenizer.convert_ids_to_tokens(stops[0])
    if tokenizer.eos_token != primary:
        tokenizer.eos_token = primary
    if tokenizer.pad_token_id is None or int(tokenizer.pad_token_id) in stops:
        tokenizer.pad_token = choose_pad_token(tokenizer, stops)
    tokenizer.padding_side = "left"
    return tokenizer
