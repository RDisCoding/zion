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
    """Primary stop first. Chat styles end turns with <|im_end|>; the base tokenizer's eos is <|endoftext|>."""
    ids: list[int] = []
    for s in stop_strings(style):
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is not None and tid != getattr(tokenizer, "unk_token_id", None) and tid not in ids:
            ids.append(int(tid))
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(int(tokenizer.eos_token_id))
    return ids


def configure_tokenizer(tokenizer, style: str):
    """Make the tokenizer consistent with the prompt style: eos = primary stop, pad distinct, left padding."""
    stops = resolve_stop_token_ids(tokenizer, style)
    primary = tokenizer.convert_ids_to_tokens(stops[0])
    if tokenizer.eos_token != primary:
        tokenizer.eos_token = primary
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        pad = ENDOFTEXT if ENDOFTEXT != tokenizer.eos_token and ENDOFTEXT in tokenizer.get_vocab() else None
        if pad is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tokenizer.pad_token = pad
    tokenizer.padding_side = "left"
    return tokenizer
