"""Reward functions for TRL's `GRPOTrainer`.

Only `grader.MathVerifyGrader` decides correctness: a completion earns 1.0 iff its LAST balanced
`\\boxed{...}` is equivalent to the dataset gold answer and the completion was not truncated. There is
deliberately no "last number in the text" fallback (the old code's silent-credit bug). The `format`
reward (a box is present) carries weight 0 in `TrainCfg.reward_weights` and exists for logging only.

TRL 1.14 calling convention (verified by introspection of `GRPOTrainer._calculate_rewards`):

    fn(prompts=..., completions=..., completion_ids=..., **reward_kwargs)

where `reward_kwargs` holds every dataset column other than prompt/completion/completion_ids, repeated once
per completion (our `answer` and `unique_id` columns arrive as lists), plus `trainer_state`, `log_extra`,
`log_metric` and sometimes `environments`. Reward functions therefore MUST accept `**kwargs`. With plain
string prompts the completions are plain strings; chat-style prompts yield lists of messages. TRL keeps the
EOS token inside `completion_ids`, so "last id is not a stop id" means the completion was cut off at
`max_completion_length`. The metric name TRL logs is `rewards/<fn.__name__>/mean`.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

from .artifacts import JsonlWriter
from .grader import MathVerifyGrader, extract_last_boxed

log = logging.getLogger(__name__)

REWARD_NAMES: tuple[str, ...] = ("correctness", "format")


def completion_text(completion: Any) -> str:
    """Plain text of a completion: a string, or the last message's content of a chat-style message list."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return _content_text(completion.get("content", ""))
    if isinstance(completion, (list, tuple)):
        if not completion:
            return ""
        last = completion[-1]
        if isinstance(last, dict):
            return _content_text(last.get("content", ""))
        return str(last)
    return str(completion)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):  # multimodal "content parts"
        return "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def per_completion(value: Any, i: int, n: int) -> Any:
    """Dataset columns arrive as lists with one entry per completion; scalars are broadcast."""
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return value[i] if i < len(value) else value[-1]
    return value


def is_truncated_ids(ids: Any, stop_token_ids: Iterable[int] | None) -> bool:
    """True when `ids` (one completion's token ids, EOS included) does not end with a stop token.
    Unknown (no ids or no stop ids) counts as not truncated, matching the spec."""
    if ids is None or not stop_token_ids:
        return False
    try:
        n = len(ids)
    except TypeError:
        return False
    if n == 0:
        return False
    try:
        last = int(ids[-1])
    except (TypeError, ValueError):
        return False
    return last not in set(int(t) for t in stop_token_ids)


def make_reward_funcs(
    grader: MathVerifyGrader,
    stop_token_ids: list[int] | None = None,
    rollout_writer: JsonlWriter | None = None,
    context: dict | None = None,
) -> list[Callable[..., list[float]]]:
    """Build `[correctness, format]` reward functions (names fixed, in `TrainCfg.reward_weights` order).

    - `correctness`: 1.0 iff `grader.grade(text, gold, truncated).correct`.
      Every completion is appended to `rollout_writer` (if given) as
      {step, optimizer_step, unique_id, correct, format_ok, boxed, method, truncated, n_tokens, text, **context}.
      `step` is `trainer_state.global_step` at generation time (0-based; rewards run before the optimizer
      step increments it), so `optimizer_step = step + 1` matches `global_step` in train_metrics.jsonl.
    - `format`: 1.0 iff a balanced `\\boxed{}` is present (logging only; weight 0 in the config).
    """
    stops = frozenset(int(t) for t in (stop_token_ids or ()))
    ctx = dict(context or {})

    def correctness(prompts, completions, completion_ids=None, answer=None, unique_id=None,
                    trainer_state=None, **kwargs) -> list[float]:
        n = len(completions)
        if answer is None:
            raise ValueError("correctness reward: the training dataset must carry an `answer` column")
        step = getattr(trainer_state, "global_step", None)
        rewards: list[float] = []
        for i, comp in enumerate(completions):
            text = completion_text(comp)
            ids = None
            if completion_ids is not None and i < len(completion_ids):
                ids = completion_ids[i]
            truncated = is_truncated_ids(ids, stops)
            gold = per_completion(answer, i, n)
            if gold is None:
                raise ValueError(f"correctness reward: missing gold answer for completion {i}")
            g = grader.grade(text, str(gold), truncated)
            rewards.append(1.0 if g.correct else 0.0)
            if rollout_writer is not None:
                rollout_writer.write({
                    "step": step,  # trainer global_step when generated (0-based)
                    "optimizer_step": None if step is None else int(step) + 1,  # == train_metrics global_step
                    "unique_id": per_completion(unique_id, i, n),
                    "correct": bool(g.correct),
                    "format_ok": bool(g.format_ok),
                    "boxed": g.boxed,
                    "method": g.method,
                    "truncated": bool(truncated),
                    "n_tokens": (len(ids) if ids is not None else None),
                    "text": text,
                    **ctx,
                })
        return rewards

    def format_reward(prompts, completions, **kwargs) -> list[float]:
        return [1.0 if extract_last_boxed(completion_text(c)) is not None else 0.0 for c in completions]

    correctness.__name__ = correctness.__qualname__ = "correctness"
    format_reward.__name__ = format_reward.__qualname__ = "format"
    return [correctness, format_reward]
