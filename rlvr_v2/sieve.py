"""Sieve: K rollouts per candidate -> graded rollouts -> `Signals`, with per-rollout audit logs and resume.

Files written when `out_dir` is given
- rollouts.jsonl  one line per rollout (unique_id, policy_tag, rollout_idx, prompt_hash, n_tokens,
                  finish_reason, truncated, boxed, correct, answer_class, text)
- signals.jsonl   one line per problem (`Signals.to_dict()` + seed + prompt_hash)

Resume: problems whose (unique_id, policy_tag) already appear in signals.jsonl are skipped and
returned from the file. Batches are seeded with `seed + batch_index` over the *remaining*
problems, so a resumed run does not reproduce the random streams of an uninterrupted one.
"""
from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from .artifacts import JsonlWriter, read_jsonl
from .config import Config
from .data import Problem
from .grader import MathVerifyGrader
from .prompts import build_prompt, prompt_hash
from .sampling import Sampler
from .signals import Signals, compute_signals, grade_rollouts

log = logging.getLogger(__name__)

ROLLOUTS_FILE = "rollouts.jsonl"
SIGNALS_FILE = "signals.jsonl"


def prompt_token_count(tokenizer, rendered: str) -> int:
    """Prompt length in tokens (`add_special_tokens=False`); whitespace-token approximation without a tokenizer."""
    if tokenizer is None:
        return len(rendered.split())
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def load_cached_signals(out_dir: str | Path, policy_tag: str | None = None) -> dict[str, Signals]:
    """unique_id -> Signals from `out_dir/signals.jsonl` (last line wins), optionally for one policy tag."""
    out: dict[str, Signals] = {}
    for row in read_jsonl(Path(out_dir) / SIGNALS_FILE):
        if policy_tag is not None and row.get("policy_tag") != policy_tag:
            continue
        out[row["unique_id"]] = Signals.from_dict(row)
    return out


def measure_signals(
    sampler: Sampler,
    tokenizer,
    problems: Sequence[Problem],
    cfg: Config,
    grader: MathVerifyGrader,
    policy_tag: str,
    k: int,
    seed: int,
    out_dir: Path | None = None,
    system: str | None = None,
    prompts_per_batch: int = 8,
) -> list[Signals]:
    """Sample `k` rollouts per problem with `cfg.gen.sieve` temperature/top_p, grade them and compute
    `Signals`. Returns one `Signals` per problem in input order."""
    if k < 2:
        raise ValueError(f"k must be >= 2 to compute signals, got {k}")
    if prompts_per_batch <= 0:
        raise ValueError("prompts_per_batch must be positive")
    problems = list(problems)
    style = cfg.prompt.style
    system = cfg.prompt.system if system is None else system
    params = dataclasses.replace(cfg.gen.sieve, n=int(k))

    cached: dict[str, Signals] = load_cached_signals(out_dir, policy_tag) if out_dir is not None else {}
    results: dict[str, Signals] = {p.unique_id: cached[p.unique_id] for p in problems if p.unique_id in cached}
    todo = [p for p in problems if p.unique_id not in results]
    if results:
        log.info("sieve resume: %d/%d problems already in %s", len(results), len(problems), out_dir)
    if not todo:
        return [results[p.unique_id] for p in problems]

    rollouts_w = signals_w = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rollouts_w = JsonlWriter(out_dir / ROLLOUTS_FILE, append=True)
        signals_w = JsonlWriter(out_dir / SIGNALS_FILE, append=True)
    t0 = time.perf_counter()
    done = 0
    try:
        for b, start in enumerate(range(0, len(todo), prompts_per_batch)):
            batch = todo[start:start + prompts_per_batch]
            rendered = [build_prompt(p.problem, style, tokenizer, system) for p in batch]
            groups = sampler.generate(rendered, k, params, int(seed) + b)
            if len(groups) != len(batch) or any(len(g) != k for g in groups):
                raise RuntimeError(
                    f"sampler returned {[len(g) for g in groups]} rollouts for {len(batch)} prompts; expected {k} each"
                )
            for p, text, group in zip(batch, rendered, groups):
                graded = grade_rollouts(group, p.answer, grader)
                sig = compute_signals(graded, p, policy_tag, prompt_token_count(tokenizer, text))
                ph = prompt_hash(text)
                if rollouts_w is not None:
                    for i, r in enumerate(graded):
                        rollouts_w.write({
                            "unique_id": p.unique_id, "policy_tag": policy_tag, "rollout_idx": i, "prompt_hash": ph,
                            "n_tokens": r.n_tokens, "finish_reason": r.finish_reason, "truncated": r.truncated,
                            "boxed": r.boxed, "correct": r.correct, "answer_class": r.answer_class, "text": r.text,
                        })
                if signals_w is not None:
                    signals_w.write({**sig.to_dict(), "seed": int(seed), "prompt_hash": ph})
                results[p.unique_id] = sig
            done += len(batch)
            stats = getattr(sampler, "stats", None) or {}
            log.info(
                "sieve [%s] %d/%d problems (k=%d) %.0fs elapsed; last batch %.1f tok/s; mean p_s so far %.3f, trunc %.3f",
                policy_tag, done, len(todo), k, time.perf_counter() - t0, stats.get("tokens_per_s", 0.0),
                sum(results[p.unique_id].p_s for p in todo[:done]) / done,
                sum(results[p.unique_id].trunc_rate for p in todo[:done]) / done,
            )
    finally:
        if rollouts_w is not None:
            rollouts_w.close()
        if signals_w is not None:
            signals_w.close()
    return [results[p.unique_id] for p in problems]


def preflight_truncation_guard(signals: Sequence[Signals], max_trunc_rate: float) -> None:
    """Raise `RuntimeError` when the mean truncation rate exceeds `max_trunc_rate` (completion cap too small
    or stop tokens wrong); the message lists the worst offending unique_ids."""
    signals = list(signals)
    if not signals:
        return
    mean = sum(s.trunc_rate for s in signals) / len(signals)
    if mean > max_trunc_rate:
        offenders = sorted((s for s in signals if s.trunc_rate > max_trunc_rate), key=lambda s: -s.trunc_rate)
        shown = [(s.unique_id, round(s.trunc_rate, 3)) for s in offenders[:20]]
        raise RuntimeError(
            f"mean truncation rate {mean:.3f} over {len(signals)} candidates exceeds {max_trunc_rate:.3f}; "
            f"{len(offenders)} candidates above the threshold, worst: {shown}. "
            "Raise gen.max_new_tokens or check the stop tokens before training."
        )
    log.info("truncation guard ok: mean trunc_rate %.3f <= %.3f over %d candidates", mean, max_trunc_rate, len(signals))
