"""Greedy evaluation with per-item logs, bootstrap CIs and paired comparisons.

Files written to `out_dir`
- per_item.jsonl  unique_id, level, subject, correct, format_ok, method, boxed, finish_reason, truncated,
                  n_tokens, answer (+ text when `cfg.eval.store_text`)
- summary.json    `EvalSummary.to_dict()`
- eval_meta.json  identity of the evaluation (policy, adapter, prompt style/hash, cap) used to decide
                  whether an interrupted per_item.jsonl may be resumed

Resume: an existing summary.json with the same `n` is returned without recomputation unless
`force=True`; an interrupted run (per_item.jsonl present, matching eval_meta.json, no summary)
continues from the items already graded.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .artifacts import JsonlWriter, atomic_write_json, read_json, read_jsonl
from .config import Config
from .data import Problem
from .grader import MathVerifyGrader
from .prompts import build_prompt, prompt_hash
from .sampling import Sampler

log = logging.getLogger(__name__)

PER_ITEM_FILE = "per_item.jsonl"
SUMMARY_FILE = "summary.json"
META_FILE = "eval_meta.json"


@dataclass
class EvalSummary:
    tag: str
    n: int
    n_correct: int
    acc: float
    ci_lo: float
    ci_hi: float
    trunc_rate: float
    format_rate: float
    by_level: dict
    by_subject: dict
    wall_s: float
    prompt_style: str
    prompt_hash: str  # hash of the first rendered prompt
    backend: str
    policy: str
    adapter_path: str | None
    max_new_tokens: int
    n_boot: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> EvalSummary:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# ---------------------------------------------------------------------- statistics
def bootstrap_ci(values: Sequence[float], n_boot: int, seed: int = 0, alpha: float = 0.05,
                 block: int = 256) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (resampling items), numpy-seeded; (nan, nan) for no items."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if n_boot <= 0:
        m = float(arr.mean())
        return m, m
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, block):  # blocks bound memory for large item sets
        stop = min(n_boot, start + block)
        idx = rng.integers(0, arr.size, size=(stop - start, arr.size))
        means[start:stop] = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _group_stats(rows: Sequence[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        v = r.get(key)
        groups["unknown" if v is None else str(v)].append(bool(r["correct"]))
    return {g: {"n": len(v), "n_correct": int(sum(v)), "acc": sum(v) / len(v)} for g, v in sorted(groups.items())}


def summarize_rows(rows: Sequence[dict], tag: str, cfg: Config, prompt_hash_value: str, policy: str,
                   adapter_path: str | None, wall_s: float) -> EvalSummary:
    """Aggregate per-item rows into an `EvalSummary` (bootstrap seed 0, percentile 2.5/97.5)."""
    n = len(rows)
    correct = [bool(r["correct"]) for r in rows]
    n_correct = int(sum(correct))
    acc = n_correct / n if n else float("nan")
    lo, hi = bootstrap_ci([1.0 if c else 0.0 for c in correct], cfg.eval.bootstrap_samples, seed=0)
    return EvalSummary(
        tag=tag, n=n, n_correct=n_correct, acc=acc, ci_lo=lo, ci_hi=hi,
        trunc_rate=(sum(1 for r in rows if r.get("finish_reason") == "length") / n) if n else float("nan"),
        format_rate=(sum(1 for r in rows if r.get("format_ok")) / n) if n else float("nan"),
        by_level=_group_stats(rows, "level"), by_subject=_group_stats(rows, "subject"), wall_s=wall_s,
        prompt_style=cfg.prompt.style, prompt_hash=prompt_hash_value, backend=cfg.gen.backend, policy=policy,
        adapter_path=adapter_path, max_new_tokens=cfg.gen.max_new_tokens, n_boot=cfg.eval.bootstrap_samples,
    )


# ---------------------------------------------------------------------- evaluation
def _row(p: Problem, rollout, grade, store_text: bool) -> dict:
    row = {
        "unique_id": p.unique_id, "level": p.level, "subject": p.subject, "correct": bool(grade.correct),
        "format_ok": bool(grade.format_ok), "method": grade.method, "boxed": grade.boxed,
        "finish_reason": rollout.finish_reason, "truncated": bool(rollout.truncated), "n_tokens": rollout.n_tokens,
        "answer": p.answer,
    }
    if store_text:
        row["text"] = rollout.text
    return row


def evaluate(
    sampler: Sampler,
    tokenizer,
    problems: Sequence[Problem],
    cfg: Config,
    grader: MathVerifyGrader,
    out_dir: Path,
    tag: str,
    policy: str = "base",
    adapter_path: str | None = None,
    prompts_per_batch: int = 64,
    force: bool = False,
) -> EvalSummary:
    """Greedy (`cfg.gen.eval`, n=1) evaluation of `problems`; see the module docstring for files and resume."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = list(problems)
    if cfg.eval.max_items:
        problems = problems[: cfg.eval.max_items]
    n = len(problems)
    if n == 0:
        raise ValueError("no problems to evaluate")
    if prompts_per_batch <= 0:
        raise ValueError("prompts_per_batch must be positive")
    summary_path, per_item_path, meta_path = out_dir / SUMMARY_FILE, out_dir / PER_ITEM_FILE, out_dir / META_FILE

    if not force and summary_path.exists():
        cached = read_json(summary_path)
        if cached and cached.get("n") == n:
            log.info("eval resume: returning cached %s (n=%d, acc=%.4f)", summary_path, n, cached.get("acc", float("nan")))
            return EvalSummary.from_dict(cached)
        log.info("existing %s has n=%s != %d; recomputing", summary_path, (cached or {}).get("n"), n)

    params = dataclasses.replace(cfg.gen.eval, n=1)
    if params.temperature != 0.0:
        log.warning("evaluation should be greedy but cfg.gen.eval.temperature=%s", params.temperature)
    style, system = cfg.prompt.style, cfg.prompt.system
    rendered = [build_prompt(p.problem, style, tokenizer, system) for p in problems]
    first_hash = prompt_hash(rendered[0])
    meta = {"tag": tag, "policy": policy, "adapter_path": adapter_path, "prompt_style": style,
            "prompt_hash": first_hash, "max_new_tokens": cfg.gen.max_new_tokens, "backend": cfg.gen.backend,
            "store_text": bool(cfg.eval.store_text), "n": n}

    done: dict[str, dict] = {}
    if not force and per_item_path.exists():
        if read_json(meta_path) == meta:
            wanted = {p.unique_id for p in problems}
            done = {r["unique_id"]: r for r in read_jsonl(per_item_path) if r["unique_id"] in wanted}
            if done:
                log.info("eval resume: %d/%d items already graded in %s", len(done), n, per_item_path)
        else:
            log.info("%s belongs to a different evaluation; recomputing all items", per_item_path)
    atomic_write_json(meta_path, meta)

    t0 = time.perf_counter()
    rows: list[dict] = []
    with JsonlWriter(per_item_path, append=False) as writer:
        for start in range(0, n, prompts_per_batch):
            batch = list(range(start, min(n, start + prompts_per_batch)))
            todo = [i for i in batch if problems[i].unique_id not in done]
            fresh: dict[int, dict] = {}
            if todo:
                groups = sampler.generate([rendered[i] for i in todo], 1, params, seed=0)
                if len(groups) != len(todo) or any(len(g) != 1 for g in groups):
                    raise RuntimeError(f"sampler returned {[len(g) for g in groups]} completions for {len(todo)} prompts")
                for i, group in zip(todo, groups):
                    r = group[0]
                    g = grader.grade(r.text, problems[i].answer, truncated=r.truncated)
                    fresh[i] = _row(problems[i], r, g, cfg.eval.store_text)
            for i in batch:
                row = fresh.get(i) or done[problems[i].unique_id]
                writer.write(row)
                rows.append(row)
            acc_so_far = sum(1 for r in rows if r["correct"]) / len(rows)
            stats = getattr(sampler, "stats", None) or {}
            log.info("eval [%s] %d/%d items, running acc %.4f, last batch %.1f tok/s", tag, len(rows), n, acc_so_far,
                     stats.get("tokens_per_s", 0.0))
    wall = time.perf_counter() - t0
    summary = summarize_rows(rows, tag, cfg, first_hash, policy, adapter_path, wall)
    atomic_write_json(summary_path, summary.to_dict())
    log.info("eval [%s] done: acc %.4f [%.4f, %.4f] n=%d trunc %.3f format %.3f in %.0fs", tag, summary.acc,
             summary.ci_lo, summary.ci_hi, n, summary.trunc_rate, summary.format_rate, wall)
    return summary


# ---------------------------------------------------------------------- comparisons
def load_per_item(path: str | Path) -> list[dict]:
    return read_jsonl(path)


def paired_delta(per_item_a: Path, per_item_b: Path, n_boot: int = 2000, seed: int = 0) -> dict:
    """acc_b - acc_a on the shared unique_ids with a paired item bootstrap CI (resampling items, so the
    per-item differences are resampled together)."""
    a = {r["unique_id"]: bool(r["correct"]) for r in load_per_item(per_item_a)}
    b = {r["unique_id"]: bool(r["correct"]) for r in load_per_item(per_item_b)}
    shared = sorted(set(a) & set(b))
    out = {"n_shared": len(shared), "n_a": len(a), "n_b": len(b), "n_boot": n_boot, "seed": seed}
    if not shared:
        out.update(delta=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), acc_a=float("nan"),
                   acc_b=float("nan"), n_a_only_correct=0, n_b_only_correct=0)
        return out
    xa = np.array([a[u] for u in shared], dtype=float)
    xb = np.array([b[u] for u in shared], dtype=float)
    diff = xb - xa
    lo, hi = bootstrap_ci(diff, n_boot, seed=seed)
    out.update(
        delta=float(diff.mean()), ci_lo=lo, ci_hi=hi, acc_a=float(xa.mean()), acc_b=float(xb.mean()),
        n_a_only_correct=int(((xa == 1) & (xb == 0)).sum()), n_b_only_correct=int(((xb == 1) & (xa == 0)).sum()),
    )
    if math.isnan(out["delta"]):  # pragma: no cover - defensive
        log.warning("paired_delta produced NaN for %s vs %s", per_item_a, per_item_b)
    return out
