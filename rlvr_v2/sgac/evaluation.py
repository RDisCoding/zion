"""Greedy evaluation with both graders per item, resumable, batched exactly like `evaluate.evaluate` (E0 G1).

`prompts_per_call` prompts go to one sampler call (HF chunks of the sampler's batch size, length-sorted inside the
call), seed 0, so for the e0 profile the generated text matches E0's evaluation path item for item. Files in
`out_dir`: per_item.jsonl (both verdicts, both golds, full text), summary.json, eval_meta.json.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from ..artifacts import JsonlWriter, atomic_write_json, read_json, read_jsonl
from ..config import SamplingParams
from ..evaluate import bootstrap_ci
from ..prompts import prompt_hash
from . import legacy
from .data import SgacItem
from .grading import DualGrader
from .model_io import render_prompt
from .spec import SgacSpec

log = logging.getLogger(__name__)

PER_ITEM_FILE, SUMMARY_FILE, META_FILE = "per_item.jsonl", "summary.json", "eval_meta.json"
N_BOOT = 10000


def _by_level(rows: Sequence[dict]) -> dict:
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("level"))].append(bool(r["correct"]))
    return {k: {"n": len(v), "acc": sum(v) / len(v)} for k, v in sorted(groups.items())}


def summarize(rows: Sequence[dict], meta: dict, wall_s: float) -> dict:
    n = len(rows)
    corr = [1.0 if r["correct"] else 0.0 for r in rows]
    lo, hi = bootstrap_ci(corr, N_BOOT, seed=0)
    return {
        **meta, "n": n, "n_correct": int(sum(corr)), "acc": sum(corr) / n if n else float("nan"), "ci_lo": lo,
        "ci_hi": hi, "n_boot": N_BOOT,
        "acc_mv": sum(1 for r in rows if r["mv_correct"]) / n if n else float("nan"),
        "acc_legacy": sum(1 for r in rows if r["legacy_correct"]) / n if n else float("nan"),
        "trunc_rate": sum(1 for r in rows if r["truncated"]) / n if n else float("nan"),
        "mv_format_rate": sum(1 for r in rows if r["mv_format_ok"]) / n if n else float("nan"),
        "mean_tokens": sum(r["n_tokens"] for r in rows) / n if n else float("nan"),
        "by_level": _by_level(rows), "wall_s": wall_s,
    }


def evaluate_items(sampler, tok, items: Sequence[SgacItem], spec: SgacSpec, grader: DualGrader, out_dir: Path,
                   tag: str, policy: str, adapter_ref: str | None = None, force: bool = False,
                   render=None, prompt_name: str | None = None) -> dict:
    """`render(item) -> str` overrides the profile's prompt (Phase-1 used its own eval prompt)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = list(items)
    if spec.loop.eval_max_items:
        items = items[: spec.loop.eval_max_items]
    n = len(items)
    if n == 0:
        raise ValueError("no items to evaluate")
    rendered = [render(it) if render else render_prompt(it, spec, tok) for it in items]
    desc = sampler.describe() if hasattr(sampler, "describe") else {}
    meta = {"tag": tag, "policy": policy, "adapter_ref": adapter_ref, "profile": spec.profile.name,
            "grader_primary": grader.primary, "prompt": prompt_name or spec.profile.prompt,
            "prompt_hash_first": prompt_hash(rendered[0]),
            "max_new_tokens": desc.get("max_new_tokens"), "batch_size": desc.get("batch_size"),
            "stop_token_ids": desc.get("stop_token_ids"), "prompts_per_call": int(spec.profile.eval_prompts_per_call),
            "n": n, "uids_hash": prompt_hash("\n".join(it.unique_id for it in items))}
    summary_path, per_item_path, meta_path = out_dir / SUMMARY_FILE, out_dir / PER_ITEM_FILE, out_dir / META_FILE

    if not force and summary_path.exists():
        cached = read_json(summary_path)
        if cached and all(cached.get(k) == v for k, v in meta.items()):
            log.info("eval %s: cached (acc %.4f)", tag, cached["acc"])
            return cached
    done: dict[str, dict] = {}
    if not force and per_item_path.exists() and read_json(meta_path) == meta:
        done = {r["unique_id"]: r for r in read_jsonl(per_item_path)}
    atomic_write_json(meta_path, meta)

    params = SamplingParams(temperature=0.0, top_p=1.0, n=1)
    step = max(1, int(spec.profile.eval_prompts_per_call))
    t0 = time.perf_counter()
    rows: list[dict] = []
    with JsonlWriter(per_item_path, append=False) as w:
        for start in range(0, n, step):
            idx = list(range(start, min(n, start + step)))
            todo = [i for i in idx if items[i].unique_id not in done]
            fresh: dict[int, dict] = {}
            if todo:
                groups = sampler.generate([rendered[i] for i in todo], 1, params, seed=0)
                for i, group in zip(todo, groups):
                    r = group[0]
                    g = grader.grade(r.text, items[i], truncated=r.truncated)
                    fresh[i] = {
                        "unique_id": items[i].unique_id, "row": items[i].row, "level": items[i].level,
                        "subject": items[i].subject, **g, "finish_reason": r.finish_reason, "truncated": bool(r.truncated),
                        "n_tokens": r.n_tokens, "stop_id": r.stop_id, "answer": items[i].answer,
                        "gold_legacy": legacy.legacy_gold(items[i].legacy_solution), "text": r.text,
                    }
            for i in idx:
                row = fresh.get(i) or done[items[i].unique_id]
                w.write(row)
                rows.append(row)
            log.info("eval [%s] %d/%d running acc %.4f", tag, len(rows), n, sum(r["correct"] for r in rows) / len(rows))
    summary = summarize(rows, meta, time.perf_counter() - t0)
    atomic_write_json(summary_path, summary)
    log.info("eval [%s] acc %.4f [%.4f, %.4f] (mv %.4f, legacy %.4f) trunc %.3f n=%d", tag, summary["acc"],
             summary["ci_lo"], summary["ci_hi"], summary["acc_mv"], summary["acc_legacy"], summary["trunc_rate"], n)
    return summary


def item_agreement(per_item_a: Path, per_item_b: Path, key: str = "correct") -> dict:
    """Share of shared unique_ids on which two evaluations give the same verdict (and identical text)."""
    a = {r["unique_id"]: r for r in read_jsonl(per_item_a)}
    b = {r["unique_id"]: r for r in read_jsonl(per_item_b)}
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n_shared": 0, "verdict_agreement": float("nan"), "text_agreement": float("nan")}
    same = sum(1 for u in shared if bool(a[u].get(key)) == bool(b[u].get(key)))
    same_text = sum(1 for u in shared if a[u].get("text") is not None and a[u].get("text") == b[u].get("text"))
    return {"n_shared": len(shared), "verdict_agreement": same / len(shared), "text_agreement": same_text / len(shared),
            "acc_a": sum(bool(a[u].get(key)) for u in shared) / len(shared),
            "acc_b": sum(bool(b[u].get(key)) for u in shared) / len(shared)}
