"""Phase A (NB-M cell 7): K rollouts per candidate with the CURRENT policy, then the four SGAC signals.

Both signal variants are computed from the same rollouts:
- legacy (NB-M verbatim): Ps = mean(binary), Var = np.var(binary + 0.5*box-substring), D = |{str(extract_answer)}|/K;
- E0 (`rlvr_v2.signals`): p_s via math-verify, v_legacy = pvar(correct + 0.5*boxed), u_ratio = math-verify classes/K.
The profile picks which variant feeds the selector (`selection_signals`); the other is logged for diagnostics.
All candidates of a step are generated in one sampler call (B*K sequences, one chunk at the default batch size).
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from ..artifacts import JsonlWriter, atomic_write_json
from ..config import SamplingParams
from ..prompts import prompt_hash
from ..sieve import prompt_token_count
from ..signals import compute_signals, grade_rollouts
from . import legacy
from .data import SgacItem
from .grading import DualGrader
from .model_io import render_prompt
from .spec import SgacSpec

ROLLOUTS_FILE = "rollouts.jsonl"
SIGNALS_FILE = "signals.json"


def selection_signals(spec: SgacSpec, lsig: dict, msig, level) -> dict:
    if spec.profile.grader == "legacy":
        ps, var = lsig["Ps"], lsig["Var"]
    else:
        ps, var = msig.p_s, msig.v_legacy
    d = msig.u_ratio if spec.profile.d_metric == "u_ratio" else lsig["D"]
    return {"Ps": float(ps), "Var": float(var), "D": float(d), "L": None if level is None else int(level)}


def sieve_candidates(sampler, tok, items: Sequence[SgacItem], spec: SgacSpec, grader: DualGrader, seed: int,
                     out_dir: Path, policy_tag: str) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k = int(spec.loop.k)
    params = SamplingParams(temperature=float(spec.loop.temperature), top_p=float(spec.loop.top_p), n=k)
    rendered = [render_prompt(it, spec, tok) for it in items]
    t0 = time.perf_counter()
    groups = sampler.generate(rendered, k, params, int(seed))
    gen_stats = dict(getattr(sampler, "stats", {}) or {})
    if len(groups) != len(items) or any(len(g) != k for g in groups):
        raise RuntimeError(f"sampler returned {[len(g) for g in groups]} rollouts for {len(items)} candidates, expected {k}")
    cands: list[dict] = []
    with JsonlWriter(out_dir / ROLLOUTS_FILE, append=False) as rw:
        for ci, (it, text, group) in enumerate(zip(items, rendered, groups)):
            lsig = legacy.legacy_signals([r.text for r in group], it.legacy_solution)
            graded = grade_rollouts(group, it.answer, grader.mv)
            msig = compute_signals(graded, it.to_problem(), policy_tag, prompt_token_count(tok, text))
            ph = prompt_hash(text)
            for ri, (r, gr) in enumerate(zip(group, graded)):
                rw.write({
                    "cand_idx": ci, "unique_id": it.unique_id, "rollout_idx": ri, "policy_tag": policy_tag,
                    "prompt_hash": ph, "n_tokens": r.n_tokens, "finish_reason": r.finish_reason,
                    "truncated": r.truncated, "stop_id": r.stop_id, "chat_marker": r.chat_marker,
                    "legacy_answer": lsig["answers"][ri], "legacy_bin": lsig["bin"][ri], "legacy_fmt": lsig["fmt"][ri],
                    "mv_boxed": gr.boxed, "mv_correct": gr.correct, "mv_answer_class": gr.answer_class,
                    "text": r.text,
                })
            cands.append({
                "cand_idx": ci, "row": it.row, "unique_id": it.unique_id, "level": it.level, "subject": it.subject,
                "prompt_hash": ph, "selection_signals": selection_signals(spec, lsig, msig, it.level),
                "legacy": {k2: lsig[k2] for k2 in ("Ps", "Var", "D", "bin", "fmt", "answers")},
                "mv": msig.to_dict(),
            })
    wall = time.perf_counter() - t0
    atomic_write_json(out_dir / SIGNALS_FILE, {
        "policy_tag": policy_tag, "seed": int(seed), "k": k, "temperature": params.temperature, "top_p": params.top_p,
        "sampler": sampler.describe() if hasattr(sampler, "describe") else None, "gen_stats": gen_stats,
        "wall_s": wall, "candidates": cands,
    })
    return cands
