"""Deployment-matched selector features from the prereg K=32 base sieve (protocol section 3; no new sieve).

The curriculum sieves K=4 rollouts per candidate and scores them with `rlvr_v2.signals.grade_rollouts ->
compute_signals -> rlvr_v2.sgac.sieve.selection_signals` (e0: Ps = p_s, Var = v_legacy, D = u_ratio, L = level). Here
the same functions run on random K=4 subsamples of the 32 stored rollouts, regraded and re-clustered on the four alone,
so the training features are the deployed estimator by construction. Before that, two checks:
- the stored rollouts were generated from the prompt the SGAC e0 sieve renders (prompt hash per candidate), and
- regrading all 32 stored rollouts reproduces the stored sieve signals (same grader as at sieve time).
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from ..artifacts import atomic_write_json, read_json, read_jsonl, utc_now
from ..data import load_math_train
from ..grader import MathVerifyGrader
from ..prompts import prompt_hash
from ..sgac.data import SgacItem
from ..sgac.sieve import selection_signals
from ..signals import Rollout, compute_signals, grade_rollouts
from .manifests import CANDIDATES, sha256_file
from .spec import CalibSpec, substrate_spec

log = logging.getLogger(__name__)

FEATURES = "features.json"
CHECK_KEYS = ("p_s", "v_legacy", "u_ratio", "n_classes")


def _raw(recs: list[dict]) -> list[Rollout]:
    return [Rollout(text=r["text"], n_tokens=int(r["n_tokens"]), truncated=r["truncated"] in (True, "True"),
                    finish_reason=r.get("finish_reason", "stop")) for r in recs]


def deploy_signals(spec, recs: list[dict], problem, grader: MathVerifyGrader) -> dict:
    """The four selection signals exactly as the curriculum's sieve computes them from these rollouts."""
    graded = grade_rollouts(_raw(recs), problem.answer, grader)
    msig = compute_signals(graded, problem, "base")
    return selection_signals(spec, {}, msig, problem.level), msig


def build_features(cs: CalibSpec, rollouts_path=None, signals_path=None, tokenizer=None, grader=None) -> dict:
    spec = substrate_spec(cs, "calib")
    cands = read_json(cs.manifest_dir / CANDIDATES)
    if cands is None:
        raise FileNotFoundError(cs.manifest_dir / CANDIDATES)
    recs_all = cands["pilot"] + cands["expansion"]
    want = {r["unique_id"] for r in recs_all}
    rp = rollouts_path or cs.path(cs.data.sieve_rollouts)
    sp = signals_path or cs.path(cs.data.sieve_signals)
    if sha256_file(sp) != cands["sieve_signals_sha256"]:
        raise RuntimeError(f"{sp} is not the sieve the candidates were drawn from")
    stored = {r["unique_id"]: r for r in read_jsonl(sp) if r["unique_id"] in want}
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(rp):
        if r["unique_id"] in want:
            by_uid[r["unique_id"]].append(r)
    problems = {p.unique_id: p for p in load_math_train() if p.unique_id in want}
    grader = grader or MathVerifyGrader()
    if tokenizer is None:
        from ..sgac.model_io import load_tokenizer

        tokenizer = load_tokenizer(spec)
    from ..sgac.model_io import render_prompt

    k, m = int(cs.features.k_deploy), int(cs.features.n_subsamples)
    out = {"created": utc_now(), "sieve_rollouts_sha256": sha256_file(rp), "sieve_signals_sha256": sha256_file(sp),
           "k_deploy": k, "n_subsamples": m, "subsample_seed": int(cs.features.subsample_seed),
           "substrate_spec_hash": spec.spec_hash(),
           "definition": "Ps = p_s, Var = v_legacy (pvar of correct + 0.5*boxed), D = u_ratio (math-verify classes / K), "
                         "L = level; computed by rlvr_v2.sgac.sieve.selection_signals on regraded rollouts",
           "candidates": []}
    for c in recs_all:
        uid, idx = c["unique_id"], int(c["index"])
        recs = sorted(by_uid[uid], key=lambda r: int(r["rollout_idx"]))
        if len(recs) != 32:
            raise RuntimeError(f"{uid}: {len(recs)} stored rollouts, expected 32")
        p = problems[uid]
        item = SgacItem(row=-1, orig_index=-1, unique_id=uid, problem=p.problem, solution=p.solution, answer=p.answer,
                        level=p.level, subject=p.subject, source="prereg_pool")
        ph = prompt_hash(render_prompt(item, spec, tokenizer))
        stored_ph = {r["prompt_hash"] for r in recs}
        if stored_ph != {ph}:
            raise RuntimeError(f"{uid}: stored prompt hash {stored_ph} != SGAC e0 sieve prompt {ph}")
        sig32, msig32 = deploy_signals(spec, recs, p, grader)
        mismatch = {kk: (getattr(msig32, kk), stored[uid][kk]) for kk in CHECK_KEYS
                    if abs(float(getattr(msig32, kk)) - float(stored[uid][kk])) > 1e-9}
        if mismatch:
            raise RuntimeError(f"{uid}: regrading the stored rollouts does not reproduce the sieve signals: {mismatch}")
        rng = np.random.default_rng([int(cs.features.subsample_seed), idx])
        subs, k4 = [], []
        for _ in range(m):
            sel = sorted(int(i) for i in rng.choice(len(recs), size=k, replace=False))
            sig, _ = deploy_signals(spec, [recs[i] for i in sel], p, grader)
            subs.append(sel)
            k4.append(sig)
        out["candidates"].append({"unique_id": uid, "index": idx, "bin": c["bin"], "level": p.level, "prompt_hash": ph,
                                  "k32": sig32, "k4": k4, "subsample_rollout_idx": subs})
        log.info("features %2d %s: K32 %s", idx, uid, sig32)
    atomic_write_json(cs.manifest_dir / FEATURES, out)
    return out


def load_features(cs: CalibSpec) -> dict:
    doc = read_json(cs.manifest_dir / FEATURES)
    if doc is None:
        raise FileNotFoundError(f"{cs.manifest_dir / FEATURES}; run `python -m rlvr_v2.sgac_calib build-features`")
    return doc
