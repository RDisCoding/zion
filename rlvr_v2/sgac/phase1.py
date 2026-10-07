"""Optional Phase-1 replication (tier 6): the old paper's signal-discovery experiment (its Table 1 and Table 6).

Mirrors `ghoul/rl-llm-phase-4-selector-experiments-results.ipynb`:
- candidates = shuffled rows 0-3 (the paper calls them hand-picked), test = rows 4-13 (10 items);
- signals on the BASE model: K=8 samples of the raw NB-M instruction, 2048 new tokens, T=1.0 (top_k 50 in as_run);
- per candidate: a fresh LoRA (r16/a32 q/k/v/o) on the base and ONE 20-update GRPO run (the burst's pinned config with
  max_steps=20, as the notebook's `max_steps=20`), then greedy evaluation at 2048 tokens on the 10 test items with that
  notebook's own prompt, plus MATH-500 with the profile's protocol for statistical power; the LoRA is unloaded before
  the next candidate (`base_model = model_with_lora.unload()`).
- strategies over the four candidates: "random" = candidate #0 (the notebook hardcoded `rand_idx = 0`), max variance,
  max disagreement, max level (ties reported), Eq. 10 as run, Table 2 as printed, the pickle's true mapping.
- the 4-row linear fit is printed under both label conventions only to show it is not identified; it is never used.
Results: results_sgac/<profile>/<spec_hash>/phase1/ (resumable per candidate) and phase1_summary.json.
"""
from __future__ import annotations

import dataclasses
import logging

import numpy as np

from ..artifacts import atomic_write_json, read_json
from . import data as sdata
from . import nbm_original as nbm
from . import selection as sel
from .evaluation import evaluate_items
from .grading import DualGrader
from .model_io import attach_lora, load_policy_model, load_tokenizer, stop_token_ids
from .sampler import SgacSampler, eval_sampler, sieve_sampler
from .sieve import sieve_candidates
from .spec import SgacSpec, group_dir

log = logging.getLogger(__name__)

# phase-4 notebook, evaluate_model: f"Solve step by step and give final answer in \\boxed{{}}:\n\n{q}\n\nSolution:\n"
PHASE1_EVAL_PROMPT = "Solve step by step and give final answer in \\boxed{{}}:\n\n{q}\n\nSolution:\n"
PHASE1_K, PHASE1_TOKENS, PHASE1_GRPO_STEPS = 8, 2048, 20


def phase1_spec(spec: SgacSpec) -> SgacSpec:
    return dataclasses.replace(
        spec, loop=dataclasses.replace(spec.loop, k=PHASE1_K),
        profile=dataclasses.replace(spec.profile, sieve_max_new_tokens=PHASE1_TOKENS),
        grpo=dataclasses.replace(spec.grpo, max_steps=PHASE1_GRPO_STEPS))


def strategies(signals: list[dict], acc: list[float]) -> dict:
    out = {"random_as_notebook": {"idx": 0, "acc": acc[0], "note": "the notebook hardcoded rand_idx = 0"},
           "random_expected": {"acc": float(np.mean(acc))}}
    for rule in ("max_var", "max_d", "max_level", "sgac_eq10", "table2_as_printed", "pickle_true_mapping"):
        scores = [sel.score(rule, s) for s in signals]
        best = max(scores)
        ties = [i for i, s in enumerate(scores) if s == best]
        idx = sel.first_max(scores)
        out[rule] = {"idx": idx, "ties": ties, "acc": acc[idx], "acc_over_ties": float(np.mean([acc[i] for i in ties]))}
    return out


def four_row_fit(signals: list[dict], acc: list[float]) -> dict:
    """The phase-4 fit (X = [L, Ps, Var, D], 4 rows, intercept): exact, rank-deficient, signs not identified."""
    from sklearn.linear_model import LinearRegression

    X = np.array([[s["L"], s["Ps"], s["Var"], s["D"]] for s in signals], dtype=float)
    reg = LinearRegression().fit(X, np.array(acc, dtype=float))
    coef = [float(c) for c in reg.coef_]
    return {"true_mapping": dict(zip(("L", "Ps", "Var", "D"), coef)),
            "as_printed_by_notebook": dict(zip(("Ps", "Var", "D", "L"), coef)),
            "intercept": float(reg.intercept_), "rank": int(np.linalg.matrix_rank(np.c_[X, np.ones(len(X))])),
            "n": len(acc), "note": "4 rows, 5 parameters: infinitely many exact fits; never used for selection"}


def run_phase1(spec: SgacSpec) -> dict:
    spec1 = phase1_spec(spec)
    out = group_dir(spec) / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    seed = int(spec.run.seeds[0])
    manifest = sdata.load_data_manifest(spec)
    ds = sdata.load_split(spec)
    cands = sdata.load_section(spec, "phase1_candidates", ds, manifest)
    test10 = sdata.load_section(spec, "phase1_test", ds, manifest)
    math500 = sdata.load_math500_items(spec) if spec.loop.math500_steps else []
    tok = load_tokenizer(spec1)
    base, load_record = load_policy_model(spec1)
    stops = stop_token_ids(spec1, tok)
    grader = DualGrader(spec1.profile.grader)
    render = lambda it: PHASE1_EVAL_PROMPT.format(q=it.problem)

    def p1_sampler(model):
        return SgacSampler(model, tok, max_new_tokens=PHASE1_TOKENS, batch_size=spec1.profile.eval_batch_size,
                           stop_token_ids=stops, top_k=0, max_prompt_tokens=spec1.profile.max_prompt_tokens)

    # 1) signals with the base model
    sig_file = out / "signals" / "signals.json"
    if read_json(sig_file) is None:
        sieve_candidates(sieve_sampler(base, tok, spec1, stops), tok, cands, spec1, grader, seed=seed * 1000,
                         out_dir=out / "signals", policy_tag="phase1-base")
    measured = read_json(sig_file)["candidates"]
    signals = [c["selection_signals"] for c in measured]
    # base model reference on the 10 items (phase-1 prompt)
    base_eval = evaluate_items(p1_sampler(base), tok, test10, spec1, grader, out / "base" / "test10", "p1-base-test10",
                               "base", render=render, prompt_name="phase1_eval")
    results = []
    for i, item in enumerate(cands):
        cdir = out / f"cand{i}"
        done = read_json(cdir / "done.json")
        if done is not None:
            results.append(done)
            continue
        from .burst import run_burst

        model = attach_lora(base, spec1, seed)
        burst = run_burst(model, tok, item, spec1, grader, cdir / "train", seed=seed,
                          context={"phase1_candidate": i}, stop_ids=stops)
        ev10 = evaluate_items(p1_sampler(model), tok, test10, spec1, grader, cdir / "test10", f"p1-cand{i}-test10",
                              f"cand{i}", render=render, prompt_name="phase1_eval")
        ev500 = (evaluate_items(eval_sampler(model, tok, spec1, stops), tok, math500, spec1, grader, cdir / "math500",
                                f"p1-cand{i}-math500", f"cand{i}") if math500 else None)
        base = model.unload()  # back to the pure base model for the next candidate, like the notebook
        rec = {"idx": i, "unique_id": item.unique_id, "signals": signals[i], "acc_test10": ev10["acc"],
               "acc_test10_mv": ev10["acc_mv"], "acc_test10_legacy": ev10["acc_legacy"],
               "acc_math500": None if ev500 is None else ev500["acc"],
               "burst": {k: burst.get(k) for k in ("losses", "loss_trl100", "reward", "frac_reward_zero_std",
                                                    "clipped_ratio", "delta_A", "delta_B", "wall_s")}}
        atomic_write_json(cdir / "done.json", rec)
        results.append(rec)
    acc10 = [r["acc_test10"] for r in results]
    summary = {
        "load_record": load_record, "spec_hash_phase1": spec1.spec_hash(), "seed": seed,
        "base_test10": base_eval["acc"], "candidates": results, "paper_table1": list(nbm.PHASE1_TABLE1),
        "strategies_test10": strategies(signals, acc10),
        "strategies_math500": strategies(signals, [r["acc_math500"] for r in results]) if math500 else None,
        "four_row_fit_test10": four_row_fit(signals, acc10),
    }
    atomic_write_json(out / "phase1_summary.json", summary)
    log.info("phase 1 done: %s", {r["unique_id"]: (r["acc_test10"], r["acc_math500"]) for r in results})
    return summary
