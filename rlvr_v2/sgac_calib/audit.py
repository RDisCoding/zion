"""Stage A (protocol section 1): the original Phase-1 candidates' signals recomputed with the E0 pipeline.

The four candidates are the historical shuffled rows 0-3 (`configs/sgac/manifests/data.json`, section
phase1_candidates). This job sieves them with the e0 substrate at K=32 (3072 tokens, math-verify) through the frozen
`rlvr_v2.sgac.sieve.sieve_candidates`, which records both the E0 and the legacy signal variants. The historical-settings
recomputation (K=8, 2048 tokens, legacy prompt and grader) and the downstream replication are the frozen
`python -m rlvr_v2.sgac phase1 --profile as_run`, run with its results root redirected into this track.
"""
from __future__ import annotations

from ..artifacts import atomic_write_json, read_json, utc_now
from ..sgac import data as sdata
from ..sgac import nbm_original as nbm
from .spec import CalibSpec, substrate_spec

AUDIT_K = 32


def run_phase1_signals(cs: CalibSpec) -> dict:
    from ..modeling import free_cuda
    from ..sgac.grading import DualGrader
    from ..sgac.model_io import load_policy_model, load_tokenizer, stop_token_ids
    from ..sgac.sampler import sieve_sampler
    from ..sgac.sieve import sieve_candidates
    from ..sgac.spec import group_dir

    spec = substrate_spec(cs, "audit", [f"loop.k={AUDIT_K}"])
    out = group_dir(spec) / "phase1_e0_signals"
    done = read_json(out / "signals.json")
    if done is None:
        items = sdata.load_section(spec, "phase1_candidates")
        tok = load_tokenizer(spec)
        model, _ = load_policy_model(spec)
        stops = stop_token_ids(spec, tok)
        sieve_candidates(sieve_sampler(model, tok, spec, stops), tok, items, spec, DualGrader(spec.profile.grader),
                         seed=20261012, out_dir=out, policy_tag="phase1-base-e0-k32")
        del model
        free_cuda()
        done = read_json(out / "signals.json")
    summary = {"created": utc_now(), "spec_hash": spec.spec_hash(), "k": AUDIT_K,
               "candidates": [{"unique_id": c["unique_id"], "level": c["level"], "e0": c["selection_signals"],
                               "legacy_variant": {k: c["legacy"][k] for k in ("Ps", "Var", "D")}}
                              for c in done["candidates"]],
               "paper_table1": list(nbm.PHASE1_TABLE1)}
    atomic_write_json(out / "summary.json", summary)
    return summary
