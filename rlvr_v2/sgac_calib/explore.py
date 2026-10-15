"""Exploratory selector fits (protocol section 13, amendments A1-A3): the frozen Stage-D pipeline applied to whatever
labels exist, clearly separated from the confirmatory track.

The confirmatory track stopped at gate G-P (C1 pilot). Everything here is EXPLORATORY: the same features, models,
cross-validation, permutation test and gate thresholds as Stage D (`selector.run_stage_d`), applied to
- E1: the 16 C1 pilot labels (unchanged data; N=16);
- E2: C2 labels (the protocol's pre-declared escalation rung, 80 bursts) when they exist;
- E3: one-shot-regime labels (prereg Study-1 training recipe) when they exist.
Each fit writes `results_sgac_calib/explore/<tag>/{stage_d.json, equation.json}`. Thresholds are never changed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..artifacts import atomic_write_json, read_json, utc_now
from .manifests import sieve_signals
from .selector import Table, build_table, freeze_selector, pool_bin_frequencies, run_stage_d
from .spec import CalibSpec

log = logging.getLogger(__name__)

EXPLORE = "explore"


def labeled_table(features_doc: dict, labels_doc: dict, feature_names, min_seeds: int) -> Table:
    """Stage-D table restricted to candidates whose label has at least `min_seeds` finished seeds."""
    done = {c["unique_id"] for c in labels_doc["candidates"]
            if c.get("delta_pp_mean") is not None and int(c.get("n_seeds_done", 0)) >= min_seeds}
    sub = {**features_doc, "candidates": [c for c in features_doc["candidates"] if c["unique_id"] in done]}
    if len(sub["candidates"]) < 8:
        raise RuntimeError(f"only {len(sub['candidates'])} labelled candidates; need >= 8 to cross-validate")
    return build_table(sub, labels_doc, feature_names)


def equation_from(stage_d: dict, t: Table) -> dict:
    """The fitted linear equation(s) on raw features, as reported by Stage D (all candidates, final alpha)."""
    co = stage_d["coefficients"]
    out = {}
    for name in ("ridge", "ols"):
        c = co[name]
        out[name] = {"intercept": c["intercept_raw"], "coef": c["coef_raw"], "coef_ci95": c["coef_raw_ci95"],
                     "coef_std": c["coef_std"], "sign_agreement": c["sign_agreement"], "alpha": c["alpha"]}
    feats = list(t.features)
    r = out["ridge"]
    terms = " ".join(f"{'+' if r['coef'][f] >= 0 else '-'} {abs(r['coef'][f]):.6g}*{f}" for f in feats)
    out["ridge_text"] = f"score = {r['intercept']:.6g} {terms}   (predicted dev ΔA, pp)"
    return out


def run_explore_fit(cs: CalibSpec, labels_doc: dict, tag: str, min_seeds: int, with_permutation: bool = True,
                    freeze: bool = False) -> dict:
    feats = read_json(cs.manifest_dir / "features.json")
    t = labeled_table(feats, labels_doc, cs.selector.features, min_seeds)
    freq = np.asarray(pool_bin_frequencies(cs, sieve_signals(cs)))
    res = run_stage_d(cs, t, freq, with_permutation=with_permutation)
    res["exploratory"] = True
    res["tag"] = tag
    res["candidates"] = t.uids
    res["labels_source"] = labels_doc.get("source", tag)
    out_dir = cs.results_dir / EXPLORE / tag
    atomic_write_json(out_dir / "stage_d.json", res)
    eq = {"created": utc_now(), "tag": tag, "exploratory": True, "n_candidates": t.n,
          "label_mean_pp": float(t.y.mean()), "label_sd_pp": float(t.y.std(ddof=1)), **equation_from(res, t),
          "out_of_fold": {m: {k: res["models"][m][k] for k in ("rho", "rho_avg", "r2_oos", "gain_pp", "oracle_pp")}
                          | {"ci95": res["models"][m]["ci95"]} for m in res["models"]},
          "rules": res["rules"], "gate_s_thresholds_applied": res["gate_s"], "model_choice": res["model_choice"],
          "permutation": res.get("permutation")}
    atomic_write_json(out_dir / "equation.json", eq)
    if freeze:
        # Amendment A4 (pre-declared): an exploratory selector is frozen and taken end to end WHATEVER the G-S
        # thresholds say; whether they were met is recorded next to it and travels into the curriculum analysis.
        doc = freeze_selector(cs, t, res, out_dir)
        eq["frozen"] = {"model": doc["model"], "training_table_sha256": doc["training_table_sha256"],
                        "sha256": doc["sha256"], "validated_by_gate_s": bool(res["gate_s"]["passed"])}
        atomic_write_json(out_dir / "equation.json", eq)
    return eq


def summary_line(eq: dict) -> str:
    r = eq["out_of_fold"]["ridge"]
    return json.dumps({"tag": eq["tag"], "n": eq["n_candidates"], "equation": eq["ridge_text"],
                       "oof_rho": round(r["rho"], 3), "oof_r2": round(r["r2_oos"], 3), "top1_gain_pp": round(r["gain_pp"], 3),
                       "perm_p": (eq.get("permutation") or {}).get("p_one_sided"),
                       "gate_s_passed": eq["gate_s_thresholds_applied"]["passed"]})


def load_labels(path: str | Path) -> dict:
    doc = read_json(path)
    if doc is None:
        raise FileNotFoundError(path)
    return doc
