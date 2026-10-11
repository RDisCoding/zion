"""Stage E confirmatory analysis (protocol section 7): H1 learned - random, H2 learned - sgac (Eq. 10), MATH-500 at
step 20, seeds paired across arms (shared batches).

Primary: the crossed seed x item test (`crossed.crossed_test`), Holm across H1 and H2. Per comparison:
- "beats X": Holm-adjusted p < alpha AND mean >= +min_effect_pp AND a positive per-seed difference in at least
  ceil(seed_consistency * S) seeds;
- "worse than X": the mirror image;
- "equivalent within ±equivalence_pp": the 90% CI lies inside [-equivalence_pp, +equivalence_pp] (TOST);
- otherwise "inconclusive". "Outperforms both" needs "beats" on H1 and H2.
Sensitivity: pigeonhole and item bootstraps (`crossed`), reported, never decisive.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..artifacts import read_jsonl, utc_now
from ..stats import holm
from .crossed import crossed_test, item_bootstrap, pigeonhole_bootstrap
from .spec import CalibSpec

CONFIRMATORY = {"H1": ("learned", "random"), "H2": ("learned", "sgac")}


def per_item(path: Path, ids: list[str]) -> np.ndarray:
    rows = {r["unique_id"]: bool(r["correct"]) for r in read_jsonl(path)}
    if set(rows) != set(ids):
        raise RuntimeError(f"{path} does not cover the evaluation set")
    return np.array([rows[i] for i in ids], dtype=float)


def step_path(group_dir: Path, arm: str, seed: int, step: int, name: str = "math500") -> Path:
    return group_dir / f"{arm}__seed{seed}" / "evals" / f"step-{step:03d}" / name / "per_item.jsonl"


def load_arm(group_dir: Path, arm: str, seeds: list[int], ids: list[str], step: int = 20, name: str = "math500"):
    """{seed: correctness vector} for the seeds whose step-`step` evaluation exists."""
    out = {}
    for s in seeds:
        p = step_path(group_dir, arm, s, step, name)
        if p.exists():
            out[s] = per_item(p, ids)
    return out


def decide(test: dict, holm_p: float, per_seed_pp: list[float], cs: CalibSpec) -> str:
    a = cs.analysis
    S = len(per_seed_pp)
    need = math.ceil(a.seed_consistency * S - 1e-9)
    mean_pp = 100 * test["mean"]
    if holm_p < a.alpha and mean_pp >= a.min_effect_pp and sum(d > 0 for d in per_seed_pp) >= need:
        return "beats"
    if holm_p < a.alpha and mean_pp <= -a.min_effect_pp and sum(d < 0 for d in per_seed_pp) >= need:
        return "worse"
    lo, hi = (100 * test["ci90"][0], 100 * test["ci90"][1])
    if -a.equivalence_pp <= lo and hi <= a.equivalence_pp:
        return "equivalent"
    return "inconclusive"


def confirmatory(cs: CalibSpec, arms: dict[str, dict[int, np.ndarray]]) -> dict:
    """`arms[arm][seed]` = per-item correctness at step 20. Uses, per comparison, the seeds both arms completed."""
    a = cs.analysis
    tests, per_seed = {}, {}
    for h, (x, y) in CONFIRMATORY.items():
        seeds = sorted(set(arms.get(x, {})) & set(arms.get(y, {})))
        if len(seeds) < 2:
            tests[h] = None
            continue
        e = np.stack([arms[x][s] - arms[y][s] for s in seeds])
        t = crossed_test(e, a.alpha)
        t["seeds"] = seeds
        t["sensitivity"] = {"pigeonhole": pigeonhole_bootstrap(e, a.n_boot, a.boot_seed, a.alpha),
                            "item_bootstrap": item_bootstrap(e, a.n_boot, a.boot_seed, a.alpha)}
        tests[h] = t
        per_seed[h] = [100 * float(v) for v in e.mean(axis=1)]
    adj = holm({h: t["p"] for h, t in tests.items() if t is not None})
    out = {"created": utc_now(), "endpoint": "MATH-500 accuracy at step 20", "units": "pp", "results": {}}
    for h, t in tests.items():
        if t is None:
            out["results"][h] = {"status": "not enough paired seeds"}
            continue
        verdict = decide(t, adj[h], per_seed[h], cs)
        out["results"][h] = {"comparison": f"{CONFIRMATORY[h][0]} - {CONFIRMATORY[h][1]}", "seeds": t["seeds"],
                             "mean_pp": 100 * t["mean"], "se_pp": 100 * t["se"], "df": t["df"], "p": t["p"],
                             "p_holm": adj[h], "ci95_pp": [100 * v for v in t["ci"]],
                             "ci90_pp": [100 * v for v in t["ci90"]], "per_seed_pp": per_seed[h],
                             "components": t["components"], "verdict": verdict,
                             "sensitivity_ci95_pp": {k: [100 * v for v in s["ci"]] for k, s in t["sensitivity"].items()}}
    v = [out["results"].get(h, {}).get("verdict") for h in CONFIRMATORY]
    out["outperforms_both"] = all(x == "beats" for x in v)
    out["rule"] = {"alpha": a.alpha, "min_effect_pp": a.min_effect_pp, "seed_consistency": a.seed_consistency,
                   "equivalence_pp": a.equivalence_pp, "multiplicity": "Holm over H1, H2"}
    return out
