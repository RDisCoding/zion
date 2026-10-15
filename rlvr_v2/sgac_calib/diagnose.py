"""Why a calibration rung's labels are (un)reliable: run-level and item-level diagnostics (EXPLORATORY).

For every finished run of a rung, the per-item dev outcome is compared with the C1 base reference:
- ΔA, flips (items whose verdict changed), gains and losses;
- effective updates (SGAC rungs: updates with a non-zero gradient, i.e. from groups whose rewards were not all equal);
- problem specificity: if training on problem x changes the model in an x-specific way, two seeds of the same problem
  should change the same items more alike than runs on different problems. Compared via the per-item Δ correlation
  and the Jaccard overlap of flipped items, within vs between candidates (with a permutation p-value);
- fragile items: items that flip in many runs regardless of the problem trained on;
- the variance decomposition of ΔA (between-candidate σ_b vs seed noise σ_w) and the best top-1-of-4 gain a perfect
  selector could reach given σ_b (E[max of 4 iid normals] = 1.029 σ_b).
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

from ..artifacts import read_json, read_jsonl
from ..stats import noise_decomposition

E_MAX4 = 1.0293753  # E[max of 4 iid standard normals]


def _corr(x, y) -> float | None:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def candidate_key(run: str) -> str:
    return run.split("__seed")[0]


def _per_item(path: Path, ids: list[str]) -> np.ndarray:
    rows = {r["unique_id"]: bool(r["correct"]) for r in read_jsonl(path)}
    return np.array([rows[i] for i in ids], dtype=float)


def sgac_runs(group: Path) -> dict[str, dict]:
    """{run_name: {"per_item": path, "eff_updates": int, "groups": int, "nonzero_groups": int}} for C1/C2 rungs."""
    out = {}
    for rd in sorted(p for p in group.iterdir() if p.is_dir() and not p.name.startswith("_")):
        st = read_json(rd / "state.json")
        evals = sorted((rd / "evals").glob("step-*/dev/per_item.jsonl")) if (rd / "evals").exists() else []
        if st is None or not evals:
            continue
        eff = groups = nz = 0
        for h in st.get("history", []):
            b = h.get("burst") or {}
            eff += sum(1 for g in (b.get("grad_norms") or []) if g and g > 0)
            fz = b.get("frac_reward_zero_std") or []
            groups += len(fz)
            nz += sum(1 for f in fz if f < 1.0)
        out[rd.name] = {"per_item": evals[-1], "eff_updates": eff, "groups": groups, "nonzero_groups": nz}
    return out


def os_runs(group: Path) -> dict[str, dict]:
    out = {}
    for rd in sorted(p for p in group.iterdir() if p.is_dir()):
        pi = rd / "eval" / "dev" / "per_item.jsonl"
        res = read_json(rd / "result.json")
        if pi.exists() and res is not None:
            burst = res.get("burst") or {}
            out[rd.name] = {"per_item": pi, "eff_updates": None, "frac_zero_std": burst.get("frac_zero_std_mean"),
                            "reward_first_last": [burst.get("reward_correct_mean_first"),
                                                  burst.get("reward_correct_mean_last")]}
    return out


def diagnose(runs: dict[str, dict], base: dict[str, bool], n_perm: int = 2000, seed: int = 0) -> dict:
    ids = sorted(base)
    b = np.array([base[i] for i in ids], dtype=float)
    D = {k: _per_item(v["per_item"], ids) - b for k, v in runs.items()}
    table = []
    for k, d in D.items():
        table.append({"run": k, "candidate": candidate_key(k), "delta_pp": 100 * float(d.mean()),
                      "flips": int((d != 0).sum()), "gains": int((d > 0).sum()), "losses": int((d < 0).sum()),
                      **{kk: v for kk, v in runs[k].items() if kk != "per_item"}})
    changed = [k for k in D if D[k].any()]
    names = sorted(changed)
    cands = [candidate_key(k) for k in names]

    def pair_stats(labels):
        w, bt = [], []
        for (i, a), (j, c) in itertools.combinations(enumerate(names), 2):
            r = _corr(D[a], D[c])
            if r is None:
                continue
            fa, fc = D[a] != 0, D[c] != 0
            jac = float((fa & fc).sum() / max(1, (fa | fc).sum()))
            (w if labels[i] == labels[j] else bt).append((r, jac))
        return np.array(w), np.array(bt)

    w, bt = pair_stats(cands)
    spec = {"within_pairs": len(w), "between_pairs": len(bt)}
    if len(w) and len(bt):
        obs = w[:, 0].mean() - bt[:, 0].mean()
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(n_perm):
            perm = list(rng.permutation(cands))
            pw, pb = pair_stats(perm)
            if len(pw) and len(pb):
                null.append(pw[:, 0].mean() - pb[:, 0].mean())
        spec.update(corr_within=float(w[:, 0].mean()), corr_between=float(bt[:, 0].mean()),
                    jaccard_within=float(w[:, 1].mean()), jaccard_between=float(bt[:, 1].mean()),
                    corr_excess=float(obs), perm_p_one_sided=float((1 + np.sum(np.array(null) >= obs)) / (1 + len(null))),
                    n_perm=len(null))
    rate = np.mean([D[k] != 0 for k in changed], axis=0) if changed else np.zeros(len(ids))
    groups: dict[str, list[float]] = {}
    for r in table:
        groups.setdefault(r["candidate"], []).append(r["delta_pp"])
    nd = noise_decomposition({k: v for k, v in groups.items() if not k.startswith(("pi1", "os_pi1"))})
    out = {"n_runs": len(D), "n_runs_changed": len(changed), "n_items": len(ids), "table": table,
           "problem_specificity": spec,
           "fragile_items": {"flip_rate_gt_0.5": int((rate > 0.5).sum()), "flip_rate_gt_0.25": int((rate > 0.25).sum()),
                             "ever": int((rate > 0).sum())},
           "mean_delta_pp_changed_runs": float(np.mean([100 * D[k].mean() for k in changed])) if changed else None,
           "variance": nd}
    sb = nd.get("sigma_between")
    if sb is not None and math.isfinite(sb):
        out["oracle_top1_of_4_gain_pp"] = E_MAX4 * sb
    effs = [r for r in table if r.get("eff_updates") is not None and r["flips"] > 0]
    if effs:
        hi = [r["delta_pp"] for r in effs if r["eff_updates"] >= 50]
        lo = [r["delta_pp"] for r in effs if r["eff_updates"] < 50]
        out["dose"] = {"corr_eff_updates_abs_delta": _corr([r["eff_updates"] for r in effs],
                                                           [abs(r["delta_pp"]) for r in effs]),
                       "corr_eff_updates_flips": _corr([r["eff_updates"] for r in effs], [r["flips"] for r in effs]),
                       "mean_delta_pp_eff_ge_50": float(np.mean(hi)) if hi else None, "n_ge_50": len(hi),
                       "mean_delta_pp_eff_lt_50": float(np.mean(lo)) if lo else None, "n_lt_50": len(lo),
                       "runs_with_zero_updates": sum(1 for r in table if r.get("eff_updates") == 0)}
    return out
