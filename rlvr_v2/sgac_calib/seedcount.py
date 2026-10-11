"""Seed count for the confirmatory curriculum arms, fixed before the protocol freeze from historical data only.

Inputs are the frozen reproduction's step-20 MATH-500 per-item files (`<arm>_seed<S>.jsonl`, copies of
`results_sgac/e0/577f3319/<arm>__seed<S>/evals/step-020/math500/per_item.jsonl`; hashes in
`paper/sgac_repro_freeze.md`). No outcome of the calibration track exists when this runs.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from ..artifacts import read_jsonl
from .crossed import seed_count
from .spec import CalibSpec

FILE_RE = re.compile(r"^(?P<arm>[a-z_0-9]+)_seed(?P<seed>\d+)\.jsonl$")


def load_step20(dirs: Iterable[str | Path], item_ids: Sequence[str]) -> tuple[dict[str, np.ndarray], list[int], dict]:
    """{arm: (seeds, items) correctness} for arms present at every seed, in `item_ids` order; plus provenance."""
    found: dict[tuple[str, int], Path] = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*.jsonl")):
            m = FILE_RE.match(p.name)
            if m:
                found[(m["arm"], int(m["seed"]))] = p
    arms = sorted({a for a, _ in found})
    seeds = sorted({s for _, s in found})
    complete = [a for a in arms if all((a, s) in found for s in seeds)]
    if len(complete) < 2 or len(seeds) < 2:
        raise ValueError(f"need >= 2 arms complete over >= 2 seeds; found arms {arms}, seeds {seeds}")
    Y: dict[str, np.ndarray] = {}
    prov: dict = {"files": {}}
    for a in complete:
        rows = []
        for s in seeds:
            p = found[(a, s)]
            recs = {r["unique_id"]: r for r in read_jsonl(p)}
            if set(recs) != set(item_ids):
                raise ValueError(f"{p} does not cover exactly the {len(item_ids)} evaluation items")
            y = np.array([1.0 if recs[i]["correct"] in (True, "True") else 0.0 for i in item_ids])
            rows.append(y)
            prov["files"][p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "acc": float(y.mean())}
        Y[a] = np.stack(rows)
    prov.update(arms=complete, seeds=seeds, n_items=len(item_ids), skipped_arms=sorted(set(arms) - set(complete)))
    return Y, seeds, prov


def run_seed_count(cs: CalibSpec, dirs: Iterable[str | Path], item_ids: Sequence[str]) -> dict:
    Y, _, prov = load_step20(dirs, item_ids)
    a = cs.analysis
    out = seed_count(Y, cs.curriculum.seeds_min, cs.curriculum.seeds_max, a.mde_alpha, a.mde_power,
                     a.mde_max_pp / 100.0)
    out["provenance"] = prov
    out["units"] = "proportions (multiply by 100 for pp)"
    return out
