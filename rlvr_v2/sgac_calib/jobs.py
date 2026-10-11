"""Pre-declared job queues per stage (protocol sections 4-6 and 9). Never reordered on results.

Each line is the argument list for `python -m rlvr_v2.sgac_calib` (or, for the frozen Phase-1 replication,
`python -m rlvr_v2.sgac` with its results root redirected into this track). A stage starts only when the previous
gate has been decided; `scripts/local/sgac_calib.sh` enforces the gate files.
"""
from __future__ import annotations

from ..artifacts import read_json
from .manifests import CANDIDATES
from .spec import CalibSpec

STAGES = ("t0", "audit", "pilot", "expansion", "selector", "curriculum", "tm")


def _calib_runs(cs: CalibSpec, recs: list[dict]) -> list[str]:
    from .calib import training_seeds

    return [f"calib-run --index {r['index']} --seed {s}" for r in recs for s in training_seeds(cs, int(r["index"]))]


def confirmatory_seeds(cs: CalibSpec) -> list[int]:
    doc = read_json(cs.manifest_dir / "seed_count.json")
    if doc is None:
        raise FileNotFoundError("seed_count.json missing; run `python -m rlvr_v2.sgac_calib seed-count` before freezing")
    return list(range(cs.curriculum.first_seed, cs.curriculum.first_seed + int(doc["chosen_seeds"])))


def stage_jobs(cs: CalibSpec, stage: str) -> list[str]:
    cands = read_json(cs.manifest_dir / CANDIDATES) or {"pilot": [], "expansion": []}
    if stage == "t0":
        return ["show-spec", "build-manifests --check", "check-features", "smoke --gpu"]
    if stage == "audit":
        return ["audit-phase1-signals",
                "SGAC phase1 --profile as_run --override run.results_root=results_sgac_calib/audit"]
    if stage == "pilot":
        from .calib import training_seeds

        jobs = ["base-dev-eval"] + _calib_runs(cs, cands["pilot"])
        jobs += [f"calib-run --pi1 --seed {s}" for s in training_seeds(cs, None)]
        return jobs + ["gate-p"]
    if stage == "expansion":
        return ["require-gate-p"] + _calib_runs(cs, cands["expansion"]) + ["labels"]
    if stage == "selector":
        return ["stage-d", "freeze-selector"]
    if stage == "curriculum":
        conf = confirmatory_seeds(cs)
        sec = list(cs.curriculum.secondary_seeds)
        arms_first = list(cs.curriculum.confirmatory_arms) + list(cs.curriculum.secondary_arms)
        jobs = ["require-gate-s", "curriculum-base-eval"]
        jobs += [f"curriculum-run --arm {a} --seed {s}" for s in sec for a in arms_first]
        jobs += [f"curriculum-run --arm {a} --seed {s}" for s in conf if s not in sec
                 for a in cs.curriculum.confirmatory_arms]
        return jobs + ["analyze"]
    if stage == "tm":
        return [f"curriculum-run --arm random_tm --seed {s}" for s in cs.curriculum.secondary_seeds]
    raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")

