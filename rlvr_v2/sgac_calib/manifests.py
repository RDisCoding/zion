"""Data roles, the outcome-blind candidate draw and the clean-pool curriculum schedules (protocol sections 2, 4, 6).

Four pairwise-disjoint roles, by exact id and by near-duplicate (difflib ratio >= `near_dup_ratio` after a token-Jaccard
prefilter, `rlvr_v2.data.near_duplicates`):
- final evaluation: MATH-500 + the historical test50 (never used for labels, model selection or gates);
- dev set: prereg held-out minus anything matching a final-evaluation item;
- calibration candidates: prereg pool minus ineligible golds, matches of final-evaluation or dev items, the whole
  historical SGAC pool (exact and near), and the prereg Study-1 candidates;
- curriculum pool: the historical SGAC pool minus matches of final-evaluation items (rows kept, so schedules use the
  frozen loop unchanged).
Everything is written to `configs/sgac_calib/manifests/` and committed; `--check` rebuilds and compares.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..artifacts import atomic_write_json, read_json, read_jsonl, utc_now
from ..data import Manifest, Problem, load_math500, load_math_train, near_duplicates, select_by_manifest
from ..sgac.schedule import schedule_path, write_schedule
from .spec import CalibSpec

log = logging.getLogger(__name__)

DATA_ROLES = "data_roles.json"
CANDIDATES = "candidates.json"
SGAC_DATA_MANIFEST = "configs/sgac/manifests/data.json"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def _exclusions(pool: list[Problem], refs: dict[str, list[Problem]], ratio: float) -> dict[str, list[dict]]:
    """{pool_id: [{"ref_set", "ref_id", "sim"}]} for every exact-id or near-duplicate match."""
    out: dict[str, list[dict]] = defaultdict(list)
    for name, ref in refs.items():
        ref_ids = {r.unique_id for r in ref}
        for p in pool:
            if p.unique_id in ref_ids:
                out[p.unique_id].append({"ref_set": name, "ref_id": p.unique_id, "sim": 1.0, "kind": "exact_id"})
        for pid, rid, sim in near_duplicates(pool, ref, ratio):
            if pid != rid:
                out[pid].append({"ref_set": name, "ref_id": rid, "sim": float(sim), "kind": "near_dup"})
    return dict(out)


def _study1_ids(cs: CalibSpec, frozen: list[str] | None) -> list[str]:
    path = cs.path(cs.data.study1_candidates)
    doc = read_json(path)
    if doc is None:
        if frozen is None:
            raise FileNotFoundError(f"{path} missing and no frozen Study-1 list to fall back on")
        log.warning("%s not found; using the Study-1 ids frozen in %s", path, DATA_ROLES)
        return list(frozen)
    ids = sorted(doc["ids"])
    if frozen is not None and ids != sorted(frozen):
        raise RuntimeError(f"{path} no longer matches the Study-1 ids frozen in {DATA_ROLES}")
    return ids


def build_data_roles(cs: CalibSpec, frozen_study1: list[str] | None = None) -> dict:
    ratio = float(cs.data.near_dup_ratio)
    train = load_math_train()
    by_id = {p.unique_id: p for p in train}
    m500 = load_math500()
    sg = read_json(cs.path(SGAC_DATA_MANIFEST))["sections"]
    test50 = [by_id[r["unique_id"]] for r in sg["test50"]["items"]]
    sgac_pool = [by_id[r["unique_id"]] for r in sg["pool"]["items"]]
    row_of = {r["unique_id"]: int(r["row"]) for r in sg["pool"]["items"]}
    pool_m = Manifest.load(cs.path(cs.data.prereg_pool_manifest))
    held_m = Manifest.load(cs.path(cs.data.prereg_heldout_manifest))
    prereg_pool = select_by_manifest(pool_m, train)
    heldout = select_by_manifest(held_m, train)
    inel = [r["unique_id"] for r in read_json(cs.path(cs.data.prereg_ineligible_manifest))["items"]]
    study1 = _study1_ids(cs, frozen_study1)
    final = {"math500": m500, "test50": test50}

    dev_ex = _exclusions(heldout, final, ratio)
    dev = [p for p in heldout if p.unique_id not in dev_ex]

    cal_ex = _exclusions(prereg_pool, {**final, "dev": dev, "sgac_pool": sgac_pool}, ratio)
    for uid in inel:
        cal_ex.setdefault(uid, []).append({"ref_set": "ineligible_gold", "ref_id": uid, "sim": None, "kind": "rule"})
    for uid in study1:
        cal_ex.setdefault(uid, []).append({"ref_set": "prereg_study1", "ref_id": uid, "sim": None, "kind": "rule"})
    calib = [p for p in prereg_pool if p.unique_id not in cal_ex]

    cur_ex = _exclusions(sgac_pool, final, ratio)
    curriculum = [p for p in sgac_pool if p.unique_id not in cur_ex]

    def role(items, excluded, extra=None):
        ids = [p.unique_id for p in items]
        return {"n": len(ids), "ids": ids, "ids_hash": _ids_hash(ids), **(extra or {}),
                "excluded": {k: excluded[k] for k in sorted(excluded)}}

    doc = {
        "created": utc_now(), "near_dup_ratio": ratio,
        "rule": "pairwise disjoint by exact id and difflib near-duplicate >= near_dup_ratio (rlvr_v2.data.near_duplicates)",
        "sources": {"sgac_data_manifest": sha256_file(cs.path(SGAC_DATA_MANIFEST)),
                    "prereg_pool": sha256_file(cs.path(cs.data.prereg_pool_manifest)),
                    "prereg_heldout": sha256_file(cs.path(cs.data.prereg_heldout_manifest)),
                    "prereg_ineligible": sha256_file(cs.path(cs.data.prereg_ineligible_manifest))},
        "final_eval": {"math500": {"n": len(m500), "ids_hash": _ids_hash([p.unique_id for p in m500])},
                       "test50": {"n": len(test50), "ids": [p.unique_id for p in test50]}},
        "dev": role(dev, dev_ex),
        "calibration_eligible": role(calib, cal_ex),
        "curriculum_pool": role(curriculum, cur_ex, {"rows": [row_of[p.unique_id] for p in curriculum]}),
        "study1_candidates": study1,
    }
    _assert_disjoint(doc)
    return doc


def _assert_disjoint(doc: dict) -> None:
    final = set(doc["final_eval"]["test50"]["ids"])
    roles = {k: set(doc[k]["ids"]) for k in ("dev", "calibration_eligible", "curriculum_pool")}
    for name, ids in roles.items():
        if ids & final:
            raise RuntimeError(f"role {name} contains test50 items: {sorted(ids & final)[:5]}")
    if roles["dev"] & roles["calibration_eligible"]:
        raise RuntimeError("dev and calibration candidates overlap")
    if roles["calibration_eligible"] & roles["curriculum_pool"]:
        raise RuntimeError("calibration candidates overlap the curriculum pool")


# ---------------------------------------------------------------------- outcome-blind candidate draw
def ps_bin(ps: float, bins) -> int:
    for j, (lo, hi) in enumerate(bins):
        if (lo == hi and ps == lo) or (lo < ps <= hi):
            return j
    raise ValueError(f"Ps {ps} falls in no bin {bins}")


def draw_candidates(cs: CalibSpec, eligible_ids: list[str], signals: dict[str, dict]) -> dict:
    """Per Ps bin (K=32 base sieve), cycle levels from 5 down to 1 among those present and draw uniformly within a level.
    Pilot = the first `pilot_per_bin` per bin, expansion = the next `expansion_per_bin`. Uses base-policy signals and
    levels only, never any training outcome; input-order invariant (items sorted by id)."""
    s = cs.sampling
    bins = [tuple(b) for b in s.ps_bins]
    rng = np.random.default_rng(int(s.rng_seed))
    per_bin = int(s.pilot_per_bin) + int(s.expansion_per_bin)
    by_bin_level: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for uid in sorted(eligible_ids):
        sig = signals[uid]
        by_bin_level[ps_bin(float(sig["p_s"]), bins)][int(sig["level"])].append(uid)
    out: dict = {"rule": ("per Ps bin: levels cycled 5..1 among those present, uniform draw within a level; "
                          "pilot = first pilot_per_bin, expansion = next expansion_per_bin"),
                 "rng_seed": int(s.rng_seed), "ps_bins": [list(b) for b in bins], "bins": [], "pilot": [], "expansion": []}
    for j in range(len(bins)):
        levels = {lv: list(ids) for lv, ids in by_bin_level[j].items()}
        drawn: list[str] = []
        while len(drawn) < per_bin and any(levels.values()):
            for lv in sorted(levels, reverse=True):
                if len(drawn) >= per_bin:
                    break
                if levels[lv]:
                    drawn.append(levels[lv].pop(int(rng.integers(len(levels[lv])))))
        if len(drawn) < per_bin:
            raise RuntimeError(f"Ps bin {bins[j]} has only {len(drawn)} eligible items < {per_bin}")
        recs = [{"unique_id": u, "bin": j, "level": int(signals[u]["level"]), "ps32": float(signals[u]["p_s"])}
                for u in drawn]
        out["bins"].append({"bin": list(bins[j]), "n_eligible": sum(len(v) for v in by_bin_level[j].values()),
                            "levels_eligible": {str(k): len(v) for k, v in sorted(by_bin_level[j].items())}})
        out["pilot"] += recs[: s.pilot_per_bin]
        out["expansion"] += recs[s.pilot_per_bin:]
    for i, r in enumerate(out["pilot"] + out["expansion"]):
        r["index"] = i  # candidate index -> training seeds (offset + index)
    return out


def sieve_signals(cs: CalibSpec, path: str | Path | None = None) -> dict[str, dict]:
    return {r["unique_id"]: r for r in read_jsonl(path or cs.path(cs.data.sieve_signals))}


# ---------------------------------------------------------------------- build / check
def curriculum_seeds(cs: CalibSpec, seed_count_doc: dict | None) -> list[int]:
    n = int(seed_count_doc["chosen_seeds"]) if seed_count_doc else int(cs.curriculum.seeds_max)
    seeds = list(range(cs.curriculum.first_seed, cs.curriculum.first_seed + n))
    return sorted(set(seeds) | set(cs.curriculum.secondary_seeds))


def build_all(cs: CalibSpec, sieve_signals_path: str | Path | None = None, steps: int = 20, batch_b: int = 4) -> dict:
    md = cs.manifest_dir
    md.mkdir(parents=True, exist_ok=True)
    roles = build_data_roles(cs, (read_json(md / DATA_ROLES) or {}).get("study1_candidates"))
    sig_path = Path(sieve_signals_path) if sieve_signals_path else cs.path(cs.data.sieve_signals)
    cands = draw_candidates(cs, roles["calibration_eligible"]["ids"], sieve_signals(cs, sig_path))
    cands["sieve_signals_sha256"] = sha256_file(sig_path)
    atomic_write_json(md / DATA_ROLES, roles)
    atomic_write_json(md / CANDIDATES, cands)
    uid_of_row = dict(zip(roles["curriculum_pool"]["rows"], roles["curriculum_pool"]["ids"]))
    rows = sorted(roles["curriculum_pool"]["rows"])
    seeds = curriculum_seeds(cs, read_json(md / "seed_count.json"))
    for s in seeds:
        write_schedule(md, s, rows, steps, batch_b, uid_of_row)
    return {"data_roles": {k: roles[k]["n"] for k in ("dev", "calibration_eligible", "curriculum_pool")},
            "candidates": {"pilot": len(cands["pilot"]), "expansion": len(cands["expansion"])}, "schedules": seeds}


def check_all(cs: CalibSpec, sieve_signals_path: str | Path | None = None, steps: int = 20, batch_b: int = 4) -> dict:
    """Rebuild every manifest in memory and compare with the committed files (ids, exclusions, draw, schedules)."""
    from ..sgac.schedule import load_schedule

    md = cs.manifest_dir
    stored_roles, stored_cands = read_json(md / DATA_ROLES), read_json(md / CANDIDATES)
    if stored_roles is None or stored_cands is None:
        raise FileNotFoundError(f"manifests missing under {md}; run `python -m rlvr_v2.sgac_calib build-manifests`")
    roles = build_data_roles(cs, stored_roles.get("study1_candidates"))
    for k in ("dev", "calibration_eligible", "curriculum_pool"):
        if roles[k]["ids"] != stored_roles[k]["ids"] or roles[k]["excluded"].keys() != stored_roles[k]["excluded"].keys():
            raise RuntimeError(f"role {k} no longer reproduces {md / DATA_ROLES}")
    sig_path = Path(sieve_signals_path) if sieve_signals_path else cs.path(cs.data.sieve_signals)
    if sha256_file(sig_path) != stored_cands["sieve_signals_sha256"]:
        raise RuntimeError(f"{sig_path} differs from the sieve the candidates were drawn from")
    cands = draw_candidates(cs, roles["calibration_eligible"]["ids"], sieve_signals(cs, sig_path))
    if [r["unique_id"] for r in cands["pilot"] + cands["expansion"]] != \
            [r["unique_id"] for r in stored_cands["pilot"] + stored_cands["expansion"]]:
        raise RuntimeError(f"the candidate draw no longer reproduces {md / CANDIDATES}")
    rows = sorted(stored_roles["curriculum_pool"]["rows"])
    seeds = curriculum_seeds(cs, read_json(md / "seed_count.json"))
    for s in seeds:
        if not schedule_path(md, s).exists():
            raise FileNotFoundError(schedule_path(md, s))
        load_schedule(md, s, rows, steps, batch_b)
    return {"data_roles": "ok", "candidates": "ok", "schedules": seeds}


def candidate_records(cs: CalibSpec, stage: str = "all") -> list[dict]:
    doc = read_json(cs.manifest_dir / CANDIDATES)
    if doc is None:
        raise FileNotFoundError(cs.manifest_dir / CANDIDATES)
    if stage == "pilot":
        return doc["pilot"]
    if stage == "expansion":
        return doc["expansion"]
    return doc["pilot"] + doc["expansion"]


def dump(obj) -> str:
    return json.dumps(obj, indent=1, default=str)
