"""Committed calibration manifests: disjoint data roles, the outcome-blind draw, schedules, the seed count."""
import json
import math

import numpy as np
import pytest

from rlvr_v2.artifacts import read_json
from rlvr_v2.sgac.schedule import candidate_schedule, load_schedule
from rlvr_v2.sgac_calib.manifests import draw_candidates, ps_bin
from rlvr_v2.sgac_calib.spec import load_calib

CS = load_calib()
MD = CS.manifest_dir
ROLES = read_json(MD / "data_roles.json")
CANDS = read_json(MD / "candidates.json")
BINS = [tuple(b) for b in CS.sampling.ps_bins]


def test_roles_pairwise_disjoint_and_free_of_final_eval():
    dev, cal, cur = (set(ROLES[k]["ids"]) for k in ("dev", "calibration_eligible", "curriculum_pool"))
    test50 = set(ROLES["final_eval"]["test50"]["ids"])
    m500 = set(read_json(CS.path("manifests/math500.json"))["ids"])
    assert not (dev & cal) and not (cal & cur)
    for ids in (dev, cal, cur):
        assert not (ids & test50) and not (ids & m500)
    assert ROLES["near_dup_ratio"] == pytest.approx(0.8)
    assert (len(dev), len(cal), len(cur)) == (486, 408, 964)


def test_exclusions_cover_every_role_rule():
    ex = ROLES["calibration_eligible"]["excluded"]
    reasons = {m["ref_set"] for v in ex.values() for m in v}
    assert {"math500", "test50", "dev", "sgac_pool", "prereg_study1", "ineligible_gold"} <= reasons
    assert len(ROLES["study1_candidates"]) == 22
    assert all(uid in ex for uid in ROLES["study1_candidates"])
    cur_ex = ROLES["curriculum_pool"]["excluded"]
    assert {m["ref_set"] for v in cur_ex.values() for m in v} <= {"math500", "test50"}
    assert all(m["sim"] >= 0.8 for v in cur_ex.values() for m in v)


def test_candidate_draw_shape():
    recs = CANDS["pilot"] + CANDS["expansion"]
    assert len(CANDS["pilot"]) == 16 and len(CANDS["expansion"]) == 32
    assert [r["index"] for r in recs] == list(range(48))
    assert {r["unique_id"] for r in recs} <= set(ROLES["calibration_eligible"]["ids"])
    for j in range(len(BINS)):
        assert sum(r["bin"] == j for r in CANDS["pilot"]) == 4
        assert sum(r["bin"] == j for r in recs) == 12
    for r in recs:
        assert ps_bin(r["ps32"], BINS) == r["bin"]


def test_draw_is_deterministic_and_input_order_invariant():
    rng = np.random.default_rng(0)
    sig = {f"train/x/{i}.json": {"p_s": float(rng.choice([0.0, 0.2, 0.5, 0.9])), "level": int(rng.integers(1, 6))}
           for i in range(400)}
    ids = list(sig)
    a = draw_candidates(CS, ids, sig)
    b = draw_candidates(CS, list(reversed(ids)), sig)
    assert [r["unique_id"] for r in a["pilot"] + a["expansion"]] == [r["unique_id"] for r in b["pilot"] + b["expansion"]]


@pytest.mark.parametrize("value,expected", [(0.0, 0), (1 / 32, 1), (0.4, 1), (0.40625, 2), (0.8, 2), (0.8125, 3), (1.0, 3)])
def test_ps_bin_edges(value, expected):
    assert ps_bin(value, BINS) == expected


def test_seed_count_follows_the_rule():
    doc = read_json(MD / "seed_count.json")
    assert doc["chosen_seeds"] == 8
    ok = [r for r in doc["table"] if r["S"] >= 6 and r["mde"] <= doc["mde_max"]]
    assert ok[0]["S"] == doc["chosen_seeds"]
    assert doc["mde_max"] == pytest.approx(0.02) and doc["alpha"] == pytest.approx(0.025)
    assert sorted(doc["provenance"]["arms"]) == ["max_d", "max_var", "random", "sgac"]


@pytest.mark.parametrize("seed", range(42, 50))
def test_schedules_verify_and_share_the_20_step_prefix(seed):
    rows = sorted(ROLES["curriculum_pool"]["rows"])
    batches = load_schedule(MD, seed, rows, 20, 4)
    assert [b["rows"] for b in batches] == candidate_schedule(rows, seed, 20, 4)
    used = [r for b in batches for r in b["rows"]]
    assert len(set(used)) == 80 and set(used) <= set(rows)
    doc = json.loads((MD / f"batches_seed{seed}.json").read_text(encoding="utf-8"))
    assert len(doc["batches"]) == 40 and math.isclose(doc["n_pool"], 964)
