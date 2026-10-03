import dataclasses
import json
import math

import pytest

from rlvr_v2.candidates import candidates_manifest, in_bin, select_candidates, summarize_selection
from rlvr_v2.config import Study1Cfg
from rlvr_v2.data import Manifest
from tests.conftest import make_signals

K = 32


def _pool(gap_pattern="wide"):
    """~120 synthetic candidates: anchors, three p_s bins x levels x high/low d_wrong, some NaN d_wrong."""
    sigs = []
    # null anchors (p_s == 1) at several levels, zero anchors with parsable and unparsable failures
    for i, level in enumerate([1, 2, 3, 4, 5, 2]):
        sigs.append(make_signals(f"null{i}", p_s=1.0, d_simpson=0.0, level=level, k=K))
    for i, level in enumerate([1, 2, 3, 4]):
        sigs.append(make_signals(f"zero{i}", p_s=0.0, d_simpson=0.5, d_wrong=0.5, level=level, k=K))
    for i in range(2):  # unparsable failures: ineligible as zero anchors
        sigs.append(dataclasses.replace(make_signals(f"zero_bad{i}", p_s=0.0, level=1, k=K), format_rate=0.2))
    grids = {0: [2, 3, 5, 6, 8], 1: [10, 12, 15, 16, 20, 24], 2: [25, 26, 27, 28]}
    uid = 0
    for b, counts in grids.items():
        for n_corr in counts:
            for level in (1, 3, 5):
                for role in ("hi", "lo"):
                    if gap_pattern == "wide":
                        dw = 0.9 if role == "hi" else 0.1
                    elif gap_pattern == "narrow_mid" and b == 1:
                        dw = 0.55 if role == "hi" else 0.3  # gap 0.25: needs relaxation to 0.2
                    elif gap_pattern == "none_mid" and b == 1:
                        dw = 0.5  # no gap at all in the middle bin
                    else:
                        dw = 0.9 if role == "hi" else 0.1
                    sigs.append(make_signals(f"c{uid}", p_s=n_corr / K, d_simpson=0.5, d_wrong=dw, level=level, k=K))
                    uid += 1
    # candidates whose d_wrong is undefined must never be paired
    for i in range(6):
        sigs.append(make_signals(f"nan{i}", p_s=0.5, d_simpson=0.5, d_wrong=float("nan"), level=3, k=K))
    return sigs


def _check_pairs(sel, cfg, by_uid):
    for p in sel["pairs"]:
        hi, lo = by_uid[p["high"]], by_uid[p["low"]]
        assert hi.level == lo.level == p["level"]
        assert abs(hi.p_s - lo.p_s) <= cfg.max_ps_diff + 1e-9
        assert hi.d_wrong - lo.d_wrong >= p["min_dwrong_gap_used"] - 1e-9
        assert p["dwrong_high"] == hi.d_wrong and p["dwrong_low"] == lo.d_wrong
        lo_b, hi_b = p["bin"]
        assert in_bin(hi.p_s, lo_b, hi_b) and in_bin(lo.p_s, lo_b, hi_b)
        assert not math.isnan(hi.d_wrong) and not math.isnan(lo.d_wrong)


def test_design_counts_and_constraints():
    cfg = Study1Cfg()
    sigs = _pool()
    assert len(sigs) >= 100
    by_uid = {s.unique_id: s for s in sigs}
    sel = select_candidates(sigs, cfg, rng_seed=0)
    assert len(sel["anchors_null"]) == cfg.n_null_anchors and all(by_uid[u].p_s == 1.0 for u in sel["anchors_null"])
    assert len(sel["anchors_zero"]) == cfg.n_zero_anchors
    assert all(by_uid[u].p_s == 0.0 and by_uid[u].format_rate >= 0.5 for u in sel["anchors_zero"])
    # level diversity: two anchors never share a level when several levels are available
    assert len({by_uid[u].level for u in sel["anchors_null"]}) == 2
    assert len({by_uid[u].level for u in sel["anchors_zero"]}) == 2
    formed = [sum(1 for p in sel["pairs"] if p["bin_index"] == b) for b in range(len(cfg.bins))]
    assert formed == list(cfg.pairs_per_bin)
    _check_pairs(sel, cfg, by_uid)
    assert sel["relaxations"] == [] and sel["shortfall"] == {}
    assert all(p["dwrong_gap"] >= cfg.min_dwrong_gap for p in sel["pairs"])
    # nobody used twice; anchors are not pair members; NaN d_wrong never paired
    ids = sel["ordered_ids"]
    assert len(ids) == len(set(ids)) == 2 + 2 + 2 * sum(cfg.pairs_per_bin)
    pair_ids = {p["high"] for p in sel["pairs"]} | {p["low"] for p in sel["pairs"]}
    assert not pair_ids & set(sel["anchors_null"] + sel["anchors_zero"])
    assert not any(u.startswith("nan") for u in pair_ids)
    assert ids[:4] == sel["anchors_null"] + sel["anchors_zero"]
    # replicates: requested count, spread across all bins, alternating roles
    reps = sel["replicate_ids"]
    assert len(reps) == cfg.replicate_count and len(set(reps)) == len(reps) and set(reps) <= set(ids)
    rep_bins = {p["bin_index"] for p in sel["pairs"] if p["high"] in reps or p["low"] in reps}
    assert rep_bins == set(range(len(cfg.bins)))
    roles = ["high" if any(p["high"] == r for p in sel["pairs"]) else "low" for r in reps[:3]]
    assert roles == ["high", "low", "high"]
    assert "anchors" in summarize_selection(sel)


def test_deterministic_and_order_invariant():
    cfg = Study1Cfg()
    sigs = _pool()
    a = select_candidates(sigs, cfg, rng_seed=0)
    b = select_candidates(list(reversed(sigs)), cfg, rng_seed=0)
    assert a == b
    c = select_candidates(sigs, cfg, rng_seed=1)
    assert len(c["pairs"]) == len(a["pairs"])  # a different seed is still a complete design
    assert json.dumps(a, sort_keys=True)  # JSON serialisable


def test_relaxation_recorded_when_gaps_are_scarce():
    cfg = Study1Cfg()
    sigs = _pool("narrow_mid")
    by_uid = {s.unique_id: s for s in sigs}
    sel = select_candidates(sigs, cfg, rng_seed=0)
    mid = [p for p in sel["pairs"] if p["bin_index"] == 1]
    assert len(mid) == cfg.pairs_per_bin[1]
    assert all(p["min_dwrong_gap_used"] == pytest.approx(0.2) for p in mid)
    gaps_tried = [r["min_dwrong_gap"] for r in sel["relaxations"] if r["bin_index"] == 1]
    assert gaps_tried == [0.3, 0.2]
    assert sum(r["pairs_added"] for r in sel["relaxations"] if r["bin_index"] == 1) == cfg.pairs_per_bin[1]
    _check_pairs(sel, cfg, by_uid)
    assert sel["shortfall"] == {}
    # no usable gap at all -> shortfall is reported, the other bins are unaffected
    none = select_candidates(_pool("none_mid"), cfg, rng_seed=0)
    assert none["shortfall"] == {"bin1": cfg.pairs_per_bin[1]}
    assert [sum(1 for p in none["pairs"] if p["bin_index"] == b) for b in range(3)] == [4, 0, 4]
    assert len(none["replicate_ids"]) == cfg.replicate_count


def test_small_config_and_missing_anchors():
    cfg = Study1Cfg(pairs_per_bin=(1, 1, 1), n_null_anchors=1, n_zero_anchors=3, replicate_count=2)
    sigs = [s for s in _pool() if not s.unique_id.startswith("zero")]  # no zero anchors available
    sel = select_candidates(sigs, cfg, rng_seed=0)
    assert sel["anchors_zero"] == [] and sel["shortfall"] == {"anchors_zero": 3}
    assert len(sel["pairs"]) == 3 and len(sel["replicate_ids"]) == 2
    with pytest.raises(ValueError):
        select_candidates(sigs, Study1Cfg(pairs_per_bin=(1, 1)), rng_seed=0)


def test_manifest_from_pool():
    cfg = Study1Cfg()
    sigs = _pool()
    sel = select_candidates(sigs, cfg, rng_seed=0)
    pool = Manifest(name="pool", source="math_train", ids=[s.unique_id for s in sigs],
                    hashes={s.unique_id: f"h{i:04d}" for i, s in enumerate(sigs)}, meta={"dataset": "ds", "n": len(sigs)})
    m = candidates_manifest(sel, pool)
    assert m.ids == sel["ordered_ids"] and m.source == "math_train"
    assert all(m.hashes[u] == pool.hashes[u] for u in m.ids) and set(m.hashes) == set(m.ids)
    assert m.meta["selection"]["pairs"] == sel["pairs"] and m.meta["replicate_ids"] == sel["replicate_ids"]
    assert m.meta["dataset"] == "ds" and m.meta["n"] == len(m.ids)
    with pytest.raises(KeyError):
        candidates_manifest(sel, Manifest(name="pool", source="math_train", ids=[], hashes={}))


def test_cli_script(tmp_path):
    import importlib.util
    import sys

    sigs = _pool()
    sig_path = tmp_path / "signals.jsonl"
    sig_path.write_text("\n".join(json.dumps(s.to_dict()) for s in sigs) + "\n")
    pool = Manifest(name="pool", source="math_train", ids=[s.unique_id for s in sigs],
                    hashes={s.unique_id: f"h{i:04d}" for i, s in enumerate(sigs)})
    pool.save(tmp_path / "pool.json")
    spec = importlib.util.spec_from_file_location("select_candidates", "scripts/select_candidates.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["select_candidates"] = mod
    spec.loader.exec_module(mod)
    out = tmp_path / "cands.json"
    mod.main(["--signals", str(sig_path), "--pool-manifest", str(tmp_path / "pool.json"), "--no-base",
              "--config", "--out", str(out), "--seed", "3"])
    m = Manifest.load(out)
    assert len(m) == 2 + 2 + 2 * sum(Study1Cfg().pairs_per_bin) and len(m.meta["replicate_ids"]) == 8
