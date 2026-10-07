"""The pre-declared verdict rule and an end-to-end report over synthetic results."""
from pathlib import Path

import pytest

from rlvr_v2.artifacts import JsonlWriter, atomic_write_json, read_json
from rlvr_v2.sgac.report import build_report, verdict
from rlvr_v2.sgac.spec import group_dir, load_profile


def ci(lo, hi):
    return {"delta": (lo + hi) / 2, "ci_lo": lo, "ci_hi": hi}


def test_verdict_rule():
    assert verdict(ci(0.01, 0.05), ci(0.002, 0.04))["label"] == "Original SGAC reproduced"
    assert verdict(ci(0.01, 0.05), ci(-0.01, 0.04))["label"] == "Original SGAC partially reproduced"
    assert verdict(ci(-0.01, 0.05), ci(0.01, 0.04))["label"] == "Original SGAC partially reproduced"
    v = verdict(ci(-0.06, -0.01), ci(-0.02, 0.01))
    assert v["label"] == "Original SGAC could not be reproduced" and any("DEGRADES" in n for n in v["notes"])
    assert verdict(None, ci(0.1, 0.2))["label"].startswith("INCOMPLETE")


def _per_item(path: Path, uids, correct_set):
    with JsonlWriter(path, append=False) as w:
        for u in uids:
            w.write({"unique_id": u, "correct": u in correct_set, "mv_correct": u in correct_set,
                     "legacy_correct": u in correct_set, "level": 3, "text": "x"})


def _run(group: Path, spec, arm, seed, uids50, uids500, c50, c500):
    d = group / f"{arm}__seed{seed}"
    hist = [{"step": t, "selection": {"selected_uid": f"p{t}", "selected_idx": 0, "selected_signals":
                                      {"Ps": 0.0, "Var": 0.0, "D": 1.0, "L": 5},
                                      "picks": {"sgac_eq10": 0, "max_level": 0, "max_var": 1, "max_d": 0, "random": 2},
                                      "scores": {"sgac_eq10": [1.0, 0.5, 0.2, 0.1]}, "eq10_picked_max_level": True},
             "burst": {"all_zero_loss": t % 2 == 0, "reward": [0.5], "frac_reward_zero_std": [1.0],
                       "clipped_ratio": [0.1], "peak_mem_gb": 5.0},
             "timing": {"sieve": {"wall_s": 10.0}, "burst": {"wall_s": 20.0}, "save": {"wall_s": 1.0}}}
            for t in range(1, 21)]
    evals = {"20": {"test50": {"acc": len(c50) / len(uids50), "timing": {"wall_s": 60.0}},
                    "math500": {"acc": len(c500) / len(uids500), "timing": {"wall_s": 600.0}}}}
    atomic_write_json(d / "state.json", {"arm": arm, "seed": seed, "spec_hash": spec.spec_hash(), "profile": "e0",
                                         "step_done": 20, "history": hist, "evals": evals})
    atomic_write_json(d / "status.json", {"state": "done"})
    _per_item(d / "evals" / "step-020" / "test50" / "per_item.jsonl", uids50, c50)
    _per_item(d / "evals" / "step-020" / "math500" / "per_item.jsonl", uids500, c500)


@pytest.mark.parametrize("sgac_k, expect", [(36, "Original SGAC reproduced"), (20, "Original SGAC could not be reproduced")])
def test_report_end_to_end(tmp_path, sgac_k, expect):
    spec = load_profile("e0", [f"run.results_root={tmp_path.as_posix()}"])
    g = group_dir(spec)
    u50 = [f"t{i}" for i in range(50)]
    u500 = [f"m{i}" for i in range(40)]
    sh = g / "_shared" / "base_eval"
    _per_item(sh / "test50" / "per_item.jsonl", u50, set(u50[:30]))
    _per_item(sh / "math500" / "per_item.jsonl", u500, set(u500[:20]))
    _run(g, spec, "sgac", 42, u50, u500, set(u50[:33]), set(u500[:sgac_k]))
    _run(g, spec, "random", 42, u50, u500, set(u50[:31]), set(u500[:20]))
    path = build_report([spec], out_dir=tmp_path / "report")
    text = path.read_text(encoding="utf-8")
    assert expect in text and "P1: SGAC − Base" in text
    data = read_json(tmp_path / "report" / "report_data.json")
    assert data["verdict"]["label"] == expect
    assert data["e0"]["diagnostics"]["eq10_picked_max_level"]["rate"] == 1.0
    assert (tmp_path / "report" / "tables" / "selection_e0_sgac_seed42.csv").exists()
