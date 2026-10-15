"""Equal-GPU-compute comparison (amendment A6): budget rule, specs, schedules, decision rule, queue."""
import numpy as np
import pytest

from rlvr_v2.sgac_calib.compute_budget import (
    DecisionCfg,
    check_schedules,
    decide,
    load_cb,
    may_run_next,
    schedule_prefix_equal,
    spec_a,
    spec_b,
    step_cost,
    training_seconds,
)
from rlvr_v2.sgac_calib.jobs import stage_jobs
from rlvr_v2.sgac_calib.spec import load_calib

CS = load_calib()
CB = load_cb()


def test_budget_rule_never_plans_past_the_budget():
    assert may_run_next(0.0, [], 100.0)
    assert may_run_next(60.0, [30.0, 30.0], 100.0)  # 60 + 30 = 90 <= 100: the next step fits
    assert may_run_next(60.0, [30.0, 30.0], 90.0)  # exactly fills the budget
    assert not may_run_next(80.0, [40.0, 40.0], 100.0)  # 80 + 40 = 120 > 100: stop
    assert not may_run_next(0.0, [], 0.0)


def test_budget_rule_corrects_itself():
    assert may_run_next(60.0, [30.0, 30.0], 90.0) and not may_run_next(61.0, [30.0, 31.0], 90.0)


def test_step_costs_count_screening_only_for_sgac():
    h = [{"timing": {"sieve": {"wall_s": 80.0}, "burst": {"wall_s": 30.0}, "save": {"wall_s": 1.0}}}]
    assert training_seconds(h, ("sieve", "burst", "save")) == 111.0
    assert step_cost(h[0], ("burst", "save")) == 31.0


def test_specs_keep_the_e0_substrate():
    assert spec_a(CS, CB).spec_hash() == "577f3319"
    b = spec_b(CS, CB)
    assert b.loop.steps == 160 and b.loop.eval_every == 10 and tuple(b.loop.math500_steps) == (0, 20)


def test_schedules_reproduce_and_share_the_frozen_prefix():
    assert set(check_schedules(CS, CB).values()) == {160}
    for s in CB.seeds:
        assert schedule_prefix_equal(CS, CB, s, 20)


@pytest.mark.parametrize("mean,ci90,p,seeds,expected", [
    (0.025, (0.012, 0.038), 0.001, [3, 2, 2, 3, 1, 2, 3, 4], "B better"),
    (-0.025, (-0.038, -0.012), 0.001, [-3] * 8, "A better"),
    (0.003, (-0.01, 0.016), 0.6, [1, -1] * 4, "equivalent"),
    (0.012, (-0.012, 0.036), 0.3, [1] * 8, "inconclusive"),
    (0.025, (0.012, 0.038), 0.001, [3, 2, -2, -3, 1, 2, 3, 4], "inconclusive"),  # 6/8 < 7/8
])
def test_decision_rule(mean, ci90, p, seeds, expected):
    t = {"mean": mean, "ci90": list(ci90), "p": p}
    assert decide(t, seeds, DecisionCfg()) == expected


def test_queue_runs_a_before_b_per_seed():
    jobs = stage_jobs(CS, "compute_budget")
    assert jobs[:3] == ["cb-schedules --check", "cb-smoke --gpu", "cb-base-eval"] and jobs[-1] == "cb-analyze"
    runs = jobs[3:-1]
    assert len(runs) == 16
    for i, s in enumerate(CB.seeds):
        assert runs[2 * i] == f"cb-run --strategy sgac --seed {s}"
        assert runs[2 * i + 1] == f"cb-run --strategy random --seed {s}"


def test_withdrawn_queue_is_not_in_the_current_plan():
    text = (CS.path("scripts/local/sgac_calib.sh")).read_text(encoding="utf-8")
    assert "STAGES=compute_budget" in text and "WITHDRAWN" in text


@pytest.mark.smoke
def test_cb_smoke_cpu(tmp_path):
    from rlvr_v2.sgac_calib.compute_budget import run_cb_smoke

    s = run_cb_smoke(tmp_path, gpu=False)
    assert s["A_steps"] == 2 and s["B_steps"] >= 1
    assert s["budget_ratio"] is not None and s["budget_ratio"] <= 1.5
    assert any(r.get("status") == "ok" for r in s["analysis_rows"])
    assert (tmp_path / "report" / "compute_budget_comparison.md").exists()
    assert (tmp_path / "report" / "compute_budget_seeds.csv").exists()
    assert np.isfinite(s["projected_B_steps_for_20_A_steps"])


def _write_pi(path, ids, y):
    import json as _j

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_j.dumps({"unique_id": i, "correct": bool(v)}) for i, v in zip(ids, y)), encoding="utf-8")


def _write_state(rd, history, evals, extra=None):
    import json as _j

    rd.mkdir(parents=True, exist_ok=True)
    (rd / "state.json").write_text(_j.dumps({"step_done": len(history), "history": history, "evals": evals,
                                             **(extra or {})}), encoding="utf-8")


def test_analysis_and_report_on_synthetic_results(tmp_path):
    from rlvr_v2.artifacts import read_json
    from rlvr_v2.sgac.spec import group_dir
    from rlvr_v2.sgac_calib.compute_budget import analyze

    cs = load_calib(overrides=[f"results_root={tmp_path.as_posix()}"])
    cb = load_cb(overrides=["seeds=[42,43,44]", "n_boot=300"])
    sa, sb = spec_a(cs, cb), spec_b(cs, cb)
    ids = read_json(cs.path("manifests/math500.json"))["ids"]
    rng = np.random.default_rng(0)
    theta = rng.normal(0.6, 1.5, len(ids))
    base = (rng.random(len(ids)) < 1 / (1 + np.exp(-theta))).astype(float)
    _write_pi(group_dir(sa) / "_shared" / "base_eval" / "math500" / "per_item.jsonl", ids, base)
    for s in cb.seeds:
        ya = (rng.random(len(ids)) < 1 / (1 + np.exp(-theta))).astype(float)
        yb20 = (rng.random(len(ids)) < 1 / (1 + np.exp(-theta))).astype(float)
        ybf = (rng.random(len(ids)) < 1 / (1 + np.exp(-(theta + 0.5)))).astype(float)
        ha = [{"step": t, "timing": {"sieve": {"wall_s": 85.0}, "burst": {"wall_s": 29.0}, "save": {"wall_s": 1.0}}}
              for t in range(1, 21)]
        ra = group_dir(sa) / f"sgac__seed{s}"
        _write_state(ra, ha, {"20": {"math500": {"timing": {"wall_s": 1900.0}}}})
        _write_pi(ra / "evals" / "step-020" / "math500" / "per_item.jsonl", ids, ya)
        hb = [{"step": t, "timing": {"burst": {"wall_s": 29.0}, "save": {"wall_s": 1.0}}} for t in range(1, 77)]
        rb = group_dir(sb) / f"random_budget__seed{s}"
        _write_state(rb, hb, {"20": {"math500": {"timing": {"wall_s": 1900.0}}},
                              "76": {"math500": {"timing": {"wall_s": 1900.0}}}},
                     {"budget_stop": {"final_step": 76, "budget_s": 2300.0, "used_s": 2280.0, "ratio": 2280 / 2300,
                                      "cap_hit": False}})
        _write_pi(rb / "evals" / "step-020" / "math500" / "per_item.jsonl", ids, yb20)
        _write_pi(rb / "evals" / "step-076" / "math500" / "per_item.jsonl", ids, ybf)
    out = tmp_path / "report"
    res = analyze(cs, cb, out_dir=out)
    assert res["seeds_complete"] == [42, 43, 44]
    assert res["primary"]["mean_pp"] > 0 and res["primary"]["verdict"] in ("B better", "inconclusive")
    assert res["compute"]["mean_B_steps"] == 76 and abs(res["compute"]["mean_budget_ratio"] - 2280 / 2300) < 1e-9
    assert res["compute"]["screening_share_of_A"] == pytest.approx(85 / 115)
    md = (out / "compute_budget_comparison.md").read_text(encoding="utf-8")
    for needle in ("## Result", "## Per seed", "## Compute", "## Secondary comparisons", "B_equal_steps_minus_A",
                   "accuracy_vs_compute.png"):
        assert needle in md
    assert (out / "figures" / "compute_budget_accuracy_vs_compute.png").stat().st_size > 1000
    rows = (out / "compute_budget_seeds.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4 and rows[0].startswith("seed,status,A_acc")
