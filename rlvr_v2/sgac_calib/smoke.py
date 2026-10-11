"""End-to-end smoke of the calibration track: base dev eval -> one C1 run with a forced resume -> labels -> a short
learned-arm curriculum and a time-matched Random run, with a synthetic (clearly labelled) selector.

CPU mode: tiny model (configs/sgac/smoke_cpu.yaml), temporary results root; plumbing and record schemas only.
GPU mode (T0): the real model and caps, 2 bursts / 2 steps, 64-item eval subsets, MATH-500 off; its timings give the
per-run projection and the pre-declared step count of the time-matched Random arm (`t0/tm_steps.json`). Committed
manifests are read, never written.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
from pathlib import Path

from ..artifacts import atomic_write_json, read_json
from ..sgac.loop import SgacRun
from ..sgac.runinfo import SgacRunDir
from ..sgac.schedule import load_schedule
from ..sgac.spec import CONFIG_DIR as SGAC_CONFIG_DIR
from .calib import (
    base_dev_reference,
    collect_labels,
    load_calib_context,
    run_base_dev_eval,
    run_calibration,
    run_name,
    training_seeds,
)
from .curriculum import LearnedRun, RandomNoSieveRun, load_curriculum_context, tm_steps
from .selector import LearnedSelector
from .spec import load_calib, substrate_spec

log = logging.getLogger(__name__)

SMOKE_SELECTOR = {"model": "ridge", "features": ["Ps", "Var", "D", "L"], "k_deploy": 4,
                  "coef_raw": {"Ps": -1.0, "Var": 0.5, "D": 0.5, "L": 0.1}, "intercept_raw": 0.0,
                  "note": "SYNTHETIC smoke-test selector; never deployed"}


def _phase_s(history: list[dict], key: str) -> float:
    vals = [((h.get("timing") or {}).get(key) or {}).get("wall_s") for h in history]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def run_smoke(gpu: bool = False, out_dir: Path | None = None, calib_path=None, overrides: list[str] | None = None) -> dict:
    prod = load_calib(calib_path, overrides)
    out_dir = Path(out_dir) if out_dir else prod.results_dir / "t0_smoke"
    cs = load_calib(calib_path, [*(overrides or []), f"results_root={out_dir.as_posix()}"])
    extra = [SGAC_CONFIG_DIR / ("smoke_gpu.yaml" if gpu else "smoke_cpu.yaml")]
    ov = ["loop.math500_steps=[]", "loop.eval_max_items=64"] if gpu else []
    summary: dict = {"gpu": gpu, "calib_hash": cs.calib_hash()}

    # --- calibration: base dev eval, one C1 run, forced resume, label
    cspec = substrate_spec(cs, "calib", ov, extra)
    ctx = load_calib_context(cs, cspec)
    summary["base_dev"] = run_base_dev_eval(ctx)
    idx = int(ctx.candidates[0]["index"])
    seed = training_seeds(cs, idx)[0]
    run_calibration(ctx, idx, seed)
    again = run_calibration(ctx, idx, seed)
    assert again["step_done"] == cspec.loop.steps, "re-running a finished calibration run changed it"
    rd = SgacRunDir(cspec, run_name(idx, ctx.candidates[0]["unique_id"], seed))
    last = cspec.loop.steps
    evaluated = {p.name for p in (rd.path / "evals").glob("*/*") if p.is_dir()}
    assert evaluated == {"dev"}, f"a calibration run evaluated {evaluated}; only the dev set is allowed"
    # Calibration runs keep only the final adapter and the resume point, so the realistic failure after completion is a
    # lost state update (and a lost evaluation): the frozen loop must re-commit the last step from its files and redo
    # the dev evaluation. (A crash mid-step resumes from the previous adapter, which is deleted only after the next
    # commit; that path is the frozen loop's and is covered by tests/test_sgac_loop_resume.py.)
    s = read_json(rd.path / "state.json")
    s["step_done"] = last - 1
    s["history"] = [h for h in s["history"] if h["step"] < last]
    s["evals"] = {}
    atomic_write_json(rd.path / "state.json", s)
    shutil.rmtree(rd.path / "evals", ignore_errors=True)
    rd.set_status(state="running")
    redo = run_calibration(ctx, idx, seed)
    assert redo["step_done"] == last and str(last) in redo["evals"], "calibration resume did not re-commit the last step"
    kept = sorted(p.parent.name for p in (rd.path / "steps").glob("*/adapter"))
    assert kept == [f"step-{last:03d}"], f"calibration run kept adapters {kept}; only the final one is expected"
    summary["calib_resume_check"] = "ok"
    labels = collect_labels(ctx, [idx])
    summary["label"] = labels["candidates"][0]
    base_ref = base_dev_reference(cspec)
    summary["base_dev_n"] = len(base_ref)

    # --- curriculum: learned arm (synthetic selector), random (sieving), random_tm (no sieve)
    kspec = substrate_spec(cs, "curriculum", ov, extra)
    kctx = load_curriculum_context(cs, kspec)
    sel_path = out_dir / "selector_smoke.json"
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    sel_path.write_text(json.dumps(SMOKE_SELECTOR, indent=1), encoding="utf-8")
    selector = LearnedSelector(sel_path, hashlib.sha256(sel_path.read_bytes()).hexdigest())
    kseed = int(prod.curriculum.first_seed)
    rows = sorted(it.row for it in kctx.pool)
    sched = load_schedule(prod.manifest_dir, kseed, rows, kspec.loop.steps, kspec.loop.batch_b)
    kw = {"pool": kctx.pool, "test50": kctx.test50, "math500": kctx.math500, "schedule": sched, "grader": kctx.grader}
    learned = LearnedRun(kspec, kseed, selector, **kw).run()
    random_st = SgacRun(kspec, "random", kseed, **kw).run()
    tm = RandomNoSieveRun(kspec, kseed, **kw).run()
    picks_l = [h["selection"]["selected_uid"] for h in learned["history"]]
    picks_r = [h["selection"]["selected_uid"] for h in random_st["history"]]
    picks_tm = [h["selection"]["selected_uid"] for h in tm["history"]]
    assert picks_tm == picks_r, "time-matched Random must reproduce the Random arm's picks on the shared prefix"
    for h in learned["history"]:
        sc = h["selection"]["scores"]["learned"]
        assert h["selection"]["selected_idx"] == max(range(len(sc)), key=lambda i: (sc[i], -i)), "learned pick != argmax"
    summary["curriculum"] = {"learned_picks": picks_l, "random_picks": picks_r, "tm_picks": picks_tm}

    # --- timings and projection
    burst_s = _phase_s(redo["history"], "burst")
    sieve_s = _phase_s(random_st["history"], "sieve")
    k_burst_s = _phase_s(random_st["history"], "burst")
    dev_eval_s = float((redo["evals"][str(last)]["dev"].get("timing") or {}).get("wall_s", float("nan")))
    n_eval = int(kspec.loop.eval_max_items or len(ctx.dev))
    full_dev_s = dev_eval_s * math.ceil(len(ctx.dev) / 64) / max(1, math.ceil(n_eval / 64))
    full = substrate_spec(prod, "calib")
    summary["timings"] = {"calib_burst_s": burst_s, "curriculum_sieve_s": sieve_s, "curriculum_burst_s": k_burst_s,
                          "dev_eval_s_subset": dev_eval_s, "dev_eval_s_full_est": full_dev_s}
    if gpu:
        per_run_h = (full.loop.steps * burst_s + full_dev_s) / 3600
        n = tm_steps(sieve_s, k_burst_s, full.loop.steps)
        summary["projection"] = {"calib_run_h": per_run_h, "pilot_runs": 34, "pilot_h": 34 * per_run_h,
                                 "expansion_runs": 64, "expansion_h": 64 * per_run_h,
                                 "re_cost_before_pilot": bool(per_run_h > 1.1),
                                 "rule": "protocol section 9: re-cost before the pilot if a calibration run exceeds 1.1 h"}
        atomic_write_json(prod.results_dir / "t0" / "tm_steps.json",
                          {"tm_steps": n, "sieve_s": sieve_s, "burst_s": k_burst_s,
                           "rule": "round(20 * (t_sieve + t_burst) / t_burst), capped at the schedule length"})
        summary["tm_steps"] = n
    out = (prod.results_dir / "t0" if gpu else out_dir) / f"smoke_{'gpu' if gpu else 'cpu'}.json"
    atomic_write_json(out, summary)
    log.info("calibration smoke ok -> %s", out)
    return summary
