"""End-to-end smoke for one profile: manifests -> base eval -> two short curricula (sgac, random) -> resume check ->
report. CPU mode uses a tiny model and a temporary results/manifest root (committed manifests are never touched);
GPU mode (`--gpu`) uses the real model, the committed manifests and writes a per-run time projection."""
from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path

from ..artifacts import atomic_write_json, read_json
from .spec import CONFIG_DIR, load_profile

log = logging.getLogger(__name__)


def _spec(profile: str, gpu: bool, out_dir: Path | None, overrides: list[str] | None):
    extra = [CONFIG_DIR / ("smoke_gpu.yaml" if gpu else "smoke_cpu.yaml")]
    ov = list(overrides or [])
    if out_dir is not None:
        ov.append(f"run.results_root={out_dir.as_posix()}")
        if not gpu:
            ov.append(f"data.manifest_dir={(out_dir / 'manifests').as_posix()}")
    return load_profile(profile, ov, extra)


def projection(spec, run_state: dict, base_eval: dict, full_spec) -> dict:
    """Rough per-run wall-clock for the full configuration from the smoke's measured phases."""
    steps = run_state.get("history", [])
    per_step = [sum((h.get("timing") or {}).get(k, {}).get("wall_s", 0.0) for k in ("sieve", "burst", "save")) for h in steps]
    step_s = sum(per_step) / len(per_step) if per_step else float("nan")
    timings = base_eval.get("timings", {})
    n_smoke = spec.loop.eval_max_items or 50
    bs = full_spec.profile.eval_batch_size
    t50 = timings.get("test50", {}).get("wall_s", float("nan"))
    m500 = timings.get("math500", {}).get("wall_s", float("nan"))
    t50_full = math.ceil(50 / bs) * t50 / max(1, math.ceil(n_smoke / bs))
    m500_full = math.ceil(500 / bs) * m500 / max(1, math.ceil(n_smoke / bs))
    n_test50 = sum(1 for t in range(1, full_spec.loop.steps + 1)
                   if t == full_spec.loop.steps or t % full_spec.loop.eval_every == 0)
    n_m500 = sum(1 for t in full_spec.loop.math500_steps if t > 0)
    per_run = full_spec.loop.steps * step_s + n_test50 * t50_full + n_m500 * m500_full
    return {"step_s": step_s, "test50_eval_s": t50_full, "math500_eval_s": m500_full, "per_run_s": per_run,
            "per_run_h": per_run / 3600, "note": "rough: eval times scaled from the smoke subset by batch count"}


def run_smoke(profile: str, gpu: bool = False, out_dir: Path | None = None, overrides: list[str] | None = None) -> dict:
    from . import data as sdata
    from .jobs import load_context, run_base_eval, run_curriculum
    from .report import build_report
    from .runinfo import SgacRunDir
    from .schedule import write_schedule

    spec = _spec(profile, gpu, out_dir, overrides)
    seed = int(spec.run.seeds[0])
    summary: dict = {"profile": profile, "gpu": gpu, "spec_hash": spec.spec_hash()}
    if not gpu:  # private manifests for the tiny-model run
        sdata.write_data_manifest(spec, mv=None)
        pool_rows = list(sdata.section_rows(spec, "pool"))
        manifest = sdata.load_data_manifest(spec)
        uid = {r["row"]: r["unique_id"] for r in manifest["sections"]["pool"]["items"]}
        write_schedule(sdata.manifest_dir(spec), seed, pool_rows, spec.loop.steps, spec.loop.batch_b, uid)
    ctx = load_context(spec)
    summary["base_eval"] = run_base_eval(spec, ctx)
    states = {}
    for arm in ("sgac", "random"):
        states[arm] = run_curriculum(spec, ctx, arm, seed, base_after_run=(arm == "sgac"))
    # resume check: a finished run is a no-op; a lost last step is redone from the previous adapter
    again = run_curriculum(spec, ctx, "sgac", seed, base_after_run=True)
    assert again["step_done"] == spec.loop.steps, "re-running a finished run changed it"
    rd = SgacRunDir(spec, f"sgac__seed{seed}")
    last = spec.loop.steps
    st = read_json(rd.path / "state.json")
    st["step_done"] = last - 1
    st["history"] = [h for h in st["history"] if h["step"] < last]
    st["evals"] = {k: v for k, v in st["evals"].items() if k != str(last)}
    atomic_write_json(rd.path / "state.json", st)
    (rd.path / "steps" / f"step-{last:03d}" / "done.flag").unlink()
    shutil.rmtree(rd.path / "evals" / f"step-{last:03d}", ignore_errors=True)
    rd.set_status(state="running")
    if not (rd.path / "steps" / f"step-{last - 1:03d}" / "adapter").exists():
        raise AssertionError("resume point adapter was not kept")
    redo = run_curriculum(spec, ctx, "sgac", seed, base_after_run=True)
    assert redo["step_done"] == last and str(last) in redo["evals"], "resume did not redo the last step"
    summary["resume_check"] = "ok"
    summary["runs"] = {arm: {"step_done": s["step_done"], "evals": s["evals"],
                             "picks": [h["selection"].get("selected_uid") for h in s["history"]]} for arm, s in states.items()}
    summary["report"] = str(build_report([spec], out_dir=(out_dir or Path(spec.run.results_root)) / "report"))
    if gpu:
        full = load_profile(profile)
        summary["projection"] = projection(spec, redo, summary["base_eval"], full)
    path = Path(spec.run.results_root) if Path(spec.run.results_root).is_absolute() else None
    out = (path or (out_dir or Path("."))) / f"smoke_{profile}.json"
    atomic_write_json(out, summary)
    log.info("smoke %s ok -> %s\n%s", profile, out, json.dumps({k: v for k, v in summary.items() if k != "base_eval"},
                                                               indent=1, default=str)[:3000])
    return summary
