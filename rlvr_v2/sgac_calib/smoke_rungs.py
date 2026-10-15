"""CPU smoke of the post-G-P rungs: base dev eval -> one C2 run -> one OS run (prereg trainer) -> labels -> diagnostics.

Tiny models and lengths (configs/sgac/smoke_cpu.yaml for the SGAC substrate, configs/smoke_cpu.yaml for the prereg
trainer); plumbing and record schemas only. Committed manifests are read, never written.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..artifacts import REPO_ROOT, atomic_write_json
from ..sgac.spec import CONFIG_DIR as SGAC_CONFIG_DIR
from ..sgac.spec import group_dir
from .calib import load_calib_context, run_base_dev_eval, training_seeds
from .diagnose import diagnose, os_runs, sgac_runs
from .rungs import (
    OS_STUDY,
    STAGE_OS,
    base_reference,
    c2_labels,
    c2_spec,
    os_config,
    os_labels,
    os_run_name,
    os_seeds,
    run_c2,
    run_os,
)
from .spec import load_calib, substrate_spec

log = logging.getLogger(__name__)


def run_rungs_smoke(out_dir: Path, gpu: bool = False) -> dict:
    """CPU: tiny models. GPU (first job of the os_pilot stage): the real model and recipes with 2 C2 bursts, 2 OS rounds
    and 64-item dev subsets; its timings project the cost of a full OS run before the queue commits to it."""
    cs = load_calib(None, [f"results_root={Path(out_dir).as_posix()}"])
    files = [] if gpu else [SGAC_CONFIG_DIR / "smoke_cpu.yaml"]
    ov = ["loop.eval_max_items=64", "loop.math500_steps=[]"] if gpu else []
    base_spec = substrate_spec(cs, "calib", ov, files)
    base = run_base_dev_eval(load_calib_context(cs, base_spec, need_candidates=False))
    summary: dict = {"gpu": gpu, "base_dev_n": len(base_reference(cs, base_spec)),
                     "base_dev_s": (base.get("timings") or {}).get("rep1", {}).get("wall_s")}

    # C2: the C1 run with more bursts (2 in the smoke), dev-only evaluation
    spec2 = c2_spec(cs, ["loop.steps=2", *ov], files)
    ctx2 = load_calib_context(cs, spec2)
    seed = training_seeds(cs, 0)[0]
    run_c2(ctx2, 0, seed)
    lab2 = c2_labels(cs, ctx2, base_spec)
    assert lab2["candidates"][0]["n_seeds_done"] == 1, "C2 label missing"
    summary["c2_label"] = lab2["candidates"][0]
    summary["c2_diag"] = diagnose(sgac_runs(group_dir(spec2)), base_reference(cs, base_spec), n_perm=10)["n_runs"]

    # OS: prereg one-shot trainer (2 rounds), then the SGAC dev evaluator
    cfg = os_config(rounds=2) if gpu else os_config(smoke_files=[REPO_ROOT / "configs" / "smoke_cpu.yaml"])
    spec_os = substrate_spec(cs, STAGE_OS, ov, files)
    ctx_os = load_calib_context(cs, spec_os)
    os_seed = os_seeds(0)[0]
    res = run_os(cs, ctx_os, 0, os_seed, cfg=cfg, spec=spec_os, check_gates=gpu, check_substrate=False)
    again = run_os(cs, ctx_os, 0, os_seed, cfg=cfg, spec=spec_os, check_gates=gpu, check_substrate=False)
    assert again == res, "re-running a finished OS run changed it"
    lab_os = os_labels(cs, ctx_os, cfg, base_spec)
    assert lab_os["candidates"][0]["n_seeds_done"] == 1, "OS label missing"
    summary["os_label"] = lab_os["candidates"][0]
    os_group = cs.results_dir / STAGE_OS / OS_STUDY / f"{cfg.run.tag}-{cfg.config_hash()}"
    summary["os_diag_runs"] = diagnose(os_runs(os_group), base_reference(cs, base_spec), n_perm=10)["n_runs"]
    if gpu:
        rounds = int(cfg.train.rounds)
        train_s = float((res.get("burst") or {}).get("wall_s") or 0.0)
        ev = __import__("json").loads((os_group / os_run_name(0, res["unique_id"], os_seed) / "eval" / "dev"
                                       / "summary.json").read_text(encoding="utf-8"))
        dev_full_s = float(ev.get("wall_s") or 0.0) * 486 / max(1, int(ev.get("n") or 64))
        summary["os_projection"] = {"train_s_per_round": train_s / rounds, "dev_eval_s_full_est": dev_full_s,
                                    "os_run_h_est": (100 * train_s / rounds + dev_full_s) / 3600,
                                    "note": "rough: 2-round timing scaled to 100 rounds; dev eval scaled by items"}
    atomic_write_json(Path(out_dir) / ("smoke_rungs_gpu.json" if gpu else "smoke_rungs.json"), summary)
    return summary
