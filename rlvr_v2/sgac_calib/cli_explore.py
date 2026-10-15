"""Commands added after gate G-P failed on the C1 pilot (protocol section 13, amendments A1-A5).

C2 rung (confirmatory, pre-declared)
  c2-run                one C2 run (80 bursts): --index I --seed S, or --pi1 --seed S
  c2-labels-gate        labels + gate G-P at C2 -> results .../calib_c2/{labels,gate}.json (informational exit 0)
One-shot regime OS (exploratory)
  os-run                one OS run (prereg Study-1 trainer, hash 84e100e2) + dev evaluation: --index I / --pi1, --seed S
  os-labels-gate        labels + the G-P thresholds on the OS pilot -> results .../oneshot/{labels,gate}.json (exit 0)
  require-os-gate       exit 3 unless the OS pilot met the G-P thresholds (guards the OS expansion)
Exploratory fits and diagnostics (CPU)
  fit-explore           the Stage-D pipeline on a rung's labels: --source c1|c2|os --tag T [--freeze]
  diagnose              run- and item-level diagnostics of a rung: --source c1|c2|os
Equal-compute comparison (amendment A6; the only GPU work after G-P)
  cb-schedules          write (or --check) the 160-step clean-pool schedules of seeds 42-49
  cb-smoke              timing/plumbing check (--gpu on the workstation); never an accuracy result
  cb-base-eval          base model on MATH-500 + test50 for this experiment (E0 G1 agreement check)
  cb-run                --strategy sgac|random --seed S (random needs the same seed's sgac run first)
  cb-analyze            analysis, CSV, plots -> reports/sgac_calib/compute_budget_comparison.md
Exploratory curriculum (GPU; withdrawn by amendment A6)
  explore-curriculum-base-eval   base model on test50 + MATH-500 for the exploratory curriculum root
  explore-curriculum-run         one arm/seed: --arm learned|random|sgac|max_var|max_d|max_level --seed S [--tag T]
  explore-analyze                H1/H2 analysis for an exploratory selector: --tag T
"""
from __future__ import annotations

import json
from pathlib import Path

from ..artifacts import atomic_write_json, read_json
from .spec import load_calib

EXIT_GATE_FAILED = 3
SOURCES = {"c1": "calib", "c2": "calib_c2", "os": "oneshot"}


def _cs(args):
    return load_calib(args.calib, args.override)


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, default=str))


def _index(args):
    if args.pi1:
        return None
    if args.index is None:
        raise SystemExit("give --index I or --pi1")
    return args.index


# ---------------------------------------------------------------------- C2
def cmd_c2_run(args) -> int:
    from .calib import training_seeds
    from .rungs import c2_context, run_c2

    cs = _cs(args)
    index = _index(args)
    if args.seed not in training_seeds(cs, index):
        raise SystemExit(f"seed {args.seed} is not a pre-declared seed {training_seeds(cs, index)}")
    st = run_c2(c2_context(cs), index, args.seed)
    _print({"rung": "C2", "index": index, "seed": args.seed, "step_done": st["step_done"], "evals": st["evals"]})
    return 0


def cmd_c2_labels_gate(args) -> int:
    from .rungs import STAGE_C2, c2_context, c2_labels, write_labels_and_gate

    cs = _cs(args)
    res = write_labels_and_gate(cs, STAGE_C2, c2_labels(cs, c2_context(cs)))
    _print({k: res.get(k) for k in ("source", "status", "icc", "icc_ci95", "sd_between_pp", "mean_delta_pp")})
    return 0


# ---------------------------------------------------------------------- OS
def _os_ctx(cs):
    from .calib import load_calib_context
    from .rungs import STAGE_OS
    from .spec import require_e0_substrate, substrate_spec

    spec = substrate_spec(cs, STAGE_OS)
    require_e0_substrate(spec)
    return load_calib_context(cs, spec)


def cmd_os_run(args) -> int:
    from .rungs import os_seeds, run_os

    cs = _cs(args)
    index = _index(args)
    if args.seed not in os_seeds(index):
        raise SystemExit(f"seed {args.seed} is not a pre-declared OS seed {os_seeds(index)}")
    ctx = _os_ctx(cs)
    _print(run_os(cs, ctx, index, args.seed, spec=ctx.spec))
    return 0


def cmd_os_labels_gate(args) -> int:
    from .rungs import STAGE_OS, os_labels, write_labels_and_gate

    cs = _cs(args)
    res = write_labels_and_gate(cs, STAGE_OS, os_labels(cs, _os_ctx(cs)))
    _print({k: res.get(k) for k in ("source", "status", "icc", "icc_ci95", "sd_between_pp", "mean_delta_pp", "pi1")})
    return 0


def cmd_require_os_gate(args) -> int:
    cs = _cs(args)
    doc = read_json(cs.results_dir / "oneshot" / "gate.json")
    if not doc or doc.get("passed") is not True:
        print("the OS pilot did not meet the G-P thresholds; the OS expansion does not run", flush=True)
        return EXIT_GATE_FAILED
    return 0


# ---------------------------------------------------------------------- fits and diagnostics
def cmd_fit_explore(args) -> int:
    from .explore import load_labels, run_explore_fit, summary_line

    cs = _cs(args)
    labels = load_labels(cs.results_dir / SOURCES[args.source] / "labels.json")
    eq = run_explore_fit(cs, labels, args.tag, min_seeds=args.min_seeds, with_permutation=not args.no_permutation,
                         freeze=args.freeze)
    print(summary_line(eq))
    return 0


def cmd_diagnose(args) -> int:
    from ..sgac.spec import group_dir
    from .diagnose import diagnose, os_runs, sgac_runs
    from .rungs import OS_STUDY, STAGE_OS, base_reference, c2_spec, os_config
    from .spec import substrate_spec

    cs = _cs(args)
    if args.source == "c1":
        runs = sgac_runs(group_dir(substrate_spec(cs, "calib")))
    elif args.source == "c2":
        runs = sgac_runs(group_dir(c2_spec(cs)))
    else:
        cfg = os_config()
        runs = os_runs(cs.results_dir / STAGE_OS / OS_STUDY / f"{cfg.run.tag}-{cfg.config_hash()}")
    res = diagnose(runs, base_reference(cs), n_perm=args.n_perm)
    atomic_write_json(cs.results_dir / "explore" / f"diagnostics_{args.source}.json", res)
    _print({k: v for k, v in res.items() if k != "table"})
    return 0


# ---------------------------------------------------------------------- exploratory curriculum
def cmd_explore_curriculum_base_eval(args) -> int:
    from ..sgac.jobs import Context, run_base_eval
    from .curriculum import load_curriculum_context
    from .explore_curriculum import explore_spec

    cs = _cs(args)
    ctx = load_curriculum_context(cs, explore_spec(cs))
    res = run_base_eval(ctx.spec, Context(spec=ctx.spec, test50=ctx.test50, math500=ctx.math500, grader=ctx.grader))
    agree = res.get("e0_g1_agreement") or {}
    _print({"test50": res["test50"]["acc"], "math500": (res.get("math500") or {}).get("acc"), "e0_g1": agree})
    return EXIT_GATE_FAILED if agree.get("passed") is False else 0


def cmd_explore_curriculum_run(args) -> int:
    from .explore_curriculum import run_explore_arm

    st = run_explore_arm(_cs(args), args.arm, args.seed, args.tag)
    _print({"arm": args.arm, "tag": args.tag, "seed": args.seed, "step_done": st["step_done"], "evals": st["evals"]})
    return 0


def cmd_explore_analyze(args) -> int:
    from .explore_curriculum import analyze_explore

    _print(analyze_explore(_cs(args), args.tag))
    return 0


def cmd_report(args) -> int:
    from .report import build_report

    print(build_report(_cs(args)))
    return 0


def cmd_smoke_rungs(args) -> int:
    from .smoke_rungs import run_rungs_smoke

    cs = _cs(args)
    out = Path(args.out) if args.out else cs.results_dir / ("t0_rungs" if args.gpu else "t0_rungs_cpu")
    _print(run_rungs_smoke(out, gpu=args.gpu))
    return 0


# ---------------------------------------------------------------------- equal-compute comparison (amendment A6)
def cmd_cb_schedules(args) -> int:
    from .compute_budget import check_schedules, load_cb, write_schedules

    cs, cb = _cs(args), load_cb()
    if args.check:
        _print({"schedules_ok": check_schedules(cs, cb)})
    else:
        _print({"written": [str(p) for p in write_schedules(cs, cb)], "check": check_schedules(cs, cb)})
    return 0


def cmd_cb_smoke(args) -> int:
    from .compute_budget import run_cb_smoke

    cs = _cs(args)
    out = Path(args.out) if args.out else cs.results_dir / ("t0_cb" if args.gpu else "t0_cb_cpu")
    s = run_cb_smoke(out, gpu=args.gpu)
    _print({k: v for k, v in s.items() if k != "analysis_rows"})
    return 0


def cmd_cb_base_eval(args) -> int:
    from ..sgac.jobs import Context, run_base_eval
    from .compute_budget import _ctx, load_cb, spec_a

    cs, cb = _cs(args), load_cb()
    sa = spec_a(cs, cb)
    ctx = _ctx(cs, sa)
    res = run_base_eval(sa, Context(spec=sa, test50=ctx.test50, math500=ctx.math500, grader=ctx.grader))
    agree = res.get("e0_g1_agreement") or {}
    _print({"test50": res["test50"]["acc"], "math500": (res.get("math500") or {}).get("acc"), "e0_g1": agree})
    return EXIT_GATE_FAILED if agree.get("passed") is False else 0


def cmd_cb_run(args) -> int:
    from .compute_budget import load_cb, run_strategy

    cs, cb = _cs(args), load_cb()
    if args.seed not in cb.seeds:
        raise SystemExit(f"seed {args.seed} is not one of the pre-declared seeds {list(cb.seeds)}")
    st = run_strategy(cs, cb, args.strategy, args.seed)
    _print({"strategy": args.strategy, "seed": args.seed, "step_done": st["step_done"],
            "budget_stop": st.get("budget_stop"), "evals": st["evals"]})
    return 0


def cmd_cb_analyze(args) -> int:
    from .compute_budget import analyze, load_cb

    res = analyze(_cs(args), load_cb())
    _print({k: res.get(k) for k in ("seeds_complete", "primary", "interpretation", "compute")})
    return 0


def register(add) -> None:
    add("report", cmd_report)
    s = add("cb-schedules", cmd_cb_schedules)
    s.add_argument("--check", action="store_true")
    s = add("cb-smoke", cmd_cb_smoke)
    s.add_argument("--gpu", action="store_true")
    s.add_argument("--out", default=None)
    add("cb-base-eval", cmd_cb_base_eval)
    s = add("cb-run", cmd_cb_run)
    s.add_argument("--strategy", required=True, choices=["sgac", "random"])
    s.add_argument("--seed", type=int, required=True)
    add("cb-analyze", cmd_cb_analyze)
    s = add("smoke-rungs", cmd_smoke_rungs)
    s.add_argument("--gpu", action="store_true")
    s.add_argument("--out", default=None)
    for name, func in (("c2-run", cmd_c2_run), ("os-run", cmd_os_run)):
        s = add(name, func)
        s.add_argument("--index", type=int, default=None)
        s.add_argument("--pi1", action="store_true")
        s.add_argument("--seed", type=int, required=True)
    add("c2-labels-gate", cmd_c2_labels_gate)
    add("os-labels-gate", cmd_os_labels_gate)
    add("require-os-gate", cmd_require_os_gate)
    s = add("fit-explore", cmd_fit_explore)
    s.add_argument("--source", required=True, choices=sorted(SOURCES))
    s.add_argument("--tag", required=True)
    s.add_argument("--min-seeds", type=int, default=2)
    s.add_argument("--freeze", action="store_true")
    s.add_argument("--no-permutation", action="store_true")
    s = add("diagnose", cmd_diagnose)
    s.add_argument("--source", required=True, choices=sorted(SOURCES))
    s.add_argument("--n-perm", type=int, default=2000)
    add("explore-curriculum-base-eval", cmd_explore_curriculum_base_eval)
    s = add("explore-curriculum-run", cmd_explore_curriculum_run)
    s.add_argument("--arm", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--tag", default=None)
    s = add("explore-analyze", cmd_explore_analyze)
    s.add_argument("--tag", required=True)
