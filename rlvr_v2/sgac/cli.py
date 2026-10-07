"""`python -m rlvr_v2.sgac <command>` -- the SGAC reproduction entry point.

Commands
  probe-env        GPU / bitsandbytes / legacy sympy / grader / cache probe -> results_sgac/env_probe.json
  build-manifests  rebuild configs/sgac/manifests/{data.json, batches_seed*.json}; --check compares, never writes
  show-spec        print the resolved spec and its hash
  base-eval        base model on test50 (+ MATH-500) for a profile (e0: E0 G1 agreement gate; as_run: batch rule)
  pi1-eval         Wang et al.'s pi1 checkpoint with the profile's evaluation protocol
  run              one curriculum (profile, arm, seed), resumable
  eval-checkpoints evaluate saved step adapters later (e.g. MATH-500 at steps 5/10/15)
  phase1           optional Phase-1 replication (as_run)
  jobs             print the pre-declared queue of a tier (used by scripts/local/sgac_repro.sh)
  report           build reports/sgac_repro/ from everything under results_sgac/
  smoke            end-to-end smoke (CPU tiny model by default; --gpu for the real model)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .spec import PROFILES, load_profile

EXIT_GATE_FAILED = 3


def _common(p: argparse.ArgumentParser, profile: bool = True) -> None:
    if profile:
        p.add_argument("--profile", required=True, help=f"{PROFILES} or a YAML path")
    p.add_argument("--override", action="append", default=[], help="dotted override, e.g. loop.steps=2 (repeatable)")
    p.add_argument("--config", action="append", default=[], help="extra YAML merged after the profile (repeatable)")


def _spec(args):
    return load_profile(args.profile, args.override, args.config)


def cmd_probe_env(args) -> int:
    from .env_probe import probe

    out = probe(_spec(args))
    bad = out.get("math_verify_self_check") != "ok"
    return 1 if bad else 0


def cmd_build_manifests(args) -> int:
    from ..artifacts import read_json
    from ..grader import MathVerifyGrader
    from . import data as sdata
    from .schedule import candidate_schedule, load_schedule, write_schedule

    spec = _spec(args)
    if args.check:
        doc = sdata.load_data_manifest(spec)
        ds = sdata.load_split(spec)
        for section in sdata.SECTIONS:
            sdata.load_section(spec, section, ds, doc)  # raises on any id/text mismatch
        pool_rows = list(sdata.section_rows(spec, "pool"))
        for s in spec.run.seeds:
            load_schedule(sdata.manifest_dir(spec), s, pool_rows, spec.loop.steps, spec.loop.batch_b)
        print(json.dumps({"data_manifest": "ok", "schedules": list(spec.run.seeds), "n_rows": doc["n_rows"]}))
        return 0
    path = sdata.write_data_manifest(spec, MathVerifyGrader())
    doc = read_json(path)
    uid = {r["row"]: r["unique_id"] for r in doc["sections"]["pool"]["items"]}
    pool_rows = list(sdata.section_rows(spec, "pool"))
    for s in spec.run.seeds:
        write_schedule(sdata.manifest_dir(spec), s, pool_rows, spec.loop.steps, spec.loop.batch_b, uid)
    assert candidate_schedule(pool_rows, spec.run.seeds[0], spec.loop.steps, spec.loop.batch_b)
    print(json.dumps({"written": str(path), "schedules": list(spec.run.seeds),
                      "sections": {k: {"n": v["n"], "levels": v["level_counts"], "golds_differ": len(v["golds_differ"]),
                                       "gold_unparseable_mv": len(v.get("gold_unparseable_mv", []))}
                                   for k, v in doc["sections"].items()}}, indent=1))
    return 0


def cmd_show_spec(args) -> int:
    spec = _spec(args)
    print(json.dumps({"spec_hash": spec.spec_hash(), **spec.to_dict()}, indent=1, default=str))
    return 0


def cmd_base_eval(args) -> int:
    from .jobs import load_context, run_base_eval

    spec = _spec(args)
    res = run_base_eval(spec, load_context(spec, need_pool=False), args.e0_g1)
    agree = res.get("e0_g1_agreement")
    if agree is not None and agree.get("passed") is False:
        print(f"E0 G1 agreement gate FAILED: {agree}", file=sys.stderr)
        return EXIT_GATE_FAILED
    if agree is not None and agree.get("passed") is None:
        print(f"E0 G1 agreement could not be checked: {agree}", file=sys.stderr)
    return 0


def cmd_pi1_eval(args) -> int:
    from .jobs import load_context, run_pi1_eval

    spec = _spec(args)
    run_pi1_eval(spec, load_context(spec, need_pool=False))
    return 0


def cmd_run(args) -> int:
    from .jobs import load_context, run_curriculum

    spec = _spec(args)
    state = run_curriculum(spec, load_context(spec), args.arm, args.seed, base_after_run=args.base_after_run)
    print(json.dumps({"arm": args.arm, "seed": args.seed, "step_done": state["step_done"], "evals": state["evals"]},
                     default=str))
    return 0


def cmd_eval_checkpoints(args) -> int:
    from .jobs import load_context, run_eval_checkpoints

    spec = _spec(args)
    sets = [s for s in args.sets.split(",") if s]
    ctx = load_context(spec, need_pool=False, need_math500="math500" in sets)
    print(json.dumps(run_eval_checkpoints(spec, ctx, args.arm, args.seed, [int(x) for x in args.steps.split(",")], sets)))
    return 0


def cmd_phase1(args) -> int:
    from .phase1 import run_phase1

    run_phase1(_spec(args))
    return 0


def cmd_jobs(args) -> int:
    from .jobs import tier_jobs

    for line in tier_jobs(args.tier):
        print(line)
    return 0


def cmd_report(args) -> int:
    from .report import build_report

    specs = [load_profile(p, args.override, args.config) for p in args.profiles.split(",") if p]
    print(build_report(specs, out_dir=Path(args.out) if args.out else None))
    return 0


def cmd_smoke(args) -> int:
    from .smoke import run_smoke

    run_smoke(args.profile, gpu=args.gpu, out_dir=Path(args.out) if args.out else None, overrides=args.override)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m rlvr_v2.sgac", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("probe-env"); _common(s); s.set_defaults(func=cmd_probe_env)
    s = sub.add_parser("build-manifests"); _common(s); s.add_argument("--check", action="store_true")
    s.set_defaults(func=cmd_build_manifests)
    s = sub.add_parser("show-spec"); _common(s); s.set_defaults(func=cmd_show_spec)
    s = sub.add_parser("base-eval"); _common(s); s.add_argument("--e0-g1", default=None,
                                                                help="E0 G1 MATH-500 per_item.jsonl (default results/e0/...)")
    s.set_defaults(func=cmd_base_eval)
    s = sub.add_parser("pi1-eval"); _common(s); s.set_defaults(func=cmd_pi1_eval)
    s = sub.add_parser("run"); _common(s)
    s.add_argument("--arm", required=True); s.add_argument("--seed", type=int, required=True)
    s.add_argument("--base-after-run", action="store_true", help="also evaluate test50 with the adapter disabled at the end")
    s.set_defaults(func=cmd_run)
    s = sub.add_parser("eval-checkpoints"); _common(s)
    s.add_argument("--arm", required=True); s.add_argument("--seed", type=int, required=True)
    s.add_argument("--steps", default="5,10,15"); s.add_argument("--sets", default="math500")
    s.set_defaults(func=cmd_eval_checkpoints)
    s = sub.add_parser("phase1"); _common(s); s.set_defaults(func=cmd_phase1)
    s = sub.add_parser("jobs"); s.add_argument("--tier", type=int, required=True); s.set_defaults(func=cmd_jobs)
    s = sub.add_parser("report"); _common(s, profile=False)
    s.add_argument("--profiles", default="e0,as_run"); s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_report)
    s = sub.add_parser("smoke"); _common(s); s.add_argument("--gpu", action="store_true")
    s.add_argument("--out", default=None, help="CPU smoke output dir (default: a fresh temp dir)")
    s.set_defaults(func=cmd_smoke)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "smoke" and not args.gpu and not args.out:
        import tempfile

        args.out = tempfile.mkdtemp(prefix="sgac_smoke_")
    return int(args.func(args) or 0)
