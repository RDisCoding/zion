"""`python -m rlvr_v2.sgac_calib <command>` -- the E0-calibrated SGAC selector track (paper/sgac_calib_protocol.md).

Preparation (CPU, committed outputs)
  seed-count            confirmatory seed count from the frozen reproduction's step-20 files -> manifests/seed_count.json
  build-manifests       data roles, candidate draw, clean-pool schedules -> configs/sgac_calib/manifests/; --check compares
  build-features        K=4 deployment-matched (and K=32) features of the 48 candidates from the prereg sieve
  check-features        verify the committed features against the sieve files present on this machine
  show-spec             the calibration spec, its hash and the substrate's SGAC spec hash
Stage A (audit)
  audit-phase1-signals  the four original Phase-1 candidates sieved with the E0 pipeline at K=32
Stage C (calibration labels, GPU)
  base-dev-eval         base model on the dev set (twice; determinism check)
  calib-run             one protocol-C1 run: --index I --seed S, or --pi1 --seed S (positive control)
  labels                collect ΔA labels of every finished calibration run
  gate-p                pilot gate G-P (label reliability); exit 3 if it fails
  require-gate-p        exit 3 unless G-P passed (guards the expansion)
Stage D (selector, CPU)
  stage-d               models, CV, baselines, coefficients, permutation test, gate G-S -> results .../selector/
  freeze-selector       refit the chosen model on all candidates; write and pin manifests/selector.json
Stage E (curriculum, GPU)
  require-gate-s        exit 3 unless G-S passed and the selector is pinned
  curriculum-base-eval  base model on test50 + MATH-500 (E0 G1 agreement check), this track's results root
  curriculum-run        one arm/seed of the 20-step curriculum
  analyze               confirmatory H1/H2 analysis -> results .../curriculum/confirmatory.json
Other
  smoke                 end-to-end smoke (CPU tiny model by default; --gpu = T0 timing projection)
  jobs                  print a stage's pre-declared queue
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ..artifacts import atomic_write_json, read_json
from .spec import load_calib

EXIT_GATE_FAILED = 3


def _cs(args):
    return load_calib(args.calib, args.override)


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, default=str))


# ---------------------------------------------------------------------- preparation
def cmd_seed_count(args) -> int:
    from .seedcount import run_seed_count

    cs = _cs(args)
    ids = read_json(cs.path("manifests/math500.json"))["ids"]
    out = run_seed_count(cs, args.dirs.split(","), ids)
    if not args.dry_run:
        atomic_write_json(cs.manifest_dir / "seed_count.json", out)
    _print({"chosen_seeds": out["chosen_seeds"], "chosen_mde_pp": 100 * out["chosen_mde"],
            "table_pp": {r["S"]: round(100 * r["mde"], 3) for r in out["table"]}})
    return 0


def cmd_build_manifests(args) -> int:
    from .curriculum import SCHEDULE_STEPS
    from .manifests import build_all, check_all

    cs = _cs(args)
    _print((check_all if args.check else build_all)(cs, args.sieve_signals, steps=SCHEDULE_STEPS))
    return 0


def cmd_build_features(args) -> int:
    from .features import build_features

    out = build_features(_cs(args), args.rollouts, args.signals)
    _print({"n": len(out["candidates"]), "sieve_rollouts_sha256": out["sieve_rollouts_sha256"]})
    return 0


def cmd_check_features(args) -> int:
    from .features import load_features
    from .manifests import sha256_file

    cs = _cs(args)
    doc = load_features(cs)
    rp = Path(args.rollouts) if args.rollouts else cs.path(cs.data.sieve_rollouts)
    sp = Path(args.signals) if args.signals else cs.path(cs.data.sieve_signals)
    got = {"rollouts": sha256_file(rp), "signals": sha256_file(sp)}
    ok = got["rollouts"] == doc["sieve_rollouts_sha256"] and got["signals"] == doc["sieve_signals_sha256"]
    _print({"features": str(cs.manifest_dir / "features.json"), "sieve_files_match": ok, **got})
    return 0 if ok else 1


def cmd_show_spec(args) -> int:
    from .spec import substrate_spec

    cs = _cs(args)
    sub = substrate_spec(cs, "calib")
    _print({"calib_hash": cs.calib_hash(), "substrate_spec_hash": sub.spec_hash(), **cs.to_dict()})
    return 0 if sub.spec_hash() == "577f3319" else 1


# ---------------------------------------------------------------------- stage A
def cmd_audit_phase1_signals(args) -> int:
    from .audit import run_phase1_signals

    _print(run_phase1_signals(_cs(args)))
    return 0


# ---------------------------------------------------------------------- stage C
def _calib_ctx(cs):
    from .calib import load_calib_context
    from .spec import require_e0_substrate, substrate_spec

    spec = substrate_spec(cs, "calib")
    require_e0_substrate(spec)
    return load_calib_context(cs, spec)


def cmd_base_dev_eval(args) -> int:
    from .calib import run_base_dev_eval

    res = run_base_dev_eval(_calib_ctx(_cs(args)))
    det = res.get("determinism") or {}
    _print({"repeats": res["repeats"], "determinism": det})
    return 0 if det.get("passed", True) else EXIT_GATE_FAILED


def cmd_calib_run(args) -> int:
    from .calib import run_calibration, training_seeds

    cs = _cs(args)
    index = None if args.pi1 else args.index
    if index is None and not args.pi1:
        raise SystemExit("give --index I or --pi1")
    if args.seed not in training_seeds(cs, index):
        raise SystemExit(f"seed {args.seed} is not a pre-declared training seed {training_seeds(cs, index)}")
    st = run_calibration(_calib_ctx(cs), index, args.seed)
    _print({"index": index, "seed": args.seed, "step_done": st["step_done"], "evals": st["evals"]})
    return 0


def cmd_labels(args) -> int:
    from .calib import collect_labels

    cs = _cs(args)
    labels = collect_labels(_calib_ctx(cs))
    atomic_write_json(cs.results_dir / "calib" / "labels.json", labels)
    _print({c["unique_id"]: (c["n_seeds_done"], c["delta_pp_mean"]) for c in labels["candidates"]})
    return 0


def cmd_gate_p(args) -> int:
    from .calib import write_gate_p

    res = write_gate_p(_calib_ctx(_cs(args)))
    _print({k: res.get(k) for k in ("status", "icc", "icc_ci95", "sd_between_pp", "mean_delta_pp", "incomplete")})
    return 0 if res.get("passed") else EXIT_GATE_FAILED


def _require(path: Path, what: str) -> int:
    doc = read_json(path)
    if not doc or doc.get("passed") is not True:
        print(f"{what} has not passed ({path}); refusing", file=sys.stderr)
        return EXIT_GATE_FAILED
    return 0


def cmd_require_gate_p(args) -> int:
    from .calib import GATE_P

    cs = _cs(args)
    return _require(cs.results_dir / "calib" / GATE_P, "gate G-P")


# ---------------------------------------------------------------------- stage D
def cmd_stage_d(args) -> int:
    import numpy as np

    from .manifests import sieve_signals
    from .selector import build_table, load_stage_inputs, pool_bin_frequencies, run_stage_d

    cs = _cs(args)
    feats, labels = load_stage_inputs(cs)
    t = build_table(feats, labels, cs.selector.features, require_seeds=len(cs.calibration.seed_offsets))
    freq = pool_bin_frequencies(cs, sieve_signals(cs))
    res = run_stage_d(cs, t, np.asarray(freq), with_permutation=not args.no_permutation)
    atomic_write_json(cs.results_dir / "selector" / "stage_d.json", res)
    _print({"model_choice": res["model_choice"], "gate_s": res["gate_s"],
            "models": {k: {m: v[m] for m in ("rho", "r2_oos", "gain_pp")} for k, v in res["models"].items()}})
    return 0 if res["gate_s"]["passed"] else EXIT_GATE_FAILED


def cmd_freeze_selector(args) -> int:
    import hashlib

    from .curriculum import SELECTOR_PIN
    from .selector import SELECTOR_JSON, build_table, freeze_selector, load_stage_inputs

    cs = _cs(args)
    stage_d = read_json(cs.results_dir / "selector" / "stage_d.json")
    if not stage_d or not stage_d["gate_s"]["passed"]:
        print("gate G-S has not passed; the selector is not frozen", file=sys.stderr)
        return EXIT_GATE_FAILED
    feats, labels = load_stage_inputs(cs)
    t = build_table(feats, labels, cs.selector.features, require_seeds=len(cs.calibration.seed_offsets))
    doc = freeze_selector(cs, t, stage_d, cs.manifest_dir)
    sha = hashlib.sha256((cs.manifest_dir / SELECTOR_JSON).read_bytes()).hexdigest()
    atomic_write_json(cs.manifest_dir / SELECTOR_PIN, {"file": SELECTOR_JSON, "sha256": sha, "model": doc["model"]})
    atomic_write_json(cs.results_dir / "selector" / "gate_s.json", stage_d["gate_s"])
    _print({"selector": str(cs.manifest_dir / SELECTOR_JSON), "sha256": sha, "model": doc["model"]})
    return 0


# ---------------------------------------------------------------------- stage E
def cmd_require_gate_s(args) -> int:
    from .curriculum import pinned_selector

    cs = _cs(args)
    rc = _require(cs.results_dir / "selector" / "gate_s.json", "gate G-S")
    if rc == 0:
        pinned_selector(cs)  # raises if the pin or the hash is wrong
    return rc


def _curr_ctx(cs):
    from .curriculum import STAGE, load_curriculum_context
    from .spec import require_e0_substrate, substrate_spec

    spec = substrate_spec(cs, STAGE)
    require_e0_substrate(spec)
    return load_curriculum_context(cs, spec)


def cmd_curriculum_base_eval(args) -> int:
    from ..sgac.jobs import Context, run_base_eval

    ctx = _curr_ctx(_cs(args))
    res = run_base_eval(ctx.spec, Context(spec=ctx.spec, test50=ctx.test50, math500=ctx.math500, grader=ctx.grader))
    agree = res.get("e0_g1_agreement") or {}
    _print({"test50": res["test50"]["acc"], "math500": (res.get("math500") or {}).get("acc"), "e0_g1": agree})
    return EXIT_GATE_FAILED if agree.get("passed") is False else 0


def cmd_curriculum_run(args) -> int:
    from .curriculum import run_curriculum_arm

    cs = _cs(args)
    steps = None
    if args.arm == "random_tm":
        tm = read_json(cs.results_dir / "t0" / "tm_steps.json")
        if tm is None:
            raise SystemExit("t0/tm_steps.json missing: run the GPU smoke (stage t0) first")
        steps = int(tm["tm_steps"])
    st = run_curriculum_arm(_curr_ctx(cs), args.arm, args.seed, steps)
    _print({"arm": args.arm, "seed": args.seed, "step_done": st["step_done"], "evals": st["evals"]})
    return 0


def cmd_analyze(args) -> int:
    from ..sgac.spec import group_dir
    from .analysis import confirmatory, load_arm
    from .jobs import confirmatory_seeds
    from .spec import substrate_spec

    cs = _cs(args)
    spec = substrate_spec(cs, "curriculum")
    ids = read_json(cs.path("manifests/math500.json"))["ids"]
    seeds = confirmatory_seeds(cs)
    arms = {a: load_arm(group_dir(spec), a, seeds, ids, step=int(spec.loop.steps))
            for a in ("learned", "random", "sgac")}
    res = confirmatory(cs, arms)
    atomic_write_json(cs.results_dir / "curriculum" / "confirmatory.json", res)
    _print(res)
    return 0


# ---------------------------------------------------------------------- other
def cmd_smoke(args) -> int:
    from .smoke import run_smoke

    s = run_smoke(gpu=args.gpu, out_dir=Path(args.out) if args.out else None, calib_path=args.calib,
                  overrides=args.override)
    _print({k: s.get(k) for k in ("calib_resume_check", "label", "curriculum", "timings", "projection", "tm_steps")})
    return 0


def cmd_jobs(args) -> int:
    from .jobs import stage_jobs

    for line in stage_jobs(_cs(args), args.stage):
        print(line)
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--calib", default=None, help="calibration YAML (default configs/sgac_calib/calib.yaml)")
    p.add_argument("--override", action="append", default=[], help="dotted override, e.g. data.sieve_rollouts=...")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m rlvr_v2.sgac_calib", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, func):
        s = sub.add_parser(name)
        _common(s)
        s.set_defaults(func=func)
        return s

    s = add("seed-count", cmd_seed_count)
    s.add_argument("--dirs", required=True, help="comma-separated dirs of <arm>_seed<S>.jsonl step-20 MATH-500 files")
    s.add_argument("--dry-run", action="store_true")
    s = add("build-manifests", cmd_build_manifests)
    s.add_argument("--check", action="store_true")
    s.add_argument("--sieve-signals", default=None, help="path of the K=32 sieve signals (default data.sieve_signals)")
    for name, func in (("build-features", cmd_build_features), ("check-features", cmd_check_features)):
        s = add(name, func)
        s.add_argument("--rollouts", default=None)
        s.add_argument("--signals", default=None)
    add("show-spec", cmd_show_spec)
    add("audit-phase1-signals", cmd_audit_phase1_signals)
    add("base-dev-eval", cmd_base_dev_eval)
    s = add("calib-run", cmd_calib_run)
    s.add_argument("--index", type=int, default=None)
    s.add_argument("--pi1", action="store_true")
    s.add_argument("--seed", type=int, required=True)
    add("labels", cmd_labels)
    add("gate-p", cmd_gate_p)
    add("require-gate-p", cmd_require_gate_p)
    s = add("stage-d", cmd_stage_d)
    s.add_argument("--no-permutation", action="store_true", help="skip the permutation test (G-S then undecided)")
    add("freeze-selector", cmd_freeze_selector)
    add("require-gate-s", cmd_require_gate_s)
    add("curriculum-base-eval", cmd_curriculum_base_eval)
    s = add("curriculum-run", cmd_curriculum_run)
    s.add_argument("--arm", required=True)
    s.add_argument("--seed", type=int, required=True)
    add("analyze", cmd_analyze)
    s = add("smoke", cmd_smoke)
    s.add_argument("--gpu", action="store_true")
    s.add_argument("--out", default=None, help="results root of the smoke (default <results_root>/t0_smoke)")
    s = add("jobs", cmd_jobs)
    s.add_argument("--stage", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return int(args.func(args) or 0)
