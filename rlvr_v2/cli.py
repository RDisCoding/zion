"""Command-line entry point.

    python -m rlvr_v2.cli <command> --config configs/study1.yaml [--override key=value ...]

`configs/base.yaml` is always merged first unless `--no-base` is given. Heavy modules are imported
lazily inside each command so `--help` and the pure-Python commands work without torch.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .artifacts import REPO_ROOT, RunDir, atomic_write_json, read_json
from .config import Config, load_config

log = logging.getLogger("rlvr_v2")


# ---------------------------------------------------------------------- helpers
def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", nargs="*", default=[], help="YAML files merged left to right after base.yaml")
    p.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. train.rounds=50")
    p.add_argument("--no-base", action="store_true", help="do not merge configs/base.yaml first")
    p.add_argument("-v", "--verbose", action="store_true")


def _cfg(args) -> Config:
    paths = [] if args.no_base else [REPO_ROOT / "configs" / "base.yaml"]
    paths += [_abs(c) for c in args.config]
    cfg = load_config(paths, args.override)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    return cfg


def _grader():
    from .grader import MathVerifyGrader

    return MathVerifyGrader()


def load_split_problems(cfg: Config, names: tuple[str, ...] = ("pool", "heldout", "math500")) -> dict[str, list]:
    """Load the datasets once and resolve the requested manifests (verifying text hashes)."""
    from .data import Manifest, load_math500, load_math_train, select_by_manifest

    out: dict[str, list] = {}
    train = math500 = None
    paths = {"pool": cfg.study1.pool_manifest, "heldout": cfg.study1.heldout_manifest, "math500": cfg.eval.manifest,
             "candidates": cfg.study1.candidates_manifest}
    for name in names:
        m = Manifest.load(_abs(paths[name]))
        if m.source == "math500":
            math500 = math500 or load_math500(cfg.data.eval_dataset)
            out[name] = select_by_manifest(m, math500)
        else:
            train = train or load_math_train(cfg.data.train_dataset)
            out[name] = select_by_manifest(m, train)
    return out


def _require_gates(cfg: Config) -> None:
    if not cfg.run.require_gates:
        return
    from .gates import gate_problems

    problems = gate_problems(read_json(REPO_ROOT / "results" / "e0" / "gates.json"))
    if problems:
        sys.exit("E0 gates do not authorise study jobs (" + "; ".join(problems) + "). Run the missing gates "
                 "(see RUNBOOK.md) or set run.require_gates=false for development.")


def _load_model_and_tokenizer(cfg: Config, adapter: str | None = None):
    from .modeling import load_adapter, load_base_model, load_tokenizer

    tok = load_tokenizer(cfg)
    model = load_base_model(cfg)
    if adapter:
        model = load_adapter(model, _abs(adapter))
    return model, tok


# ---------------------------------------------------------------------- commands
def cmd_manifests(args) -> None:
    from .data import load_math500, load_math_train, make_splits

    cfg = _cfg(args)
    train = load_math_train(cfg.data.train_dataset)
    m500 = load_math500(cfg.data.eval_dataset)
    manifests = make_splits(cfg.data, train, m500)
    out_dir = REPO_ROOT / "manifests"
    for name, m in manifests.items():
        m.save(out_dir / f"{name}.json")
        log.info("manifest %s: %d problems -> %s", name, len(m), out_dir / f"{name}.json")
    dups = manifests["pool"].meta.get("removed_near_duplicates", [])
    log.info("near-duplicates removed from the pool: %d", len(dups))


def cmd_sieve(args) -> None:
    from .sampling import make_sampler
    from .sieve import measure_signals

    cfg = _cfg(args)
    probs = load_split_problems(cfg, ("pool",))["pool"]
    if args.max_items:
        probs = probs[: args.max_items]
    model, tok = _load_model_and_tokenizer(cfg)
    sampler = make_sampler(cfg, model=model, tokenizer=tok)
    out = _abs(args.out)
    signals = measure_signals(sampler, tok, probs, cfg, _grader(), policy_tag="base", k=cfg.study1.k_sieve,
                              seed=cfg.run.seed, out_dir=out)
    log.info("sieved %d problems at K=%d -> %s", len(signals), cfg.study1.k_sieve, out)


def cmd_eval(args) -> None:
    from .evaluate import evaluate
    from .sampling import make_sampler

    cfg = _cfg(args)
    probs = load_split_problems(cfg, (args.items,))[args.items]
    limit = args.max_items or cfg.eval.max_items
    if limit:
        probs = probs[:limit]
    model, tok = _load_model_and_tokenizer(cfg, args.adapter)
    sampler = make_sampler(cfg, model=model, tokenizer=tok, adapter_path=args.adapter)
    summary = evaluate(sampler, tok, probs, cfg, _grader(), _abs(args.out) / args.items, tag=args.tag,
                       policy="adapter" if args.adapter else "base", adapter_path=args.adapter)
    log.info("eval %s on %s: acc=%.4f [%.4f, %.4f] n=%d", args.tag, args.items, summary.acc, summary.ci_lo,
             summary.ci_hi, summary.n)


def cmd_study1_jobs(args) -> None:
    from .data import Manifest
    from .study1 import expand_jobs, write_jobs_csv

    cfg = _cfg(args)
    jobs = expand_jobs(cfg, Manifest.load(_abs(cfg.study1.candidates_manifest)))
    write_jobs_csv(jobs, _abs(args.out))
    log.info("%d Study-1 jobs (%d replicates) -> %s", len(jobs), sum(j.replicate for j in jobs), args.out)


def cmd_study1(args) -> None:
    from .data import Manifest
    from .study1 import expand_jobs, run_job

    cfg = _cfg(args)
    _require_gates(cfg)
    jobs = expand_jobs(cfg, Manifest.load(_abs(cfg.study1.candidates_manifest)))
    if args.job_index < 0 or args.job_index >= len(jobs):
        sys.exit(f"--job-index {args.job_index} out of range (0..{len(jobs) - 1})")
    job = jobs[args.job_index]
    splits = load_split_problems(cfg, ("pool", "math500", "heldout"))
    stages = tuple(args.stages.split(","))
    result = run_job(cfg, job, splits["pool"], splits["math500"], splits["heldout"] if args.heldout else None,
                     _grader(), stages=stages)
    log.info("Study-1 job %d (%s, seed %d) done: %s", job.index, job.unique_id, job.seed,
             json.dumps({k: v for k, v in result.items() if not isinstance(v, (dict, list))}, default=str)[:500])


def cmd_curriculum(args) -> None:
    from .curriculum import Curriculum, expand_study2_jobs
    from .data import load_pi1

    cfg = _cfg(args)
    _require_gates(cfg)
    if args.job_index is not None:
        jobs = expand_study2_jobs(cfg)
        if args.job_index < 0 or args.job_index >= len(jobs):
            sys.exit(f"--job-index {args.job_index} out of range (0..{len(jobs) - 1})")
        arm, seed = jobs[args.job_index]["arm"], int(jobs[args.job_index]["seed"])
    else:
        if args.arm is None or args.seed is None:
            sys.exit("give --job-index or both --arm and --seed")
        arm, seed = args.arm, int(args.seed)
    splits = load_split_problems(cfg, ("pool", "math500"))
    repeat = None
    if cfg.study2.repeat_one_uid:
        if cfg.study2.repeat_one_uid == "oneshot/pi1":
            repeat = load_pi1()
        else:
            repeat = next((p for p in splits["pool"] if p.unique_id == cfg.study2.repeat_one_uid), None)
    run_dir = RunDir(cfg, "study2", f"{arm}__seed{seed}")
    cur = Curriculum(cfg, arm, seed, run_dir, splits["pool"], splits["math500"], _grader(), repeat_problem=repeat)
    cur.run()
    log.info("curriculum %s seed %d finished -> %s", arm, seed, run_dir.path)


def cmd_gates(args) -> None:
    from .gates import run_gates

    cfg = _cfg(args)
    report = run_gates(cfg, gates=tuple(args.gates.split(",")), out=_abs(args.out), extra=vars(args))
    passed = report.get("all_passed")
    log.info("E0 gates all_passed=%s -> %s", passed, args.out)
    if passed is False:
        sys.exit(3)


def cmd_aggregate(args) -> None:
    from .aggregate import aggregate

    cfg = _cfg(args)
    tables = aggregate(_abs(args.results), _abs(args.base_per_item) if args.base_per_item else None)
    for name, df in tables.items():
        log.info("table %s: %d rows", name, len(df))


def cmd_smoke(args) -> None:
    from .smoke import run_smoke

    cfg = _cfg(args)
    report = run_smoke(cfg, _abs(args.out))
    atomic_write_json(_abs(args.out) / "smoke_report.json", report)
    log.info("smoke test ok: %s", json.dumps(report, default=str)[:800])


def cmd_show_config(args) -> None:
    cfg = _cfg(args)
    print(json.dumps(cfg.to_dict(), indent=1, default=str))
    print("config_hash:", cfg.config_hash())


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rlvr_v2", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("manifests", help="build pool/heldout/math500 manifests (downloads datasets)")
    _add_common(s)
    s.set_defaults(func=cmd_manifests)

    s = sub.add_parser("sieve", help="measure base-policy signals for the pool at K=study1.k_sieve")
    _add_common(s)
    s.add_argument("--out", default="results/pool")
    s.add_argument("--max-items", type=int, default=None)
    s.set_defaults(func=cmd_sieve)

    s = sub.add_parser("eval", help="evaluate the base model or an adapter on a manifest")
    _add_common(s)
    s.add_argument("--items", choices=["math500", "heldout"], default="math500")
    s.add_argument("--adapter", default=None)
    s.add_argument("--tag", default="base")
    s.add_argument("--out", default="results/eval")
    s.add_argument("--max-items", type=int, default=None)
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("study1-jobs", help="expand Study-1 (candidate, seed) jobs to a CSV")
    _add_common(s)
    s.add_argument("--out", default="manifests/study1_jobs.csv")
    s.set_defaults(func=cmd_study1_jobs)

    s = sub.add_parser("study1", help="run one Study-1 job (train + eval), resumable")
    _add_common(s)
    s.add_argument("--job-index", type=int, required=True)
    s.add_argument("--stages", default="train,eval")
    s.add_argument("--heldout", action="store_true", help="also evaluate on the held-out train slice")
    s.set_defaults(func=cmd_study1)

    s = sub.add_parser("curriculum", help="run one Study-2 arm x seed, resumable")
    _add_common(s)
    s.add_argument("--job-index", type=int, default=None)
    s.add_argument("--arm", default=None)
    s.add_argument("--seed", type=int, default=None)
    s.set_defaults(func=cmd_curriculum)

    s = sub.add_parser("gates", help="run E0 infrastructure gates")
    _add_common(s)
    s.add_argument("--gates", default="g1,g2,g3,g5,g4")
    s.add_argument("--out", default="results/e0/gates.json")
    s.add_argument("--n-items", type=int, default=None, help="limit eval items for quick dry runs")
    s.set_defaults(func=cmd_gates)

    s = sub.add_parser("aggregate", help="collect results/ into Parquet tables")
    _add_common(s)
    s.add_argument("--results", default="results")
    s.add_argument("--base-per-item", default=None)
    s.set_defaults(func=cmd_aggregate)

    s = sub.add_parser("smoke", help="end-to-end CPU smoke test with a tiny model")
    _add_common(s)
    s.add_argument("--out", default="results/smoke")
    s.set_defaults(func=cmd_smoke)

    s = sub.add_parser("show-config", help="print the merged config and its hash")
    _add_common(s)
    s.set_defaults(func=cmd_show_config)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
