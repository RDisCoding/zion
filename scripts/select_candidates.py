#!/usr/bin/env python
"""Select the pre-registered Study-1 candidates from pool signals and write their manifest.

    python scripts/select_candidates.py --signals results/pool/signals.jsonl \
        --pool-manifest manifests/pool.json --config configs/study1.yaml \
        --out manifests/study1_candidates.json

`configs/base.yaml` is merged first unless `--no-base` is given (same convention as `rlvr_v2.cli`).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlvr_v2.artifacts import read_jsonl  # noqa: E402
from rlvr_v2.candidates import candidates_manifest, select_candidates, summarize_selection  # noqa: E402
from rlvr_v2.config import load_config  # noqa: E402
from rlvr_v2.data import Manifest  # noqa: E402
from rlvr_v2.signals import Signals  # noqa: E402

log = logging.getLogger("select_candidates")


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_signals(path: Path, policy_tag: str | None = "base") -> list[Signals]:
    rows = read_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"no signal records found in {path}")
    if policy_tag:
        rows = [r for r in rows if r.get("policy_tag", policy_tag) == policy_tag]
    return [Signals.from_dict(r) for r in rows]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signals", default="results/pool/signals.jsonl")
    ap.add_argument("--pool-manifest", default="manifests/pool.json")
    ap.add_argument("--config", nargs="*", default=["configs/study1.yaml"])
    ap.add_argument("--override", nargs="*", action="extend", default=[])
    ap.add_argument("--no-base", action="store_true", help="do not merge configs/base.yaml first")
    ap.add_argument("--out", default=None, help="defaults to study1.candidates_manifest from the config")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for tie-breaks (default run.seed)")
    ap.add_argument("--policy-tag", default="base", help="keep only signals with this policy_tag ('' = all)")
    ap.add_argument("--ineligible", default="manifests/pool_ineligible.json",
                    help="committed list of pool items with unparseable gold answers (prereg §5)")
    ap.add_argument("--skip-recheck", action="store_true",
                    help="tests only: do not recompute the ineligibility rule from the dataset")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s | %(message)s")

    paths = [] if args.no_base else [p for p in [ROOT / "configs" / "base.yaml"] if p.exists()]
    paths += [_abs(c) for c in args.config]
    cfg = load_config(paths, args.override)
    signals = load_signals(_abs(args.signals), args.policy_tag or None)
    pool = Manifest.load(_abs(args.pool_manifest))
    unknown = sorted({s.unique_id for s in signals} - set(pool.ids))
    if unknown:
        log.warning("%d signal records are not in the pool manifest and are ignored, e.g. %s", len(unknown), unknown[:3])
        signals = [s for s in signals if s.unique_id in pool.hashes]
    from rlvr_v2.cli import load_pool_ineligible

    ineligible = load_pool_ineligible(cfg, _abs(args.ineligible), recheck=not args.skip_recheck)
    seed = cfg.run.seed if args.seed is None else args.seed
    selection = select_candidates(signals, cfg.study1, rng_seed=seed, ineligible=ineligible)
    print(summarize_selection(selection))
    manifest = candidates_manifest(selection, pool)
    out = _abs(args.out or cfg.study1.candidates_manifest)
    manifest.save(out)
    log.info("wrote %d candidates (%d replicates) -> %s", len(manifest), len(selection["replicate_ids"]), out)


if __name__ == "__main__":
    main()
