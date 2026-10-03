#!/usr/bin/env python
"""Collect results/ into Parquet tables under results/tables/ (see `rlvr_v2.aggregate`).

    python scripts/aggregate.py --results results --base-per-item results/study1/<group>/_shared/eval_base/math500/per_item.jsonl
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlvr_v2.aggregate import aggregate  # noqa: E402


def _abs(p: str | Path | None) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--base-per-item", default=None,
                    help="base-model per_item.jsonl; default: results/study1/<group>/_shared/eval_base or results/e0/**")
    ap.add_argument("--signals", default=None, help="pool signals.jsonl (default <results>/pool/signals.jsonl)")
    ap.add_argument("--out", default=None, help="output directory (default <results>/tables)")
    ap.add_argument("--n-boot", type=int, default=2000, help="item-bootstrap resamples for per-run deltas")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s | %(message)s")
    tables = aggregate(_abs(args.results), _abs(args.base_per_item), _abs(args.signals), _abs(args.out),
                       n_boot=args.n_boot, seed=args.seed)
    for name, df in tables.items():
        print(f"{name:<14}{len(df):>8} rows  {len(df.columns):>3} cols")


if __name__ == "__main__":
    main()
