"""Re-grade saved SGAC evaluation outputs with the legacy grader's SYMBOLIC path switched on (diagnostic only).

The repo pins antlr4-python3-runtime 4.13.2 (math-verify), so sympy's `parse_latex` raises inside the legacy grader
and every symbolic check degrades to whitespace-stripped string equality. Whether the Kaggle run had antlr4 4.11 is
unknown, so this script measures how much that matters: run it inside the isolated environment that has sympy 1.12 +
antlr4 4.11.1 (built by scripts/local/grader_crosscheck.sh):

    ~/envs/qwen_grader/bin/python scripts/sgac_legacy_regrade.py results_sgac

It loads rlvr_v2/sgac/legacy.py by file path (no rlvr_v2 import, so no torch/transformers needed), regrades every
per_item.jsonl found under the given root with a per-item timeout (the original code had none), and writes
per_item_legacy_symbolic.jsonl + legacy_symbolic_summary.json next to each file. Nothing else is modified.
"""
from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import sys
from pathlib import Path

LEGACY_PATH = Path(__file__).resolve().parent.parent / "rlvr_v2" / "sgac" / "legacy.py"
TIMEOUT_S = 10.0


def _load_legacy():
    spec = importlib.util.spec_from_file_location("sgac_legacy", LEGACY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verdict(args):
    text, gold = args
    legacy = _load_legacy()
    return bool(legacy.is_correct(legacy.extract_answer(text or ""), gold))


def regrade_file(path: Path, ctx) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out, n_timeout = [], 0
    pool = ctx.Pool(1)
    try:
        for r in rows:
            gold = r.get("gold_legacy") or r.get("answer") or ""
            try:
                ok = pool.apply_async(_verdict, ((r.get("text"), gold),)).get(timeout=TIMEOUT_S)
            except mp.TimeoutError:  # a hung simplify would block every later item: restart the worker
                ok, n_timeout = None, n_timeout + 1
                pool.terminate()
                pool = ctx.Pool(1)
            out.append({"unique_id": r["unique_id"], "legacy_symbolic_correct": ok,
                        "legacy_correct": r.get("legacy_correct"), "mv_correct": r.get("mv_correct")})
    finally:
        pool.terminate()
    dst = path.with_name("per_item_legacy_symbolic.jsonl")
    dst.write_text("".join(json.dumps(o) + "\n" for o in out), encoding="utf-8")
    graded = [o for o in out if o["legacy_symbolic_correct"] is not None]
    summary = {"n": len(out), "n_timeout": n_timeout,
               "acc_legacy_symbolic": sum(o["legacy_symbolic_correct"] for o in graded) / len(graded) if graded else None,
               "acc_legacy_string": sum(bool(o["legacy_correct"]) for o in out) / len(out) if out else None,
               "acc_mv": sum(bool(o["mv_correct"]) for o in out) / len(out) if out else None,
               "changed_vs_string": sum(1 for o in graded if bool(o["legacy_symbolic_correct"]) != bool(o["legacy_correct"]))}
    path.with_name("legacy_symbolic_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    legacy = _load_legacy()
    status = legacy.symbolic_status()
    print("legacy symbolic status:", status)
    if not status.get("available"):
        print("sympy's LaTeX parser is not usable in this environment (needs antlr4-python3-runtime==4.11); aborting")
        return 1
    ctx = mp.get_context("spawn")
    for f in sorted(Path(argv[1]).rglob("per_item.jsonl")):
        print(f, json.dumps(regrade_file(f, ctx)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
