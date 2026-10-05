"""E0 G2 manual cross-check (prereg section 7): our grader vs the Qwen2.5-Math grader on 200 base outputs,
"<= 2% disagreement, adjudicated".

Samples `--n` rows of the pinned-style G1 base eval (per_item.jsonl, which stores the generated text),
re-grades them with MathVerifyGrader (and checks the stored verdicts), grades them with Qwen2.5-Math's own
evaluation code in an isolated environment (scripts/crosscheck_qwen_side.py), and writes

  results/e0/g2_crosscheck/summary.json          counts, rates, categories
  results/e0/g2_crosscheck/items.jsonl           every sampled item with both verdicts
  results/e0/g2_crosscheck/disagreements.jsonl   the cases to adjudicate (adjudication field left empty)

Disagreement categories:
  rule_truncated   our pre-registered rule: truncated output => incorrect (Qwen grades the partial text)
  rule_unboxed     our pre-registered rule: no \\boxed{} => incorrect (Qwen falls back to "the answer is"/last number)
  extraction       Qwen extracts a different string from the text, but agrees with us when judging our extracted
                   boxed answer (e.g. its "final answer is $...$. I hope" branch keeps a literal \\boxed{...})
  equivalence      same extracted answer, different verdict: the case that tests equivalence judgement itself
The script reports; it does not decide G2. The decision follows adjudication of every disagreement.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rlvr_v2.artifacts import REPO_ROOT, atomic_write_json, read_json, read_jsonl  # noqa: E402
from rlvr_v2.config import load_config  # noqa: E402
from rlvr_v2.grader import MathVerifyGrader  # noqa: E402


def categorize(row: dict) -> str | None:
    if bool(row["ours_correct"]) == bool(row["qwen_correct"]):
        return None
    if row.get("truncated"):
        return "rule_truncated"
    if not row.get("our_boxed"):
        return "rule_unboxed"
    if row.get("qwen_on_our_box") is not None and bool(row["qwen_on_our_box"]) == bool(row["ours_correct"]):
        return "extraction"  # Qwen agrees on our extracted answer; it pulled a different string from the text
    return "equivalence"


def summarize(rows: list[dict], meta: dict) -> dict:
    n = len(rows)
    cats = Counter(r["category"] for r in rows if r["category"])
    n_dis = sum(cats.values())
    boxed = [r for r in rows if r.get("our_boxed") and not r.get("truncated")]
    eq_split = sum(1 for r in boxed if r.get("qwen_on_our_box") is not None
                   and bool(r["qwen_on_our_box"]) != bool(r["ours_correct"]))
    return {
        **meta,
        "n": n,
        "ours_correct": sum(bool(r["ours_correct"]) for r in rows),
        "qwen_correct": sum(bool(r["qwen_correct"]) for r in rows),
        "agree": n - n_dis,
        "disagree": n_dis,
        "raw_disagreement_rate": n_dis / n if n else float("nan"),
        "by_category": dict(cats),
        "equivalence_disagreement_rate": cats.get("equivalence", 0) / n if n else float("nan"),
        "same_answer_judgement": {
            "n_boxed_not_truncated": len(boxed),
            "disagree": eq_split,
            "note": "Qwen math_equal applied to OUR extracted boxed answer vs our verdict: isolates equivalence "
                    "judgement from answer-extraction rules",
        },
        "stored_vs_regraded_mismatches": sum(1 for r in rows if r["stored_correct"] != r["ours_correct"]),
        "qwen_timeouts_or_errors": sum(1 for r in rows if r.get("qwen_status") != "ok"),
        "criterion": "prereg G2: <= 2% disagreement, adjudicated. Every disagreement in disagreements.jsonl must be "
                     "adjudicated (which grader is right, and why) before G2's cross-check is recorded as passed.",
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qwen-python", required=True, help="python of the isolated Qwen-grader environment")
    ap.add_argument("--qwen-eval-dir", required=True, help="Qwen2.5-Math/evaluation at the pinned commit")
    ap.add_argument("--qwen-commit", default="unknown")
    ap.add_argument("--per-item", default=None, help="default: gates.json g1.base_per_item")
    ap.add_argument("--gates", default="results/e0/gates.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--out", default="results/e0/g2_crosscheck")
    a = ap.parse_args(argv)

    out = (REPO_ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_item = a.per_item or (read_json(REPO_ROOT / a.gates, {}) or {}).get("gates", {}).get("g1", {}).get("base_per_item")
    if not per_item:
        sys.exit("no --per-item given and gates.json has no g1.base_per_item")
    per_item = Path(per_item)
    rows = [r for r in read_jsonl(per_item) if r.get("text") is not None]
    if len(rows) < a.n:
        sys.exit(f"{per_item} has {len(rows)} rows with stored text; need {a.n} (eval.store_text must be true)")

    from rlvr_v2.cli import load_split_problems

    cfg = load_config([REPO_ROOT / "configs" / "base.yaml"])
    problems = {p.unique_id: p for p in load_split_problems(cfg, ("math500",))["math500"]}
    rows = sorted(rows, key=lambda r: r["unique_id"])
    sample = random.Random(a.seed).sample(rows, a.n)

    grader = MathVerifyGrader()
    merged, qwen_in = [], out / "qwen_input.jsonl"
    with open(qwen_in, "w", encoding="utf-8") as fh:
        for r in sample:
            p = problems[r["unique_id"]]
            g = grader.grade(r["text"], p.answer, truncated=bool(r.get("truncated")))
            merged.append({"unique_id": r["unique_id"], "level": p.level, "subject": p.subject, "gold": p.answer,
                           "truncated": bool(r.get("truncated")), "stored_correct": bool(r["correct"]),
                           "ours_correct": g.correct, "our_boxed": g.boxed, "our_method": g.method,
                           "text": r["text"]})
            fh.write(json.dumps({"unique_id": r["unique_id"], "text": r["text"], "answer": p.answer,
                                 "solution": p.solution or "", "our_boxed": g.boxed}, ensure_ascii=False) + "\n")

    qwen_out = out / "qwen_output.jsonl"
    cmd = [a.qwen_python, str(ROOT / "scripts" / "crosscheck_qwen_side.py"), "--eval-dir", a.qwen_eval_dir,
           "--inp", str(qwen_in), "--out", str(qwen_out)]
    print("running Qwen side:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    qwen = {r["unique_id"]: r for r in read_jsonl(qwen_out)}
    for m in merged:
        m.update({k: v for k, v in qwen[m["unique_id"]].items() if k != "unique_id"})
        m["category"] = categorize(m)

    with open(out / "items.jsonl", "w", encoding="utf-8") as fh:
        for m in merged:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(out / "disagreements.jsonl", "w", encoding="utf-8") as fh:
        for m in merged:
            if m["category"]:
                fh.write(json.dumps({**m, "adjudication": None}, ensure_ascii=False) + "\n")
    summary = summarize(merged, {"per_item": str(per_item), "seed": a.seed, "qwen_commit": a.qwen_commit,
                                 "grader_errors": dict(grader.errors)})
    atomic_write_json(out / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "criterion"}, indent=1))


if __name__ == "__main__":
    main()
