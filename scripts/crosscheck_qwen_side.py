"""Qwen2.5-Math reference grading for the E0 G2 cross-check (prereg section 7).

Runs in an ISOLATED environment with Qwen's own evaluation dependencies (sympy 1.12, antlr4 4.11.1, Qwen's bundled
latex2sympy2), never in the rlvr_v2 environment, whose math-verify stack pins different versions. Mirrors
QwenLM/Qwen2.5-Math evaluation/evaluate.py: ground truth via parse_ground_truth, prediction via extract_answer,
judgement via math_equal_process in a pebble ProcessPool with a 3 s timeout (timeouts count as incorrect).

Input JSONL rows: unique_id, text, answer, solution, our_boxed. Output JSONL rows add: qwen_gt, qwen_pred,
qwen_correct, qwen_on_our_box (math_equal(strip_string(our_boxed), qwen_gt), isolating the equivalence judgement
from answer extraction), qwen_status / qwen_on_our_box_status (ok | timeout | error).
"""
import argparse
import json
import os
import sys

if os.environ.get("QWEN_EVAL_DIR"):  # also needed in spawned pool workers
    sys.path.insert(0, os.environ["QWEN_EVAL_DIR"])


def judge_all(params, timeout):
    from concurrent.futures import TimeoutError

    from grader import math_equal_process
    from pebble import ProcessPool

    results, statuses = [], []
    with ProcessPool(max_workers=1) as pool:
        it = pool.map(math_equal_process, params, timeout=timeout).result()
        while True:
            try:
                results.append(bool(next(it)))
                statuses.append("ok")
            except StopIteration:
                break
            except TimeoutError:
                results.append(False)
                statuses.append("timeout")
            except Exception as e:  # noqa: BLE001 - recorded, counted as incorrect like a timeout
                results.append(False)
                statuses.append(f"error: {e!r}"[:200])
    return results, statuses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="Qwen2.5-Math/evaluation directory")
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-name", default="math")
    ap.add_argument("--timeout", type=float, default=3.0)
    a = ap.parse_args()
    os.environ["QWEN_EVAL_DIR"] = os.path.abspath(a.eval_dir)
    if os.environ["QWEN_EVAL_DIR"] not in sys.path:
        sys.path.insert(0, os.environ["QWEN_EVAL_DIR"])
    from parser import extract_answer, parse_ground_truth, strip_string

    with open(a.inp, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    gts, preds, ours = [], [], []
    for r in rows:
        _, gt = parse_ground_truth({"solution": r["solution"], "answer": r["answer"]}, a.data_name)
        gts.append(gt)
        preds.append(extract_answer(r["text"] or "", a.data_name))
        ours.append(strip_string(r["our_boxed"]) if r.get("our_boxed") else "")
    full, full_status = judge_all([(i, preds[i], gts[i]) for i in range(len(rows))], a.timeout)
    on_box, on_box_status = judge_all([(i, ours[i], gts[i]) for i in range(len(rows))], a.timeout)
    with open(a.out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps({
                "unique_id": r["unique_id"], "qwen_gt": gts[i], "qwen_pred": preds[i],
                "qwen_correct": full[i], "qwen_status": full_status[i],
                "qwen_on_our_box": on_box[i] if ours[i] else None,
                "qwen_on_our_box_status": on_box_status[i] if ours[i] else None,
            }, ensure_ascii=False) + "\n")
    print(f"qwen side: {len(rows)} items, {sum(full)} correct, "
          f"{sum(s != 'ok' for s in full_status)} timeouts/errors")


if __name__ == "__main__":
    main()
