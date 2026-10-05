#!/bin/bash
# E0 G2 manual cross-check: our grader vs the Qwen2.5-Math grader on 200 G1 base outputs (prereg section 7).
#   scripts/local/grader_crosscheck.sh
# Builds (once) an ISOLATED Qwen-grader environment: Qwen's evaluation code pins sympy 1.12 and antlr4 4.11.1,
# which must never be installed into the rlvr_v2 environment (math-verify needs antlr4 4.13.2). Needs internet once.
# Output: results/e0/g2_crosscheck/{summary.json,items.jsonl,disagreements.jsonl}. CPU only, a few minutes.
source "$(dirname "$0")/common.sh"
QWEN_COMMIT="a45202bd16f1ec06f433442dc1152d0074773465"
QWEN_DIR="${QWEN_DIR:-$HOME/qwen25_math_eval}"
QWEN_ENV="${QWEN_ENV:-$HOME/envs/qwen_grader}"
test -f results/e0/gates.json || { echo "results/e0/gates.json missing: run the E0 gates first"; exit 2; }
if [[ ! -d "$QWEN_DIR/.git" ]]; then
  git clone -q https://github.com/QwenLM/Qwen2.5-Math.git "$QWEN_DIR"
fi
git -C "$QWEN_DIR" fetch -q origin "$QWEN_COMMIT" 2>/dev/null || true
git -C "$QWEN_DIR" checkout -q "$QWEN_COMMIT"
if [[ ! -x "$QWEN_ENV/bin/python" ]]; then
  log "creating isolated Qwen-grader environment at $QWEN_ENV"
  python -m venv "$QWEN_ENV"
  "$QWEN_ENV/bin/pip" install -q --upgrade pip
  "$QWEN_ENV/bin/pip" install -q "sympy==1.12" "antlr4-python3-runtime==4.11.1" regex word2number pebble timeout-decorator numpy
  "$QWEN_ENV/bin/pip" install -q -e "$QWEN_DIR/evaluation/latex2sympy"
fi
QWEN_EVAL_DIR="$QWEN_DIR/evaluation" "$QWEN_ENV/bin/python" -c "import os, sys; sys.path.insert(0, os.environ['QWEN_EVAL_DIR']); import sympy, parser, grader; print('qwen grader env ok: sympy', sympy.__version__)"
run_logged grader_crosscheck python scripts/grader_crosscheck.py --qwen-python "$QWEN_ENV/bin/python" \
  --qwen-eval-dir "$QWEN_DIR/evaluation" --qwen-commit "$QWEN_COMMIT" "$@"
log "done: results/e0/g2_crosscheck/summary.json and disagreements.jsonl (send both)"
