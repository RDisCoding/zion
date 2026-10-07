"""Verbatim port of the original SGAC notebook's answer extraction, rewards and correctness check.

Source: `ghoul/rl-llm-phase-5-the-loop-modified-results.ipynb` ("NB-M"), cell 3 (notebook sha256 in
`nbm_original.NBM_SHA256`). The six functions below are the notebook's, unchanged; tests/test_sgac_legacy.py
compares their syntax trees with the frozen copy in tests/data/sgac_nbm_cell3.py.txt. Do not "fix" anything in
them: FIRST-box extraction, the last-number fallback, whitespace-only normalisation and the swallowed sympy
exceptions ARE the behaviour being reproduced.

The module-level `sympy` / `parse_latex` imports mirror NB-M cell 2 (`is_correct` uses them). Whether sympy's LaTeX
parser works depends on antlr4-python3-runtime==4.11; with the repo's pinned 4.13.2 it raises ImportError, every
symbolic check is swallowed and grading degrades to whitespace-stripped string equality (as on the old cluster).
`symbolic_status()` records which case applies.

Import-light on purpose (re, numpy, sympy only): scripts/sgac_legacy_regrade.py loads this file by path inside the
isolated `~/envs/qwen_grader` environment (sympy 1.12 + antlr4 4.11.1) to regrade outputs with the symbolic path on.
"""
import re

import numpy as np
import sympy
from sympy.parsing.latex import parse_latex

# NB-M cell 4: f"Solve the following math problem step by step and give the final answer in \\boxed{{}}:\n\n{example['problem']}"
LEGACY_PROMPT_TEMPLATE = "Solve the following math problem step by step and give the final answer in \\boxed{{}}:\n\n{problem}"


def legacy_prompt(problem: str) -> str:
    """The user text NB-M builds for every problem. Sieve and evaluation fed it to the model raw (no chat
    template); GRPO training passed it as a one-message conversation, so TRL applied Qwen's chat template."""
    return LEGACY_PROMPT_TEMPLATE.format(problem=problem)


# ---------------------------------------------------------------------- NB-M cell 3 (verbatim)
# Extraction logic
def extract_box(text):
    match = re.search(r'\\boxed{', text)
    if not match: return None
    start = match.end()
    open_braces = 1
    for i, char in enumerate(text[start:]):
        if char == '{': open_braces += 1
        elif char == '}': open_braces -= 1
        if open_braces == 0: return text[start:start+i]
    return text[start:]

def extract_answer(text):
    match = extract_box(text)
    if match is not None: return match.strip()
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1].strip() if numbers else None

def extract_gt(gt_text):
    match = extract_box(gt_text)
    return match.strip() if match else gt_text.strip()

# Reward Functions for GRPO
def binary_match_reward(completions, **kwargs):
    answers = kwargs.get('solution', [])
    rewards = []
    import sympy
    from sympy.parsing.latex import parse_latex

    for comp, gt in zip(completions, answers):
        comp_text = comp[0]['content'] if isinstance(comp, list) else comp
        pred = extract_answer(comp_text)
        if pred is None:
            rewards.append(0.0)
            continue
        pred_cl = str(pred).strip()
        gt_cl = str(extract_gt(gt)).strip()
        if pred_cl.replace(' ', '') == gt_cl.replace(' ', ''):
            rewards.append(1.0)
            continue
        try:
            if sympy.simplify(parse_latex(pred_cl) - parse_latex(gt_cl)) == 0:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(1.0 if pred_cl.replace(' ','') == gt_cl.replace(' ','') else 0.0)
    return rewards

def format_reward(completions, **kwargs):
    return [0.5 if '\\boxed{' in (c[0]['content'] if isinstance(c, list) else c) else 0.0 for c in completions]

def is_correct(pred, gt):
    if pred is None: return False
    pred_cl = str(pred).strip()
    gt_cl = str(extract_gt(gt)).strip()
    if pred_cl.replace(' ', '') == gt_cl.replace(' ', ''): return True
    try:
        return sympy.simplify(parse_latex(pred_cl) - parse_latex(gt_cl)) == 0
    except:
        return pred_cl.replace(' ','') == gt_cl.replace(' ','')
# ---------------------------------------------------------------------- end of verbatim NB-M code

LEGACY_FUNCTIONS = ("extract_box", "extract_answer", "extract_gt", "binary_match_reward", "format_reward", "is_correct")


def legacy_signals(texts, solution) -> dict:
    """NB-M cell 7's sieve signals for one candidate from its K completion texts.

    Ps = np.mean(binary rewards); Var = np.var(binary + format) (population variance, values in {0, .5, 1, 1.5});
    D = |{str(extract_answer(t))}| / K, so an answer-less completion contributes the string "None" as one answer.
    """
    texts = list(texts)
    k = len(texts)
    if k == 0:
        raise ValueError("legacy_signals needs at least one completion")
    generations = [[{"content": t}] for t in texts]
    bin_rews = binary_match_reward(generations, solution=[solution] * k)
    fmt_rews = format_reward(generations)
    total = [b + f for b, f in zip(bin_rews, fmt_rews)]
    answers = [extract_answer(t) for t in texts]
    return {
        "bin": [float(b) for b in bin_rews],
        "fmt": [float(f) for f in fmt_rews],
        "total": [float(t) for t in total],
        "Ps": float(np.mean(bin_rews)),
        "Var": float(np.var(total)),
        "D": len(set(str(a) for a in answers)) / float(k),
        "answers": [None if a is None else str(a) for a in answers],
    }


def legacy_eval_correct(response: str, solution: str) -> bool:
    """NB-M `evaluate_model_acc` verdict for one greedy response: `is_correct(extract_answer(response), solution)`."""
    return bool(is_correct(extract_answer(response), solution))


def legacy_gold(solution: str) -> str:
    """The gold answer the legacy grader compares against (first box of the reference solution)."""
    return str(extract_gt(solution)).strip()


def symbolic_status() -> dict:
    """Whether the legacy sympy path can work in this process (antlr4 4.11 present) and the versions involved."""
    import importlib.metadata as md

    info: dict = {"sympy": getattr(sympy, "__version__", "?")}
    try:
        info["antlr4"] = md.version("antlr4-python3-runtime")
    except Exception:  # pragma: no cover - environment dependent
        info["antlr4"] = None
    try:
        ok = sympy.simplify(parse_latex("0.5") - parse_latex("\\frac{1}{2}")) == 0
        info.update(available=bool(ok), error=None)
    except Exception as e:  # noqa: BLE001 - this is the diagnostic
        info.update(available=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    return info
