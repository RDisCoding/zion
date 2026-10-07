"""The original SGAC run, transcribed from the notebooks, for side-by-side comparison in the report.

Primary source: `ghoul/rl-llm-phase-5-the-loop-modified-results.ipynb` ("NB-M"), the Kaggle T4 run behind the
paper's 64 -> 68 % (cell 7 output ends with `Historical Accuracies: [0.66, 0.62, 0.6, 0.68]`). Values below are as
printed by NB-M cell 7 (`Selector Picked Problem #i (Level L) | P(success)=.., Var=.., Disagree=..` plus the GRPO
loss table); `var` is the exact population variance consistent with the printed (Ps, Var) pair, and `score`
recomputes the hardcoded rule for the selected candidate. Unselected candidates' signals, problem ids, rewards and
completion lengths were never printed, so the original selections cannot be recovered.

Base 64.0 % and pi1 66.0 % exist only in `ghoul/phase-5-results/autonomous_rlvr_model/outputs.png` (NB-M cell 11
measured base AFTER training, with the adapter disabled; pi1 in fp16).
"""
from __future__ import annotations

NBM_PATH = "ghoul/rl-llm-phase-5-the-loop-modified-results.ipynb"
NBM_SHA256 = "08818a2f4747ce879c8f4466526a679406129044c86781091e84407f5d1be41b"
NBM_LOOP_WALL_S = 39442.97  # papermill duration of cell 7 (2026-04-11T17:30:27Z -> 2026-04-12T04:27:50Z)

# Hardcoded in NB-M cell 7 (index map s = [Level, Ps, Var, D]):
#   score = (0.0050 * s[1]) + (0.1832 * s[2]) - (0.0751 * s[3]) + (0.2188 * s[0]);  best_idx = np.argmax(scores)
EQ10_AS_RUN = {"Ps": 0.0050, "Var": 0.1832, "D": -0.0751, "L": 0.2188}

# learned_selector.pkl (phase-4-modified cell 8, N=20 rows, X = [level, Ps, Var, D]); decoded without unpickling.
PICKLE_FEATURE_ORDER = ("L", "Ps", "Var", "D")
PICKLE_COEF = (0.005039982936240752, 0.18323669846886836, -0.07505765067125274, 0.21881256616444827)
PICKLE_INTERCEPT = 0.2790726550110435

# Paper Table 2 = phase-4 N=4 fit, printed with labels shifted by one (true order [L, Ps, Var, D]).
TABLE2_AS_PRINTED = {"Ps": -0.0574, "Var": -0.2511, "D": 0.0393, "L": 0.1095}
TABLE2_TRUE_MAPPING = {"L": -0.0574, "Ps": -0.2511, "Var": 0.0393, "D": 0.1095}

# step, picked index within the batch of 4, level, Ps, Var (exact), D, score of the pick, the 5 GRPO losses
# (TRL 1.0.0 logging_steps=1), eval accuracy after the step (None when not evaluated).
STEPS: tuple[dict, ...] = (
    {"step": 1, "idx": 0, "L": 5, "Ps": 1.00, "Var": 0.0, "D": 0.25, "score": 1.0802,
     "losses": (-0.202772, -0.085766, -0.203260, 0.026689, 0.124950), "eval": None},
    {"step": 2, "idx": 3, "L": 5, "Ps": 0.00, "Var": 0.0, "D": 0.75, "score": 1.0377,
     "losses": (0.0, 0.0, 0.0, 0.0, 0.156226), "eval": None},
    {"step": 3, "idx": 2, "L": 4, "Ps": 0.75, "Var": 0.1875, "D": 0.50, "score": 0.8758,
     "losses": (0.0, 0.0, -0.379030, 0.378868, 0.191325), "eval": None},
    {"step": 4, "idx": 3, "L": 4, "Ps": 0.00, "Var": 0.0, "D": 1.00, "score": 0.8001,
     "losses": (0.124950, 0.124862, -0.374945, 0.125261, 0.147577), "eval": None},
    {"step": 5, "idx": 3, "L": 5, "Ps": 0.00, "Var": 0.0, "D": 1.00, "score": 1.0189,
     "losses": (0.0, 0.0, 0.0, 0.0, 0.0), "eval": 0.66},
    {"step": 6, "idx": 1, "L": 5, "Ps": 0.50, "Var": 0.25, "D": 0.75, "score": 1.0860,
     "losses": (-0.088389, 0.410737, -0.136774, -0.137579, -0.216855), "eval": None},
    {"step": 7, "idx": 1, "L": 5, "Ps": 0.50, "Var": 0.421875, "D": 0.75, "score": 1.1175,
     "losses": (0.101768, -0.356856, 0.357401, -0.050446, 0.156226), "eval": None},
    {"step": 8, "idx": 0, "L": 5, "Ps": 0.00, "Var": 0.0, "D": 1.00, "score": 1.0189,
     "losses": (0.0, 0.0, 0.0, 0.0, 0.159936), "eval": None},
    {"step": 9, "idx": 2, "L": 5, "Ps": 0.00, "Var": 0.0, "D": 0.25, "score": 1.0752,
     "losses": (0.216431, 0.216661, -0.216501, -0.216742, 0.0), "eval": None},
    {"step": 10, "idx": 3, "L": 4, "Ps": 0.75, "Var": 0.1875, "D": 0.50, "score": 0.8758,
     "losses": (0.374850, -0.124797, -0.125118, -0.125360, 0.220994), "eval": 0.62},
    {"step": 11, "idx": 3, "L": 5, "Ps": 0.00, "Var": 0.0, "D": 0.50, "score": 1.0564,
     "losses": (0.0, 0.0, 0.0, 0.0, 0.0), "eval": None},
    {"step": 12, "idx": 3, "L": 5, "Ps": 0.00, "Var": 0.046875, "D": 1.00, "score": 1.0275,
     "losses": (0.0, 0.0, 0.0, 0.0, 0.0), "eval": None},
    {"step": 13, "idx": 2, "L": 5, "Ps": 0.00, "Var": 0.0625, "D": 1.00, "score": 1.0303,
     "losses": (0.0, 0.0, 0.0, 0.0, 0.0), "eval": None},
    {"step": 14, "idx": 0, "L": 5, "Ps": 0.50, "Var": 0.5625, "D": 0.75, "score": 1.1432,
     "losses": (0.075146, -0.262423, -0.263001, 0.263898, 0.170826), "eval": None},
    {"step": 15, "idx": 1, "L": 4, "Ps": 0.25, "Var": 0.375, "D": 0.75, "score": 0.8888,
     "losses": (0.0, 0.0, 0.0, 0.0, 0.0), "eval": 0.60},
    {"step": 16, "idx": 0, "L": 5, "Ps": 0.00, "Var": 0.046875, "D": 1.00, "score": 1.0275,
     "losses": (0.0, 0.0, 0.0, 0.0, 0.0), "eval": None},
    {"step": 17, "idx": 0, "L": 4, "Ps": 0.75, "Var": 0.421875, "D": 0.50, "score": 0.9187,
     "losses": (0.124950, 0.125070, 0.124954, -0.375029, -0.374850), "eval": None},
    {"step": 18, "idx": 0, "L": 5, "Ps": 0.25, "Var": 0.375, "D": 1.00, "score": 1.0888,
     "losses": (-0.250769, 0.137245, 0.139674, 0.139978, 0.0), "eval": None},
    {"step": 19, "idx": 2, "L": 4, "Ps": 0.00, "Var": 0.0625, "D": 1.00, "score": 0.8115,
     "losses": (0.0, 0.0, 0.0, 0.0, 0.0), "eval": None},
    {"step": 20, "idx": 0, "L": 4, "Ps": 1.00, "Var": 0.0, "D": 0.25, "score": 0.8614,
     "losses": (-0.374850, 0.124585, 0.124927, 0.125029, 0.0), "eval": 0.68},
)

TRAJECTORY = {0: 0.64, 5: 0.66, 10: 0.62, 15: 0.60, 20: 0.68}  # step 0 = "base" (screenshot, measured after training)
PI1_ACC = 0.66  # ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1 in fp16, same 50 items and protocol (screenshot)
TEST_N = 50

# Phase-1 signal discovery (phase-4-selector-experiments-results.ipynb): shuffled rows 0-3 of the same split, K=8 at
# 2048 tokens; downstream accuracy on shuffled rows 4-13 (10 items) after a fresh 20-step GRPO run per candidate.
PHASE1_TABLE1 = (
    {"row": 0, "unique_id": "test/prealgebra/1820.json", "L": 5, "Ps": 0.375, "Var": 0.25, "D": 0.625, "acc": 0.4},
    {"row": 1, "unique_id": "test/geometry/554.json", "L": 2, "Ps": 0.875, "Var": 0.109375, "D": 0.25, "acc": 0.4},
    {"row": 2, "unique_id": "test/counting_and_probability/559.json", "L": 5, "Ps": 0.125, "Var": 0.15234375, "D": 1.0,
     "acc": 0.5},
    {"row": 3, "unique_id": "test/number_theory/457.json", "L": 4, "Ps": 0.25, "Var": 0.1875, "D": 0.75, "acc": 0.5},
)


def zero_loss_bursts() -> int:
    """Bursts whose 5 logged losses are all exactly zero (paper: 8/20; log: 7/20)."""
    return sum(1 for s in STEPS if all(v == 0.0 for v in s["losses"]))


def pick_distribution() -> dict:
    """Level / Ps / D distribution of the 20 logged picks (the paper's Table 9 does not match these)."""
    n = len(STEPS)
    lv = [s["L"] for s in STEPS]
    ps = [s["Ps"] for s in STEPS]
    d = [s["D"] for s in STEPS]
    return {
        "level5": sum(v == 5 for v in lv) / n, "level4": sum(v == 4 for v in lv) / n, "level_le3": sum(v <= 3 for v in lv) / n,
        "ps0": sum(v == 0 for v in ps) / n, "ps_0_half": sum(0 < v <= 0.5 for v in ps) / n, "ps_gt_half": sum(v > 0.5 for v in ps) / n,
        "d_ge_075": sum(v >= 0.75 for v in d) / n, "d_05_075": sum(0.5 <= v < 0.75 for v in d) / n, "d_lt_05": sum(v < 0.5 for v in d) / n,
        "zero_loss_bursts": zero_loss_bursts() / n,
    }
