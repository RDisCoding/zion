"""Aggregate everything under results_sgac/ into reports/sgac_repro/ (REPORT.md + CSV + PNG + report_data.json).

Sections follow the brief: A configuration, B coefficients, C 20-step selection tables (next to the NB-M log),
D accuracy trajectories, E comparison (Base / pi1 / Random / Variance / Disagreement / Difficulty / SGAC), F runtime,
G deviations, H verdict computed by the pre-declared rule of paper/sgac_repro_protocol.md:
  P1 = SGAC - Base and P2 = SGAC - Random, e0 profile, MATH-500 at step 20, per-item correctness averaged over the
  completed seeds (P2: seeds completed by both arms), paired item bootstrap (10,000 resamples, seed 0).
  Reproduced: both 95% CI lower bounds > 0; partially: exactly one; otherwise could not be reproduced.
"""
from __future__ import annotations

import csv
import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..artifacts import REPO_ROOT, atomic_write_json, read_json, read_jsonl
from ..stats import bootstrap_mean_ci, paired_item_bootstrap
from . import nbm_original as nbm
from .spec import SgacSpec, group_dir

log = logging.getLogger(__name__)

ARM_ORDER = ("sgac", "random", "max_var", "max_d", "max_level", "sgac_label_corrected", "fixed_pi1")
ARM_LABEL = {"sgac": "SGAC (Eq. 10 as run)", "random": "Random", "max_var": "Max variance",
             "max_d": "Max disagreement", "max_level": "Max difficulty", "sgac_label_corrected": "SGAC label-corrected",
             "fixed_pi1": "Fixed pi1 (same budget)"}
# dataviz reference palette, categorical slots 1-7 in fixed order (validated light mode; contrast relief = labels/tables)
ARM_COLOR = dict(zip(ARM_ORDER, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")))
ARM_MARKER = dict(zip(ARM_ORDER, ("o", "s", "^", "D", "v", "P", "X")))
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
       "surface": "#fcfcfb"}
N_BOOT, BOOT_SEED = 10000, 0
EVAL_STEPS = (0, 5, 10, 15, 20)


# ---------------------------------------------------------------------- loading
def _per_item(path: Path) -> dict[str, bool] | None:
    rows = read_jsonl(path)
    return {r["unique_id"]: bool(r["correct"]) for r in rows} if rows else None


def _per_item_key(path: Path, key: str) -> dict[str, bool] | None:
    rows = read_jsonl(path)
    return {r["unique_id"]: bool(r.get(key)) for r in rows} if rows else None


def collect(spec: SgacSpec) -> dict:
    g = group_dir(spec)
    runs: dict[tuple[str, int], dict] = {}
    if g.exists():
        for d in sorted(p for p in g.iterdir() if p.is_dir() and not p.name.startswith("_")):
            st = read_json(d / "state.json")
            if st is None:
                continue
            runs[(st["arm"], int(st["seed"]))] = {"dir": d, "state": st, "status": read_json(d / "status.json", {}),
                                                  "run": read_json(d / "run.json", {})}
    sh = g / "_shared"
    shared = {"base": read_json(sh / "base_eval" / "base_eval.json"), "pi1": read_json(sh / "pi1_eval" / "pi1_eval.json"),
              "test50_decision": read_json(sh / "base_eval" / "test50_batch_decision.json"),
              "e0_g1": read_json(sh / "base_eval" / "e0_g1_agreement.json")}
    return {"spec": spec, "group": g, "runs": runs, "shared": shared}


def eval_path(col: dict, who: str, step: int | str, name: str, arm: str | None = None, seed: int | None = None) -> Path:
    g = col["group"]
    if who == "base":
        return g / "_shared" / "base_eval" / name / "per_item.jsonl"
    if who == "pi1":
        return g / "_shared" / "pi1_eval" / name / "per_item.jsonl"
    d = col["runs"][(arm, seed)]["dir"]
    tag = step if isinstance(step, str) else f"step-{int(step):03d}"
    return d / "evals" / tag / name / "per_item.jsonl"


def seeds_with(col: dict, arm: str, step: int, name: str) -> list[int]:
    return sorted(s for (a, s) in col["runs"] if a == arm and eval_path(col, "run", step, name, a, s).exists())


def item_matrix(col: dict, arm: str, step: int, name: str, seeds: Sequence[int], uids: Sequence[str],
                key: str = "correct") -> np.ndarray:
    rows = []
    for s in seeds:
        d = _per_item_key(eval_path(col, "run", step, name, arm, s), key) or {}
        rows.append([float(d[u]) if u in d else np.nan for u in uids])
    return np.array(rows, dtype=float)


def base_vector(col: dict, name: str, uids: Sequence[str], who: str = "base", key: str = "correct") -> np.ndarray | None:
    d = _per_item_key(eval_path(col, who, 0, name), key)
    return None if d is None else np.array([float(d[u]) if u in d else np.nan for u in uids], dtype=float)


def uids_of(col: dict, name: str) -> list[str]:
    rows = read_jsonl(eval_path(col, "base", 0, name))
    return [r["unique_id"] for r in rows]


# ---------------------------------------------------------------------- statistics
def arm_summary(col: dict, arm: str, step: int, name: str, uids: Sequence[str], key: str = "correct") -> dict | None:
    seeds = seeds_with(col, arm, step, name)
    if not seeds:
        return None
    m = item_matrix(col, arm, step, name, seeds, uids, key)
    per_seed = [float(np.nanmean(r)) for r in m]
    pooled = np.nanmean(m, axis=0)
    mean, lo, hi = bootstrap_mean_ci(pooled, n_boot=N_BOOT, seed=BOOT_SEED)
    return {"arm": arm, "step": step, "set": name, "seeds": seeds, "per_seed": per_seed, "mean": mean, "ci": [lo, hi],
            "pooled_items": pooled}


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    r = paired_item_bootstrap(a, b, n_boot=N_BOOT, seed=BOOT_SEED)
    return {k: r[k] for k in ("delta", "ci_lo", "ci_hi", "p_two_sided", "n", "mean_a", "mean_b")}


def primary_endpoints(col: dict, final_step: int) -> dict:
    """P1 (SGAC - Base) and P2 (SGAC - Random) on MATH-500 at the final step, seeds pooled."""
    out: dict = {"set": "math500", "step": final_step}
    if not eval_path(col, "base", 0, "math500").exists():
        out["error"] = "base MATH-500 evaluation missing"
        return out
    uids = uids_of(col, "math500")
    base = base_vector(col, "math500", uids)
    s_seeds = seeds_with(col, "sgac", final_step, "math500")
    r_seeds = seeds_with(col, "random", final_step, "math500")
    if s_seeds:
        sg = np.nanmean(item_matrix(col, "sgac", final_step, "math500", s_seeds, uids), axis=0)
        out["P1"] = {**paired(sg, base), "seeds": s_seeds}
    both = sorted(set(s_seeds) & set(r_seeds))
    if both:
        sg2 = np.nanmean(item_matrix(col, "sgac", final_step, "math500", both, uids), axis=0)
        rn = np.nanmean(item_matrix(col, "random", final_step, "math500", both, uids), axis=0)
        out["P2"] = {**paired(sg2, rn), "seeds": both}
    return out


def verdict(p1: dict | None, p2: dict | None) -> dict:
    """The pre-declared rule (protocol section 4)."""
    if p1 is None or p2 is None:
        return {"label": "INCOMPLETE: primary endpoints not available", "p1_positive": None, "p2_positive": None}
    pos1, pos2 = p1["ci_lo"] > 0, p2["ci_lo"] > 0
    if pos1 and pos2:
        label = "Original SGAC reproduced"
    elif pos1 or pos2:
        label = "Original SGAC partially reproduced"
    else:
        label = "Original SGAC could not be reproduced"
    note = []
    if not pos1:
        note.append("no training effect: SGAC - Base 95% CI includes 0" if p1["ci_hi"] >= 0
                    else "SGAC DEGRADES accuracy: SGAC - Base 95% CI entirely below 0")
    if not pos2:
        note.append("no selection effect: SGAC - Random 95% CI includes 0" if p2["ci_hi"] >= 0
                    else "SGAC is WORSE than random selection: 95% CI entirely below 0")
    return {"label": label, "p1_positive": pos1, "p2_positive": pos2, "notes": note}


# ---------------------------------------------------------------------- diagnostics
def selection_diagnostics(col: dict) -> dict:
    """Counterfactual agreement of rules on identical batches, and pick distributions per arm."""
    pairs = (("sgac_eq10", "max_level"), ("sgac_eq10", "max_var"), ("sgac_eq10", "max_d"), ("sgac_eq10", "random"),
             ("sgac_eq10", "pickle_true_mapping"), ("sgac_eq10", "table2_as_printed"))
    agree = {f"{a}=={b}": [] for a, b in pairs}
    maxlevel = []
    per_arm: dict[str, dict] = {}
    for (arm, seed), r in col["runs"].items():
        hist = r["state"].get("history", [])
        acc = per_arm.setdefault(arm, {"levels": [], "ps": [], "d": [], "var": [], "zero_loss": [], "zero_std": [],
                                       "clipped": [], "reward": [], "trunc": []})
        for h in hist:
            s = h.get("selection") or {}
            b = h.get("burst") or {}
            picks = s.get("picks") or {}
            for a, bb in pairs:
                if a in picks and bb in picks:
                    agree[f"{a}=={bb}"].append(picks[a] == picks[bb])
            if "eq10_picked_max_level" in s:
                maxlevel.append(bool(s["eq10_picked_max_level"]))
            sig = s.get("selected_signals") or {}
            if sig:
                acc["levels"].append(sig.get("L"))
                acc["ps"].append(sig.get("Ps"))
                acc["d"].append(sig.get("D"))
                acc["var"].append(sig.get("Var"))
            acc["zero_loss"].append(bool(b.get("all_zero_loss")))
            for key, field in (("zero_std", "frac_reward_zero_std"), ("clipped", "clipped_ratio"), ("reward", "reward")):
                vals = [v for v in (b.get(field) or []) if v is not None]
                if vals:
                    acc[key].append(float(np.mean(vals)))
    dist = {}
    for arm, a in per_arm.items():
        n = len(a["levels"]) or 1
        ps, d = [v for v in a["ps"] if v is not None], [v for v in a["d"] if v is not None]
        dist[arm] = {
            "n_picks": len(a["levels"]), "level5": sum(v == 5 for v in a["levels"]) / n,
            "level4": sum(v == 4 for v in a["levels"]) / n, "level_le3": sum(v is not None and v <= 3 for v in a["levels"]) / n,
            "ps0": sum(v == 0 for v in ps) / n, "ps_0_half": sum(0 < v <= 0.5 for v in ps) / n,
            "ps_gt_half": sum(v > 0.5 for v in ps) / n, "d_ge_075": sum(v >= 0.75 for v in d) / n,
            "d_05_075": sum(0.5 <= v < 0.75 for v in d) / n, "d_lt_05": sum(v < 0.5 for v in d) / n,
            "zero_loss_bursts": (sum(a["zero_loss"]) / len(a["zero_loss"])) if a["zero_loss"] else None,
            "mean_frac_reward_zero_std": float(np.mean(a["zero_std"])) if a["zero_std"] else None,
            "mean_clipped_ratio": float(np.mean(a["clipped"])) if a["clipped"] else None,
            "mean_train_reward": float(np.mean(a["reward"])) if a["reward"] else None,
        }
    return {"agreement": {k: {"n": len(v), "rate": (sum(v) / len(v)) if v else None} for k, v in agree.items()},
            "eq10_picked_max_level": {"n": len(maxlevel), "rate": (sum(maxlevel) / len(maxlevel)) if maxlevel else None},
            "pick_distribution": dist, "nbm_original": nbm.pick_distribution()}


def runtime(col: dict) -> dict:
    out = {}
    for (arm, seed), r in col["runs"].items():
        st = r["state"]
        step_wall = sum(sum((h.get("timing") or {}).get(k, {}).get("wall_s", 0.0) for k in ("sieve", "burst", "save"))
                        for h in st.get("history", []))
        eval_wall = sum(v.get("timing", {}).get("wall_s", 0.0) for e in st.get("evals", {}).values() for v in e.values())
        peaks = [(h.get("burst") or {}).get("peak_mem_gb") for h in st.get("history", [])]
        peaks += [v.get("timing", {}).get("peak_mem_gb") for e in st.get("evals", {}).values() for v in e.values()]
        peaks = [p for p in peaks if p is not None]
        out[f"{arm}__seed{seed}"] = {"steps_done": st.get("step_done"), "curriculum_s": step_wall, "eval_s": eval_wall,
                                     "total_h": (step_wall + eval_wall) / 3600, "peak_mem_gb": max(peaks) if peaks else None,
                                     "status": r["status"].get("state")}
    return out


# ---------------------------------------------------------------------- formatting
def pct(v) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.1f}"


def ci_str(ci) -> str:
    return "—" if not ci or any(c is None or math.isnan(c) for c in ci) else f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def delta_str(d: dict | None) -> str:
    if not d:
        return "—"
    return f"{100 * d['delta']:+.1f} [{100 * d['ci_lo']:+.1f}, {100 * d['ci_hi']:+.1f}]"


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _md_table(header: Sequence[str], rows: Sequence[Sequence]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def selection_table(run: dict, eq10_key: str = "sgac_eq10") -> tuple[list[str], list[list]]:
    header = ["Step", "Problem", "Ps", "Var", "D", "Level", "Score (Eq.10)", "Arm pick = Eq.10?", "Zero-loss burst",
              "Train reward", "Eval acc (test50)"]
    evals = run["state"].get("evals", {})
    rows = []
    for h in run["state"].get("history", []):
        s, b = h.get("selection") or {}, h.get("burst") or {}
        sig = s.get("selected_signals") or {}
        idx = s.get("selected_idx")
        eq = (s.get("scores") or {}).get(eq10_key)
        score = eq[idx] if (eq is not None and idx is not None and idx < len(eq)) else None
        same = (s.get("picks") or {}).get(eq10_key) == idx if s.get("picks") else None
        rew = [v for v in (b.get("reward") or []) if v is not None]
        ev = evals.get(str(h["step"]), {}).get("test50", {}).get("acc")
        rows.append([h["step"], s.get("selected_uid"), _f(sig.get("Ps")), _f(sig.get("Var")), _f(sig.get("D")),
                     sig.get("L"), _f(score, 4), {True: "yes", False: "no", None: "—"}[same],
                     {True: "yes", False: "no", None: "—"}[b.get("all_zero_loss")],
                     _f(float(np.mean(rew)) if rew else None), pct(ev) if ev is not None else ""])
    return header, rows


def _f(v, nd: int = 3) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "—" if math.isnan(f) else f"{f:.{nd}f}"


def nbm_table() -> tuple[list[str], list[list]]:
    header = ["Step", "Pick #", "Ps", "Var", "D", "Level", "Score (Eq.10)", "Zero-loss burst", "Eval acc (test50)"]
    rows = [[s["step"], s["idx"], f"{s['Ps']:.2f}", f"{s['Var']:.4f}", f"{s['D']:.2f}", s["L"], f"{s['score']:.4f}",
             "yes" if all(v == 0 for v in s["losses"]) else "no", pct(s["eval"]) if s["eval"] is not None else ""]
            for s in nbm.STEPS]
    return header, rows


# ---------------------------------------------------------------------- figures
def _style(ax) -> None:
    ax.set_facecolor(INK["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])
    ax.tick_params(colors=INK["muted"], labelsize=8.5)
    ax.yaxis.grid(True, color=INK["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def plot_trajectory(path: Path, traj: dict, profile: str, title: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=200)
    fig.patch.set_facecolor(INK["surface"])
    _style(ax)
    ends = []
    for arm in ARM_ORDER:
        pts = traj.get(arm)
        if not pts:
            continue
        xs = [x for x, _ in pts]
        ys = [100 * y for _, y in pts]
        ax.plot(xs, ys, color=ARM_COLOR[arm], linewidth=1.6, marker=ARM_MARKER[arm], markersize=5.5,
                markeredgecolor=INK["surface"], markeredgewidth=0.8, label=ARM_LABEL[arm], zorder=3)
        ends.append((xs[-1], ys[-1], ARM_LABEL[arm]))
    if profile == "as_run" or traj.get("_show_nbm"):
        xs = sorted(nbm.TRAJECTORY)
        ax.plot(xs, [100 * nbm.TRAJECTORY[x] for x in xs], color=INK["muted"], linewidth=1.2, linestyle="--",
                marker="o", markersize=4, label="Original NB-M run (T4, 1 run)", zorder=2)
    ends.sort(key=lambda e: e[1])
    last_y = -1e9
    for x, y, lab in ends:  # direct labels at line ends, nudged apart, in secondary ink
        y_lab = max(y, last_y + 2.2)
        ax.annotate(lab, (x, y), xytext=(x + 0.6, y_lab), color=INK["secondary"], fontsize=7.5, va="center")
        last_y = y_lab
    ax.set_xticks(list(EVAL_STEPS))
    ax.set_xlim(-0.5, 26.5)
    ax.set_xlabel("Curriculum step", color=INK["secondary"], fontsize=9)
    ax.set_ylabel("Accuracy on the original 50-item test set (%)", color=INK["secondary"], fontsize=9)
    ax.set_title(title, color=INK["primary"], fontsize=10, loc="left")
    if ax.get_legend_handles_labels()[0]:
        leg = ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)
        for t in leg.get_texts():
            t.set_color(INK["secondary"])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=INK["surface"])
    plt.close(fig)
    return path


def plot_final(path: Path, rows: list[dict], base: float | None, pi1: float | None, title: str, xlabel: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return None
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(rows) + 1.4), dpi=200)
    fig.patch.set_facecolor(INK["surface"])
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=INK["grid"], linewidth=0.6)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = r["ci"]
        ax.plot([100 * lo, 100 * hi], [y, y], color=ARM_COLOR[r["arm"]], linewidth=1.6, zorder=2)
        ax.plot([100 * r["mean"]], [y], marker=ARM_MARKER[r["arm"]], color=ARM_COLOR[r["arm"]], markersize=7,
                markeredgecolor=INK["surface"], markeredgewidth=1.0, zorder=3)
        ax.annotate(f"{100 * r['mean']:.1f}", (100 * hi, y), xytext=(4, 0), textcoords="offset points",
                    color=INK["secondary"], fontsize=7.5, va="center")
    for val, lab in ((base, "Base"), (pi1, "pi1 checkpoint")):
        if val is not None:
            ax.axvline(100 * val, color=INK["muted"], linewidth=1.0, linestyle="--", zorder=1)
            ax.annotate(f"{lab} {100 * val:.1f}", (100 * val, len(rows) - 0.45), color=INK["secondary"], fontsize=7.5,
                        ha="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([ARM_LABEL[r["arm"]] for r in reversed(rows)], color=INK["secondary"], fontsize=8.5)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.set_xlabel(xlabel, color=INK["secondary"], fontsize=9)
    ax.set_title(title, color=INK["primary"], fontsize=10, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=INK["surface"])
    plt.close(fig)
    return path


# ---------------------------------------------------------------------- the report
def profile_section(col: dict, out_dir: Path) -> tuple[str, dict]:
    spec: SgacSpec = col["spec"]
    name = spec.profile.name
    final = int(spec.loop.steps)
    data: dict = {"profile": name, "spec_hash": spec.spec_hash(), "group": str(col["group"]), "runs": {}}
    md = [f"## Profile `{name}` (spec {spec.spec_hash()})", ""]
    if not col["runs"] and not col["shared"]["base"]:
        md.append("_No results yet._")
        return "\n".join(md), data

    # --- A. configuration as executed
    p = spec.profile
    any_run = next(iter(col["runs"].values()), None)
    load = (any_run or {}).get("state", {}).get("load_records") or []
    md += ["### A. Configuration", "",
           _md_table(["Setting", "Value"], [
               ["Model", f"{spec.model.name} (loaded: {', '.join(sorted({str(l.get('effective')) for l in load})) or '—'};"
                         f" commit {', '.join(sorted({str(l.get('commit')) for l in load})) or '—'})"],
               ["Quantisation requested / fallback", f"{p.quant} / {', '.join(sorted({str(l.get('fallback')) for l in load if l.get('fallback')})) or 'none'}"],
               ["Prompt (sieve, eval) / training", f"{p.prompt} / {p.train_prompt}"],
               ["Grader (primary) / D metric", f"{p.grader} / {p.d_metric}"],
               ["Token caps sieve / GRPO / eval", f"{p.sieve_max_new_tokens} / {p.grpo_max_completion_length} / {p.eval_max_new_tokens}"],
               ["Sieve top_k / stops / pad", f"{p.sieve_top_k} / {p.stops} / {p.tokenizer_pad}"],
               ["Loop", f"{spec.loop.steps} steps, B={spec.loop.batch_b}, K={spec.loop.k}, T={spec.loop.temperature}, eval every {spec.loop.eval_every}"],
               ["Burst", f"{spec.grpo.max_steps} single-completion updates (pdbs {spec.grpo.per_device_train_batch_size}, gen batch {spec.grpo.generation_batch_size}, G {spec.grpo.num_generations}), lr {spec.grpo.learning_rate} {spec.grpo.lr_scheduler_type}"],
               ["LoRA", f"r={spec.lora.r}, alpha={spec.lora.alpha}, {list(spec.lora.target_modules)}"],
               ["test50 batch decision", str((col["shared"]["test50_decision"] or {}).get("batch_size", "n/a"))],
               ["E0 G1 agreement (base MATH-500)", str((col["shared"]["e0_g1"] or {}).get("verdict_agreement", "n/a"))],
           ]), ""]

    # --- D. trajectories (test50, primary grader)
    traj: dict = {}
    traj_rows = []
    uids50 = uids_of(col, "test50") if eval_path(col, "base", 0, "test50").exists() else []
    base50 = base_vector(col, "test50", uids50) if uids50 else None
    for arm in ARM_ORDER:
        pts = []
        row = [ARM_LABEL[arm]]
        for t in EVAL_STEPS:
            if t == 0:
                v = float(np.nanmean(base50)) if base50 is not None else None
                row.append(pct(v))
                if v is not None:
                    pts.append((0, v))
                continue
            s = arm_summary(col, arm, t, "test50", uids50) if uids50 else None
            if s:
                pts.append((t, s["mean"]))
                row.append(f"{pct(s['mean'])} (n={len(s['seeds'])})")
            else:
                row.append("—")
        if len(pts) > 1:
            traj[arm] = pts
            traj_rows.append(row)
    data["trajectory_test50"] = {a: [(int(x), float(y)) for x, y in v] for a, v in traj.items()}
    _write_csv(out_dir / "tables" / f"trajectory_test50_{name}.csv", ["Arm"] + [f"step {t}" for t in EVAL_STEPS], traj_rows)
    fig1 = plot_trajectory(out_dir / "figures" / f"trajectory_test50_{name}.png", traj, name,
                           f"{name}: accuracy trajectory on the original 50 items (mean over seeds)")
    md += ["### D. Accuracy trajectory (original 50-item test set; seeds averaged)", "",
           _md_table(["Arm"] + ["Base" if t == 0 else f"Step {t}" for t in EVAL_STEPS], traj_rows), "",
           f"Original NB-M run (one unseeded T4 run): {', '.join(f'{t}: {pct(v)}' for t, v in nbm.TRAJECTORY.items())}.",
           ""]
    if fig1:
        md += [f"![trajectory]({fig1.relative_to(out_dir).as_posix()})", ""]

    # --- E. comparison at the final step
    comp: dict = {}
    md += [f"### E. Comparison at step {final}", ""]
    for set_name in ("math500", "test50"):
        if not eval_path(col, "base", 0, set_name).exists():
            continue
        uids = uids_of(col, set_name)
        base = base_vector(col, set_name, uids)
        pi1 = base_vector(col, set_name, uids, who="pi1")
        sg = arm_summary(col, "sgac", final, set_name, uids)
        rows, frows = [], []
        rows.append(["Base", pct(float(np.nanmean(base))), "—", "—", "—"])
        if pi1 is not None:
            rows.append(["pi1 checkpoint (Wang et al.)", pct(float(np.nanmean(pi1))), "—", delta_str(paired(pi1, base)), "—"])
        for arm in ARM_ORDER:
            s = arm_summary(col, arm, final, set_name, uids)
            if not s:
                continue
            vs_base = paired(s["pooled_items"], base)
            vs_sgac = None
            if arm != "sgac" and sg:
                both = sorted(set(sg["seeds"]) & set(s["seeds"]))
                if both:
                    a = np.nanmean(item_matrix(col, "sgac", final, set_name, both, uids), axis=0)
                    b = np.nanmean(item_matrix(col, arm, final, set_name, both, uids), axis=0)
                    vs_sgac = paired(a, b)
            rows.append([f"{ARM_LABEL[arm]} (seeds {','.join(map(str, s['seeds']))})",
                         f"{pct(s['mean'])} {ci_str(s['ci'])}", ", ".join(pct(v) for v in s["per_seed"]),
                         delta_str(vs_base), delta_str(vs_sgac)])
            frows.append({"arm": arm, "mean": s["mean"], "ci": s["ci"]})
            comp.setdefault(set_name, {})[arm] = {"mean": s["mean"], "ci": s["ci"], "per_seed": s["per_seed"],
                                                  "seeds": s["seeds"], "vs_base": vs_base, "sgac_minus_arm": vs_sgac}
        header = ["Model / arm", "Accuracy % [95% CI]", "Per seed", "vs Base (pp) [95% CI]", "SGAC − arm (pp) [95% CI]"]
        _write_csv(out_dir / "tables" / f"comparison_{set_name}_step{final}_{name}.csv", header, rows)
        label = "MATH-500 (500 items)" if set_name == "math500" else "original 50-item test set"
        md += [f"**{label}**, primary grader `{spec.profile.grader}`:", "", _md_table(header, rows), ""]
        figp = plot_final(out_dir / "figures" / f"final_{set_name}_{name}.png", frows,
                          float(np.nanmean(base)), None if pi1 is None else float(np.nanmean(pi1)),
                          f"{name}: accuracy at step {final} on {label} (95% item-bootstrap CI, seeds pooled)",
                          "Accuracy (%)")
        if figp:
            md += [f"![final {set_name}]({figp.relative_to(out_dir).as_posix()})", ""]
    data["comparison"] = comp

    # --- other-grader view (sensitivity)
    sens_rows = []
    for (arm, seed), r in sorted(col["runs"].items()):
        e = r["state"].get("evals", {}).get(str(final), {})
        for set_name, v in e.items():
            sens_rows.append([f"{arm} seed {seed}", set_name, pct(v.get("acc")), pct(v.get("acc_mv")), pct(v.get("acc_legacy")),
                              pct(v.get("trunc_rate"))])
    if sens_rows:
        md += ["Grader sensitivity at the final step (same outputs, both graders):", "",
               _md_table(["Run", "Set", "Primary", "math-verify", "legacy", "Truncated"], sens_rows), ""]

    # --- C. selection tables
    md += ["### C. Selection trajectories (20 steps per run)", ""]
    for (arm, seed), r in sorted(col["runs"].items(), key=lambda kv: (ARM_ORDER.index(kv[0][0]) if kv[0][0] in ARM_ORDER else 99, kv[0][1])):
        header, rows = selection_table(r)
        _write_csv(out_dir / "tables" / f"selection_{name}_{arm}_seed{seed}.csv", header, rows)
        md += [f"<details><summary>{ARM_LABEL.get(arm, arm)}, seed {seed} ({r['state'].get('step_done')} steps done, "
               f"status {r['status'].get('state')})</summary>", "", _md_table(header, rows), "", "</details>", ""]
        data["runs"][f"{arm}__seed{seed}"] = {"step_done": r["state"].get("step_done"), "evals": r["state"].get("evals")}

    # --- diagnostics
    diag = selection_diagnostics(col)
    data["diagnostics"] = diag
    ag = diag["agreement"]
    md += ["### Selection diagnostics (counterfactual picks on identical batches, all runs)", "",
           _md_table(["Rule pair", "Steps", "Same pick"],
                     [[k, v["n"], pct(v["rate"])] for k, v in ag.items()]), "",
           f"Eq. 10 picked a maximum-level candidate in {pct(diag['eq10_picked_max_level']['rate'])}% of "
           f"{diag['eq10_picked_max_level']['n']} steps (analytically 100% at K=4).", "",
           _md_table(["Arm", "Picks", "L5", "L4", "L<=3", "Ps=0", "0<Ps<=.5", "Ps>.5", "D>=.75", "zero-loss bursts",
                      "zero-std groups", "clipped", "train reward"],
                     [[ARM_LABEL.get(a, a), d["n_picks"], pct(d["level5"]), pct(d["level4"]), pct(d["level_le3"]), pct(d["ps0"]),
                       pct(d["ps_0_half"]), pct(d["ps_gt_half"]), pct(d["d_ge_075"]), pct(d["zero_loss_bursts"]),
                       pct(d["mean_frac_reward_zero_std"]), pct(d["mean_clipped_ratio"]), _f(d["mean_train_reward"])]
                      for a, d in diag["pick_distribution"].items()]
                     + [["Original NB-M log", 20, pct(diag["nbm_original"]["level5"]), pct(diag["nbm_original"]["level4"]),
                         pct(diag["nbm_original"]["level_le3"]), pct(diag["nbm_original"]["ps0"]),
                         pct(diag["nbm_original"]["ps_0_half"]), pct(diag["nbm_original"]["ps_gt_half"]),
                         pct(diag["nbm_original"]["d_ge_075"]), pct(diag["nbm_original"]["zero_loss_bursts"]), "—", "—", "—"]]),
           ""]

    # --- F. runtime
    rt = runtime(col)
    data["runtime"] = rt
    md += ["### F. Runtime and memory", "",
           _md_table(["Run", "Steps", "Curriculum (h)", "Evals (h)", "Total (h)", "Peak GPU mem (GB)", "Status"],
                     [[k, v["steps_done"], _f(v["curriculum_s"] / 3600, 2), _f(v["eval_s"] / 3600, 2), _f(v["total_h"], 2),
                       _f(v["peak_mem_gb"], 2), v["status"]] for k, v in sorted(rt.items())]), "",
           f"Original NB-M loop: {nbm.NBM_LOOP_WALL_S / 3600:.2f} h on a Kaggle T4 (one run).", ""]
    return "\n".join(md), data


def build_report(specs: Sequence[SgacSpec], out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "reports" / "sgac_repro"
    out_dir.mkdir(parents=True, exist_ok=True)
    sections, data = [], {}
    primary = None
    for spec in specs:
        col = collect(spec)
        md, d = profile_section(col, out_dir)
        sections.append(md)
        data[spec.profile.name] = d
        if spec.profile.name == "e0":
            primary = primary_endpoints(col, int(spec.loop.steps))
    v = verdict((primary or {}).get("P1"), (primary or {}).get("P2"))
    data["primary_endpoints"] = primary
    data["verdict"] = v
    head = [
        "# SGAC reproduction report", "",
        "Generated by `python -m rlvr_v2.sgac report`. Protocol, verdict rule and deviations: "
        "`paper/sgac_repro_protocol.md`. Every number below is computed from `results_sgac/`.", "",
        "## H. Verdict (pre-declared rule, e0 profile, MATH-500, step 20, seeds pooled)", "",
        f"**{v['label']}**" + (f" — {'; '.join(v.get('notes') or [])}" if v.get("notes") else ""), "",
        _md_table(["Endpoint", "Δ accuracy (pp) [95% CI]", "Seeds", "Items"],
                  [["P1: SGAC − Base", delta_str((primary or {}).get("P1")), ",".join(map(str, ((primary or {}).get("P1") or {}).get("seeds", []))),
                    ((primary or {}).get("P1") or {}).get("n", "—")],
                   ["P2: SGAC − Random", delta_str((primary or {}).get("P2")), ",".join(map(str, ((primary or {}).get("P2") or {}).get("seeds", []))),
                    ((primary or {}).get("P2") or {}).get("n", "—")]]), "",
        "## B. Selector coefficients actually used", "",
        "`Score = 0.0050·Ps + 0.1832·Var − 0.0751·D + 0.2188·L`, first-maximum argmax, no intercept: the hardcoded line of "
        "the published notebook (paper Eq. 10). It is the N=20 phase-4 regression printed with labels shifted by one "
        "feature; the regression itself (`learned_selector.pkl`) means `0.0050·L + 0.1832·Ps − 0.0751·Var + 0.2188·D + "
        "0.279` (optional label-corrected arm). Paper Table 2 is the N=4 fit with the same shift and was never run.", "",
    ]
    report = "\n".join(head + sections)
    path = out_dir / "REPORT.md"
    path.write_text(report + "\n", encoding="utf-8")
    atomic_write_json(out_dir / "report_data.json", _jsonable(data))
    return path


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [None if (isinstance(x, float) and math.isnan(x)) else float(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj
