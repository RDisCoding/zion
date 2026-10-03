#!/usr/bin/env python
"""Study-2 analysis: curriculum arms x seeds x checkpoints on MATH-500.

Inputs   results/tables/{study2_evals,eval_items,selections,train_metrics,runs}.parquet (from `rlvr_v2.aggregate`).
Outputs  paper/tables/study2_*.tex, paper/tables/study2_summary.json, paper/figures/study2_curves.pdf.

Pre-registered comparison: every arm versus the random baseline at the final checkpoint, by a paired item
bootstrap (10,000 resamples) over the seed-averaged per-item correctness; secondarily the area under the
accuracy-vs-step curve (AUC, normalised by the step span) compared the same way on per-item AUCs (AUC is linear
in the per-item correctness, so item resampling applies unchanged). Curves report the mean over seeds with an
item-bootstrap band on the seed-averaged correctness. Selection diagnostics per arm come from the per-step
selections (mean p_s, shares with p_s == 1 / p_s == 0, mean d_simpson, repeat rate) and the mean fraction of
zero-reward-std groups from the training metrics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlvr_v2 import stats as st  # noqa: E402
from rlvr_v2.aggregate import read_tables  # noqa: E402
from rlvr_v2.reporting import (ARM_COLORS, PALETTE, Raw, fmt_ci, fmt_num, fmt_p, setup_matplotlib, tex_table,  # noqa: E402
                               write_json)

log = logging.getLogger("study2_analysis")

ARM_ORDER: tuple[str, ...] = ("random", "variance", "disagreement", "ps_band", "learned", "repeat_one")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--candidates", default="manifests/study1_candidates.json", help="unused; accepted for symmetry")
    ap.add_argument("--tables", default=None, help="tables directory (default <results>/tables)")
    ap.add_argument("--baseline", default="random")
    ap.add_argument("--final-step", type=int, default=None, help="override the final checkpoint step")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def arm_sort_key(arm: str):
    return (ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER), str(arm))


# ----------------------------------------------------------------------------------------- inputs
def curriculum_items(eval_items: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Per-item correctness of curriculum checkpoints with arm/seed attached, seed-averaged per (arm, step, item)."""
    ev = eval_items[eval_items["step"].notna()].copy()
    if ev.empty:
        return pd.DataFrame(columns=["arm", "step", "unique_id", "correct", "n_seeds"])
    meta = runs[["run_id", "arm", "seed"]].drop_duplicates("run_id")
    ev = ev.merge(meta, on="run_id", how="left")
    ev = ev[ev["arm"].notna()]
    ev["step"] = ev["step"].astype(int)
    ev["correct"] = ev["correct"].astype(float)
    out = ev.groupby(["arm", "step", "unique_id"], as_index=False).agg(correct=("correct", "mean"), n_seeds=("run_id", "nunique"))
    return out


def curve_rows(evals: pd.DataFrame, items: pd.DataFrame, n_boot: int, seed: int) -> list[dict]:
    rows = []
    for (arm, step), g in evals.groupby(["arm", "step"]):
        acc = g["acc"].to_numpy(dtype=float)
        sub = items[(items["arm"] == arm) & (items["step"] == step)]
        if len(sub):
            mean, lo, hi = st.bootstrap_mean_ci(sub["correct"].to_numpy(dtype=float), n_boot=min(n_boot, 2000), seed=seed)
        else:
            mean, lo, hi = float("nan"), float("nan"), float("nan")
        rows.append({"arm": arm, "step": int(step), "n_seeds": int(len(acc)), "acc_mean": float(np.nanmean(acc)),
                     "acc_sd": float(np.nanstd(acc, ddof=1)) if len(acc) > 1 else float("nan"),
                     "acc_se": float(np.nanstd(acc, ddof=1) / np.sqrt(len(acc))) if len(acc) > 1 else float("nan"),
                     "item_mean": mean, "item_ci_lo": lo, "item_ci_hi": hi, "n_items": int(len(sub)),
                     "seeds": sorted(int(s) for s in g["seed"].dropna().unique())})
    rows.sort(key=lambda r: (arm_sort_key(r["arm"]), r["step"]))
    return rows


def common_steps(evals: pd.DataFrame) -> list[int]:
    """Steps evaluated by every (arm, seed) run."""
    sets = [set(g["step"].astype(int)) for _, g in evals.groupby(["arm", "seed"], dropna=False)]
    return sorted(set.intersection(*sets)) if sets else []


def _item_matrix(items: pd.DataFrame, arm: str, steps: list[int]) -> pd.DataFrame:
    sub = items[(items["arm"] == arm) & (items["step"].isin(steps))]
    return sub.pivot_table(index="unique_id", columns="step", values="correct", aggfunc="mean").reindex(columns=steps)


def comparisons(items: pd.DataFrame, arms: list[str], baseline: str, final: int, steps: list[int], n_boot: int,
                seed: int) -> list[dict]:
    rows = []
    if baseline not in arms:
        log.warning("baseline arm %r has no evaluations; comparisons skipped", baseline)
        return rows
    base_final = items[(items["arm"] == baseline) & (items["step"] == final)].set_index("unique_id")["correct"]
    base_auc = _item_matrix(items, baseline, steps).dropna()
    for arm in arms:
        if arm == baseline:
            continue
        row: dict = {"arm": arm, "baseline": baseline, "final_step": final}
        af = items[(items["arm"] == arm) & (items["step"] == final)].set_index("unique_id")["correct"]
        common = af.index.intersection(base_final.index)
        if len(common):
            res = st.paired_item_bootstrap(af.loc[common].to_numpy(dtype=float), base_final.loc[common].to_numpy(dtype=float),
                                           n_boot=n_boot, seed=seed)
            row.update({f"final_{k}": v for k, v in res.items()})
        else:
            log.warning("arm %s: no items shared with the baseline at step %s", arm, final)
        if len(steps) >= 2:
            am = _item_matrix(items, arm, steps).dropna()
            idx = am.index.intersection(base_auc.index)
            if len(idx):
                span = float(steps[-1] - steps[0])
                auc_a = np.trapezoid(am.loc[idx].to_numpy(dtype=float), np.array(steps, dtype=float), axis=1) / span
                auc_b = np.trapezoid(base_auc.loc[idx].to_numpy(dtype=float), np.array(steps, dtype=float), axis=1) / span
                res = st.paired_item_bootstrap(auc_a, auc_b, n_boot=n_boot, seed=seed)
                row.update({f"auc_{k}": v for k, v in res.items()})
        rows.append(row)
    return rows


def per_run_auc(evals: pd.DataFrame, steps: list[int]) -> pd.DataFrame:
    rows = []
    for (arm, seed, run_id), g in evals.groupby(["arm", "seed", "run_id"], dropna=False):
        g = g[g["step"].isin(steps)].sort_values("step")
        if len(g) == 0:
            continue
        rows.append({"arm": arm, "seed": seed, "run_id": run_id,
                     "auc": st.auc_over_steps(g["step"].to_numpy(dtype=float), g["acc"].to_numpy(dtype=float)),
                     "final_acc": float(g["acc"].iloc[-1]), "n_steps": int(len(g))})
    return pd.DataFrame(rows, columns=["arm", "seed", "run_id", "auc", "final_acc", "n_steps"])


def selection_diagnostics(selections: pd.DataFrame, train_metrics: pd.DataFrame, runs: pd.DataFrame) -> list[dict]:
    rows = []
    arms = sorted(set(selections["arm"].dropna()) | set(runs.loc[runs["kind"] == "curriculum", "arm"].dropna()), key=arm_sort_key)
    tm = train_metrics.merge(runs[["run_id", "arm", "kind"]].drop_duplicates("run_id"), on="run_id", how="left")
    tm = tm[tm["kind"] == "curriculum"]
    for arm in arms:
        s = selections[selections["arm"] == arm]
        ps = pd.to_numeric(s["p_s"], errors="coerce")
        dw = pd.to_numeric(s["d_wrong"], errors="coerce")
        ds = pd.to_numeric(s["d_simpson"], errors="coerce")
        t = pd.to_numeric(tm.loc[tm["arm"] == arm, "frac_reward_zero_std"], errors="coerce")
        rows.append({
            "arm": arm, "n_selections": int(len(s)), "n_runs": int(s["run_id"].nunique()),
            "mean_p_s": float(ps.mean()) if ps.notna().any() else float("nan"),
            "share_p_s_1": float((ps >= 1.0 - 1e-9).mean()) if ps.notna().any() else float("nan"),
            "share_p_s_0": float((ps <= 1e-9).mean()) if ps.notna().any() else float("nan"),
            "mean_d_simpson": float(ds.mean()) if ds.notna().any() else float("nan"),
            "mean_d_wrong": float(dw.mean()) if dw.notna().any() else float("nan"),
            "share_d_wrong_defined": float(dw.notna().mean()) if len(s) else float("nan"),
            "repeat_rate": float(1.0 - s["selected_uid"].nunique() / len(s)) if len(s) else float("nan"),
            "mean_level": float(pd.to_numeric(s["level"], errors="coerce").mean()) if len(s) else float("nan"),
            "zero_std_frac_mean": float(t.mean()) if t.notna().any() else float("nan"),
            "n_train_steps": int(t.notna().sum()),
        })
    return rows


# ----------------------------------------------------------------------------------------- outputs
def write_tables(paper: Path, summary: dict) -> None:
    t = paper / "tables"
    final_by_arm = {r["arm"]: r for r in summary["curves"] if r["step"] == summary["final_step"]}
    comp = {r["arm"]: r for r in summary["comparisons"]}
    rows = []
    for arm in sorted(final_by_arm, key=arm_sort_key):
        c, f = comp.get(arm, {}), final_by_arm[arm]
        rows.append([arm, str(f["n_seeds"]), f"{fmt_num(f['acc_mean'], 4)} ({fmt_num(f['acc_sd'], 4)})",
                     fmt_ci(f["item_ci_lo"], f["item_ci_hi"], 4),
                     "--" if arm == summary["baseline"] else fmt_num(c.get("final_delta"), 4, signed=True),
                     "--" if arm == summary["baseline"] else fmt_ci(c.get("final_ci_lo"), c.get("final_ci_hi"), 4),
                     "--" if arm == summary["baseline"] else fmt_p(c.get("final_p_two_sided"))])
    tex_table(["Arm", "seeds", "final acc (sd)", "item 95% CI", Raw(f"$\\Delta$ vs {summary['baseline']}"), "95% CI", "p"], rows,
              path=t / "study2_final.tex",
              caption=f"Final-checkpoint (step {summary['final_step']}) MATH-500 accuracy per arm; paired item bootstrap "
                      f"over seed-averaged correctness ({summary['args']['n_boot']} resamples).", label="tab:study2-final")
    auc = {r["arm"]: r for r in summary["auc_per_arm"]}
    rows = []
    for arm in sorted(auc, key=arm_sort_key):
        a, c = auc[arm], comp.get(arm, {})
        rows.append([arm, str(a["n_runs"]), f"{fmt_num(a['auc_mean'], 4)} ({fmt_num(a['auc_sd'], 4)})",
                     "--" if arm == summary["baseline"] else fmt_num(c.get("auc_delta"), 4, signed=True),
                     "--" if arm == summary["baseline"] else fmt_ci(c.get("auc_ci_lo"), c.get("auc_ci_hi"), 4),
                     "--" if arm == summary["baseline"] else fmt_p(c.get("auc_p_two_sided"))])
    tex_table(["Arm", "runs", "AUC (sd)", Raw(f"$\\Delta$AUC vs {summary['baseline']}"), "95% CI", "p"], rows,
              path=t / "study2_auc.tex",
              caption=f"Area under the accuracy-vs-step curve over steps {summary['steps_common']} (normalised by the span).",
              label="tab:study2-auc")
    rows = [[d["arm"], str(d["n_selections"]), fmt_num(d["mean_p_s"]), fmt_num(d["share_p_s_1"]), fmt_num(d["share_p_s_0"]),
             fmt_num(d["mean_d_simpson"]), fmt_num(d["mean_d_wrong"]), fmt_num(d["repeat_rate"]), fmt_num(d["zero_std_frac_mean"])]
            for d in summary["selection_diagnostics"]]
    tex_table(["Arm", "selections", Raw("mean $p_s$"), Raw("share $p_s{=}1$"), Raw("share $p_s{=}0$"),
               Raw("mean $d_{simpson}$"), Raw("mean $d_{wrong}$"), "repeat rate", "zero-std frac."], rows,
              path=t / "study2_selection.tex",
              caption="Selection diagnostics per arm and mean fraction of zero-reward-std groups during training.",
              label="tab:study2-selection")
    rows = [[r["arm"], str(r["step"]), str(r["n_seeds"]), fmt_num(r["acc_mean"], 4), fmt_num(r["acc_sd"], 4),
             fmt_ci(r["item_ci_lo"], r["item_ci_hi"], 4)] for r in summary["curves"]]
    tex_table(["Arm", "step", "seeds", "mean acc", "sd", "item 95% CI"], rows, path=t / "study2_curves.tex",
              caption="Accuracy per arm and checkpoint (mean over seeds).", label="tab:study2-curves")


def fig_curves(plt, curves: list[dict], baseline: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    arms = sorted({r["arm"] for r in curves}, key=arm_sort_key)
    free = [c for c in PALETTE["series"] if c not in ARM_COLORS.values()]
    for arm in arms:
        rows = sorted((r for r in curves if r["arm"] == arm), key=lambda r: r["step"])
        x = np.array([r["step"] for r in rows], dtype=float)
        y = np.array([r["acc_mean"] for r in rows], dtype=float)
        lo = np.array([r["item_ci_lo"] for r in rows], dtype=float)
        hi = np.array([r["item_ci_hi"] for r in rows], dtype=float)
        colour = ARM_COLORS.get(arm) or (free.pop(0) if free else PALETTE["muted"])
        if np.isfinite(lo).any():
            ax.fill_between(x, lo, hi, color=colour, alpha=0.12, linewidth=0)
        ax.plot(x, y, color=colour, marker="o", ms=4, markeredgecolor=PALETTE["surface"], markeredgewidth=0.6,
                label=arm, zorder=3)
        if len(x):
            ax.annotate(arm, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7,
                        color=PALETTE["ink2"], va="center")
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("curriculum step")
    ax.set_ylabel("MATH-500 accuracy (mean over seeds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=min(4, max(1, len(arms))), title=None)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s | %(message)s")
    results, paper = _abs(args.results), _abs(args.paper)
    tables_dir = _abs(args.tables) if args.tables else results / "tables"
    tabs = read_tables(tables_dir, ("study2_evals", "eval_items", "selections", "train_metrics", "runs"))
    evals = tabs["study2_evals"].copy()
    evals = evals[evals["step"].notna() & evals["arm"].notna()]
    evals["step"] = evals["step"].astype(int)
    evals["acc"] = pd.to_numeric(evals["acc"], errors="coerce")
    items = curriculum_items(tabs["eval_items"], tabs["runs"])
    arms = sorted(evals["arm"].unique(), key=arm_sort_key)
    steps = common_steps(evals)
    if args.final_step is not None:
        final = int(args.final_step)
    elif steps:
        final = steps[-1]
    else:
        final = int(evals["step"].max()) if len(evals) else -1
        log.warning("no checkpoint common to all runs; using the overall last step %s", final)
    log.info("study2: %d evals, arms %s, common steps %s, final step %s", len(evals), arms, steps, final)

    summary: dict = {"args": vars(args), "baseline": args.baseline, "arms": list(arms), "steps_common": steps,
                     "final_step": final, "n_evals": int(len(evals)),
                     "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    summary["curves"] = curve_rows(evals, items, args.n_boot, args.seed)
    summary["comparisons"] = comparisons(items, list(arms), args.baseline, final, steps, args.n_boot, args.seed)
    pra = per_run_auc(evals, steps)
    summary["auc_per_run"] = pra.to_dict(orient="records")
    summary["auc_per_arm"] = [
        {"arm": arm, "n_runs": int(len(g)), "auc_mean": float(g["auc"].mean()),
         "auc_sd": float(g["auc"].std(ddof=1)) if len(g) > 1 else float("nan"),
         "final_acc_mean": float(g["final_acc"].mean())}
        for arm, g in pra.groupby("arm")]
    summary["auc_per_arm"].sort(key=lambda r: arm_sort_key(r["arm"]))
    summary["selection_diagnostics"] = selection_diagnostics(tabs["selections"], tabs["train_metrics"], tabs["runs"])

    (paper / "tables").mkdir(parents=True, exist_ok=True)
    (paper / "figures").mkdir(parents=True, exist_ok=True)
    write_tables(paper, summary)
    write_json(paper / "tables" / "study2_summary.json", summary)
    if not args.no_figures:
        plt = setup_matplotlib()
        if plt is None:
            log.warning("matplotlib not installed; figures skipped")
        else:
            fig_curves(plt, summary["curves"], args.baseline, paper / "figures" / "study2_curves.pdf")
    for c in summary["comparisons"]:
        log.info("%s vs %s at step %s: delta %s %s p=%s; AUC delta %s", c["arm"], c["baseline"], c["final_step"],
                 fmt_num(c.get("final_delta"), 4, signed=True), fmt_ci(c.get("final_ci_lo"), c.get("final_ci_hi"), 4),
                 fmt_p(c.get("final_p_two_sided")), fmt_num(c.get("auc_delta"), 4, signed=True))
    return summary


if __name__ == "__main__":
    main()
