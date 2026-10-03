#!/usr/bin/env python
"""Study-1 analysis: per-example transfer of 1-shot GRPO (pool signals -> Delta MATH-500).

Inputs   results/tables/study1.parquet (from `rlvr_v2.aggregate`) and the candidates manifest (pair structure).
Outputs  paper/tables/study1_*.tex, paper/tables/study1_summary.json, paper/figures/study1_*.pdf and the learned
         selector JSONs under configs/selectors/ (learned_v1_candidate.json always; learned_v1.json only when the
         pre-registered gate passes).

Pre-registered quantities (all thresholds are CLI arguments with the pre-registered defaults)
- H1 hump: Spearman rho(v_bin, Delta) with bootstrap CI; quadratic coefficient of (p_s - 0.5)^2 (<= 0 expected);
  Mann-Whitney middle bin vs the rest.
- H2 diversity beyond p_s: matched pairs (high minus low d_wrong) Wilcoxon + bootstrap CI of the mean difference;
  partial Spearman of d_simpson / entropy_bits / d_wrong with Delta given p_s (polynomial in rank space).
- H3 learned selector: ridge/lasso on standardised features with LOO; gate = LOO Spearman >= 0.30 and
  permutation p < 0.05 (1,000 label shuffles); LOO R^2 gain over the p_s-only model (p_s, v_bin) reported.
  p_s^2 enters through v_bin = p_s - p_s^2 (same linear span) so the exported selector only uses NUMERIC_FEATURES.
- Null results are accompanied by TOST equivalence tests (|rho| <= bound) and Holm adjustment is applied over the
  two primary tests (H1 Spearman, H2 matched pairs) and, secondarily, over all H1/H2 tests.
Outcomes are seed-averaged per candidate before any correlation; replicate seeds feed the noise decomposition.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlvr_v2 import stats as st  # noqa: E402
from rlvr_v2.aggregate import read_tables  # noqa: E402
from rlvr_v2.candidates import in_bin  # noqa: E402
from rlvr_v2.data import Manifest  # noqa: E402
from rlvr_v2.reporting import PALETTE, Raw, fmt_ci, fmt_num, fmt_p, setup_matplotlib, tex_table, write_json  # noqa: E402
from rlvr_v2.selectors import LearnedLinearSelector  # noqa: E402
from rlvr_v2.signals import NUMERIC_FEATURES  # noqa: E402

log = logging.getLogger("study1_analysis")

SELECTOR_FEATURES: tuple[str, ...] = ("p_s", "v_bin", "d_simpson", "d_wrong", "entropy_bits", "level", "len_mean")
PS_ONLY_FEATURES: tuple[str, ...] = ("p_s", "v_bin")
PARTIAL_FEATURES: tuple[str, ...] = ("d_simpson", "entropy_bits", "d_wrong")
PRIMARY_TESTS: tuple[str, ...] = ("H1_spearman_v_bin", "H2_pairs_wilcoxon")
DEFAULT_BINS: tuple[tuple[float, float], ...] = ((0.0, 0.25), (0.25, 0.75), (0.75, 0.9))


# ------------------------------------------------------------------------------------------ args
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--candidates", default="manifests/study1_candidates.json")
    ap.add_argument("--tables", default=None, help="tables directory (default <results>/tables)")
    ap.add_argument("--selectors-dir", default="configs/selectors")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--gate-spearman", type=float, default=0.30)
    ap.add_argument("--gate-perm-p", type=float, default=0.05)
    ap.add_argument("--gate-r2-gain", type=float, default=0.05)
    ap.add_argument("--tost-bound", type=float, default=0.2)
    ap.add_argument("--primary-model", choices=["ridge", "lasso"], default="ridge")
    ap.add_argument("--min-n-selector", type=int, default=8, help="minimum candidates for the LOO selector")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# ----------------------------------------------------------------------------------------- inputs
def load_manifest(path: Path) -> Manifest | None:
    if not path.exists():
        log.warning("candidates manifest %s not found; pair-based analyses are skipped", path)
        return None
    return Manifest.load(path)


def design_bins(manifest: Manifest | None) -> list[tuple[float, float]]:
    sel = (manifest.meta.get("selection") if manifest else None) or {}
    bins = sel.get("bins") or (sel.get("config") or {}).get("bins")
    return [(float(lo), float(hi)) for lo, hi in bins] if bins else list(DEFAULT_BINS)


def design_roles(manifest: Manifest | None) -> dict[str, dict]:
    roles: dict[str, dict] = {}
    sel = (manifest.meta.get("selection") if manifest else None) or {}
    for u in sel.get("anchors_null", []):
        roles[u] = {"role": "null_anchor", "bin_index": None, "pair_index": None}
    for u in sel.get("anchors_zero", []):
        roles[u] = {"role": "zero_anchor", "bin_index": None, "pair_index": None}
    for i, p in enumerate(sel.get("pairs", [])):
        for role in ("high", "low"):
            roles[p[role]] = {"role": role, "bin_index": p.get("bin_index"), "pair_index": i}
    return roles


def bin_index_of(p_s: float, bins: list[tuple[float, float]]) -> int | None:
    if p_s is None or not np.isfinite(p_s):
        return None
    for i, (lo, hi) in enumerate(bins):
        if in_bin(float(p_s), lo, hi):
            return i
    return None


def per_candidate(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-average the outcome per candidate; signal columns are taken from the lowest-seed run."""
    d = df.copy()
    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")
    d = d[np.isfinite(d["delta"]) & d["unique_id"].notna()]
    if d.empty:
        return pd.DataFrame(columns=["unique_id", "delta", "delta_sd", "n_seeds", "acc", "base_acc", *NUMERIC_FEATURES])
    for c in ("acc", "base_acc", "seed"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    agg = d.groupby("unique_id").agg(delta=("delta", "mean"), delta_sd=("delta", "std"), n_seeds=("delta", "size"),
                                     acc=("acc", "mean"), base_acc=("base_acc", "mean"))
    sig_cols = [c for c in (*NUMERIC_FEATURES, "subject", "k") if c in d.columns]
    first = d.sort_values("seed", na_position="last").groupby("unique_id")[sig_cols].first()
    out = agg.join(first).reset_index()
    for c in NUMERIC_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce") if c in out.columns else np.nan
    return out


# --------------------------------------------------------------------------------------- analyses
def spearman_block(pc: pd.DataFrame, n_boot: int, seed: int) -> list[dict]:
    rows = []
    y = pc["delta"].to_numpy(dtype=float)
    for f in NUMERIC_FEATURES:
        res = st.spearman_ci(pc[f].to_numpy(dtype=float), y, n_boot=n_boot, seed=seed)
        rows.append({"feature": f, **res})
    return rows


def partial_block(pc: pd.DataFrame, n_boot: int, seed: int) -> list[dict]:
    """Partial Spearman given [p_s, p_s^2]: both control columns are rank-transformed with squares added in
    rank space (``stats.partial_spearman``), so passing p_s^2 explicitly mirrors the pre-registration text."""
    rows = []
    ps = pc["p_s"].to_numpy(dtype=float)
    controls = np.column_stack([ps, ps**2])
    y = pc["delta"].to_numpy(dtype=float)
    for f in PARTIAL_FEATURES:
        res = st.partial_spearman(pc[f].to_numpy(dtype=float), y, controls, n_boot=n_boot, seed=seed)
        rows.append({"feature": f, **res})
    return rows


def pairs_block(pc: pd.DataFrame, manifest: Manifest | None, bins: list[tuple[float, float]], n_boot: int, seed: int) -> dict:
    sel = (manifest.meta.get("selection") if manifest else None) or {}
    delta = dict(zip(pc["unique_id"], pc["delta"]))
    rows, missing = [], 0
    for i, p in enumerate(sel.get("pairs", [])):
        if p["high"] in delta and p["low"] in delta:
            rows.append({"pair_index": i, "bin_index": p.get("bin_index"), "bin": p.get("bin"), "level": p.get("level"),
                         "high": p["high"], "low": p["low"], "delta_high": float(delta[p["high"]]),
                         "delta_low": float(delta[p["low"]]), "diff": float(delta[p["high"]] - delta[p["low"]]),
                         "dwrong_high": p.get("dwrong_high"), "dwrong_low": p.get("dwrong_low"),
                         "ps_high": p.get("ps_high"), "ps_low": p.get("ps_low")})
        else:
            missing += 1
    out: dict = {"n_pairs_design": len(sel.get("pairs", [])), "n_pairs_complete": len(rows), "n_pairs_missing": missing,
                 "pairs": rows, "overall": None, "per_bin": {}}
    if rows:
        hi = np.array([r["delta_high"] for r in rows])
        lo = np.array([r["delta_low"] for r in rows])
        out["overall"] = st.matched_pairs_test(hi, lo, n_boot=n_boot, seed=seed)
        for bi in sorted({r["bin_index"] for r in rows if r["bin_index"] is not None}):
            sub = [r for r in rows if r["bin_index"] == bi]
            res = st.matched_pairs_test(np.array([r["delta_high"] for r in sub]), np.array([r["delta_low"] for r in sub]),
                                        n_boot=n_boot, seed=seed)
            res["bin"] = list(bins[bi]) if bi < len(bins) else sub[0].get("bin")
            out["per_bin"][str(bi)] = res
    else:
        log.warning("no complete matched pairs available")
    return out


def selector_block(pc: pd.DataFrame, args: argparse.Namespace) -> dict:
    n = len(pc)
    out: dict = {"n": n, "features": list(SELECTOR_FEATURES), "baseline_features": list(PS_ONLY_FEATURES),
                 "alphas": [0.01, 0.1, 1.0, 10.0, 100.0], "models": {}, "primary_model": args.primary_model,
                 "gate": {"spearman_min": args.gate_spearman, "perm_p_max": args.gate_perm_p, "r2_gain_min": args.gate_r2_gain,
                          "passed": False, "reason": None}}
    if n < args.min_n_selector:
        out["gate"]["reason"] = f"skipped: {n} candidates < --min-n-selector {args.min_n_selector}"
        log.warning(out["gate"]["reason"])
        return out
    y = pc["delta"].to_numpy(dtype=float)
    X_full = pc[list(SELECTOR_FEATURES)].to_numpy(dtype=float)
    X_ps = pc[list(PS_ONLY_FEATURES)].to_numpy(dtype=float)
    for model in ("ridge", "lasso"):
        try:
            full = st.loo_linear_selector(X_full, y, SELECTOR_FEATURES, alphas=out["alphas"], model=model,
                                          n_perm=args.n_perm, seed=args.seed)
            base = st.loo_linear_selector(X_ps, y, PS_ONLY_FEATURES, alphas=out["alphas"], model=model,
                                          n_perm=args.n_perm, seed=args.seed)
        except (ValueError, RuntimeError) as e:
            log.warning("selector %s skipped: %s", model, e)
            out["models"][model] = {"error": str(e)}
            continue
        gain = full["loo_r2"] - base["loo_r2"] if np.isfinite(full["loo_r2"]) and np.isfinite(base["loo_r2"]) else float("nan")
        out["models"][model] = {"full": full, "ps_only": base, "r2_gain": gain,
                                "pass_spearman": bool(full["loo_spearman"] >= args.gate_spearman),
                                "pass_perm": bool(full["perm_p"] is not None and full["perm_p"] < args.gate_perm_p),
                                "pass_r2_gain": bool(np.isfinite(gain) and gain >= args.gate_r2_gain)}
    prim = out["models"].get(args.primary_model, {})
    if "full" in prim:
        out["gate"]["passed"] = bool(prim["pass_spearman"] and prim["pass_perm"])
        out["gate"]["r2_gain_passed"] = prim["pass_r2_gain"]
        out["gate"]["reason"] = (f"{args.primary_model}: LOO Spearman {prim['full']['loo_spearman']:.3f} "
                                 f"(>= {args.gate_spearman}: {prim['pass_spearman']}), perm p {prim['full']['perm_p']} "
                                 f"(< {args.gate_perm_p}: {prim['pass_perm']}), R2 gain {prim['r2_gain']:.3f} "
                                 f"(>= {args.gate_r2_gain}: {prim['pass_r2_gain']})")
    else:
        out["gate"]["reason"] = f"primary model {args.primary_model} unavailable"
    return out


def export_selectors(selector: dict, out_dir: Path, args: argparse.Namespace, manifest_name: str | None) -> dict:
    prim = selector["models"].get(args.primary_model, {})
    paths: dict[str, str | None] = {"candidate": None, "learned_v1": None}
    if "full" not in prim:
        return paths
    full = prim["full"]
    meta = {
        "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "analysis/study1_analysis.py", "candidates_manifest": manifest_name, "model": full["model"],
        "alpha": full["alpha"], "n": full["n"], "loo_spearman": full["loo_spearman"], "loo_r2": full["loo_r2"],
        "perm_p": full["perm_p"], "n_perm": full["n_perm"], "r2_gain_over_ps_only": prim["r2_gain"],
        "gate": {k: v for k, v in selector["gate"].items()}, "coef_units": "Delta MATH-500 accuracy per 1 SD of feature",
        "note": "v_bin = p_s * (1 - p_s) carries the quadratic p_s term",
    }
    sel = LearnedLinearSelector(features=tuple(full["features"]), coef=dict(full["coef"]), intercept=float(full["intercept"]),
                                standardize=full["standardize"], meta=meta)
    cand = out_dir / "learned_v1_candidate.json"
    sel.to_json(cand)
    paths["candidate"] = str(cand)
    if selector["gate"]["passed"]:
        final = out_dir / "learned_v1.json"
        sel.to_json(final)
        paths["learned_v1"] = str(final)
        log.info("learned selector passed the gate -> %s", final)
    else:
        log.info("learned selector did NOT pass the gate (%s); only the candidate file was written", selector["gate"]["reason"])
    return paths


def noise_block(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["delta"] = pd.to_numeric(d["delta"], errors="coerce")
    d = d[np.isfinite(d["delta"]) & d["unique_id"].notna()]
    groups = {u: g["delta"].tolist() for u, g in d.groupby("unique_id") if len(g) >= 2}
    res = st.noise_decomposition(groups)
    res["replicated_candidates"] = sorted(groups)
    res["n_runs"] = int(len(d))
    return res


def tost_block(spearman_rows: list[dict], partial_rows: list[dict], selector: dict, args: argparse.Namespace) -> dict:
    out = {}

    def add(name: str, rho, n, p):
        if n is None or n <= 3 or rho is None or not np.isfinite(rho):
            out[name] = {"equivalent": False, "p_lower": None, "p_upper": None, "rho": rho, "n": n, "note": "n too small"}
            return
        res = st.tost_correlation(float(rho), int(n), bound=args.tost_bound, alpha=args.alpha)
        res["significant"] = bool(p is not None and p < args.alpha)
        out[name] = res

    for r in spearman_rows:
        if r["feature"] in ("v_bin", "p_s", "d_simpson", "entropy_bits", "d_wrong"):
            add(f"spearman_{r['feature']}", r["rho"], r["n"], r["p"])
    for r in partial_rows:
        add(f"partial_{r['feature']}", r["rho"], r["n"], r["p"])
    for model, m in selector.get("models", {}).items():
        if "full" in m:
            add(f"loo_{model}", m["full"]["loo_spearman"], m["full"]["n"], m["full"]["perm_p"])
    return out


def holm_block(spearman_rows: list[dict], hump: dict, pairs: dict, partial_rows: list[dict]) -> dict:
    sp = {r["feature"]: r for r in spearman_rows}
    raw = {
        "H1_spearman_v_bin": sp.get("v_bin", {}).get("p"),
        "H1_quad_coef": hump.get("quad_p"),
        "H1_mid_vs_extreme": hump.get("mid_vs_extreme_p"),
        "H2_pairs_wilcoxon": (pairs.get("overall") or {}).get("wilcoxon_p"),
    }
    for r in partial_rows:
        raw[f"H2_partial_{r['feature']}"] = r["p"]
    return {"raw": raw, "holm_primary": st.holm({k: raw[k] for k in PRIMARY_TESTS}), "holm_all": st.holm(raw)}


# ----------------------------------------------------------------------------------------- tables
def _bin_label(b) -> str:
    return "--" if b is None else f"({b[0]:.2f}, {b[1]:.2f}]"


def write_tables(paper: Path, pc: pd.DataFrame, roles: dict, bins, summary: dict) -> None:
    t = paper / "tables"
    rows = [[r["feature"], str(r["n"]), fmt_num(r["rho"], signed=True), fmt_ci(r["ci_lo"], r["ci_hi"]), fmt_p(r["p"])]
            for r in summary["spearman"]]
    tex_table(["Feature", "n", Raw("Spearman $\\rho$"), "95% CI", "p"], rows, path=t / "study1_spearman.tex",
              caption=Raw("Spearman correlation of every pool signal with the seed-averaged $\\Delta$ MATH-500 accuracy."),
              label="tab:study1-spearman")
    rows = [[r["feature"], str(r["n"]), fmt_num(r["rho"], signed=True), fmt_ci(r["ci_lo"], r["ci_hi"]), fmt_p(r["p"])]
            for r in summary["partial"]]
    tex_table(["Feature", "n", Raw("Partial $\\rho$ given $p_s, p_s^2$"), "95% CI", "p"], rows,
              path=t / "study1_partial.tex",
              caption=Raw("Partial Spearman correlation with $\\Delta$ controlling for $p_s$ and $p_s^2$ "
                          "(polynomial in rank space)."), label="tab:study1-partial")
    h = summary["hump"]
    rows = [[Raw("quadratic coefficient ($c = p_s - 0.5$)"), fmt_num(h["quad_coef"], 4, signed=True),
             fmt_ci(h["quad_ci_lo"], h["quad_ci_hi"], 4), fmt_p(h["quad_p"]) + " / " + fmt_p(h["quad_p_one_sided"])],
            ["linear coefficient", fmt_num(h["lin_coef"], 4, signed=True), "--", "--"],
            ["intercept", fmt_num(h["intercept"], 4, signed=True), "--", "--"],
            [Raw("$R^2$"), fmt_num(h["r2"]), "--", "--"],
            [Raw(f"Mann--Whitney $U$, middle bin {_bin_label(h['mid_bin'])} vs rest ($n$ = {h['n_mid']} / {h['n_extreme']})"),
             fmt_num(h["mid_vs_extreme_u"], 1), f"means {fmt_num(h['mean_mid'], 4)} vs {fmt_num(h['mean_extreme'], 4)}",
             fmt_p(h["mid_vs_extreme_p"]) + " / " + fmt_p(h["mid_vs_extreme_p_greater"])]]
    tex_table(["Quantity", "Estimate", "95% CI / detail", "p (two-sided / one-sided)"], rows, colspec="lrrr",
              path=t / "study1_hump.tex",
              caption=Raw("H1 hump test: OLS $\\Delta \\sim 1 + c + c^2$ with $c = p_s - 0.5$, and the middle-bin contrast."),
              label="tab:study1-hump")
    p = summary["pairs"]
    rows = []

    def pair_row(name, res):
        return [name, str(res["n_pairs"]), fmt_num(res["mean_diff"], 4, signed=True), fmt_ci(res["ci_lo"], res["ci_hi"], 4),
                f"{res['n_positive']}/{res['n_negative']}/{res['n_zero']}", fmt_p(res["wilcoxon_p"]), fmt_p(res["sign_test_p"])]

    for bi, res in sorted(p["per_bin"].items(), key=lambda kv: int(kv[0])):
        rows.append(pair_row(_bin_label(res.get("bin")), res))
    if p["overall"]:
        rows.append(pair_row("all bins", p["overall"]))
    tex_table([Raw("$p_s$ bin"), "pairs", Raw("mean $\\Delta$(high $-$ low)"), "95% CI", Raw("$+/-/0$"), "Wilcoxon p", "sign p"],
              rows, path=t / "study1_pairs.tex",
              caption=Raw("H2 matched pairs: high minus low $d_{wrong}$ member within level and $p_s$ bin."),
              label="tab:study1-pairs")
    s = summary["selector"]
    rows = []
    for model, m in s["models"].items():
        if "full" not in m:
            rows.append([model, "--", "--", "--", "--", "--", "--", m.get("error", "skipped")])
            continue
        f, b = m["full"], m["ps_only"]
        rows.append([model, str(f["alpha"]), fmt_num(f["loo_spearman"]), fmt_p(f["perm_p"]), fmt_num(f["loo_r2"]),
                     fmt_num(b["loo_r2"]), fmt_num(m["r2_gain"], signed=True),
                     "pass" if (m["pass_spearman"] and m["pass_perm"]) else "fail"])
    tex_table(["Model", Raw("$\\alpha$"), "LOO Spearman", "perm. p", Raw("LOO $R^2$"), Raw("LOO $R^2$ ($p_s$ only)"),
               Raw("$R^2$ gain"), "gate"], rows, path=t / "study1_selector.tex",
              caption=Raw(f"H3 learned selector (features: {', '.join(SELECTOR_FEATURES).replace('_', chr(92) + '_')}); "
                          f"gate = LOO Spearman $\\ge {s['gate']['spearman_min']}$ and permutation "
                          f"$p < {s['gate']['perm_p_max']}$."),
              label="tab:study1-selector")
    coef_models = [m for m in ("ridge", "lasso") if "full" in s["models"].get(m, {})]
    if coef_models:
        rows = []
        for feat in SELECTOR_FEATURES:
            rows.append([feat] + [fmt_num(s["models"][m]["full"]["coef"].get(feat), 4, signed=True) for m in coef_models])
        rows.append(["intercept"] + [fmt_num(s["models"][m]["full"]["intercept"], 4, signed=True) for m in coef_models])
        tex_table(["Feature"] + [f"{m} coef." for m in coef_models], rows, path=t / "study1_selector_coef.tex",
                  caption="Learned selector coefficients (Delta per one SD of the standardised feature).",
                  label="tab:study1-selector-coef")
    nz = summary["noise"]
    rows = [["sigma_seed (within-candidate SD across seeds)", fmt_num(nz["sigma_seed"], 4)],
            ["sigma_between (SD of candidate effects)", fmt_num(nz["sigma_between"], 4)],
            ["ICC(1)", fmt_num(nz["icc"])], ["replicated candidates", str(nz["n_groups"])],
            ["runs used", str(nz["n_runs"])]]
    tex_table(["Quantity", "Value"], rows, path=t / "study1_noise.tex",
              caption="Seed-noise decomposition from replicated candidates.", label="tab:study1-noise")
    hb = summary["holm"]
    rows = [[k, fmt_p(v), fmt_p(hb["holm_primary"].get(k)), fmt_p(hb["holm_all"].get(k))] for k, v in hb["raw"].items()]
    tex_table(["Test", "raw p", "Holm (primary)", "Holm (all)"], rows, path=t / "study1_holm.tex",
              caption="Holm step-down adjustment over the primary H1/H2 tests and over all H1/H2 tests.",
              label="tab:study1-holm")
    rows = [[k, fmt_num(v.get("rho"), signed=True), str(v.get("n")), fmt_ci(v.get("ci_lo"), v.get("ci_hi")),
             "yes" if v.get("equivalent") else "no", "yes" if v.get("significant") else "no"]
            for k, v in summary["tost"].items()]
    tex_table(["Correlation", Raw("$\\rho$"), "n", "90% CI", "equivalent", "significant"], rows, path=t / "study1_tost.tex",
              caption=Raw(f"TOST equivalence tests ($|\\rho| \\le {summary['args']['tost_bound']}$) via Fisher $z$."),
              label="tab:study1-tost")
    rows = []
    for _, r in pc.sort_values(["p_s", "unique_id"]).iterrows():
        role = roles.get(r["unique_id"], {})
        bi = role.get("bin_index")
        if bi is None:
            bi = bin_index_of(r["p_s"], bins)
        rows.append([r["unique_id"], role.get("role", "--"), _bin_label(bins[bi]) if bi is not None and bi < len(bins) else "--",
                     "--" if pd.isna(r.get("level")) else str(int(r["level"])), fmt_num(r["p_s"]), fmt_num(r["d_wrong"]),
                     fmt_num(r["d_simpson"]), fmt_num(r["delta"], 4, signed=True), str(int(r["n_seeds"]))])
    tex_table(["Candidate", "role", "bin", "level", Raw("$p_s$"), Raw("$d_{wrong}$"), Raw("$d_{simpson}$"), Raw("$\\Delta$"),
               "seeds"], rows, path=t / "study1_candidates.tex", caption="Study-1 candidates with seed-averaged outcomes.",
              label="tab:study1-candidates")


# ---------------------------------------------------------------------------------------- figures
def _binned_mean(x: np.ndarray, y: np.ndarray, edges: np.ndarray, min_n: int = 2):
    xs, ys, es = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & ((x < hi) if hi < edges[-1] else (x <= hi))
        if m.sum() >= min_n:
            xs.append(0.5 * (lo + hi))
            ys.append(float(y[m].mean()))
            es.append(float(y[m].std(ddof=1) / math.sqrt(m.sum())) if m.sum() > 1 else 0.0)
    return np.array(xs), np.array(ys), np.array(es)


def fig_delta_vs_ps(plt, pc: pd.DataFrame, roles: dict, bins, path: Path) -> None:
    x = pc["p_s"].to_numpy(dtype=float)
    y = pc["delta"].to_numpy(dtype=float)
    anchor = np.array([roles.get(u, {}).get("role") in ("null_anchor", "zero_anchor") for u in pc["unique_id"]])
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    ax.axhline(0.0, color=PALETTE["axis"], lw=0.8, zorder=1)
    for _, hi in bins:
        ax.axvline(hi, color=PALETTE["grid"], lw=0.8, zorder=1)
    ax.scatter(x[~anchor], y[~anchor], s=24, color=PALETTE["series"][0], alpha=0.9, edgecolor=PALETTE["surface"],
               linewidth=0.6, label="candidate (seed mean)", zorder=3)
    if anchor.any():
        ax.scatter(x[anchor], y[anchor], s=30, facecolor=PALETTE["surface"], edgecolor=PALETTE["series"][0], linewidth=1.2,
                   label="anchor (p_s = 0 or 1)", zorder=3)
    bx, by, be = _binned_mean(x, y, np.linspace(0.0, 1.0, 11))
    if bx.size:
        ax.errorbar(bx, by, yerr=be, color=PALETTE["ink2"], lw=1.4, marker="o", ms=4, capsize=0, label="binned mean (s.e.)",
                    zorder=4)
    ax.set_xlabel("base pass rate $p_s$")
    ax.set_ylabel("$\\Delta$ MATH-500 accuracy")
    ax.set_xlim(-0.03, 1.03)
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def fig_pairs(plt, pairs: dict, bins, path: Path) -> None:
    rows = pairs.get("pairs", [])
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    ax.axhline(0.0, color=PALETTE["axis"], lw=0.8, zorder=1)
    if rows:
        order = sorted(rows, key=lambda r: ((r["bin_index"] if r["bin_index"] is not None else 99), r["pair_index"]))
        xpos, last_bin, x = [], None, 0.0
        ticks, tick_labels = [], []
        start = 0.0
        for r in order:
            if last_bin is not None and r["bin_index"] != last_bin:
                ticks.append(0.5 * (start + x - 1))
                tick_labels.append(_bin_label(bins[last_bin]) if last_bin is not None and last_bin < len(bins) else "--")
                x += 1.0
                start = x
            xpos.append(x)
            last_bin = r["bin_index"]
            x += 1.0
        ticks.append(0.5 * (start + x - 1))
        tick_labels.append(_bin_label(bins[last_bin]) if last_bin is not None and last_bin < len(bins) else "--")
        lo_c, hi_c = PALETTE["ordinal3"][0], PALETTE["ordinal3"][2]
        for xp, r in zip(xpos, order):
            ax.plot([xp, xp], [r["delta_low"], r["delta_high"]], color=PALETTE["axis"], lw=1.0, zorder=2)
        ax.scatter(xpos, [r["delta_low"] for r in order], s=26, color=lo_c, edgecolor=PALETTE["surface"], linewidth=0.6,
                   label="low d_wrong", zorder=3)
        ax.scatter(xpos, [r["delta_high"] for r in order], s=26, color=hi_c, edgecolor=PALETTE["surface"], linewidth=0.6,
                   label="high d_wrong", zorder=3)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("matched pairs by $p_s$ bin")
    else:
        ax.text(0.5, 0.5, "no complete pairs", ha="center", va="center", transform=ax.transAxes, color=PALETTE["muted"])
    ax.set_ylabel("$\\Delta$ MATH-500 accuracy")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def fig_delta_vs_dwrong(plt, pc: pd.DataFrame, bins, path: Path) -> None:
    d = pc[np.isfinite(pc["d_wrong"].to_numpy(dtype=float))]
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    ax.axhline(0.0, color=PALETTE["axis"], lw=0.8, zorder=1)
    colours = PALETTE["ordinal3"]
    for bi, (lo, hi) in enumerate(bins):
        m = np.array([bin_index_of(v, bins) == bi for v in d["p_s"].to_numpy(dtype=float)])
        if m.any():
            ax.scatter(d["d_wrong"].to_numpy(dtype=float)[m], d["delta"].to_numpy(dtype=float)[m], s=26,
                       color=colours[bi % len(colours)], edgecolor=PALETTE["surface"], linewidth=0.6,
                       label=f"$p_s$ in {_bin_label((lo, hi))}", zorder=3)
    ax.set_xlabel("wrong-answer diversity $d_{wrong}$")
    ax.set_ylabel("$\\Delta$ MATH-500 accuracy")
    ax.set_xlim(-0.03, 1.03)
    ax.legend(loc="best", title=None)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s | %(message)s")
    results, paper = _abs(args.results), _abs(args.paper)
    tables_dir = _abs(args.tables) if args.tables else results / "tables"
    df = read_tables(tables_dir, ("study1",))["study1"]
    manifest = load_manifest(_abs(args.candidates))
    bins = design_bins(manifest)
    roles = design_roles(manifest)
    pc = per_candidate(df)
    log.info("study1: %d runs, %d candidates with a finite delta", len(df), len(pc))
    if len(pc) < 3:
        log.error("fewer than 3 candidates with outcomes; nothing to analyse")
    for _, r in pc.iterrows():
        if r["unique_id"] in roles:
            roles[r["unique_id"]].setdefault("bin_index", bin_index_of(r["p_s"], bins))

    summary: dict = {"args": vars(args), "n_runs": int(len(df)), "n_candidates": int(len(pc)),
                     "n_runs_without_delta": int(len(df) - pd.to_numeric(df["delta"], errors="coerce").notna().sum()),
                     "candidates_manifest": manifest.name if manifest else None, "bins": [list(b) for b in bins],
                     "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    mid = bins[1] if len(bins) >= 3 else (0.25, 0.75)
    summary["spearman"] = spearman_block(pc, args.n_boot, args.seed) if len(pc) >= 3 else []
    summary["partial"] = partial_block(pc, args.n_boot, args.seed) if len(pc) >= 5 else []
    summary["hump"] = st.hump_test(pc["p_s"].to_numpy(dtype=float), pc["delta"].to_numpy(dtype=float), n_boot=args.n_boot,
                                   seed=args.seed, mid=mid)
    summary["pairs"] = pairs_block(pc, manifest, bins, args.n_boot, args.seed)
    summary["selector"] = selector_block(pc, args)
    summary["selector_files"] = export_selectors(summary["selector"], _abs(args.selectors_dir), args,
                                                 manifest.name if manifest else None)
    summary["noise"] = noise_block(df)
    summary["tost"] = tost_block(summary["spearman"], summary["partial"], summary["selector"], args)
    summary["holm"] = holm_block(summary["spearman"], summary["hump"], summary["pairs"], summary["partial"])
    summary["per_candidate"] = pc.to_dict(orient="records")

    (paper / "tables").mkdir(parents=True, exist_ok=True)
    (paper / "figures").mkdir(parents=True, exist_ok=True)
    write_tables(paper, pc, roles, bins, summary)
    write_json(paper / "tables" / "study1_summary.json", summary)
    if not args.no_figures:
        plt = setup_matplotlib()
        if plt is None:
            log.warning("matplotlib not installed; figures skipped")
        else:
            fig_delta_vs_ps(plt, pc, roles, bins, paper / "figures" / "study1_delta_vs_ps.pdf")
            fig_pairs(plt, summary["pairs"], bins, paper / "figures" / "study1_pairs.pdf")
            fig_delta_vs_dwrong(plt, pc, bins, paper / "figures" / "study1_delta_vs_dwrong.pdf")
    sp = {r["feature"]: r for r in summary["spearman"]}
    log.info("H1 Spearman(v_bin, Delta) = %s; quad coef = %s (p = %s); H2 pairs mean diff = %s; H3 gate: %s",
             fmt_num(sp.get("v_bin", {}).get("rho")), fmt_num(summary["hump"]["quad_coef"], 4), fmt_p(summary["hump"]["quad_p"]),
             fmt_num((summary["pairs"]["overall"] or {}).get("mean_diff"), 4), summary["selector"]["gate"]["reason"])
    return summary


if __name__ == "__main__":
    main()
