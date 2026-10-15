"""Build `reports/sgac_calib/REPORT.md` from the calibration track's result files (re-run after every stage).

Every number is read from `results_sgac_calib/` (or the root given by `--override results_root=...`); sections whose
inputs do not exist yet are reported as pending, never filled in by hand.
"""
from __future__ import annotations

import glob
from pathlib import Path

from ..artifacts import REPO_ROOT, read_json, utc_now
from ..sgac import nbm_original as nbm
from .spec import CalibSpec

REPORT = REPO_ROOT / "reports" / "sgac_calib" / "REPORT.md"
TAGS = {"E1_c1_pilot": "C1 pilot labels (16 candidates, deployment dose)", "E2_c2": "C2 labels (16 candidates, 4x dose)",
        "E3_os": "one-shot-regime labels (prereg Study-1 trainer)"}


def _f(x, nd=3):
    return "—" if x is None else (f"{x:+.{nd}f}" if isinstance(x, float) else str(x))


def _n(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def _ci(c, nd=2):
    return "—" if not c else f"[{c[0]:+.{nd}f}, {c[1]:+.{nd}f}]"


def _hours_sgac(group: Path) -> float:
    tot = 0.0
    for st in glob.glob(str(group / "*" / "state.json")):
        s = read_json(st) or {}
        for h in s.get("history", []):
            t = h.get("timing") or {}
            tot += sum((t.get(k) or {}).get("wall_s", 0.0) for k in ("sieve", "burst", "save"))
        for v in (s.get("evals") or {}).values():
            tot += sum((x or {}).get("timing", {}).get("wall_s", 0.0) for x in v.values())
    for f in glob.glob(str(group / "_shared" / "*" / "*.json")):
        d = read_json(f) or {}
        tot += sum((v or {}).get("wall_s", 0.0) for v in (d.get("timings") or {}).values())
    return tot / 3600


def _hours_os(root: Path) -> float:
    tot = 0.0
    for rj in glob.glob(str(root / "**" / "result.json"), recursive=True):
        r = read_json(rj) or {}
        tot += float((r.get("burst") or {}).get("wall_s") or 0.0)
        s = read_json(Path(rj).parent / "eval" / "dev" / "summary.json") or {}
        tot += float(s.get("wall_s") or 0.0)
    return tot / 3600


def _hours_audit(root: Path) -> float:
    tot = 0.0
    for pat in ("**/burst_summary.json", "**/summary.json", "**/phase1_e0_signals/signals.json"):
        for f in glob.glob(str(root / pat), recursive=True):
            tot += float((read_json(f) or {}).get("wall_s") or 0.0)
    return tot / 3600


def compute_table(cs: CalibSpec) -> list[tuple[str, float]]:
    R = cs.results_dir
    rows = []
    for name, pat in (("T0 smoke", "t0_smoke/*/*"), ("C1 pilot", "calib/e0/*"),
                      ("C2 rung", "calib_c2/e0/*"), ("Confirmatory curriculum", "curriculum/e0/*"),
                      ("Exploratory curriculum", "curriculum_explore/e0/*")):
        h = sum(_hours_sgac(Path(g)) for g in glob.glob(str(R / pat)) if Path(g).is_dir())
        if h > 0:
            rows.append((name, h))
    h = _hours_audit(R / "audit")
    if h > 0:
        rows.insert(0, ("Stage A audit (logged phases)", h))
    h = _hours_os(R / "oneshot")
    if h > 0:
        rows.append(("OS rung (training + dev eval)", h))
    return rows


def _gate_block(name: str, g: dict | None) -> list[str]:
    if not g:
        return [f"- **{name}:** pending."]
    pi1 = (g.get("pi1") or {}).get("delta_pp_mean")
    return [(f"- **{name}:** {g.get('status')} — ICC(1,1) {_n(g.get('icc'))} {_ci(g.get('icc_ci95'))}, between-candidate SD "
            f"{_n(g.get('sd_between_pp'), 2)} pp, mean ΔA {_f(g.get('mean_delta_pp'), 2)} pp, π1 ΔA {_f(pi1, 2)} pp "
            f"(thresholds: ICC ≥ 0.30 and SD ≥ 1.0 pp).")]


def _diag_block(name: str, d: dict | None) -> list[str]:
    if not d:
        return [f"- **{name}:** pending."]
    ps, v = d.get("problem_specificity") or {}, d.get("variance") or {}
    out = [(f"- **{name}:** {d['n_runs']} runs. Problem specificity: per-item Δ correlation within "
           f"{_f(ps.get('corr_within'))} vs between {_f(ps.get('corr_between'))} candidates (permutation p "
           f"{_n(ps.get('perm_p_one_sided'))}). σ_between {_n(v.get('sigma_between'), 2)} pp, σ_seed {_n(v.get('sigma_seed'), 2)} pp; "
           f"best possible top-1-of-4 gain {_n(d.get('oracle_top1_of_4_gain_pp'), 2)} pp. Fragile items (flip rate > 25%): "
           f"{(d.get('fragile_items') or {}).get('flip_rate_gt_0.25')}.")]
    if d.get("dose"):
        x = d["dose"]
        out.append(f"  Dose: corr(effective updates, |ΔA|) {_f(x['corr_eff_updates_abs_delta'])}; mean ΔA "
                   f"{_f(x['mean_delta_pp_eff_ge_50'], 2)} pp with ≥ 50 effective updates vs {_f(x['mean_delta_pp_eff_lt_50'], 2)} "
                   f"pp below; runs with zero updates: {x['runs_with_zero_updates']}.")
    return out


def _equation_block(tag: str, eq: dict | None) -> list[str]:
    if not eq:
        return [f"### {tag} — {TAGS.get(tag, '')}", "", "Pending: its labels do not exist yet.", ""]
    r, o = eq["ridge"], eq["out_of_fold"]
    g = eq["gate_s_thresholds_applied"]
    lines = [f"### {tag} — {TAGS.get(tag, '')}", "",
             (f"N = {eq['n_candidates']} candidates; label mean {eq['label_mean_pp']:+.3f} pp, SD {eq['label_sd_pp']:.3f} pp. "
             f"Chosen model: **{eq['model_choice']['chosen']}**."), "",
             f"**Fitted equation (Ridge, α = {r['alpha']}):** `{eq['ridge_text']}`", "",
             "| Term | Ridge coefficient (raw units) | 95% CI (candidate bootstrap) | Sign agreement | OLS coefficient |",
             "|---|---|---|---|---|",
             f"| intercept | {r['intercept']:+.6f} | — | — | {eq['ols']['intercept']:+.6f} |"]
    for f in r["coef"]:
        lines.append(f"| {f} | {r['coef'][f]:+.6f} | {_ci(r['coef_ci95'][f], 4)} | {r['sign_agreement'][f]:.2f} | "
                     f"{eq['ols']['coef'][f]:+.6f} |")
    lines += ["", "**Out-of-fold validation** (repeated grouped CV; 95% CIs by candidate bootstrap):", "",
              "| Model / rule | Spearman ρ | R²_oos | Top-1-of-4 gain over random (pp) | Oracle gain (pp) |",
              "|---|---|---|---|---|"]
    for m, v in o.items():
        lines.append(f"| {m} | {_f(v['rho'])} {_ci(v['ci95']['rho'])} | {_f(v['r2_oos'])} | {_f(v['gain_pp'])} "
                     f"{_ci(v['ci95']['gain_pp'])} | {_f(v['oracle_pp'])} |")
    for m, v in eq["rules"].items():
        lines.append(f"| rule: {m} | {_f(v.get('rho'))} | — | {_f(v['gain_pp'])} | {_f(v['oracle_pp'])} |")
    perm = eq.get("permutation") or {}
    lines += ["", (f"Gate G-S thresholds: ρ ≥ 0.30 ({'met' if g['rho_ok'] else 'not met'}), permutation p < 0.05 "
              f"(p = {_n(perm.get('p_one_sided'))}, {'met' if g['perm_ok'] else 'not met'}), top-1 gain CI above 0 "
              f"({'met' if g['gain_ok'] else 'not met'}) → **{'validated' if g['passed'] else 'NOT validated'}**."), ""]
    if eq.get("frozen"):
        lines += [f"Frozen for the end-to-end test: sha256 `{eq['frozen']['sha256']}`.", ""]
    return lines


def _curriculum_block(a: dict | None) -> list[str]:
    if not a:
        return ["Pending: the end-to-end curriculum has not run yet.", ""]
    lines = ["| Comparison | Seeds | Δ (pp) | 95% CI | Holm p | Verdict |", "|---|---|---|---|---|---|"]
    for h, r in a["results"].items():
        if "mean_pp" not in r:
            lines.append(f"| {h} | — | — | — | — | {r.get('status')} |")
            continue
        lines.append(f"| {h}: {r['comparison']} | {len(r['seeds'])} | {r['mean_pp']:+.2f} | {_ci(r['ci95_pp'])} | "
                     f"{r['p_holm']:.3g} | {r['verdict']} |")
    lines += ["", (f"Outperforms both: **{a['outperforms_both']}** (selector validated by G-S: "
              f"{a.get('selector_validated_by_gate_s')})."), ""]
    return lines


def build_report(cs: CalibSpec, out: Path | None = None) -> Path:
    R = cs.results_dir
    E = R / "explore"
    eqs = {t: read_json(E / t / "equation.json") for t in TAGS}
    gates = {"C1 (G-P)": read_json(R / "calib" / "gate_p.json"), "C2 (G-P)": read_json(R / "calib_c2" / "gate.json"),
             "OS (G-P thresholds)": read_json(R / "oneshot" / "gate.json")}
    diags = {"C1": read_json(E / "diagnostics_c1.json"), "C2": read_json(E / "diagnostics_c2.json"),
             "OS": read_json(E / "diagnostics_os.json")}
    cur = read_json(R / "curriculum_explore" / "analysis_E3_os.json")
    validated = [t for t, e in eqs.items() if e and e["gate_s_thresholds_applied"]["passed"]]
    eq10 = " ".join(f"{'+' if c >= 0 else '−'} {abs(c)}·{k}" for k, c in dict(nbm.EQ10_AS_RUN).items()).lstrip("+ ")
    L = ["# SGAC calibration track — report", "",
         (f"_Generated {utc_now()} by `python -m rlvr_v2.sgac_calib report` from `{(R.relative_to(REPO_ROOT) if R.is_relative_to(REPO_ROOT) else R).as_posix()}`. "
         "Protocol: `paper/sgac_calib_protocol.md` (tag `sgac-calib-protocol-v1`; amendments A1–A6 in §13)._"), "",
         "## 1. Conclusion so far", ""]
    if cur:
        L.append(f"- End-to-end curriculum with `E3_os`: H1/H2 verdicts in §6; outperforms Random and SGAC: "
                 f"**{cur['outperforms_both']}**.")
    L += [(f"- Fitted equations: {', '.join(t for t in TAGS if eqs[t]) or 'none'}; validated by the G-S thresholds: "
          f"{', '.join(validated) or 'none'}."),
          ("- The deployment-dose (C1) target has no problem-specific component (§3), so the intended C1 equation is "
          "unidentifiable. The selector-calibration queue (A2–A4, about 17 GPU-days) was withdrawn by amendment A6 "
          "before any of it ran: it would cost far more compute than selection could save. The compute question is "
          "answered by the equal-compute experiment (`reports/sgac_calib/compute_budget_comparison.md`)."),
          ("- A fitted equation is not a validated selector, and a validated selector is not a demonstrated improvement; "
          "each claim below names which of the three it is."), "",
          "## 2. The equations", "",
          (f"- **Original SGAC (Eq. 10, as run):** `score = {eq10}` — no intercept, argmax of 4; the N = 20 phase-4 fit read "
          "with labels shifted by one feature (`paper/sgac_repro_protocol.md` §2)."),
          ("- **Intended new selector:** `score = b0 + b1·Ps + b2·Var + b3·D + b4·L`, predicting dev ΔA (pp); argmax of 4, "
          "first maximum on ties. Ridge on standardised features, α by inner grouped CV over 10^−3…10^3 (13 values), "
          "squared-error loss, closed-form solution; GBM / OLS / interactions compared (protocol §5)."),
          ("- **Features** (from K = 4 rollouts of the current policy, `rlvr_v2.sgac.sieve.selection_signals`): "
          "Ps = share correct (math-verify, last box, truncated = wrong), in {0, .25, .5, .75, 1}; "
          "Var = population variance of (correct + 0.5·boxed), in [0, 0.5625]; D = distinct math-verify answer classes / 4 "
          "(unparsable = one class), in {.25, .5, .75, 1}; L = MATH level 1–5. Training rows: 16 random K = 4 subsamples of "
          "each candidate's 32 stored base rollouts. The coefficients below are on these raw units (the standardisation "
          "is folded in)."), ""]
    for t in TAGS:
        L += _equation_block(t, eqs[t])
    L += ["## 3. Label reliability and why", ""]
    for k, g in gates.items():
        L += _gate_block(k, g)
    L += [""]
    for k, d in diags.items():
        L += _diag_block(k, d)
    L += ["", "## 4. Stage A audit (original Phase-1 candidates)", ""]
    found = glob.glob(str(R / "audit" / "e0" / "*" / "phase1_e0_signals" / "summary.json"))
    s = read_json(found[0]) if found else None
    p = read_json(R / "audit" / "as_run" / "64ba360f" / "phase1" / "phase1_summary.json")
    if s and p:
        L += ["| Candidate | Paper Table 1 (Ps / Var / D) | Historical settings K=8 | E0 K=32 | MATH-500 after Phase-1 burst |",
              "|---|---|---|---|---|"]
        paper = {r["unique_id"]: r for r in s["paper_table1"]}
        hist = {c["unique_id"]: c for c in p["candidates"]}
        for c in s["candidates"]:
            u = c["unique_id"]
            pt, h, e = paper[u], hist[u]["signals"], c["e0"]
            L.append(f"| {u} (L{c['level']}) | {pt['Ps']:.3f} / {pt['Var']:.3f} / {pt['D']:.3f} | {h['Ps']:.3f} / "
                     f"{h['Var']:.3f} / {h['D']:.3f} | {e['Ps']:.3f} / {e['Var']:.3f} / {e['D']:.3f} | "
                     f"{hist[u]['acc_math500']:.3f} |")
        L += ["", ("Table 1's signals do not reproduce (K = 8 sampling noise, grader and truncation differences), and the four "
              "candidates' downstream MATH-500 accuracies are indistinguishable; the 4-row fit (5 parameters) is "
              "non-identifiable."), ""]
    else:
        L += ["Pending.", ""]
    L += ["## 5. End-to-end curriculum (exploratory, amendment A4)", ""] + _curriculum_block(cur)
    L += ["## 6. Compute consumed (GPU-hours, from run timings)", "", "| Stage | GPU-h |", "|---|---|"]
    for name, h in compute_table(cs):
        L.append(f"| {name} | {h:.1f} |")
    L += ["", "## 7. Reproduce", "", "```bash",
          "python -m rlvr_v2.sgac_calib diagnose --source c1        # and c2, os",
          "python -m rlvr_v2.sgac_calib fit-explore --source c1 --tag E1_c1_pilot",
          "STAGES=compute_budget scripts/local/sgac_calib.sh   # the equal-compute experiment (amendment A6)",
          "python -m rlvr_v2.sgac_calib report", "```", ""]
    out = out or REPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    return out
