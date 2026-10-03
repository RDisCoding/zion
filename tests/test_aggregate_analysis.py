"""Aggregation + analysis on a tiny synthetic results/ tree."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from rlvr_v2.aggregate import TABLE_COLUMNS, aggregate, read_tables
from rlvr_v2.data import Manifest
from rlvr_v2.selectors import LearnedLinearSelector
from tests.conftest import make_signals

ROOT = Path(__file__).resolve().parents[1]
GROUP = "dev-abc12345"
ITEMS = [f"m{i:03d}" for i in range(40)]
# uid, p_s, d_wrong, level  (4 matched pairs across the three bins + a null and a zero anchor)
CANDS = [("c00", 4 / 32, 0.90, 2), ("c01", 4 / 32, 0.10, 2), ("c02", 16 / 32, 0.95, 3), ("c03", 16 / 32, 0.15, 3),
         ("c04", 20 / 32, 0.80, 4), ("c05", 18 / 32, 0.20, 4), ("c06", 26 / 32, 0.85, 1), ("c07", 26 / 32, 0.05, 1),
         ("c08", 1.0, float("nan"), 3), ("c09", 0.0, 0.60, 5)]
PAIRS = [("c00", "c01", 0), ("c02", "c03", 1), ("c04", "c05", 1), ("c06", "c07", 2)]
BINS = [[0.0, 0.25], [0.25, 0.75], [0.75, 0.9]]
STEPS = 5


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def _per_item(correct):
    return [{"unique_id": u, "level": 1 + i % 5, "subject": "Algebra", "correct": bool(c), "format_ok": True,
             "method": "string", "boxed": "1", "finish_reason": "stop", "truncated": False, "n_tokens": 100, "answer": "1"}
            for i, (u, c) in enumerate(zip(ITEMS, correct))]


def _summary(correct, tag="math500"):
    acc = float(np.mean(correct))
    return {"tag": tag, "n": len(correct), "n_correct": int(sum(correct)), "acc": acc, "ci_lo": acc - 0.1, "ci_hi": acc + 0.1,
            "trunc_rate": 0.0, "format_rate": 1.0, "by_level": {}, "by_subject": {}, "wall_s": 1.0,
            "prompt_style": "qwen_math_chat", "prompt_hash": "abc", "backend": "hf", "policy": "adapter",
            "adapter_path": None, "max_new_tokens": 3072, "n_boot": 200}


def _run_dir(root: Path, study: str, run_name: str, seed: int, extra: dict, state: str = "done") -> Path:
    d = root / study / GROUP / run_name
    d.mkdir(parents=True, exist_ok=True)
    _json(d / "run.json", {"run_id": f"{study}/{GROUP}/{run_name}", "study": study, "run_name": run_name,
                           "created": "2026-10-03T00:00:00Z", "seed": seed, "config_hash": "abc12345", "git_hash": "deadbeef",
                           **extra})
    _json(d / "status.json", {"state": state, "stage": "eval", "updated": "2026-10-03T00:00:00Z"})
    (d / "config.yaml").write_text(yaml.safe_dump({"run": {"tag": "dev", "seed": seed}, "prompt": {"style": "qwen_math_chat"},
                                                   "selector": {"name": extra.get("arm", "random")},
                                                   "train": {"learning_rate": 2e-5, "rounds": 100}}), encoding="utf-8")
    return d


def _train_rows(n, offset=0.0):
    return [{"step": s + 1, "loss": 0.01 * s, "reward": 0.3 + 0.05 * s + offset, "rewards/correctness/mean": 0.3 + 0.05 * s,
             "rewards/format/mean": 1.0, "frac_reward_zero_std": 0.2, "completions/clipped_ratio": 0.01,
             "completions/mean_length": 400 - 5 * s} for s in range(n)]


def _shifted(base: np.ndarray, delta: float, rng) -> np.ndarray:
    """Flip items so that mean(out) - mean(base) == delta (up to rounding)."""
    out = base.copy()
    k = int(round(abs(delta) * len(base)))
    pool = np.flatnonzero(~base) if delta > 0 else np.flatnonzero(base)
    for i in rng.choice(pool, size=min(k, len(pool)), replace=False):
        out[i] = not out[i]
    return out


def make_tree(root: Path) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = np.zeros(len(ITEMS), dtype=bool)
    base[rng.choice(len(ITEMS), size=20, replace=False)] = True
    shared = root / "study1" / GROUP / "_shared" / "eval_base" / "math500"
    _jsonl(shared / "per_item.jsonl", _per_item(base))
    _json(shared / "summary.json", _summary(base, "base"))
    sig_rows = []
    for uid, p_s, dw, level in CANDS:
        s = make_signals(uid, p_s=p_s, d_simpson=min(0.95, 0.4 + 0.5 * (dw if not math.isnan(dw) else 0.0)), d_wrong=dw,
                         level=level, k=32)
        sig_rows.append({**s.to_dict(), "seed": 0, "prompt_hash": "p", "policy_tag": "base"})
    sig_rows.append({**make_signals("extra", p_s=0.3, k=32).to_dict(), "policy_tag": "base"})
    _jsonl(root / "pool" / "signals.jsonl", sig_rows)

    def target(p_s, dw):
        hump = 0.4 * p_s * (1 - p_s)  # peak 0.10 at p_s = 0.5
        return hump + (0.06 * (dw - 0.5) if not math.isnan(dw) else 0.0)

    def study1_run(i, uid, p_s, dw, level, seed, replicate, jitter):
        d = _run_dir(root, "study1", f"job{i:03d}__{uid}__seed{seed}", seed,
                     {"unique_id": uid, "replicate": replicate, "job_index": i})
        corr = _shifted(base, target(p_s, dw) + jitter, rng)
        _jsonl(d / "train" / "train_metrics.jsonl", _train_rows(STEPS))
        _jsonl(d / "eval" / "math500" / "per_item.jsonl", _per_item(corr))
        _json(d / "eval" / "math500" / "summary.json", _summary(corr))
        _json(d / "result.json", {"unique_id": uid, "seed": seed, "acc": float(corr.mean())})

    for i, (uid, p_s, dw, level) in enumerate(CANDS):
        study1_run(i, uid, p_s, dw, level, 1000, False, 0.0)
    for j, uid in enumerate(["c02", "c06"]):
        p_s, dw, level = next((c[1], c[2], c[3]) for c in CANDS if c[0] == uid)
        study1_run(50 + j, uid, p_s, dw, level, 101000, True, 0.025)
    _run_dir(root, "study1", "job099__c04__seed101000", 101000, {"unique_id": "c04", "replicate": True}, state="failed")

    for arm, bonus in (("random", 0.0), ("variance", 0.10)):
        d = _run_dir(root, "study2", f"{arm}__seed1234", 1234, {"arm": arm})
        _json(d / "state.json", {"step": 2, "done": True})
        _json(d / "steps" / "step-000" / "selection.json",
              {"step": 0, "selected_uid": "c03", "p_s": 0.5, "d_simpson": 0.6, "d_wrong": 0.15, "level": 3,
               "scores": [0.1, 0.9, 0.3]})
        _json(d / "steps" / "step-001" / "selection.json",
              {"step": 1, "selected": {"unique_id": "c04"}, "scores": {"c04": 0.7, "c01": 0.2}})
        _jsonl(d / "steps" / "step-001" / "sieve" / "signals.jsonl",
               [{**make_signals("c04", p_s=20 / 32, d_simpson=0.7, d_wrong=0.8, level=4, k=16).to_dict(), "policy_tag": "current"}])
        for k in range(2):
            _jsonl(d / "steps" / f"step-{k:03d}" / "train" / "train_metrics.jsonl", _train_rows(3, bonus))
            corr = _shifted(base, 0.05 * k + bonus, rng)
            _jsonl(d / "evals" / f"step-{k:03d}" / "per_item.jsonl", _per_item(corr))
            _json(d / "evals" / f"step-{k:03d}" / "summary.json", _summary(corr, f"step-{k:03d}"))
    return base


def make_manifest(path: Path) -> Manifest:
    pairs = []
    for i, (hi, lo, b) in enumerate(PAIRS):
        chi, clo = next(c for c in CANDS if c[0] == hi), next(c for c in CANDS if c[0] == lo)
        pairs.append({"bin": BINS[b], "bin_index": b, "high": hi, "low": lo, "level": chi[3], "ps_high": chi[1], "ps_low": clo[1],
                      "dwrong_high": chi[2], "dwrong_low": clo[2], "dwrong_gap": chi[2] - clo[2], "min_dwrong_gap_used": 0.4})
    ids = ["c08", "c09"] + [u for p in pairs for u in (p["high"], p["low"])]
    sel = {"anchors_null": ["c08"], "anchors_zero": ["c09"], "pairs": pairs, "bins": BINS, "replicate_ids": ["c02", "c06"],
           "ordered_ids": ids, "relaxations": [], "shortfall": {}}
    m = Manifest(name="study1_candidates", source="math_train", ids=ids, hashes={u: "h" for u in ids},
                 meta={"replicate_ids": ["c02", "c06"], "selection": sel})
    m.save(path)
    return m


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("results")
    base = make_tree(root)
    tables = aggregate(root, None, n_boot=200, seed=0)
    return root, base, tables


def test_aggregate_tables(tree):
    root, base, tables = tree
    assert set(tables) == set(TABLE_COLUMNS)
    for name, df in tables.items():
        assert list(df.columns) == TABLE_COLUMNS[name], name
        assert (root / "tables" / f"{name}.parquet").exists()
    runs = tables["runs"]
    assert len(runs) == 15 and set(runs["kind"]) == {"study1", "curriculum"}
    assert sorted(runs.loc[runs["kind"] == "curriculum", "arm"]) == ["random", "variance"]
    assert set(runs["state"]) == {"done", "failed"} and runs["git_hash"].eq("deadbeef").all()

    s1 = tables["study1"]
    assert len(s1) == 13 and s1["delta"].notna().sum() == 12
    assert s1.loc[s1["unique_id"] == "c02", "n_seeds"].tolist() == [2, 2]
    assert s1.loc[s1["run_name"].str.contains("seed101000"), "replicate"].all()
    assert not s1.loc[s1["seed"] == 1000, "replicate"].any()
    row = s1[s1["run_name"].str.startswith("job002")].iloc[0]
    assert row["p_s"] == pytest.approx(0.5) and row["d_wrong"] == pytest.approx(0.95) and row["level"] == 3
    assert row["n_train_steps"] == STEPS and row["reward_last"] == pytest.approx(0.3 + 0.05 * (STEPS - 1))
    assert row["base_acc"] == pytest.approx(base.mean()) and row["n_common"] == 40
    assert row["delta"] == pytest.approx(row["acc"] - row["base_acc"])
    assert row["delta_ci_lo"] <= row["delta"] <= row["delta_ci_hi"] and row["base_run_id"] == f"_base/{GROUP}"
    null_anchor = s1[s1["unique_id"] == "c08"].iloc[0]
    assert null_anchor["delta"] == pytest.approx(0.0) and math.isnan(float(null_anchor["d_wrong"]))
    failed = s1[s1["run_name"].str.startswith("job099")].iloc[0]
    assert failed["state"] == "failed" and math.isnan(float(failed["delta"])) and failed["n_train_steps"] == 0

    ev = tables["eval_items"]
    assert len(ev) == 12 * 40 + 40 + 2 * 2 * 40
    assert (ev["run_id"] == f"_base/{GROUP}").sum() == 40 and set(ev.loc[ev["step"].notna(), "tag"]) == {"step-000", "step-001"}
    tm = tables["train_metrics"]
    assert len(tm) == 12 * STEPS + 2 * 2 * 3 and tm["curriculum_step"].notna().sum() == 12
    assert tm["rewards/correctness/mean"].notna().all() and tm["global_step"].min() == 1
    sel = tables["selections"]
    assert len(sel) == 4 and sorted(sel["step"]) == [0, 0, 1, 1]
    joined = sel[sel["selected_uid"] == "c04"]
    assert len(joined) == 2 and joined["p_s"].tolist() == pytest.approx([20 / 32, 20 / 32]) and joined["d_wrong"].iloc[0] == 0.8
    assert json.loads(sel[sel["step"] == 0]["scores"].iloc[0]) == [0.1, 0.9, 0.3]
    s2 = tables["study2_evals"]
    assert len(s2) == 4 and set(s2["arm"]) == {"random", "variance"} and s2["acc"].between(0, 1).all()
    back = read_tables(root / "tables")
    assert {k: v.shape for k, v in back.items()} == {k: v.shape for k, v in tables.items()}


def test_aggregate_explicit_base_and_missing_root(tree, tmp_path):
    root, base, _ = tree
    explicit = root / "study1" / GROUP / "_shared" / "eval_base" / "math500" / "per_item.jsonl"
    t = aggregate(root, explicit, out_dir=tmp_path / "t", n_boot=50)
    assert (t["study1"]["base_run_id"].dropna() == "_base").all() and (t["eval_items"]["run_id"] == "_base").sum() == 40
    with pytest.raises(FileNotFoundError):
        aggregate(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    e = aggregate(empty, None, write=False)
    assert all(len(df) == 0 for df in e.values())


def test_study1_analysis_end_to_end(tree, tmp_path):
    root, _, _ = tree
    paper = tmp_path / "paper"
    manifest_path = tmp_path / "study1_candidates.json"
    make_manifest(manifest_path)
    mod = _load_script("study1_analysis")
    summary = mod.main(["--results", str(root), "--paper", str(paper), "--candidates", str(manifest_path),
                        "--selectors-dir", str(tmp_path / "selectors"), "--n-boot", "100", "--n-perm", "30",
                        "--min-n-selector", "6"])
    for name in ("spearman", "partial", "hump", "pairs", "selector", "selector_coef", "noise", "holm", "tost", "candidates"):
        tex = (paper / "tables" / f"study1_{name}.tex").read_text(encoding="utf-8")
        assert "\\toprule" in tex and "\\bottomrule" in tex, name
    assert "p\\_s" in (paper / "tables" / "study1_spearman.tex").read_text(encoding="utf-8")
    for fig in ("study1_delta_vs_ps.pdf", "study1_pairs.pdf", "study1_delta_vs_dwrong.pdf"):
        assert (paper / "figures" / fig).stat().st_size > 500
    js = json.loads((paper / "tables" / "study1_summary.json").read_text(encoding="utf-8"))
    assert js["n_candidates"] == 10 and js["n_runs"] == 13 and len(js["spearman"]) == 16
    assert js["pairs"]["n_pairs_complete"] == 4 and js["pairs"]["overall"]["n_pairs"] == 4
    assert js["pairs"]["overall"]["mean_diff"] > 0  # high d_wrong members were constructed to gain more
    assert js["hump"]["quad_coef"] < 0 and js["noise"]["n_groups"] == 2
    assert set(js["selector"]["models"]) == {"ridge", "lasso"}
    for m in js["selector"]["models"].values():
        assert "loo_spearman" in m["full"] and set(m["full"]["coef"]) == set(mod.SELECTOR_FEATURES)
    assert set(js["holm"]["raw"]) >= {"H1_spearman_v_bin", "H2_pairs_wilcoxon"}
    assert summary["selector"]["gate"]["reason"]
    cand = tmp_path / "selectors" / "learned_v1_candidate.json"
    sel = LearnedLinearSelector.from_json(cand)
    assert set(sel.features) == set(mod.SELECTOR_FEATURES) and sel.meta["gate"]["passed"] in (True, False)
    sigs = [make_signals("a", p_s=0.5, d_wrong=0.9, level=3), make_signals("b", p_s=0.9, d_wrong=float("nan"), level=1)]
    assert len(sel.scores(sigs, np.random.default_rng(0))) == 2
    assert (tmp_path / "selectors" / "learned_v1.json").exists() == sel.meta["gate"]["passed"]


def test_study2_analysis_end_to_end(tree, tmp_path):
    root, _, _ = tree
    paper = tmp_path / "paper2"
    mod = _load_script("study2_analysis")
    summary = mod.main(["--results", str(root), "--paper", str(paper), "--n-boot", "300"])
    for name in ("final", "auc", "selection", "curves"):
        tex = (paper / "tables" / f"study2_{name}.tex").read_text(encoding="utf-8")
        assert "\\toprule" in tex, name
    assert (paper / "figures" / "study2_curves.pdf").stat().st_size > 500
    js = json.loads((paper / "tables" / "study2_summary.json").read_text(encoding="utf-8"))
    assert js["final_step"] == 1 and js["steps_common"] == [0, 1] and js["arms"] == ["random", "variance"]
    assert len(js["comparisons"]) == 1
    c = js["comparisons"][0]
    assert c["arm"] == "variance" and c["final_delta"] == pytest.approx(0.10, abs=0.03) and c["final_n"] == 40
    assert c["auc_delta"] > 0.05 and c["final_ci_lo"] <= c["final_delta"] <= c["final_ci_hi"]
    diag = {d["arm"]: d for d in js["selection_diagnostics"]}
    assert diag["random"]["n_selections"] == 2 and diag["random"]["mean_p_s"] == pytest.approx((0.5 + 20 / 32) / 2)
    assert diag["variance"]["zero_std_frac_mean"] == pytest.approx(0.2)
    assert len(js["curves"]) == 4 and all(r["n_items"] == 40 for r in js["curves"])
    assert summary["auc_per_arm"][0]["arm"] == "random"
