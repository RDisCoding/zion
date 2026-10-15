"""Post-G-P rungs and exploratory fits: substrate guards, seeds, diagnostics, the exploratory fit and freeze."""
import json

import numpy as np
import pytest

from rlvr_v2.artifacts import read_json
from rlvr_v2.sgac_calib import explore as explore_mod
from rlvr_v2.sgac_calib.diagnose import diagnose
from rlvr_v2.sgac_calib.jobs import stage_jobs
from rlvr_v2.sgac_calib.rungs import (
    C2_STEPS,
    OS_SEED_OFFSETS,
    PREREG_STUDY1_HASH,
    c2_spec,
    os_config,
    os_run_name,
    os_seeds,
    require_e0_except,
)
from rlvr_v2.sgac_calib.spec import load_calib, substrate_spec

CS = load_calib()


def test_c2_substrate_differs_from_e0_only_in_steps():
    spec = c2_spec(CS)
    assert spec.loop.steps == C2_STEPS == 80
    altered = substrate_spec(CS, "calib_c2", ["loop.steps=80", "grpo.learning_rate=5e-05"])
    with pytest.raises(RuntimeError, match="only"):
        require_e0_except(altered, {"loop.steps": 80})


def test_os_uses_the_unchanged_prereg_study1_config():
    cfg = os_config()
    assert cfg.config_hash() == PREREG_STUDY1_HASH == "84e100e2"
    assert cfg.train.rounds == 100 and cfg.train.learning_rate == pytest.approx(2e-5)


def test_os_seeds_and_names_are_distinct_from_c1():
    assert os_seeds(5) == [o + 5 for o in OS_SEED_OFFSETS]
    assert not set(os_seeds(5)) & {7005, 17005}
    assert os_run_name(3, "train/algebra/1351.json", 27003) == "os_c03__train-algebra-1351__seed27003"
    assert os_run_name(None, "pi1", 29000) == "os_pi1__seed29000"


@pytest.mark.parametrize("stage,n", [("os_pilot", 37), ("os_expansion", 67), ("os_selector", 1),
                                     ("os_curriculum", 35), ("c2", 37)])
def test_new_stage_queues(stage, n):
    jobs = stage_jobs(CS, stage)
    assert len(jobs) == n
    if stage == "os_expansion":
        assert jobs[0] == "require-os-gate"


def _write_runs(tmp_path, deltas: dict[str, np.ndarray], base: np.ndarray, ids):
    runs = {}
    for name, d in deltas.items():
        y = (base + d).astype(bool)
        p = tmp_path / f"{name}.jsonl"
        p.write_text("\n".join(json.dumps({"unique_id": i, "correct": bool(v)}) for i, v in zip(ids, y)), encoding="utf-8")
        runs[name] = {"per_item": p, "eff_updates": 10}
    return runs


def _flip(base, idx):
    d = np.zeros(base.size)
    d[idx] = np.where(base[idx] == 1, -1, 1)
    return d


@pytest.mark.parametrize("specific", [True, False])
def test_diagnose_detects_problem_specific_effects(tmp_path, specific):
    rng = np.random.default_rng(0)
    ids = [f"q{i}" for i in range(300)]
    base = (rng.random(300) < 0.6).astype(float)
    fragile = rng.choice(300, 60, replace=False)
    deltas = {}
    for c in range(10):
        own = rng.choice(300, 25, replace=False)
        for s in range(2):
            idx = own if specific else rng.choice(fragile, 25, replace=False)
            deltas[f"c{c:02d}__seed{s}"] = _flip(base, idx)
    out = diagnose(_write_runs(tmp_path, deltas, base, ids), dict(zip(ids, base.astype(bool))), n_perm=200)
    p = out["problem_specificity"]["perm_p_one_sided"]
    assert (p < 0.05) if specific else (p > 0.05)


def test_explore_fit_freezes_and_reports(tmp_path, monkeypatch):
    feats = read_json(CS.manifest_dir / "features.json")
    rng = np.random.default_rng(1)
    cands = []
    for c in feats["candidates"][:24]:
        d = 1.5 * (c["level"] - 3) + rng.normal(0, 0.3)
        cands.append({"index": c["index"], "unique_id": c["unique_id"], "n_seeds_done": 2, "delta_pp_mean": d,
                      "runs": [{"delta_pp": d}, {"delta_pp": d}]})
    labels = {"source": "synthetic", "candidates": cands}
    cs = load_calib(overrides=[f"results_root={tmp_path.as_posix()}", "selector.cv_repeats=3",
                               "selector.n_bootstrap=50", "selector.n_permutations=20", "selector.perm_cv_repeats=2",
                               "selector.n_mc_batches=2000", "selector.alpha_cv_repeats=2"])
    monkeypatch.setattr(explore_mod, "pool_bin_frequencies", lambda cs, sig: [0.088, 0.314, 0.48, 0.118])
    monkeypatch.setattr(explore_mod, "sieve_signals", lambda cs: {})
    eq = explore_mod.run_explore_fit(cs, labels, "T_synth", min_seeds=2, freeze=True)
    assert eq["n_candidates"] == 24 and eq["exploratory"] is True
    assert eq["ridge"]["coef"]["L"] > 0 and "score =" in eq["ridge_text"]
    assert eq["frozen"]["validated_by_gate_s"] in (True, False) and len(eq["frozen"]["sha256"]) == 64
    from rlvr_v2.sgac_calib.explore_curriculum import explore_selector

    sel = explore_selector(cs, "T_synth")
    assert sel.score({"Ps": 0.5, "Var": 0.25, "D": 0.5, "L": 5}) > sel.score({"Ps": 0.5, "Var": 0.25, "D": 0.5, "L": 1})


def test_report_builds_with_pending_sections(tmp_path):
    from rlvr_v2.sgac_calib.report import build_report

    cs = load_calib(overrides=[f"results_root={tmp_path.as_posix()}"])
    out = build_report(cs, tmp_path / "REPORT.md")
    text = out.read_text(encoding="utf-8")
    assert "Original SGAC (Eq. 10, as run)" in text and text.count("Pending") >= 3
