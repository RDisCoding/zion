import json

import pytest

from rlvr_v2 import cli
from rlvr_v2.config import Config
from rlvr_v2.gates import REQUIRED_GATES, frozen_overrides, gate_problems


def _report(**passed):
    return {"gates": {g: {"passed": p} for g, p in passed.items()}}


def test_missing_or_empty_report_blocks():
    assert gate_problems(None)
    assert gate_problems({})


def test_partial_report_blocks_even_when_every_gate_run_so_far_passed():
    assert gate_problems(_report(g1=True, g2=True, g3=True, g5=True)) == ["g4: not run"]


def test_failed_gate_blocks():
    problems = gate_problems(_report(g1=True, g2=True, g3=True, g4=False, g5=True))
    assert problems == ["g4: failed"]


def test_all_passed_authorises():
    assert gate_problems(_report(**{g: True for g in REQUIRED_GATES})) == []


def test_require_gates_exits_unless_all_five_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    cfg = Config()
    assert cfg.run.require_gates
    with pytest.raises(SystemExit):
        cli._require_gates(cfg)  # no gates.json at all
    gates_path = tmp_path / "results" / "e0" / "gates.json"
    gates_path.parent.mkdir(parents=True)
    gates_path.write_text(json.dumps({"all_passed": True, **_report(g1=True, g2=True, g3=True, g5=True)}))
    with pytest.raises(SystemExit) as exc:
        cli._require_gates(cfg)  # G4 missing: a stale all_passed flag must not matter
    assert "g4: not run" in str(exc.value)
    gates_path.write_text(json.dumps(_report(**{g: True for g in REQUIRED_GATES})))
    cli._require_gates(cfg)  # all five passed: returns normally


def test_frozen_overrides_carry_style_and_budget_and_parse_back():
    report = {"gates": {"g1": {"pinned_style": "oneshot_rlvr_chat"},
                        "g4": {"frozen_budget": {"rounds": 200, "learning_rate": 5e-5}}}}
    ov = frozen_overrides(report)
    assert ov == ["prompt.style=oneshot_rlvr_chat", "train.rounds=200", "train.learning_rate=5e-05"]
    from rlvr_v2.config import parse_overrides

    parsed = parse_overrides(ov)
    assert parsed["train"]["learning_rate"] == 5e-5 and parsed["train"]["rounds"] == 200
    assert parsed["prompt"]["style"] == "oneshot_rlvr_chat"
    assert frozen_overrides(None) == [] and frozen_overrides({"gates": {"g4": {"frozen_budget": None}}}) == []
