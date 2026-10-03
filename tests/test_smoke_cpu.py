"""End-to-end CPU smoke test with a tiny model. Downloads ~270 MB; run explicitly:

    ./.venv/Scripts/python -m pytest -m smoke tests/test_smoke_cpu.py -s
"""
import json
from pathlib import Path

import pytest

from rlvr_v2.artifacts import REPO_ROOT
from rlvr_v2.config import load_config

pytestmark = pytest.mark.smoke


def test_smoke_end_to_end(tmp_path: Path):
    pytest.importorskip("trl")
    cfg = load_config([REPO_ROOT / "configs" / "base.yaml", REPO_ROOT / "configs" / "smoke_cpu.yaml"])
    from rlvr_v2.smoke import run_smoke

    report = run_smoke(cfg, tmp_path / "smoke")
    assert report["ok"]
    steps = report["steps"]
    assert steps["sieve"]["n"] == 3 and steps["manifests"]["eval"] == cfg.eval.max_items
    assert set(steps["selectors"]) == {"random", "variance", "disagreement", "ps_band", "learned"}
    assert steps["train"]["steps_logged"] >= 1 and steps["train"]["has_clipped_ratio"]
    assert steps["train"]["has_reward_correct"]
    assert 0.0 <= steps["eval"]["acc"] <= 1.0
    assert (tmp_path / "smoke" / "results").exists()
    print(json.dumps(report, indent=1, default=str)[:2000])
