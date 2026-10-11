"""The calibration track leaves the historical reproduction untouched and trains on its exact substrate."""
import shutil
import subprocess
import sys

import pytest

from rlvr_v2.artifacts import REPO_ROOT
from rlvr_v2.sgac.spec import load_profile
from rlvr_v2.sgac_calib.spec import E0_SPEC_HASH, load_calib, require_e0_substrate, substrate_spec

HISTORICAL_CODE = ["rlvr_v2/sgac", "configs/sgac", "paper/sgac_repro_protocol.md"]
FROZEN_RECORD = ["report.md", "paper/sgac_repro_freeze.md"]


def test_historical_spec_hashes_unchanged():
    assert load_profile("e0").spec_hash() == "577f3319"
    assert load_profile("as_run").spec_hash() == "64ba360f"


@pytest.mark.parametrize("stage", ["calib", "curriculum"])
def test_substrate_is_the_historical_e0_spec(stage):
    spec = substrate_spec(load_calib(), stage)
    assert spec.spec_hash() == E0_SPEC_HASH
    assert "results_sgac_calib" in spec.run.results_root
    require_e0_substrate(spec)


def test_altered_substrate_is_refused():
    spec = substrate_spec(load_calib(), "calib", ["grpo.learning_rate=5e-05"])
    with pytest.raises(RuntimeError, match="577f3319"):
        require_e0_substrate(spec)


def _git_diff(tag: str, paths: list[str]) -> str:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not available")
    if subprocess.run([git, "rev-parse", "--verify", "--quiet", tag], cwd=REPO_ROOT, capture_output=True, check=False).returncode:
        pytest.skip(f"tag {tag} not present in this clone")
    out = subprocess.run([git, "diff", "--name-only", tag, "--", *paths], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert out.returncode == 0
    return out.stdout.strip()


def test_historical_protocol_identical_to_protocol_tag():
    assert _git_diff("sgac-protocol-v1", ["paper/sgac_repro_protocol.md"]) == ""


def test_historical_code_and_record_identical_to_freeze_tag():
    # the code that produced the frozen results (pre-run fixes after sgac-protocol-v1 included)
    assert _git_diff("sgac-repro-frozen", HISTORICAL_CODE + FROZEN_RECORD) == ""


def test_importing_the_track_patches_nothing():
    code = (
        "import rlvr_v2.sgac.loop as L, rlvr_v2.sgac.selection as S, rlvr_v2.sgac.sieve as V, rlvr_v2.signals as G\n"
        "names = ('_select', '_evaluate_due', '_all_evals_present', 'keep_adapter', 'run', '_run_step')\n"
        "before = [getattr(L.SgacRun, n) for n in names] + [S.pick, S.all_picks, S.first_max, V.selection_signals,"
        " G.compute_signals, G.grade_rollouts]\n"
        "import importlib\n"
        "for m in ('spec','crossed','seedcount','manifests','features','calib','selector','curriculum','analysis',"
        "'audit','jobs','smoke','cli'):\n"
        "    importlib.import_module('rlvr_v2.sgac_calib.' + m)\n"
        "after = [getattr(L.SgacRun, n) for n in names] + [S.pick, S.all_picks, S.first_max, V.selection_signals,"
        " G.compute_signals, G.grade_rollouts]\n"
        "assert all(a is b for a, b in zip(before, after)), 'a calibration import patched a frozen callable'\n"
        "assert 'learned' not in S.ARMS and 'learned' not in S.RULES\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)
