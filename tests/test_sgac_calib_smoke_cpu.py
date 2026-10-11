"""End-to-end CPU smoke of the calibration track (tiny model, real data, committed manifests read-only)."""
import pytest

pytestmark = pytest.mark.smoke


def test_calibration_smoke(tmp_path):
    from rlvr_v2.sgac_calib.smoke import run_smoke

    s = run_smoke(gpu=False, out_dir=tmp_path)
    assert s["calib_resume_check"] == "ok"
    assert s["label"]["n_seeds_done"] == 1
    assert s["curriculum"]["tm_picks"] == s["curriculum"]["random_picks"]
    assert len(s["curriculum"]["learned_picks"]) == 2
