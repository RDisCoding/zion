"""CPU smoke of the post-G-P rungs (tiny models): C2 run, OS run via the prereg trainer, labels, diagnostics."""
import pytest

pytestmark = pytest.mark.smoke


def test_rungs_smoke(tmp_path):
    from rlvr_v2.sgac_calib.smoke_rungs import run_rungs_smoke

    s = run_rungs_smoke(tmp_path)
    assert s["c2_label"]["n_seeds_done"] == 1 and s["os_label"]["n_seeds_done"] == 1
    assert s["c2_diag"] == 1 and s["os_diag_runs"] == 1
