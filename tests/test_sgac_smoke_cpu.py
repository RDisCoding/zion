"""End-to-end CPU smoke for both profiles (tiny model, real data, temporary manifests): base eval, two 2-step
curricula, a forced resume, and the report."""
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("profile", ["e0", "as_run"])
def test_smoke(tmp_path, profile):
    from rlvr_v2.sgac.smoke import run_smoke

    s = run_smoke(profile, gpu=False, out_dir=tmp_path)
    assert s["resume_check"] == "ok"
    assert Path(s["report"]).exists()
    assert all(r["step_done"] == 2 for r in s["runs"].values())
