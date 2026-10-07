"""Phase-1 replication: strategy table and the 4-row fit on the paper's own Table 1 numbers, plus a tiny CPU run."""
import pytest

from rlvr_v2.sgac import nbm_original as nbm
from rlvr_v2.sgac import phase1

PAPER_SIGNALS = [{"Ps": r["Ps"], "Var": r["Var"], "D": r["D"], "L": r["L"]} for r in nbm.PHASE1_TABLE1]
PAPER_ACC = [r["acc"] for r in nbm.PHASE1_TABLE1]


def test_strategies_on_paper_table1():
    s = phase1.strategies(PAPER_SIGNALS, PAPER_ACC)
    assert s["random_as_notebook"]["idx"] == 0 and s["random_as_notebook"]["acc"] == 0.4
    assert s["max_var"]["idx"] == 0 and s["max_var"]["acc"] == 0.4  # the paper's "variance fallacy" pick
    assert s["max_d"]["idx"] == 2 and s["max_d"]["acc"] == 0.5
    assert s["max_level"]["ties"] == [0, 2] and s["max_level"]["acc_over_ties"] == pytest.approx(0.45)
    assert s["sgac_eq10"]["idx"] == 0  # Eq. 10 as run would have picked the 40 % candidate


def test_four_row_fit_reproduces_table2_under_the_true_mapping():
    fit = phase1.four_row_fit(PAPER_SIGNALS, PAPER_ACC)
    for k, v in nbm.TABLE2_TRUE_MAPPING.items():
        assert fit["true_mapping"][k] == pytest.approx(v, abs=1e-4)
    for k, v in nbm.TABLE2_AS_PRINTED.items():  # ...and the notebook's shifted labels give the paper's Table 2
        assert fit["as_printed_by_notebook"][k] == pytest.approx(v, abs=1e-4)
    assert fit["rank"] <= 4  # 4 rows, 5 parameters


@pytest.mark.smoke
def test_phase1_cpu(tmp_path, monkeypatch):
    from rlvr_v2.sgac import data as sdata
    from rlvr_v2.sgac.spec import CONFIG_DIR, load_profile

    monkeypatch.setattr(phase1, "PHASE1_K", 2)
    monkeypatch.setattr(phase1, "PHASE1_TOKENS", 12)
    monkeypatch.setattr(phase1, "PHASE1_GRPO_STEPS", 5)
    spec = load_profile("as_run", [f"run.results_root={tmp_path.as_posix()}",
                                   f"data.manifest_dir={(tmp_path / 'manifests').as_posix()}"],
                        [CONFIG_DIR / "smoke_cpu.yaml"])
    sdata.write_data_manifest(spec, mv=None)
    s = phase1.run_phase1(spec)
    assert len(s["candidates"]) == 4 and all(c["acc_test10"] is not None for c in s["candidates"])
    assert set(s["strategies_test10"]) >= {"max_var", "max_d", "max_level", "sgac_eq10"}
    again = phase1.run_phase1(spec)  # resumable: every candidate is already done
    assert [c["unique_id"] for c in again["candidates"]] == [c["unique_id"] for c in s["candidates"]]
