"""Candidate schedules and the rebuilt original split (pool rows 0-999, test rows 1000-1049 of shuffle(seed=42))."""
import pytest

from rlvr_v2.sgac import data as sdata
from rlvr_v2.sgac.schedule import candidate_schedule, load_schedule
from rlvr_v2.sgac.spec import load_profile

# Independently reconstructed from the cached nlile revision (shuffled order 1000 -> 1049).
TEST50_IDS = [u + ".json" for u in (
    "train/intermediate_algebra/862", "train/algebra/495", "test/number_theory/155", "train/prealgebra/511",
    "test/intermediate_algebra/641", "train/intermediate_algebra/1190", "train/geometry/767", "train/algebra/908",
    "test/intermediate_algebra/547", "train/counting_and_probability/225", "train/algebra/1196", "test/geometry/808",
    "train/algebra/29", "train/algebra/741", "test/prealgebra/1718", "train/algebra/2179", "train/algebra/281",
    "train/algebra/2224", "train/prealgebra/1469", "test/intermediate_algebra/686", "train/intermediate_algebra/1366",
    "train/algebra/465", "train/precalculus/946", "train/geometry/6009", "test/geometry/638",
    "train/counting_and_probability/426", "test/prealgebra/1624", "test/algebra/1770", "test/intermediate_algebra/1969",
    "test/intermediate_algebra/55", "test/intermediate_algebra/1873", "train/algebra/854", "test/algebra/1928",
    "train/intermediate_algebra/1188", "test/counting_and_probability/503", "train/intermediate_algebra/587",
    "train/algebra/1109", "train/counting_and_probability/5010", "test/geometry/764", "train/algebra/2622",
    "train/prealgebra/385", "test/prealgebra/1663", "test/number_theory/447", "test/prealgebra/982",
    "train/algebra/1935", "train/number_theory/605", "test/number_theory/40", "train/prealgebra/1112",
    "test/algebra/583", "train/geometry/225")]


@pytest.fixture(scope="module")
def spec():
    return load_profile("e0")


@pytest.fixture(scope="module")
def split(spec):
    try:
        return sdata.load_split(spec)
    except Exception as e:  # dataset not cached on this machine
        pytest.skip(f"nlile dataset unavailable: {e!r}")


def test_schedule_is_deterministic_and_consumes_all_four():
    pool = list(range(1000))
    a = candidate_schedule(pool, 42, 20, 4)
    assert a == candidate_schedule(pool, 42, 20, 4)
    flat = [r for b in a for r in b]
    assert len(flat) == 80 and len(set(flat)) == 80 and all(0 <= r < 1000 for r in flat)
    assert a != candidate_schedule(pool, 43, 20, 4)
    with pytest.raises(ValueError):
        candidate_schedule(list(range(10)), 0, 3, 4)


def test_committed_schedules_verify(spec):
    pool_rows = list(sdata.section_rows(spec, "pool"))
    for seed in spec.run.seeds:
        batches = load_schedule(sdata.manifest_dir(spec), seed, pool_rows, spec.loop.steps, spec.loop.batch_b)
        assert len(batches) == 20 and all(len(b["rows"]) == 4 == len(b["unique_ids"]) for b in batches)


def test_manifest_sections(spec):
    doc = sdata.load_data_manifest(spec)
    s = doc["sections"]
    assert doc["n_rows"] == 12000 and s["pool"]["n"] == 1000 and s["test50"]["n"] == 50
    assert [r["unique_id"] for r in s["test50"]["items"]] == TEST50_IDS
    assert s["test50"]["n_test_ids"] == 20
    assert s["pool"]["level_counts"] == {"1": 87, "2": 152, "3": 203, "4": 259, "5": 299}
    # paper Table 1: candidates #0..#3 are levels 5, 2, 5, 4
    assert [r["level"] for r in s["phase1_candidates"]["items"]] == [5, 2, 5, 4]
    assert [r["level"] for r in s["phase1_test"]["items"]] == [5, 2, 4, 1, 1, 5, 4, 1, 2, 2]


def test_rebuilt_items_match_manifest_and_hf_shuffle(spec, split):
    items = sdata.load_section(spec, "test50", split)
    assert [it.unique_id for it in items] == TEST50_IDS
    assert all(it.answer for it in items)
    phase1 = sdata.load_section(spec, "phase1_candidates", split)
    assert [it.unique_id for it in phase1] == ["test/prealgebra/1820.json", "test/geometry/554.json",
                                               "test/counting_and_probability/559.json", "test/number_theory/457.json"]


def test_verify_items_refuses_a_mismatch(spec, split):
    doc = sdata.load_data_manifest(spec)
    items = sdata.items_from_rows(split, range(1000, 1050), spec.data.shuffle_seed)
    sdata.verify_items(items, doc["sections"]["test50"])
    with pytest.raises(RuntimeError):
        sdata.verify_items(items[1:] + items[:1], doc["sections"]["test50"])
