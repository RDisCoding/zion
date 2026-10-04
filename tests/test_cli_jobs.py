from rlvr_v2 import cli
from rlvr_v2.data import Manifest


def test_study2_jobs_count_and_rows(capsys):
    cli.main(["study2-jobs", "--config", "configs/study2.yaml", "--count"])
    assert capsys.readouterr().out.strip().splitlines()[-1] == "9"
    cli.main(["study2-jobs", "--config", "configs/study2.yaml", "--override", "study2.arms=[random,learned]"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "index,arm,seed" and len(lines) == 1 + 2 * 3
    assert lines[1] == "0,random,1234" and lines[-1] == "5,learned,3456"


def test_manifests_differ_detects_id_and_hash_changes():
    a = Manifest(name="pool", source="math_train", ids=["x", "y"], hashes={"x": "1", "y": "2"})
    assert not cli.manifests_differ(a, Manifest(name="pool", source="math_train", ids=["x", "y"], hashes={"x": "1", "y": "2"}))
    assert cli.manifests_differ(a, Manifest(name="pool", source="math_train", ids=["y", "x"], hashes={"x": "1", "y": "2"}))
    assert cli.manifests_differ(a, Manifest(name="pool", source="math_train", ids=["x", "y"], hashes={"x": "1", "y": "9"}))
