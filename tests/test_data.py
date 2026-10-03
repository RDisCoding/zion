import pytest

from rlvr_v2.data import (ONESHOT_SUFFIX, Manifest, Problem, assert_disjoint, build_manifest, load_pi1,
                          near_duplicates, parse_level, select_by_manifest, text_hash)


def P(uid, text, ans="1", level=3):
    return Problem(uid, text, ans, None, level, "Algebra", "math_train")


def test_parse_level():
    assert parse_level("Level 5") == 5 and parse_level(2) == 2
    assert parse_level(None) is None and parse_level("?") is None


def test_manifest_roundtrip_and_disjoint(tmp_path):
    a = build_manifest([P("a1", "x + 1 = 2"), P("a2", "y + 1 = 3")], "a", "math_train")
    b = build_manifest([P("b1", "z + 1 = 4")], "b", "math_train")
    a.save(tmp_path / "a.json")
    assert Manifest.load(tmp_path / "a.json") == a
    assert_disjoint(a, b)
    with pytest.raises(AssertionError):
        assert_disjoint(a, build_manifest([P("a2", "anything")], "c", "math_train"))
    with pytest.raises(AssertionError):
        assert_disjoint(a, build_manifest([P("other", "X + 1 = 2  ")], "d", "math500"))  # same normalised text


def test_select_by_manifest_verifies_hash():
    probs = [P("a1", "x + 1 = 2"), P("a2", "y + 1 = 3")]
    m = build_manifest(probs, "a", "math_train")
    assert [p.unique_id for p in select_by_manifest(m, probs)] == ["a1", "a2"]
    changed = [P("a1", "x + 1 = 2"), P("a2", "DIFFERENT")]
    with pytest.raises(ValueError):
        select_by_manifest(m, changed)
    with pytest.raises(KeyError):
        select_by_manifest(m, probs[:1])


def test_near_duplicates():
    pool = [P("p1", "Find the value of x such that 2x + 3 = 11 and x is an integer."),
            P("p2", "Compute the area of a circle with radius 3."),
            P("p3", "Completely unrelated question about primes below one hundred.")]
    ref = [P("r1", "Find the value of x such that 2x + 3 = 11 and x is an integer!"),
           P("r2", "compute the area of a circle with radius 3.")]
    hits = near_duplicates(pool, ref, ratio=0.9)
    pairs = {(a, b) for a, b, _ in hits}
    assert ("p1", "r1") in pairs and ("p2", "r2") in pairs and all(a != "p3" for a, _, _ in hits)


def test_load_pi1():
    p = load_pi1()
    assert p.answer == "12.8" and "sail" in p.problem and not p.problem.endswith(ONESHOT_SUFFIX.strip())
    assert p.unique_id == "oneshot/pi1" and p.source == "oneshot"
    assert text_hash(p.problem) == text_hash(p.problem.upper())
