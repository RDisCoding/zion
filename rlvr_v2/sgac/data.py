"""The original pool / test split, rebuilt exactly, plus MATH-500 and pi1.

NB-M cell 4:
    full_ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="train").shuffle(seed=42)
    global_pool = rows 0..999 ; test_ds = rows 1000..1049
`datasets.shuffle(seed=42)` equals `np.random.default_rng(42).permutation(len(ds))`; both are computed and must agree.
Raw split rows are used (rlvr_v2.data.load_math_train drops answer-less rows and would shift the indices).

Every item carries both golds: `answer` (dataset column; the e0 profile's gold) and `solution` (the legacy grader takes
the FIRST `\\boxed{}` of it, NB-M's `extract_gt`). `configs/sgac/manifests/data.json` freezes ids, text hashes and
gold diagnostics; runs verify the rebuilt items against it.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..artifacts import REPO_ROOT, atomic_write_json, read_json, utc_now
from ..data import Manifest, Problem, load_math500, load_pi1, parse_level, select_by_manifest, text_hash
from . import legacy
from .spec import SgacSpec

log = logging.getLogger(__name__)

DATA_MANIFEST = "data.json"
SECTIONS = ("pool", "test50", "phase1_candidates", "phase1_test")


@dataclass(frozen=True)
class SgacItem:
    row: int  # position after .shuffle(seed=42); -1 for items from other sources
    orig_index: int  # row in the unshuffled split
    unique_id: str
    problem: str
    solution: str | None
    answer: str
    level: int | None
    subject: str | None
    source: str = "nlile_train"

    @property
    def legacy_solution(self) -> str:
        """What NB-M's grader received as `gt` (it extracts the first box; falls back to the whole text)."""
        return self.solution if self.solution else self.answer

    def to_problem(self) -> Problem:
        return Problem(self.unique_id, self.problem, self.answer, self.solution, self.level, self.subject, self.source)

    def to_dict(self) -> dict:
        return asdict(self)


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def manifest_dir(spec: SgacSpec) -> Path:
    return _abs(spec.data.manifest_dir)


def shuffled_order(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).permutation(int(n))


def load_split(spec: SgacSpec):
    from datasets import load_dataset

    return load_dataset(spec.data.dataset, split=spec.data.split, revision=spec.data.revision)


def items_from_rows(ds, rows: Iterable[int], seed: int, check_hf_shuffle: bool = True) -> list[SgacItem]:
    """Items at shuffled positions `rows`; the numpy permutation is the source of truth and is cross-checked
    against `datasets.Dataset.shuffle(seed)` for every requested row."""
    rows = [int(r) for r in rows]
    perm = shuffled_order(len(ds), seed)
    sh = ds.shuffle(seed=int(seed)) if check_hf_shuffle else None
    out: list[SgacItem] = []
    for r in rows:
        oi = int(perm[r])
        rec = ds[oi]
        if sh is not None and sh[r]["unique_id"] != rec["unique_id"]:
            raise RuntimeError(f"datasets.shuffle(seed={seed}) disagrees with the numpy permutation at row {r}")
        ans = rec.get("answer")
        if ans is None or str(ans).strip() == "":
            raise ValueError(f"row {r} ({rec.get('unique_id')}) has no answer column value")
        out.append(SgacItem(row=r, orig_index=oi, unique_id=str(rec["unique_id"]), problem=str(rec["problem"]),
                            solution=rec.get("solution"), answer=str(ans).strip(), level=parse_level(rec.get("level")),
                            subject=rec.get("subject") or rec.get("type")))
    return out


def section_rows(spec: SgacSpec, section: str) -> range:
    lo, hi = {"pool": spec.data.pool, "test50": spec.data.test, "phase1_candidates": spec.data.phase1_candidates,
              "phase1_test": spec.data.phase1_test}[section]
    return range(int(lo), int(hi))


def _item_record(it: SgacItem, mv=None) -> dict:
    gold_legacy = legacy.legacy_gold(it.legacy_solution)
    rec = {"row": it.row, "orig_index": it.orig_index, "unique_id": it.unique_id, "level": it.level,
           "subject": it.subject, "text_hash": text_hash(it.problem), "answer": it.answer, "gold_legacy": gold_legacy,
           "golds_equal_nospace": gold_legacy.replace(" ", "") == it.answer.replace(" ", "")}
    if mv is not None:
        rec["gold_parseable_mv"] = bool(mv.gold_parseable(it.answer))
        rec["golds_equivalent_mv"] = bool(gold_legacy.replace(" ", "") == it.answer.replace(" ", "")
                                          or mv.equivalent(gold_legacy, it.answer))
    return rec


def build_data_manifest(spec: SgacSpec, mv=None) -> dict:
    """Rebuild every section from the pinned dataset revision and describe it (ids, hashes, both golds)."""
    ds = load_split(spec)
    doc = {"dataset": spec.data.dataset, "revision": spec.data.revision, "split": spec.data.split, "n_rows": len(ds),
           "shuffle_seed": spec.data.shuffle_seed, "created": utc_now(),
           "rule": "load_dataset(dataset, split, revision).shuffle(seed) == np.random.default_rng(seed).permutation(n)",
           "sections": {}}
    for section in SECTIONS:
        items = items_from_rows(ds, section_rows(spec, section), spec.data.shuffle_seed)
        recs = [_item_record(it, mv) for it in items]
        levels = [r["level"] for r in recs]
        doc["sections"][section] = {
            "rows": [section_rows(spec, section).start, section_rows(spec, section).stop], "n": len(recs),
            "level_counts": {str(lv): levels.count(lv) for lv in sorted(set(levels), key=lambda v: (v is None, v))},
            "n_test_ids": sum(1 for r in recs if r["unique_id"].startswith("test/")),
            "golds_differ": [r["unique_id"] for r in recs if not r["golds_equal_nospace"]],
            "items": recs,
        }
        if mv is not None:
            doc["sections"][section]["gold_unparseable_mv"] = [r["unique_id"] for r in recs if not r["gold_parseable_mv"]]
    return doc


def write_data_manifest(spec: SgacSpec, mv=None) -> Path:
    path = manifest_dir(spec) / DATA_MANIFEST
    atomic_write_json(path, build_data_manifest(spec, mv))
    return path


def load_data_manifest(spec: SgacSpec) -> dict:
    doc = read_json(manifest_dir(spec) / DATA_MANIFEST)
    if doc is None:
        raise FileNotFoundError(f"{manifest_dir(spec) / DATA_MANIFEST} missing; run `python -m rlvr_v2.sgac build-manifests`")
    if doc["dataset"] != spec.data.dataset or doc["revision"] != spec.data.revision or int(doc["shuffle_seed"]) != spec.data.shuffle_seed:
        raise RuntimeError("data manifest was built for a different dataset/revision/seed than the spec")
    return doc


def verify_items(items: Sequence[SgacItem], manifest_section: dict) -> None:
    recs = manifest_section["items"]
    if len(recs) != len(items):
        raise RuntimeError(f"manifest section has {len(recs)} items, rebuilt {len(items)}")
    for it, rec in zip(items, recs):
        if it.row != rec["row"] or it.unique_id != rec["unique_id"] or text_hash(it.problem) != rec["text_hash"]:
            raise RuntimeError(f"rebuilt row {it.row} ({it.unique_id}) does not match the frozen manifest ({rec['unique_id']})")


def load_section(spec: SgacSpec, section: str, ds=None, manifest: dict | None = None) -> list[SgacItem]:
    """Rebuild one section and verify it against the frozen manifest."""
    ds = ds if ds is not None else load_split(spec)
    manifest = manifest if manifest is not None else load_data_manifest(spec)
    items = items_from_rows(ds, section_rows(spec, section), spec.data.shuffle_seed)
    verify_items(items, manifest["sections"][section])
    return items


def load_math500_items(spec: SgacSpec) -> list[SgacItem]:
    """MATH-500 in the order of rlvr_v2's frozen manifests/math500.json (text hashes verified)."""
    manifest = Manifest.load(_abs(spec.data.math500_manifest))
    probs = select_by_manifest(manifest, load_math500())
    return [SgacItem(row=-1, orig_index=i, unique_id=p.unique_id, problem=p.problem, solution=p.solution,
                     answer=p.answer, level=p.level, subject=p.subject, source="math500") for i, p in enumerate(probs)]


def pi1_item() -> SgacItem:
    p = load_pi1()
    return SgacItem(row=-1, orig_index=-1, unique_id=p.unique_id, problem=p.problem, solution=None, answer=p.answer,
                    level=p.level, subject=p.subject, source="oneshot")
