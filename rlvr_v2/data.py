"""Datasets, problems, manifests and split hygiene.

- Candidates/pools come from the MATH *train* split only; evaluation uses MATH-500 only.
- Gold is always the dataset `answer` column (never re-parsed from `solution`).
- Manifests record unique_ids and normalised-text hashes so disjointness can be asserted
  everywhere and dataset revisions are detected.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .config import DataCfg

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

PI1_PROBLEM = (
    "The pressure \\( P \\) exerted by wind on a sail varies jointly as the area \\( A \\) of the sail and "
    "the cube of the wind's velocity \\( V \\). When the velocity is \\( 8 \\) miles per hour, the pressure "
    "on a sail of \\( 2 \\) square feet is \\( 4 \\) pounds. Find the wind velocity when the pressure on "
    "\\( 4 \\) square feet of sail is \\( 32 \\) pounds."
)
PI1_ANSWER = "12.8"
ONESHOT_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."


@dataclass(frozen=True)
class Problem:
    unique_id: str
    problem: str
    answer: str
    solution: str | None
    level: int | None
    subject: str | None
    source: str  # "math_train" | "math500" | "oneshot"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------- normalisation
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    return _WS.sub(" ", (s or "").strip()).lower()


def text_hash(s: str) -> str:
    return hashlib.sha1(normalize_text(s).encode("utf-8")).hexdigest()[:16]


def parse_level(x) -> int | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x)
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else None


def _row_to_problem(row: dict, source: str, idx: int) -> Problem:
    uid = row.get("unique_id") or f"{source}/{idx}"
    answer = row.get("answer")
    if answer is None or str(answer).strip() == "":
        raise ValueError(f"{source} row {uid} has no `answer` column; refusing to re-parse solution text")
    return Problem(
        unique_id=str(uid),
        problem=str(row["problem"]),
        answer=str(answer).strip(),
        solution=row.get("solution"),
        level=parse_level(row.get("level")),
        subject=row.get("subject") or row.get("type"),
        source=source,
    )


# ---------------------------------------------------------------------- loaders
def load_math_train(name: str = "nlile/hendrycks-MATH-benchmark") -> list[Problem]:
    from datasets import load_dataset

    ds = load_dataset(name, split="train")
    return [_row_to_problem(r, "math_train", i) for i, r in enumerate(ds)]


def load_math500(name: str = "HuggingFaceH4/MATH-500") -> list[Problem]:
    from datasets import load_dataset

    ds = load_dataset(name, split="test")
    return [_row_to_problem(r, "math500", i) for i, r in enumerate(ds)]


def load_pi1(path: str | Path | None = None) -> Problem:
    """Wang et al.'s pi1 example. The parquet stores the problem with the one-shot suffix appended;
    we strip it so prompts are rendered by `prompts.build_prompt` like every other problem."""
    p = Path(path) if path else ASSETS_DIR / "pi1_r128.parquet"
    problem, answer = PI1_PROBLEM, PI1_ANSWER
    if p.exists():
        try:
            import pandas as pd

            df = pd.read_parquet(p)
            content = df.iloc[0]["prompt"][0]["content"]
            if content.endswith(ONESHOT_SUFFIX):
                content = content[: -len(ONESHOT_SUFFIX)]
            problem = content.strip()
            answer = str(df.iloc[0]["reward_model"]["ground_truth"]).strip()
        except Exception:  # pragma: no cover - fall back to the inline constant
            pass
    return Problem("oneshot/pi1", problem, answer, None, None, "Algebra", "oneshot")


# ---------------------------------------------------------------------- manifests
@dataclass
class Manifest:
    name: str
    source: str
    ids: list[str]
    hashes: dict[str, str]
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=1)

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(**d)

    def __len__(self) -> int:
        return len(self.ids)


def build_manifest(problems: Iterable[Problem], name: str, source: str, **meta) -> Manifest:
    probs = list(problems)
    return Manifest(
        name=name,
        source=source,
        ids=[p.unique_id for p in probs],
        hashes={p.unique_id: text_hash(p.problem) for p in probs},
        meta={"n": len(probs), **meta},
    )


def assert_disjoint(*manifests: Manifest) -> None:
    """Raise if any two manifests share a unique_id or a normalised problem-text hash."""
    for i in range(len(manifests)):
        for j in range(i + 1, len(manifests)):
            a, b = manifests[i], manifests[j]
            shared_ids = set(a.ids) & set(b.ids)
            if shared_ids:
                raise AssertionError(f"Manifests {a.name} and {b.name} share unique_ids: {sorted(shared_ids)[:5]}")
            ha = {h: u for u, h in a.hashes.items()}
            shared_hash = [(ha[h], u) for u, h in b.hashes.items() if h in ha]
            if shared_hash:
                raise AssertionError(f"Manifests {a.name} and {b.name} share problem text: {shared_hash[:5]}")


def select_by_manifest(manifest: Manifest, problems: Iterable[Problem], verify_hash: bool = True) -> list[Problem]:
    by_id = {p.unique_id: p for p in problems}
    missing = [u for u in manifest.ids if u not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} manifest ids missing from the loaded dataset, e.g. {missing[:3]}")
    out = [by_id[u] for u in manifest.ids]
    if verify_hash:
        bad = [p.unique_id for p in out if text_hash(p.problem) != manifest.hashes.get(p.unique_id)]
        if bad:
            raise ValueError(f"Problem text changed since manifest {manifest.name} was built: {bad[:3]}")
    return out


# ---------------------------------------------------------------------- near duplicates
def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_text(s)))


def near_duplicates(pool: list[Problem], ref: list[Problem], ratio: float = 0.9) -> list[tuple[str, str, float]]:
    """Pairs (pool_id, ref_id, similarity) with exact-hash matches or difflib ratio >= `ratio`.
    A token-Jaccard prefilter (>= 0.5) keeps the quadratic difflib pass cheap."""
    hits: list[tuple[str, str, float]] = []
    ref_hash = {text_hash(r.problem): r for r in ref}
    ref_tokens = [(r, _tokens(r.problem)) for r in ref]
    for p in pool:
        h = text_hash(p.problem)
        if h in ref_hash:
            hits.append((p.unique_id, ref_hash[h].unique_id, 1.0))
            continue
        pt = _tokens(p.problem)
        if not pt:
            continue
        for r, rt in ref_tokens:
            if not rt:
                continue
            jacc = len(pt & rt) / len(pt | rt)
            if jacc < 0.5:
                continue
            sim = difflib.SequenceMatcher(None, normalize_text(p.problem), normalize_text(r.problem)).ratio()
            if sim >= ratio:
                hits.append((p.unique_id, r.unique_id, round(sim, 4)))
    return hits


# ---------------------------------------------------------------------- splits
def make_splits(cfg: DataCfg, train: list[Problem], math500: list[Problem]) -> dict[str, Manifest]:
    """Shuffle the train split with `cfg.shuffle_seed`; pool = first `pool_size` problems that are not
    near-duplicates of MATH-500; held-out = the next `heldout_size`. Returns manifests (not saved)."""
    import random

    rng = random.Random(cfg.shuffle_seed)
    order = list(range(len(train)))
    rng.shuffle(order)
    shuffled = [train[i] for i in order]
    candidates = shuffled[: (cfg.pool_size + cfg.heldout_size) * 2]
    dups = near_duplicates(candidates, math500, cfg.near_dup_ratio)
    dup_ids = {d[0] for d in dups}
    clean = [p for p in shuffled if p.unique_id not in dup_ids]
    pool = clean[: cfg.pool_size]
    heldout = clean[cfg.pool_size : cfg.pool_size + cfg.heldout_size]
    manifests = {
        "math500": build_manifest(math500, "math500", "math500", dataset=cfg.eval_dataset),
        "pool": build_manifest(pool, "pool", "math_train", dataset=cfg.train_dataset, shuffle_seed=cfg.shuffle_seed,
                               removed_near_duplicates=dups),
        "heldout": build_manifest(heldout, "heldout", "math_train", dataset=cfg.train_dataset,
                                  shuffle_seed=cfg.shuffle_seed),
    }
    assert_disjoint(*manifests.values())
    return manifests
