"""Candidate schedules and derived seeds.

NB-M drew each batch with an unseeded `random.sample(global_pool, 4)` and removed ALL four candidates from the pool
(80 problems over 20 steps); every `GRPOTrainer(...)` then reset Python's global RNG with `set_seed(42)`. The original
batches are therefore unrecoverable. Here one `random.Random(seed)` stream, independent of the model and the arm,
draws the batches; the schedule is precomputed into `configs/sgac/manifests/batches_seed{S}.json` and every run reads
and verifies it, so all arms and both profiles of a seed see identical batches (paired design).
"""
from __future__ import annotations

import platform
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..artifacts import atomic_write_json, read_json, utc_now

RANDOM_ARM_STREAM = 0x5A17  # keeps the random arm's draws independent of every other seeded stream


def candidate_schedule(pool_rows: Sequence[int], seed: int, steps: int, size: int) -> list[list[int]]:
    """`steps` batches of `size` pool rows: `random.Random(seed).sample(remaining, size)`, then all are removed."""
    rng = random.Random(int(seed))
    remaining = list(pool_rows)
    if steps * size > len(remaining):
        raise ValueError(f"pool of {len(remaining)} cannot supply {steps} x {size} candidates")
    out: list[list[int]] = []
    for _ in range(int(steps)):
        batch = rng.sample(remaining, int(size))
        for r in batch:
            remaining.remove(r)
        out.append([int(r) for r in batch])
    return out


def schedule_path(schedule_dir: str | Path, seed: int) -> Path:
    return Path(schedule_dir) / f"batches_seed{int(seed)}.json"


def write_schedule(schedule_dir: str | Path, seed: int, pool_rows: Sequence[int], steps: int, size: int,
                   uid_of_row: dict[int, str]) -> Path:
    batches = candidate_schedule(pool_rows, seed, steps, size)
    doc = {
        "seed": int(seed), "steps": int(steps), "batch_size": int(size), "n_pool": len(pool_rows),
        "rule": "random.Random(seed).sample(remaining, batch_size); all batch rows removed after each step",
        "python": platform.python_version(), "created": utc_now(),
        "batches": [{"step": t + 1, "rows": b, "unique_ids": [uid_of_row[r] for r in b]} for t, b in enumerate(batches)],
    }
    path = schedule_path(schedule_dir, seed)
    atomic_write_json(path, doc)
    return path


def load_schedule(schedule_dir: str | Path, seed: int, pool_rows: Sequence[int], steps: int, size: int) -> list[dict]:
    """Stored batches for `seed`, verified against a fresh recomputation (refuses on any mismatch)."""
    path = schedule_path(schedule_dir, seed)
    doc = read_json(path)
    if doc is None:
        raise FileNotFoundError(f"no candidate schedule at {path}; run `python -m rlvr_v2.sgac build-manifests`")
    stored = [b["rows"] for b in doc["batches"]]
    fresh = candidate_schedule(pool_rows, seed, steps, size)
    if stored[:steps] != fresh or int(doc["batch_size"]) != int(size):
        raise RuntimeError(f"stored schedule {path} does not match random.Random({seed}) on this Python "
                           f"({platform.python_version()} vs {doc.get('python')}); refusing to run")
    return doc["batches"][:steps]


def random_arm_rng(seed: int, step: int) -> np.random.Generator:
    """RNG for the random arm's pick at `step` (also used for its counterfactual pick in every arm's log)."""
    return np.random.default_rng([int(seed), RANDOM_ARM_STREAM, int(step)])


def derived_seeds(seed: int, step: int) -> dict[str, int]:
    """sieve: per step; grpo: the run seed at every burst (NB-M used TRL's default 42 at every burst); lora: run seed."""
    return {"sieve": int(seed) * 1000 + int(step), "grpo": int(seed), "lora_init": int(seed)}
