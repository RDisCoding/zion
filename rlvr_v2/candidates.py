"""Pre-registered Study-1 candidate design: anchors plus d_wrong-matched pairs per p_s bin.

Design (see `Study1Cfg`)
- `n_null_anchors` candidates with p_s == 1 (no learning signal) and `n_zero_anchors` with p_s == 0 whose
  failures are at least half parsable (format_rate >= 0.5), each set chosen for level diversity.
- For every p_s bin (lower bound exclusive, upper bound inclusive) `pairs_per_bin[i]` matched pairs:
  same `level`, |p_s difference| <= `max_ps_diff`, and a d_wrong gap >= `min_dwrong_gap` between the
  "high" and the "low" member. Pairs are chosen greedily by the largest d_wrong gap (ties broken by a
  seeded RNG); every candidate is used at most once. When a bin is short the gap threshold is relaxed
  stepwise (configured value -> 0.3 -> 0.2) and every relaxation is recorded.
- `replicate_count` candidates receive a second seed; they are drawn round-robin across bins, alternating
  the high/low role, so seed noise is estimated across the whole design.

The procedure is deterministic given `rng_seed` and independent of the input order (candidates are
sorted by `unique_id` first).
"""
from __future__ import annotations

import dataclasses
import logging
import math
from collections import defaultdict
from typing import Sequence

import numpy as np

from .config import Study1Cfg
from .data import Manifest
from .signals import Signals

log = logging.getLogger(__name__)

RELAXATION_STEPS: tuple[float, ...] = (0.3, 0.2)
_TOL = 1e-9


def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def in_bin(p_s: float, lo: float, hi: float) -> bool:
    """Bin membership with the design convention lo < p_s <= hi."""
    return (p_s > lo + _TOL) and (p_s <= hi + _TOL)


def _shuffled(items: list, rng: np.random.Generator) -> list:
    return [items[i] for i in rng.permutation(len(items))] if items else []


def _pick_diverse(cands: list[Signals], n: int, rng: np.random.Generator) -> list[Signals]:
    """Pick up to `n` candidates round-robin over `level` (level order and within-level order from `rng`)."""
    by_level: dict = defaultdict(list)
    for s in cands:
        by_level[s.level].append(s)
    levels = sorted(by_level, key=lambda L: (L is None, L if L is not None else 0))
    levels = _shuffled(levels, rng)
    queues = {L: _shuffled(by_level[L], rng) for L in levels}
    picked: list[Signals] = []
    while len(picked) < n and any(queues.values()):
        for L in levels:
            if queues[L] and len(picked) < n:
                picked.append(queues[L].pop())
    return picked


def _form_pairs(elig: list[Signals], need: int, used: set[str], max_ps_diff: float, min_gap: float,
                rng: np.random.Generator) -> list[dict]:
    """Greedy matching by the largest d_wrong gap among unused eligible candidates."""
    cands = [s for s in elig if s.unique_id not in used]
    options = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            if a.level != b.level or abs(a.p_s - b.p_s) > max_ps_diff + _TOL:
                continue
            gap = abs(float(a.d_wrong) - float(b.d_wrong))
            if gap + _TOL < min_gap:
                continue
            hi, lo = (a, b) if a.d_wrong >= b.d_wrong else (b, a)
            options.append((gap, float(rng.random()), hi, lo))
    options.sort(key=lambda t: (-t[0], t[1]))
    pairs: list[dict] = []
    for gap, _, hi, lo in options:
        if len(pairs) >= need:
            break
        if hi.unique_id in used or lo.unique_id in used:
            continue
        used.update({hi.unique_id, lo.unique_id})
        pairs.append({
            "high": hi.unique_id, "low": lo.unique_id, "level": hi.level,
            "ps_high": float(hi.p_s), "ps_low": float(lo.p_s),
            "dwrong_high": float(hi.d_wrong), "dwrong_low": float(lo.d_wrong), "dwrong_gap": float(gap),
            "min_dwrong_gap_used": float(min_gap),
        })
    return pairs


def _replicate_ids(pairs: list[dict], anchors: list[str], count: int) -> list[str]:
    """Round-robin over bins, alternating the high/low role across bins and rounds; anchors fill any remainder."""
    by_bin: dict = defaultdict(list)
    for p in pairs:
        by_bin[p["bin_index"]].append(p)
    bins = sorted(by_bin)
    out: list[str] = []
    r = 0
    while len(out) < count and any(r < len(by_bin[b]) for b in bins):
        for k, b in enumerate(bins):
            if r < len(by_bin[b]) and len(out) < count:
                role = "high" if (r + k) % 2 == 0 else "low"
                out.append(by_bin[b][r][role])
        r += 1
    for uid in anchors:
        if len(out) < count and uid not in out:
            out.append(uid)
    return out


def select_candidates(signals: Sequence[Signals], cfg: Study1Cfg, rng_seed: int = 0) -> dict:
    """Apply the pre-registered design to pool signals; see the module docstring for the rules."""
    if len(cfg.pairs_per_bin) != len(cfg.bins):
        raise ValueError("cfg.pairs_per_bin and cfg.bins must have the same length")
    rng = np.random.default_rng(rng_seed)
    by_uid: dict[str, Signals] = {}
    for s in signals:
        by_uid[s.unique_id] = s
    pool = [by_uid[u] for u in sorted(by_uid)]
    used: set[str] = set()

    nulls = _pick_diverse([s for s in pool if s.p_s >= 1.0 - _TOL], cfg.n_null_anchors, rng)
    zeros = _pick_diverse([s for s in pool if s.p_s <= _TOL and (s.format_rate or 0.0) >= 0.5],
                          cfg.n_zero_anchors, rng)
    used.update(s.unique_id for s in nulls + zeros)

    pairs: list[dict] = []
    relaxations: list[dict] = []
    availability: list[dict] = []
    shortfall: dict[str, int] = {}
    if len(nulls) < cfg.n_null_anchors:
        shortfall["anchors_null"] = cfg.n_null_anchors - len(nulls)
    if len(zeros) < cfg.n_zero_anchors:
        shortfall["anchors_zero"] = cfg.n_zero_anchors - len(zeros)
    gaps = [float(cfg.min_dwrong_gap)] + [g for g in RELAXATION_STEPS if g < cfg.min_dwrong_gap - _TOL]
    for bi, ((lo, hi), need) in enumerate(zip(cfg.bins, cfg.pairs_per_bin)):
        in_b = [s for s in pool if in_bin(s.p_s, lo, hi)]
        elig = [s for s in in_b if not _is_nan(s.d_wrong) and s.level is not None and s.unique_id not in used]
        bin_pairs: list[dict] = []
        for step, gap in enumerate(gaps):
            if len(bin_pairs) >= need:
                break
            new = _form_pairs(elig, need - len(bin_pairs), used, cfg.max_ps_diff, gap, rng)
            for p in new:
                p.update({"bin": [float(lo), float(hi)], "bin_index": bi})
            bin_pairs.extend(new)
            if step > 0:
                relaxations.append({"bin": [float(lo), float(hi)], "bin_index": bi, "min_dwrong_gap": gap,
                                    "pairs_added": len(new)})
                log.warning("bin %d (%.3f, %.3f]: relaxed min d_wrong gap to %.2f, added %d pair(s)",
                            bi, lo, hi, gap, len(new))
        availability.append({"bin": [float(lo), float(hi)], "bin_index": bi, "needed": int(need),
                             "n_in_bin": len(in_b), "n_eligible": len(elig), "formed": len(bin_pairs)})
        if len(bin_pairs) < need:
            shortfall[f"bin{bi}"] = need - len(bin_pairs)
            log.warning("bin %d (%.3f, %.3f]: only %d of %d pairs could be formed", bi, lo, hi, len(bin_pairs), need)
        pairs.extend(bin_pairs)

    anchors_null = [s.unique_id for s in nulls]
    anchors_zero = [s.unique_id for s in zeros]
    ordered = anchors_null + anchors_zero
    for p in pairs:
        ordered += [p["high"], p["low"]]
    if len(set(ordered)) != len(ordered):
        raise RuntimeError("internal error: a candidate was selected twice")
    return {
        "anchors_null": anchors_null,
        "anchors_zero": anchors_zero,
        "pairs": pairs,
        "replicate_ids": _replicate_ids(pairs, anchors_null + anchors_zero, cfg.replicate_count),
        "ordered_ids": ordered,
        "relaxations": relaxations,
        "shortfall": shortfall,
        "availability": availability,
        "bins": [[float(lo), float(hi)] for lo, hi in cfg.bins],
        "n_pool": len(pool),
        "rng_seed": int(rng_seed),
        "config": dataclasses.asdict(cfg),
    }


def candidates_manifest(selection: dict, pool_manifest: Manifest, name: str = "study1_candidates") -> Manifest:
    """Manifest of the selected candidates in design order; hashes are copied from the pool manifest and
    the full selection (pairs, anchors, replicates, relaxations) is kept in `meta`."""
    ids = list(selection["ordered_ids"])
    missing = [u for u in ids if u not in pool_manifest.hashes]
    if missing:
        raise KeyError(f"{len(missing)} selected ids are not in the pool manifest, e.g. {missing[:3]}")
    meta = {
        "n": len(ids),
        "pool_manifest": pool_manifest.name,
        "pool_source": pool_manifest.source,
        "pool_n": len(pool_manifest.ids),
        "replicate_ids": list(selection["replicate_ids"]),
        "selection": selection,
    }
    for key in ("dataset", "shuffle_seed"):
        if key in pool_manifest.meta:
            meta[key] = pool_manifest.meta[key]
    return Manifest(name=name, source=pool_manifest.source, ids=ids,
                    hashes={u: pool_manifest.hashes[u] for u in ids}, meta=meta)


def summarize_selection(selection: dict) -> str:
    """Plain-text summary table (per bin: available, eligible, pairs formed, relaxations)."""
    relax_by_bin: dict = defaultdict(list)
    for r in selection.get("relaxations", []):
        relax_by_bin[r["bin_index"]].append(f"{r['min_dwrong_gap']:.2f}(+{r['pairs_added']})")
    lines = [f"{'bin':<16}{'needed':>8}{'in_bin':>8}{'eligible':>10}{'formed':>8}  relaxations"]
    for a in selection.get("availability", []):
        lo, hi = a["bin"]
        lines.append(f"({lo:.2f}, {hi:.2f}]{a['needed']:>8}{a['n_in_bin']:>8}{a['n_eligible']:>10}{a['formed']:>8}"
                     f"  {', '.join(relax_by_bin.get(a['bin_index'], [])) or '-'}")
    cfg = selection.get("config", {})
    lines.append(f"anchors: null {len(selection['anchors_null'])}/{cfg.get('n_null_anchors', '?')}, "
                 f"zero {len(selection['anchors_zero'])}/{cfg.get('n_zero_anchors', '?')}")
    lines.append(f"replicates: {len(selection['replicate_ids'])}; candidates: {len(selection['ordered_ids'])}; "
                 f"pool: {selection.get('n_pool', '?')}")
    if selection.get("shortfall"):
        lines.append(f"SHORTFALL: {selection['shortfall']}")
    return "\n".join(lines)
