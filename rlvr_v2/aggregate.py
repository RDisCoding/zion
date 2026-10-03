"""Collect ``results/<study>/<group>/<run>/`` artifacts into Parquet tables (``<results>/tables/``).

Tables
- ``runs``          one row per run directory: run.json + status.json + config.yaml essentials
- ``study1``        one row per Study-1 run (candidate x seed): pool signals, MATH-500 accuracy, paired delta
                    versus the base evaluation (item bootstrap), burst (training) summary fields
- ``eval_items``    long per-item correctness for every evaluation found; the base evaluation appears with
                    ``run_id`` ``_base`` (global) or ``_base/<group>`` (group-local ``_shared/eval_base``)
- ``train_metrics`` long per-optimizer-step metrics (``curriculum_step`` is NaN for Study-1 bursts)
- ``selections``    Study-2 per-step selections with the chosen candidate's signals and the score vector (JSON)
- ``study2_evals``  Study-2 per-checkpoint accuracy with bootstrap CI

Run directories are recognised by ``run.json`` (see ``artifacts.RunDir``). Study-1 runs are those under
``results/study1`` with ``eval/<tag>/per_item.jsonl``; curriculum runs are those with ``steps/`` or ``evals/``.
Missing or malformed files are skipped with a warning. Schema tolerance: the candidate id of a Study-1 run is
taken from run.json / result.json / status.json keys ``unique_id|candidate_uid|candidate_id|candidate|uid``,
then from the first training rollout; curriculum ``selection.json`` may carry the selected id at the top
level, under ``selected`` (dict / str / list) and the signals inline or in ``sieve/signals.jsonl``.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

from .artifacts import read_json, read_jsonl
from .signals import Signals
from .stats import bootstrap_mean_ci, paired_item_bootstrap

log = logging.getLogger(__name__)

SIGNAL_FIELDS: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(Signals))
UID_KEYS = ("unique_id", "candidate_uid", "candidate_id", "candidate", "uid", "problem_id")
STEP_RE = re.compile(r"step-?(\d+)")
BASE_GLOBS = ("study1/{group}/_shared/eval_base/math500/per_item.jsonl", "study1/{group}/_shared/eval_base/*/per_item.jsonl",
              "e0/**/base_eval/{style}/per_item.jsonl", "e0/**/base_eval/*/per_item.jsonl",
              "e0/**/g1/{style}/math500/per_item.jsonl", "e0/**/g1/*/math500/per_item.jsonl")

METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "loss": ("loss", "train/loss"),
    "reward": ("reward", "train/reward", "rewards/mean"),
    "rewards/correctness/mean": ("rewards/correctness/mean", "rewards/correctness_reward/mean",
                                 "rewards/correctness_reward_func/mean"),
    "frac_reward_zero_std": ("frac_reward_zero_std", "train/frac_reward_zero_std"),
    "completions/clipped_ratio": ("completions/clipped_ratio", "clipped_ratio"),
    "completions/mean_length": ("completions/mean_length", "completion_length", "completions/mean_len"),
}
_CORRECTNESS_RE = re.compile(r"^rewards/.*correct.*/mean$")

RUNS_COLUMNS = ["run_id", "study", "group", "run_name", "path", "seed", "config_hash", "git_hash", "created", "state",
                "stage", "updated", "error", "tag", "arm", "unique_id", "replicate", "selector", "prompt_style",
                "learning_rate", "rounds", "n_curriculum_steps", "kind"]
BURST_COLUMNS = ["n_train_steps", "loss_mean", "reward_mean", "reward_first", "reward_last", "correctness_mean",
                 "correctness_first", "correctness_last", "zero_std_frac_mean", "clipped_ratio_mean",
                 "clipped_ratio_max", "mean_length_first", "mean_length_last"]
STUDY1_COLUMNS = (["run_id", "group", "run_name", "unique_id", "seed", "replicate", "n_seeds", "state"]
                  + [f for f in SIGNAL_FIELDS if f != "unique_id"]
                  + ["acc", "ci_lo", "ci_hi", "n_items", "base_acc", "delta", "delta_ci_lo", "delta_ci_hi", "delta_p",
                     "n_common", "base_run_id", "has_signals"] + BURST_COLUMNS)
EVAL_ITEM_COLUMNS = ["run_id", "tag", "step", "unique_id", "correct", "level", "subject"]
TRAIN_METRIC_COLUMNS = ["run_id", "curriculum_step", "global_step", "loss", "reward", "rewards/correctness/mean",
                        "frac_reward_zero_std", "completions/clipped_ratio", "completions/mean_length"]
SELECTION_COLUMNS = ["run_id", "arm", "seed", "step", "selected_uid", "p_s", "v_bin", "d_simpson", "d_wrong",
                     "entropy_bits", "level", "n_candidates", "scores"]
STUDY2_EVAL_COLUMNS = ["run_id", "arm", "seed", "step", "acc", "ci_lo", "ci_hi", "n"]
TABLE_COLUMNS = {"runs": RUNS_COLUMNS, "study1": STUDY1_COLUMNS, "eval_items": EVAL_ITEM_COLUMNS,
                 "train_metrics": TRAIN_METRIC_COLUMNS, "selections": SELECTION_COLUMNS,
                 "study2_evals": STUDY2_EVAL_COLUMNS}


# ---------------------------------------------------------------------------------------- small io
def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as e:  # noqa: BLE001
        log.warning("could not parse %s: %s", path, e)
        return {}


def _first(d: Any, keys: Iterable[str]) -> Any:
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _step_of(path: Path) -> int | None:
    m = STEP_RE.search(path.name)
    return int(m.group(1)) if m else None


def _per_item_map(path: Path) -> dict[str, bool] | None:
    rows = read_jsonl(path)
    if not rows:
        return None
    out: dict[str, bool] = {}
    for r in rows:
        uid = _first(r, UID_KEYS)
        if uid is None or "correct" not in r:
            continue
        out[str(uid)] = bool(r["correct"])
    return out or None


def _eval_item_rows(path: Path, run_id: str, tag: str, step: int | None) -> list[dict]:
    rows = []
    for r in read_jsonl(path):
        uid = _first(r, UID_KEYS)
        if uid is None or "correct" not in r:
            continue
        rows.append({"run_id": run_id, "tag": tag, "step": step, "unique_id": str(uid), "correct": bool(r["correct"]),
                     "level": r.get("level"), "subject": r.get("subject")})
    return rows


def _summary_stats(summary_path: Path, per_item: dict[str, bool] | None, n_boot: int, seed: int) -> dict:
    s = read_json(summary_path, None) or {}
    acc = _first(s, ("acc", "accuracy"))
    if acc is not None:
        ci = s.get("ci") if isinstance(s.get("ci"), (list, tuple)) and len(s["ci"]) == 2 else None
        return {"acc": float(acc), "ci_lo": float(_first(s, ("ci_lo", "ci_low")) if ci is None else ci[0]),
                "ci_hi": float(_first(s, ("ci_hi", "ci_high")) if ci is None else ci[1]),
                "n": int(s.get("n", len(per_item) if per_item else 0))}
    if per_item:
        vals = np.array(list(per_item.values()), dtype=float)
        mean, lo, hi = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
        return {"acc": mean, "ci_lo": lo, "ci_hi": hi, "n": int(vals.size)}
    return {"acc": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}


# ------------------------------------------------------------------------------------- train metrics
def _metric(row: dict, name: str) -> float:
    for k in METRIC_KEYS[name]:
        if k in row and row[k] is not None:
            try:
                return float(row[k])
            except (TypeError, ValueError):
                return float("nan")
    if name == "rewards/correctness/mean":
        for k, v in row.items():
            if _CORRECTNESS_RE.match(str(k)) and v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return float("nan")
    return float("nan")


def _train_rows(path: Path, run_id: str, curriculum_step: int | None) -> list[dict]:
    out = []
    for i, r in enumerate(read_jsonl(path)):
        if not isinstance(r, dict):
            continue
        step = _first(r, ("global_step", "step"))
        out.append({"run_id": run_id, "curriculum_step": curriculum_step,
                    "global_step": int(step) if step is not None else i + 1,
                    **{name: _metric(r, name) for name in METRIC_KEYS}})
    return out


def _burst_summary(rows: Sequence[dict]) -> dict:
    out = {c: float("nan") for c in BURST_COLUMNS}
    out["n_train_steps"] = len(rows)
    if not rows:
        return out

    def col(name: str) -> np.ndarray:
        return np.array([r.get(name, float("nan")) for r in rows], dtype=float)

    def nanmean(a: np.ndarray) -> float:
        return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")

    def first(a: np.ndarray) -> float:
        f = a[np.isfinite(a)]
        return float(f[0]) if f.size else float("nan")

    def last(a: np.ndarray) -> float:
        f = a[np.isfinite(a)]
        return float(f[-1]) if f.size else float("nan")

    reward, corr = col("reward"), col("rewards/correctness/mean")
    clipped, length = col("completions/clipped_ratio"), col("completions/mean_length")
    out.update({
        "loss_mean": nanmean(col("loss")), "reward_mean": nanmean(reward), "reward_first": first(reward),
        "reward_last": last(reward), "correctness_mean": nanmean(corr), "correctness_first": first(corr),
        "correctness_last": last(corr), "zero_std_frac_mean": nanmean(col("frac_reward_zero_std")),
        "clipped_ratio_mean": nanmean(clipped),
        "clipped_ratio_max": float(np.nanmax(clipped)) if np.isfinite(clipped).any() else float("nan"),
        "mean_length_first": first(length), "mean_length_last": last(length),
    })
    return out


# ------------------------------------------------------------------------------------ base evaluation
class _BaseResolver:
    """Finds and caches the base-model per-item file for a Study-1 group / prompt style."""

    def __init__(self, results_root: Path, explicit: Path | None):
        self.root = results_root
        self.explicit = explicit
        self._cache: dict[Path, tuple[str, dict[str, bool] | None]] = {}
        self.used: dict[str, str] = {}

    def _load(self, path: Path, run_id: str) -> tuple[str, dict[str, bool] | None]:
        if path not in self._cache:
            items = _per_item_map(path)
            if items is None:
                log.warning("base per-item file %s is empty or unreadable", path)
            self._cache[path] = (run_id, items)
            self.used[run_id] = str(path)
        return self._cache[path]

    def resolve(self, group: str, style: str | None) -> tuple[str | None, dict[str, bool] | None, Path | None]:
        if self.explicit is not None:
            rid, items = self._load(self.explicit, "_base")
            return rid, items, self.explicit
        for pattern in BASE_GLOBS:
            pat = pattern.format(group=group, style=style or "*")
            hits = sorted(self.root.glob(pat))
            if hits:
                rid = f"_base/{group}" if pattern.startswith("study1") else "_base"
                if len(hits) > 1:
                    log.warning("several base per-item files match %s; using %s", pat, hits[0])
                rid_, items = self._load(hits[0], rid)
                return rid_, items, hits[0]
        return None, None, None

    def eval_item_rows(self) -> list[dict]:
        rows: list[dict] = []
        for path, (rid, _) in self._cache.items():
            rows += _eval_item_rows(path, rid, "math500", None)
        return rows


# ----------------------------------------------------------------------------------- pool signals
def load_pool_signals(path: Path | None, policy_tag: str = "base") -> dict[str, dict]:
    """unique_id -> Signals fields from a sieve ``signals.jsonl`` (rows of the requested policy tag win;
    last line wins among duplicates). Missing file -> empty dict (warning)."""
    if path is None or not Path(path).exists():
        if path is not None:
            log.warning("pool signals file %s not found; study1 rows will lack signals", path)
        return {}
    out: dict[str, dict] = {}
    other: dict[str, dict] = {}
    for r in read_jsonl(path):
        uid = r.get("unique_id")
        if uid is None:
            continue
        try:
            d = Signals.from_dict(r).to_dict()
        except TypeError:
            d = {k: r.get(k) for k in SIGNAL_FIELDS}
        (out if r.get("policy_tag", policy_tag) == policy_tag else other)[str(uid)] = d
    for uid, d in other.items():
        out.setdefault(uid, d)
    return out


# ------------------------------------------------------------------------------------------- runs
def discover_runs(results_root: Path) -> list[Path]:
    return sorted(p.parent for p in Path(results_root).glob("*/*/*/run.json"))


def _run_record(run_dir: Path, results_root: Path) -> dict:
    meta = read_json(run_dir / "run.json", {}) or {}
    status = read_json(run_dir / "status.json", {}) or {}
    cfg = _read_yaml(run_dir / "config.yaml")
    rel = run_dir.relative_to(results_root).parts
    study, group, run_name = (rel + ("", "", ""))[:3]
    run_id = meta.get("run_id") or "/".join(rel)
    arm = meta.get("arm")
    if arm is None and "__seed" in run_name:
        arm = run_name.split("__seed")[0]
    seed = meta.get("seed")
    if seed is None and "__seed" in run_name:
        try:
            seed = int(run_name.split("__seed")[-1])
        except ValueError:
            seed = None
    step_dirs = sorted(d for d in (run_dir / "steps").glob("step-*") if d.is_dir()) if (run_dir / "steps").exists() else []
    is_curriculum = bool(step_dirs) or (run_dir / "evals").exists() or (run_dir / "state.json").exists()
    looks_study1 = study == "study1" or (run_dir / "eval").exists() or (run_dir / "result.json").exists()
    kind = "curriculum" if is_curriculum else ("study1" if looks_study1 else "other")
    if kind == "curriculum" and arm is None:
        arm = (cfg.get("selector") or {}).get("name")
    rec = {
        "run_id": run_id, "study": meta.get("study", study), "group": group, "run_name": meta.get("run_name", run_name),
        "path": str(run_dir), "seed": seed, "config_hash": meta.get("config_hash"), "git_hash": meta.get("git_hash"),
        "created": meta.get("created"), "state": status.get("state"), "stage": status.get("stage"),
        "updated": status.get("updated"), "error": status.get("error"), "tag": (cfg.get("run") or {}).get("tag"),
        "arm": arm, "unique_id": _first(meta, UID_KEYS), "replicate": meta.get("replicate"),
        "selector": (cfg.get("selector") or {}).get("name"), "prompt_style": (cfg.get("prompt") or {}).get("style"),
        "learning_rate": (cfg.get("train") or {}).get("learning_rate"), "rounds": (cfg.get("train") or {}).get("rounds"),
        "n_curriculum_steps": len(step_dirs) if kind == "curriculum" else None, "kind": kind,
        "_meta": meta, "_status": status, "_cfg": cfg, "_step_dirs": step_dirs,
    }
    return rec


def _candidate_uid(run_dir: Path, rec: dict) -> str | None:
    for src in (rec["_meta"], read_json(run_dir / "result.json", {}) or {}, rec["_status"]):
        uid = _first(src, UID_KEYS)
        if uid is not None:
            return str(uid)
    rollouts = run_dir / "train" / "train_rollouts.jsonl"
    if rollouts.exists():
        with open(rollouts, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    uid = _first(json.loads(line), UID_KEYS)
                    if uid is not None:
                        return str(uid)
                    break
    return None


# ----------------------------------------------------------------------------------------- study 1
def _collect_study1(run_dir: Path, rec: dict, base: _BaseResolver, signals: dict[str, dict], n_boot: int, seed: int,
                    eval_rows: list[dict], train_rows: list[dict]) -> dict | None:
    uid = _candidate_uid(run_dir, rec)
    rec["unique_id"] = uid
    if uid is None:
        log.warning("%s: could not determine the candidate unique_id", rec["run_id"])
    tm = run_dir / "train" / "train_metrics.jsonl"
    rows = _train_rows(tm, rec["run_id"], None) if tm.exists() else []
    if not rows:
        log.warning("%s: no training metrics found", rec["run_id"])
    train_rows.extend(rows)
    per_item_paths = sorted((run_dir / "eval").glob("*/per_item.jsonl")) if (run_dir / "eval").exists() else []
    for p in per_item_paths:
        eval_rows.extend(_eval_item_rows(p, rec["run_id"], p.parent.name, None))
    main_eval = next((p for p in per_item_paths if p.parent.name == "math500"), None)
    row: dict[str, Any] = {"run_id": rec["run_id"], "group": rec["group"], "run_name": rec["run_name"], "unique_id": uid,
                           "seed": rec["seed"], "replicate": rec["replicate"], "state": rec["state"]}
    sig = signals.get(uid) if uid is not None else None
    row["has_signals"] = sig is not None
    if sig is None and uid is not None:
        log.warning("%s: candidate %s has no pool signals", rec["run_id"], uid)
    for f in SIGNAL_FIELDS:
        if f != "unique_id":
            row[f] = sig.get(f) if sig else None
    row.update(_burst_summary(rows))
    row.update({"acc": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_items": 0, "base_acc": float("nan"),
                "delta": float("nan"), "delta_ci_lo": float("nan"), "delta_ci_hi": float("nan"), "delta_p": None,
                "n_common": 0, "base_run_id": None})
    if main_eval is None:
        log.warning("%s: no eval/math500/per_item.jsonl (status %s)", rec["run_id"], rec["state"])
        return row
    per_item = _per_item_map(main_eval)
    stats = _summary_stats(main_eval.parent / "summary.json", per_item, n_boot, seed)
    row.update({"acc": stats["acc"], "ci_lo": stats["ci_lo"], "ci_hi": stats["ci_hi"], "n_items": stats["n"]})
    base_rid, base_items, base_path = base.resolve(rec["group"], rec.get("prompt_style"))
    if base_items is None:
        log.warning("%s: no base per-item evaluation found; delta is NaN", rec["run_id"])
        return row
    row["base_run_id"] = base_rid
    if per_item:
        common = sorted(set(per_item) & set(base_items))
        if common:
            a = np.array([per_item[u] for u in common], dtype=float)
            b = np.array([base_items[u] for u in common], dtype=float)
            res = paired_item_bootstrap(a, b, n_boot=n_boot, seed=seed)
            row.update({"base_acc": float(b.mean()), "delta": res["delta"], "delta_ci_lo": res["ci_lo"],
                        "delta_ci_hi": res["ci_hi"], "delta_p": res["p_two_sided"], "n_common": len(common)})
            if len(common) < len(base_items):
                log.warning("%s: only %d of %d base items are shared with the run evaluation", rec["run_id"],
                            len(common), len(base_items))
    return row


# ---------------------------------------------------------------------------------------- study 2
def _selection_records(sel: Any, step_dir: Path) -> list[dict]:
    """Normalise one selection.json (dict with the selected candidate, or a list of such dicts)."""
    if isinstance(sel, list):
        out = []
        for s in sel:
            out += _selection_records(s, step_dir)
        return out
    if not isinstance(sel, dict):
        return []
    selected = _first(sel, ("selected", "candidate", "pick", "chosen"))
    if isinstance(selected, list):
        base = {k: v for k, v in sel.items() if k not in ("selected", "candidate", "pick", "chosen")}
        out = []
        for s in selected:
            out += _selection_records({**base, "selected": s}, step_dir)
        return out
    uid = _first(sel, ("selected_uid", "unique_id", "uid", "selected_id", "chosen_uid"))
    if uid is None and isinstance(selected, dict):
        uid = _first(selected, UID_KEYS)
    if uid is None and isinstance(selected, str):
        uid = selected
    sig = None
    for src in (sel, selected if isinstance(selected, dict) else None,
                sel.get("signals") if isinstance(sel.get("signals"), dict) else None,
                sel.get("selected_signals") if isinstance(sel.get("selected_signals"), dict) else None):
        if src and src.get("p_s") is not None:
            sig = src
            break
    if sig is None and isinstance(sel.get("signals"), dict) and uid in sel["signals"]:
        sig = sel["signals"][uid]
    if sig is None and uid is not None:
        sieve = step_dir / "sieve" / "signals.jsonl"
        for r in read_jsonl(sieve):
            if str(r.get("unique_id")) == str(uid):
                sig = r
        if sig is None:
            log.warning("%s: no signals found for selected candidate %s", step_dir, uid)
    scores = sel.get("scores")
    n_cand = len(scores) if isinstance(scores, (list, dict)) else _first(sel, ("n_candidates", "batch_size"))
    step = sel.get("step", _step_of(step_dir))

    def g(name: str):
        return sig.get(name) if isinstance(sig, dict) else None

    return [{"step": int(step) if step is not None else None, "selected_uid": None if uid is None else str(uid),
             "p_s": g("p_s"), "v_bin": g("v_bin"), "d_simpson": g("d_simpson"), "d_wrong": g("d_wrong"),
             "entropy_bits": g("entropy_bits"), "level": g("level"), "n_candidates": n_cand,
             "scores": json.dumps(scores) if scores is not None else None}]


def _collect_curriculum(run_dir: Path, rec: dict, n_boot: int, seed: int, eval_rows: list[dict], train_rows: list[dict],
                        selection_rows: list[dict], study2_rows: list[dict]) -> None:
    rid, arm, run_seed = rec["run_id"], rec["arm"], rec["seed"]
    for sd in rec["_step_dirs"]:
        cstep = _step_of(sd)
        sel_path = sd / "selection.json"
        if sel_path.exists():
            for r in _selection_records(read_json(sel_path, None), sd):
                if r["step"] is None:
                    r["step"] = cstep
                selection_rows.append({"run_id": rid, "arm": arm, "seed": run_seed, **r})
        else:
            log.warning("%s: %s has no selection.json", rid, sd.name)
        tm = sd / "train" / "train_metrics.jsonl"
        if tm.exists():
            train_rows.extend(_train_rows(tm, rid, cstep))
    evals_root = run_dir / "evals"
    if not evals_root.exists():
        log.warning("%s: no evals/ directory", rid)
        return
    for ed in sorted(d for d in evals_root.glob("step-*") if d.is_dir()):
        step = _step_of(ed)
        per_item_path = next(iter(sorted(ed.rglob("per_item.jsonl"))), None)
        summary_path = per_item_path.parent / "summary.json" if per_item_path else ed / "summary.json"
        per_item = _per_item_map(per_item_path) if per_item_path else None
        if per_item_path is None and not summary_path.exists():
            log.warning("%s: %s has neither per_item.jsonl nor summary.json", rid, ed.name)
            continue
        if per_item_path is not None:
            eval_rows.extend(_eval_item_rows(per_item_path, rid, ed.name, step))
        stats = _summary_stats(summary_path, per_item, n_boot, seed)
        study2_rows.append({"run_id": rid, "arm": arm, "seed": run_seed, "step": step, **stats})


# ------------------------------------------------------------------------------------------ main
def aggregate(results_root: Path | str, base_per_item: Path | str | None = None, signals_path: Path | str | None = None,
              out_dir: Path | str | None = None, n_boot: int = 2000, seed: int = 0, write: bool = True
              ) -> dict[str, pd.DataFrame]:
    """Walk ``results_root`` and build all tables; writes Parquet files to ``out_dir`` (default
    ``<results_root>/tables``) unless ``write=False``. ``signals_path`` defaults to
    ``<results_root>/pool/signals.jsonl``."""
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"results root {root} does not exist")
    base = _BaseResolver(root, Path(base_per_item) if base_per_item else None)
    sig_path = Path(signals_path) if signals_path else root / "pool" / "signals.jsonl"
    signals = load_pool_signals(sig_path)
    run_rows, study1_rows, eval_rows, train_rows, selection_rows, study2_rows = [], [], [], [], [], []
    run_dirs = discover_runs(root)
    if not run_dirs:
        log.warning("no run directories (run.json) found under %s", root)
    for run_dir in run_dirs:
        try:
            rec = _run_record(run_dir, root)
        except Exception as e:  # noqa: BLE001
            log.warning("skipping %s: %s", run_dir, e)
            continue
        if rec["kind"] == "curriculum":
            _collect_curriculum(run_dir, rec, n_boot, seed, eval_rows, train_rows, selection_rows, study2_rows)
        elif rec["study"] == "study1" and rec["kind"] == "study1":
            row = _collect_study1(run_dir, rec, base, signals, n_boot, seed, eval_rows, train_rows)
            if row is not None:
                study1_rows.append(row)
        else:
            for p in sorted(run_dir.glob("eval/*/per_item.jsonl")):
                eval_rows.extend(_eval_item_rows(p, rec["run_id"], p.parent.name, None))
            tm = run_dir / "train" / "train_metrics.jsonl"
            if tm.exists():
                train_rows.extend(_train_rows(tm, rec["run_id"], None))
        run_rows.append({k: v for k, v in rec.items() if not k.startswith("_")})
    eval_rows = base.eval_item_rows() + eval_rows

    study1 = pd.DataFrame(study1_rows, columns=STUDY1_COLUMNS)
    if len(study1):
        counts = study1.groupby("unique_id", dropna=True)["seed"].transform("count")
        study1["n_seeds"] = counts.fillna(1).astype(int)
        first_seed = study1.groupby("unique_id", dropna=True)["seed"].transform("min")
        missing = study1["replicate"].isna()
        study1.loc[missing, "replicate"] = (study1.loc[missing, "seed"] != first_seed[missing])
        study1["replicate"] = study1["replicate"].astype(bool)
    tables = {
        "runs": pd.DataFrame(run_rows, columns=RUNS_COLUMNS),
        "study1": study1,
        "eval_items": pd.DataFrame(eval_rows, columns=EVAL_ITEM_COLUMNS),
        "train_metrics": pd.DataFrame(train_rows, columns=TRAIN_METRIC_COLUMNS),
        "selections": pd.DataFrame(selection_rows, columns=SELECTION_COLUMNS),
        "study2_evals": pd.DataFrame(study2_rows, columns=STUDY2_EVAL_COLUMNS),
    }
    for name, df in tables.items():
        log.info("table %s: %d rows", name, len(df))
    if base.used:
        log.info("base evaluations used: %s", base.used)
    if write:
        write_tables(tables, Path(out_dir) if out_dir else root / "tables")
    return tables


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path | str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in tables.items():
        p = out / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths[name] = p
    log.info("wrote %d tables -> %s", len(paths), out)
    return paths


def read_tables(tables_dir: Path | str, names: Iterable[str] = tuple(TABLE_COLUMNS)) -> dict[str, pd.DataFrame]:
    """Read the Parquet tables back; a missing table becomes an empty frame with the expected columns."""
    d = Path(tables_dir)
    out = {}
    for n in names:
        p = d / f"{n}.parquet"
        if p.exists():
            out[n] = pd.read_parquet(p)
        else:
            log.warning("table %s not found under %s", n, d)
            out[n] = pd.DataFrame(columns=TABLE_COLUMNS[n])
    return out
