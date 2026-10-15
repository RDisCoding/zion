"""Calibration rungs after the C1 pilot failed gate G-P (protocol section 13).

- C2 (confirmatory, pre-declared in protocol section 4): the C1 run with 80 bursts instead of 20 (400 updates), the
  same 16 pilot candidates x the same 2 seeds + pi1 x 2, the same gate G-P. The substrate must equal the e0 spec in
  everything except `loop.steps`.
- OS (EXPLORATORY, amendment A3): one-shot RLVR in the regime validated by the E0 positive control (prereg Study-1
  training: fresh LoRA r32 on all linear layers, 100 rounds x G=64, lr 2e-5, correctness reward; config hash
  84e100e2), trained by the unchanged prereg trainer `rlvr_v2.train_grpo.run_one_shot_grpo`. The label is the same
  dev-set ΔA, evaluated by the same SGAC evaluator against the same base reference as C1, so C1, C2 and OS labels
  are directly comparable. Same candidates, same gate thresholds; seeds 27000+i and 37000+i (pi1: 29000, 39000).

Every rung's ΔA uses the C1 base dev evaluation (repeat 1) as its reference: same base model, same evaluator.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from ..artifacts import REPO_ROOT, RunDir, atomic_write_json, read_json, read_jsonl, utc_now
from ..sgac.runinfo import shared_dir
from ..sgac.spec import SgacSpec, load_profile
from .calib import (
    BASE_DEV,
    DEV,
    CalibContext,
    gate_p,
    load_calib_context,
    run_calibration,
    run_label,
    run_name,
    training_seeds,
)
from .manifests import CANDIDATES
from .spec import CalibSpec, require_e0_substrate, substrate_spec

log = logging.getLogger(__name__)

C2_STEPS = 80
STAGE_C2 = "calib_c2"
STAGE_OS = "oneshot"
OS_SEED_OFFSETS = (27000, 37000)
OS_PI1_SEEDS = (29000, 39000)
OS_STUDY = "oneshot"
PREREG_STUDY1_HASH = "84e100e2"
PREREG_FROZEN_ARGS = ("prompt.style=oneshot_rlvr_chat", "train.rounds=100", "train.learning_rate=2e-05")


# ---------------------------------------------------------------------- shared base reference
def base_reference(cs: CalibSpec, spec: SgacSpec | None = None) -> dict[str, bool]:
    """The C1 base dev evaluation, repeat 1 (the reference of every rung's ΔA). `spec` is for smoke tests only."""
    path = shared_dir(spec or substrate_spec(cs, "calib"), BASE_DEV) / "rep1" / "per_item.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path}: the C1 base dev evaluation is the reference for every rung")
    return {r["unique_id"]: bool(r["correct"]) for r in read_jsonl(path)}


def pilot_indices(cs: CalibSpec) -> list[int]:
    return [int(r["index"]) for r in read_json(cs.manifest_dir / CANDIDATES)["pilot"]]


def _labels_doc(cs: CalibSpec, ctx: CalibContext, rows_for, pi1_rows, source: str, extra: dict) -> dict:
    out = {"created": utc_now(), "source": source, "calib_hash": cs.calib_hash(), **extra, "candidates": [], "pi1": None}
    for c in ctx.candidates:
        done = [r for r in rows_for(c) if r is not None]
        out["candidates"].append({"index": int(c["index"]), "unique_id": c["unique_id"], "bin": c["bin"],
                                  "level": c["level"], "runs": done, "n_seeds_done": len(done),
                                  "delta_pp_mean": float(np.mean([r["delta_pp"] for r in done])) if done else None})
    pr = [r for r in pi1_rows if r is not None]
    out["pi1"] = {"runs": pr, "delta_pp_mean": float(np.mean([r["delta_pp"] for r in pr])) if pr else None}
    return out


def write_labels_and_gate(cs: CalibSpec, stage: str, labels: dict) -> dict:
    res = gate_p(cs, labels, pilot_indices(cs))
    res["source"] = labels["source"]
    atomic_write_json(cs.results_dir / stage / "labels.json", labels)
    atomic_write_json(cs.results_dir / stage / "gate.json", res)
    return res


# ---------------------------------------------------------------------- C2 (pre-declared escalation rung)
def require_e0_except(spec: SgacSpec, allowed: dict) -> None:
    """The substrate must equal the e0 spec except for the `allowed` dotted keys (with exactly those values)."""
    ref = load_profile("e0").to_dict()
    got = spec.to_dict()
    ref.pop("run"), got.pop("run")
    diffs = {}
    for sec in ref:
        for k in ref[sec]:
            if ref[sec][k] != got[sec][k]:
                diffs[f"{sec}.{k}"] = got[sec][k]
    if diffs != allowed:
        raise RuntimeError(f"substrate differs from e0 by {diffs}; only {allowed} is allowed for this rung")


def c2_spec(cs: CalibSpec, smoke_overrides: list[str] | None = None, extra_files=None) -> SgacSpec:
    spec = substrate_spec(cs, STAGE_C2, [f"loop.steps={C2_STEPS}", *(smoke_overrides or [])], extra_files)
    if not smoke_overrides and not extra_files:
        require_e0_except(spec, {"loop.steps": C2_STEPS})
    return spec


def c2_context(cs: CalibSpec) -> CalibContext:
    return load_calib_context(cs, c2_spec(cs))


def run_c2(ctx: CalibContext, index: int | None, seed: int) -> dict:
    return run_calibration(ctx, index, seed)


def c2_labels(cs: CalibSpec, ctx: CalibContext, base_spec: SgacSpec | None = None) -> dict:
    base = base_reference(cs, base_spec)
    spec = ctx.spec

    def rows(c):
        return [run_label(spec, run_name(int(c["index"]), c["unique_id"], s), base)
                for s in training_seeds(cs, int(c["index"]))]

    pi1 = [run_label(spec, run_name(None, ctx.pi1.unique_id, s), base) for s in training_seeds(cs, None)]
    return _labels_doc(cs, ctx, rows, pi1, "C2", {"substrate_spec_hash": spec.spec_hash(), "bursts": C2_STEPS})


# ---------------------------------------------------------------------- OS (exploratory one-shot regime)
def os_config(smoke_overrides: list[str] | None = None, smoke_files=None, rounds: int | None = None):
    """The prereg Study-1 config (hash 84e100e2). `smoke_files` (CPU) or `rounds` (GPU smoke) only for smoke tests."""
    from ..config import load_config

    c = REPO_ROOT / "configs"
    if rounds is not None:  # GPU smoke: the real recipe, fewer rounds
        return load_config([c / "base.yaml", c / "study1.yaml"], [*PREREG_FROZEN_ARGS, f"train.rounds={int(rounds)}"])
    if smoke_files:  # tiny smoke: the frozen budget (100 rounds) would override the smoke's 2 rounds, so it is not applied
        return load_config([c / "base.yaml", c / "study1.yaml", *smoke_files],
                           ["prompt.style=oneshot_rlvr_chat", "train.rounds=2", *(smoke_overrides or [])])
    cfg = load_config([c / "base.yaml", c / "study1.yaml"], list(PREREG_FROZEN_ARGS))
    if cfg.config_hash() != PREREG_STUDY1_HASH:
        raise RuntimeError(f"prereg Study-1 config hash {cfg.config_hash()} != {PREREG_STUDY1_HASH}; refusing")
    return cfg


def os_seeds(index: int | None) -> list[int]:
    return list(OS_PI1_SEEDS) if index is None else [o + int(index) for o in OS_SEED_OFFSETS]


def os_run_name(index: int | None, uid: str, seed: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", uid.replace(".json", "")).strip("-")
    return f"os_pi1__seed{seed}" if index is None else f"os_c{index:02d}__{slug}__seed{seed}"


def os_rundir(cs: CalibSpec, cfg, name: str) -> RunDir:
    return RunDir(cfg, OS_STUDY, name, root=cs.results_dir / STAGE_OS)


def run_os(cs: CalibSpec, ctx: CalibContext, index: int | None, seed: int, cfg=None, spec: SgacSpec | None = None,
           check_gates: bool = True, check_substrate: bool = True) -> dict:
    """Train one candidate with the prereg one-shot recipe, then evaluate the dev set with the SGAC evaluator."""
    from ..data import Problem, load_math_train, load_pi1
    from ..grader import MathVerifyGrader
    from ..modeling import free_cuda
    from ..sgac.evaluation import SUMMARY_FILE, evaluate_items
    from ..sgac.model_io import load_policy_model, load_tokenizer, stop_token_ids
    from ..sgac.sampler import eval_sampler
    from ..train_grpo import run_one_shot_grpo

    cfg = cfg or os_config()
    if check_gates:
        from ..cli import _require_gates

        _require_gates(cfg)
    spec = spec or substrate_spec(cs, STAGE_OS)
    if check_substrate:  # production: the dev evaluation must use the e0 substrate (smokes use subsets / tiny models)
        require_e0_substrate(spec)
    if index is None:
        problem: Problem = load_pi1()
    else:
        uid = next(c["unique_id"] for c in ctx.candidates if int(c["index"]) == int(index))
        problem = next(p for p in load_math_train() if p.unique_id == uid)
    name = os_run_name(index, problem.unique_id, seed)
    rd = os_rundir(cs, cfg, name)
    if not rd.exists():
        rd.init(seed, extra={"candidate_index": index, "unique_id": problem.unique_id, "calib_hash": cs.calib_hash(),
                             "rung": "OS", "exploratory": True})
    out = rd.path / "eval" / DEV
    done = read_json(rd.path / "result.json")
    if rd.status().get("state") == "done" and done is not None:
        return done
    try:
        adapter_dir, burst = run_one_shot_grpo(problem, cfg, seed, rd, grader=MathVerifyGrader())
        free_cuda()
        summary = read_json(out / SUMMARY_FILE)
        if summary is None:
            from peft import PeftModel

            from ..sgac.grading import DualGrader

            rd.set_status(stage="eval_dev")
            tok = load_tokenizer(spec)
            base, _ = load_policy_model(spec)
            model = PeftModel.from_pretrained(base, str(adapter_dir))
            model.eval()
            sampler = eval_sampler(model, tok, spec, stop_token_ids(spec, tok), int(spec.profile.eval_batch_size))
            summary = evaluate_items(sampler, tok, ctx.dev, spec, DualGrader(spec.profile.grader), out,
                                     tag=f"{name}-dev", policy=f"oneshot/{name}", adapter_ref=str(adapter_dir))
            del model, base, sampler
            free_cuda()
        result = {"run": name, "unique_id": problem.unique_id, "index": index, "seed": seed,
                  "burst": burst.summary() if hasattr(burst, "summary") else None,
                  "acc_dev": summary["acc"], "n_dev": summary["n"]}
        atomic_write_json(rd.path / "result.json", result)
        rd.mark_done()
        return result
    except Exception as e:
        rd.mark_failed(repr(e))
        raise


def os_label(cs: CalibSpec, cfg, name: str, base: dict[str, bool]) -> dict | None:
    rd = os_rundir(cs, cfg, name)
    path = rd.path / "eval" / DEV / "per_item.jsonl"
    if not (path.exists() and (rd.path / "result.json").exists()):
        return None
    rows = {r["unique_id"]: bool(r["correct"]) for r in read_jsonl(path)}
    if set(rows) != set(base):
        raise RuntimeError(f"{path} does not cover the dev set")
    ids = sorted(base)
    y = np.array([rows[i] for i in ids], dtype=float)
    b = np.array([base[i] for i in ids], dtype=float)
    res = read_json(rd.path / "result.json", {}) or {}
    burst = res.get("burst") or {}
    return {"run": name, "acc": float(y.mean()), "acc_base": float(b.mean()), "delta_pp": 100 * float((y - b).mean()),
            "n": len(ids), "discordant": int((y != b).sum()),
            "train_reward_first_last": [burst.get("reward_correct_mean_first"), burst.get("reward_correct_mean_last")],
            "frac_zero_std": burst.get("frac_zero_std_mean")}


def os_labels(cs: CalibSpec, ctx: CalibContext, cfg=None, base_spec: SgacSpec | None = None) -> dict:
    from ..data import load_pi1

    cfg = cfg or os_config()
    base = base_reference(cs, base_spec)

    def rows(c):
        return [os_label(cs, cfg, os_run_name(int(c["index"]), c["unique_id"], s), base) for s in os_seeds(int(c["index"]))]

    pi1_uid = load_pi1().unique_id
    pi1 = [os_label(cs, cfg, os_run_name(None, pi1_uid, s), base) for s in os_seeds(None)]
    return _labels_doc(cs, ctx, rows, pi1, "OS", {"prereg_config_hash": cfg.config_hash(), "exploratory": True})


def stage_dir(cs: CalibSpec, stage: str) -> Path:
    return cs.results_dir / stage
