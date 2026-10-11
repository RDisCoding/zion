"""Calibration labels (protocol section 4): protocol C1 runs, the shared base dev evaluation, the ΔA labels and gate G-P.

C1 = a fresh LoRA plus `loop.steps` (20) SGAC bursts on ONE candidate, persisting the adapter: the frozen loop's
`fixed_pi1` path with the candidate in place of pi1, so every burst, seed and save is the historical code. The only
changes are what is evaluated -- the dev set at the final step, nothing else -- and which adapters are kept (the
final one and the resume point). MATH-500 and test50 are never passed in, so a calibration run cannot touch them.

Label: ΔA = acc_dev(trained) - acc_dev(base), per run; a candidate's label is the mean over its training seeds.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

from ..artifacts import atomic_write_json, read_json, read_jsonl, utc_now
from ..data import load_math_train
from ..sgac.data import SgacItem, pi1_item
from ..sgac.loop import SgacRun
from ..sgac.runinfo import SgacRunDir, phase, run_manifest, shared_dir
from ..sgac.spec import SgacSpec
from ..stats import icc_oneway
from .manifests import CANDIDATES, DATA_ROLES
from .spec import CalibSpec, substrate_spec

log = logging.getLogger(__name__)

STAGE = "calib"
DEV = "dev"
BASE_DEV = "base_dev"
GATE_P = "gate_p.json"


# ---------------------------------------------------------------------- items
def _slug(uid: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", uid.replace(".json", "")).strip("-")


def run_name(index: int | None, uid: str, seed: int) -> str:
    return f"pi1__seed{seed}" if index is None else f"c{index:02d}__{_slug(uid)}__seed{seed}"


def _items(ids: list[str], source: str) -> list[SgacItem]:
    by_id = {p.unique_id: p for p in load_math_train() if p.unique_id in set(ids)}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} ids not found in the MATH train split: {missing[:3]}")
    return [SgacItem(row=-1, orig_index=-1, unique_id=i, problem=by_id[i].problem, solution=by_id[i].solution,
                     answer=by_id[i].answer, level=by_id[i].level, subject=by_id[i].subject, source=source) for i in ids]


@dataclass
class CalibContext:
    cs: CalibSpec
    spec: SgacSpec
    dev: list = field(default_factory=list)
    candidates: list = field(default_factory=list)  # [{"index", "unique_id", "bin", "level", "item"}]
    pi1: SgacItem | None = None
    grader: object | None = None


def load_calib_context(cs: CalibSpec, spec: SgacSpec | None = None, need_candidates: bool = True) -> CalibContext:
    from ..sgac.grading import DualGrader

    spec = spec or substrate_spec(cs, STAGE)
    roles = read_json(cs.manifest_dir / DATA_ROLES)
    if roles is None:
        raise FileNotFoundError(cs.manifest_dir / DATA_ROLES)
    dev_ids = roles["dev"]["ids"]
    final = set(roles["final_eval"]["test50"]["ids"]) | set(read_json(cs.path("manifests/math500.json"))["ids"])
    if set(dev_ids) & final:
        raise RuntimeError("dev set overlaps a final-evaluation set; refusing (protocol section 2)")
    ctx = CalibContext(cs=cs, spec=spec, dev=_items(dev_ids, "prereg_heldout"), grader=DualGrader(spec.profile.grader))
    if need_candidates:
        cands = read_json(cs.manifest_dir / CANDIDATES)
        recs = cands["pilot"] + cands["expansion"]
        items = {it.unique_id: it for it in _items([r["unique_id"] for r in recs], "prereg_pool")}
        ctx.candidates = [{**r, "item": items[r["unique_id"]]} for r in recs]
    ctx.pi1 = pi1_item()
    return ctx


# ---------------------------------------------------------------------- the C1 run
class CalibRun(SgacRun):
    """Protocol C1 for one (candidate, seed): the frozen loop's fixed-item path, evaluated on the dev set only."""

    def __init__(self, spec: SgacSpec, item: SgacItem, index: int | None, seed: int, dev, grader, calib_hash: str):
        super().__init__(spec, "fixed_pi1", seed, pool=[], test50=[], math500=[], schedule=[], grader=grader, pi1=item)
        self.item, self.index, self.dev, self.calib_hash = item, index, list(dev), calib_hash
        self.arm = "calib_pi1" if index is None else "calib"
        self.rd = SgacRunDir(spec, run_name(index, item.unique_id, self.seed))
        self.state_path = self.rd.path / "state.json"
        self.state = self._new_state()

    def _select(self, t: int, sd: Path, timings: dict):
        selection = {"step": t, "arm": self.arm, "rule": None, "selected_idx": 0, "selected_uid": self.item.unique_id,
                     "selected_row": self.item.row, "candidates": [], "picks": {}, "scores": {}}
        atomic_write_json(sd / "selection.json", selection)
        return self.item, selection

    def keep_adapter(self, t: int) -> bool:
        return t == self.spec.loop.steps or t == self.state["step_done"]

    def _evaluate_due(self, t: int) -> None:
        if t == self.spec.loop.steps and not self._has_eval(t, DEV):
            self._evaluate(t, DEV, self.dev, int(self.spec.profile.eval_batch_size))

    def _all_evals_present(self) -> bool:
        return self._has_eval(int(self.spec.loop.steps), DEV)

    def run(self) -> dict:
        if not self.rd.exists():
            self.rd.init(self.seed, extra={"arm": self.arm, "candidate_index": self.index,
                                           "candidate_uid": self.item.unique_id, "calib_hash": self.calib_hash,
                                           "protocol": "C1"})
        return super().run()


def run_calibration(ctx: CalibContext, index: int | None, seed: int) -> dict:
    """One C1 run. `index=None` is the pi1 positive control."""
    if index is None:
        item = ctx.pi1
    else:
        item = next(c["item"] for c in ctx.candidates if int(c["index"]) == int(index))
    run = CalibRun(ctx.spec, item, index, seed, ctx.dev, ctx.grader, ctx.cs.calib_hash())
    try:
        return run.run()
    except Exception as e:
        if run.rd.exists():
            run.rd.mark_failed(repr(e))
        raise


def training_seeds(cs: CalibSpec, index: int | None) -> list[int]:
    offs = cs.calibration.pi1_seed_offsets if index is None else cs.calibration.seed_offsets
    return [int(o) + (0 if index is None else int(index)) for o in offs]


# ---------------------------------------------------------------------- base dev evaluation
def run_base_dev_eval(ctx: CalibContext) -> dict:
    """The base model on the dev set, `dev_eval_repeats` times; G5-style determinism check (>= 98% item agreement and
    accuracies within 0.5 pp). Repeat 1 is the reference for every ΔA."""
    from ..modeling import free_cuda
    from ..sgac.evaluation import evaluate_items, item_agreement
    from ..sgac.model_io import load_policy_model, load_tokenizer, stop_token_ids
    from ..sgac.sampler import eval_sampler

    spec = ctx.spec
    out = shared_dir(spec, BASE_DEV)
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(spec)
    model, load_record = load_policy_model(spec)
    result: dict = {"manifest": run_manifest(spec, None, {"job": "base_dev_eval", "load_record": load_record,
                                                          "calib_hash": ctx.cs.calib_hash()}), "repeats": {}}
    timings: dict = {}
    reps = int(ctx.cs.calibration.dev_eval_repeats)
    for r in range(1, reps + 1):
        with phase(timings, f"rep{r}"):
            s = evaluate_items(eval_sampler(model, tok, spec, stop_token_ids(spec, tok), int(spec.profile.eval_batch_size)),
                               tok, ctx.dev, spec, ctx.grader, out / f"rep{r}", tag=f"base-dev-rep{r}", policy="base")
        result["repeats"][r] = {"acc": s["acc"], "n": s["n"], "trunc_rate": s["trunc_rate"]}
    if reps >= 2:
        agree = item_agreement(out / "rep1" / "per_item.jsonl", out / "rep2" / "per_item.jsonl")
        gap = abs(result["repeats"][1]["acc"] - result["repeats"][2]["acc"])
        result["determinism"] = {**agree, "acc_gap": gap,
                                 "passed": agree["verdict_agreement"] >= 0.98 and gap <= 0.005,
                                 "rule": "item verdict agreement >= 0.98 and accuracy gap <= 0.5 pp (E0 G5)"}
    result["timings"] = timings
    atomic_write_json(out / "base_dev_eval.json", result)
    del model
    free_cuda()
    return result


def base_dev_reference(spec: SgacSpec) -> dict[str, bool]:
    path = shared_dir(spec, BASE_DEV) / "rep1" / "per_item.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path}; run `python -m rlvr_v2.sgac_calib base-dev-eval` first")
    return {r["unique_id"]: bool(r["correct"]) for r in read_jsonl(path)}


# ---------------------------------------------------------------------- labels
def run_label(spec: SgacSpec, name: str, base: dict[str, bool]) -> dict | None:
    rd = SgacRunDir(spec, name)
    path = rd.path / "evals" / f"step-{int(spec.loop.steps):03d}" / DEV / "per_item.jsonl"
    if not (rd.is_done() and path.exists()):
        return None
    rows = {r["unique_id"]: bool(r["correct"]) for r in read_jsonl(path)}
    if set(rows) != set(base):
        raise RuntimeError(f"{path} does not cover the dev set")
    ids = sorted(base)
    y = np.array([rows[i] for i in ids], dtype=float)
    b = np.array([base[i] for i in ids], dtype=float)
    state = read_json(rd.path / "state.json", {}) or {}
    hist = state.get("history", [])
    return {"run": name, "acc": float(y.mean()), "acc_base": float(b.mean()), "delta_pp": 100 * float((y - b).mean()),
            "n": len(ids), "discordant": int((y != b).sum()),
            "zero_loss_bursts": sum(1 for h in hist if (h.get("burst") or {}).get("all_zero_loss")),
            "train_reward_last": ((hist[-1].get("burst") or {}).get("reward") if hist else None)}


def collect_labels(ctx: CalibContext, indices: list[int] | None = None) -> dict:
    """{candidates: [{index, uid, bin, level, runs: [...], delta_pp_mean}], pi1: {...}} for completed runs."""
    spec, cs = ctx.spec, ctx.cs
    base = base_dev_reference(spec)
    out = {"created": utc_now(), "calib_hash": cs.calib_hash(), "substrate_spec_hash": spec.spec_hash(),
           "candidates": [], "pi1": None}
    for c in ctx.candidates:
        if indices is not None and int(c["index"]) not in indices:
            continue
        runs = [run_label(spec, run_name(int(c["index"]), c["unique_id"], s), base)
                for s in training_seeds(cs, int(c["index"]))]
        done = [r for r in runs if r is not None]
        out["candidates"].append({"index": int(c["index"]), "unique_id": c["unique_id"], "bin": c["bin"],
                                  "level": c["level"], "runs": done, "n_seeds_done": len(done),
                                  "delta_pp_mean": float(np.mean([r["delta_pp"] for r in done])) if done else None})
    pi1_runs = [run_label(spec, run_name(None, ctx.pi1.unique_id, s), base) for s in training_seeds(cs, None)]
    pi1_done = [r for r in pi1_runs if r is not None]
    out["pi1"] = {"runs": pi1_done, "delta_pp_mean": float(np.mean([r["delta_pp"] for r in pi1_done])) if pi1_done else None}
    return out


# ---------------------------------------------------------------------- gate G-P
def icc_ci(groups: list[list[float]], alpha: float = 0.05) -> tuple[float, float]:
    """Shrout & Fleiss F-based CI for ICC(1,1) with k replicates per group (balanced)."""
    k = len(groups[0])
    n = len(groups)
    arr = np.array(groups, dtype=float)
    grand = arr.mean()
    msb = k * float(((arr.mean(axis=1) - grand) ** 2).sum()) / (n - 1)
    msw = float(((arr - arr.mean(axis=1, keepdims=True)) ** 2).sum()) / (n * (k - 1))
    if msw <= 0:
        return float("nan"), float("nan")
    f = msb / msw
    fl = f / stats.f.ppf(1 - alpha / 2, n - 1, n * (k - 1))
    fu = f * stats.f.ppf(1 - alpha / 2, n * (k - 1), n - 1)
    return (fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1)


def gate_p(cs: CalibSpec, labels: dict, pilot_indices: list[int]) -> dict:
    """G-P on the pilot (label reliability only; never looks at features). PASS iff ICC(1,1) of single-seed ΔA
    >= icc_min AND the SD across candidates of the seed-mean ΔA >= sd_min_pp."""
    g = cs.calibration.gate_p
    n_seeds = len(cs.calibration.seed_offsets)
    rows = [c for c in labels["candidates"] if c["index"] in pilot_indices]
    incomplete = [c["unique_id"] for c in rows if c["n_seeds_done"] < n_seeds]
    res: dict = {"created": utc_now(), "rule": g, "n_candidates": len(rows), "incomplete": incomplete,
                 "pi1": labels.get("pi1")}
    if incomplete or len(rows) < len(pilot_indices):
        res.update(status="incomplete", passed=None)
        return res
    groups = [[r["delta_pp"] for r in c["runs"]] for c in rows]
    icc = icc_oneway(groups)
    lo, hi = icc_ci(groups)
    means = [float(np.mean(gr)) for gr in groups]
    sd = float(np.std(means, ddof=1))
    within = [abs(gr[0] - gr[1]) for gr in groups] if n_seeds == 2 else []
    res.update(icc=icc, icc_ci95=[lo, hi], sd_between_pp=sd, mean_delta_pp=float(np.mean(means)),
               deltas_pp={c["unique_id"]: [r["delta_pp"] for r in c["runs"]] for c in rows},
               mean_abs_seed_gap_pp=float(np.mean(within)) if within else None,
               icc_pass=bool(math.isfinite(icc) and icc >= float(g["icc_min"])),
               sd_pass=bool(sd >= float(g["sd_min_pp"])))
    res["passed"] = res["icc_pass"] and res["sd_pass"]
    res["status"] = "pass" if res["passed"] else "fail"
    res["consequence"] = ("run the expansion" if res["passed"] else
                          "stop: write the diagnosis; the team chooses (i) a negative calibration result or (ii) rung C2")
    return res


def write_gate_p(ctx: CalibContext) -> dict:
    pilot = [int(r["index"]) for r in read_json(ctx.cs.manifest_dir / CANDIDATES)["pilot"]]
    labels = collect_labels(ctx)
    res = gate_p(ctx.cs, labels, pilot)
    atomic_write_json(ctx.cs.results_dir / STAGE / GATE_P, res)
    atomic_write_json(ctx.cs.results_dir / STAGE / "labels.json", labels)
    return res
