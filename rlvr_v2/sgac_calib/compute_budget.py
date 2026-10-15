"""Equal-GPU-compute comparison (protocol section 13, amendment A6): is SGAC's selection worth its compute, or is the
same compute better spent on more Random-selected training?

- Strategy A: the frozen 20-step SGAC curriculum (`sgac` arm, Eq. 10 as run): every step sieves 4 candidates x K=4
  rollouts, picks one, trains one burst. e0 substrate (hash 577f3319), clean pool, seeds 42-49.
- Strategy B: Random without screening, trained for as many steps as fit in A's MEASURED training-side GPU time for
  the same seed. Its picks are the random arm's RNG stream on the same batches (identical prefix), so the pairing is
  by seed and its first 20 picks equal the ordinary Random arm's.

Budget = sum over A's 20 steps of the measured wall time of screening (sieve), training (burst) and adapter save.
Model loading and evaluation are identical for both strategies and reported separately. B runs step t+1 only while
`used + mean(step cost so far) <= budget`, so it never knowingly exceeds A's compute; the realised ratio is recorded.

Evaluation: MATH-500 at step 0 (shared base), A at step 20, B at step 20 (equal steps) and at its final, budget-matched
step; test50 for trajectories. Primary endpoint: MATH-500, B(final) - A(20), crossed seed x item test.
"""
from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..artifacts import REPO_ROOT, atomic_write_json, read_json, read_jsonl, utc_now
from ..config import dataclass_from_dict, deep_merge, parse_overrides
from ..sgac.loop import SgacRun, test50_batch_size
from ..sgac.runinfo import SgacRunDir
from ..sgac.schedule import candidate_schedule, load_schedule, random_arm_rng, schedule_path, write_schedule
from ..sgac.spec import SgacSpec, group_dir
from .crossed import crossed_test, item_bootstrap, pigeonhole_bootstrap
from .manifests import DATA_ROLES
from .rungs import require_e0_except
from .spec import CalibSpec, require_e0_substrate, substrate_spec

log = logging.getLogger(__name__)

CONFIG = REPO_ROOT / "configs" / "sgac_calib" / "compute_budget.yaml"
REPORT_DIR = REPO_ROOT / "reports" / "sgac_calib"
A_COST_KEYS = ("sieve", "burst", "save")
B_COST_KEYS = ("burst", "save")


@dataclass(frozen=True)
class DecisionCfg:
    alpha: float = 0.05
    min_effect_pp: float = 1.0
    seed_consistency: float = 0.875
    equivalence_pp: float = 2.0


@dataclass(frozen=True)
class CBConfig:
    stage: str = "compute_budget"
    schedule_dir: str = "configs/sgac_calib/compute_budget"
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46, 47, 48, 49)
    sgac_steps: int = 20
    random_max_steps: int = 160
    random_eval_every: int = 10
    equal_steps_checkpoint: int = 20
    decision: DecisionCfg = field(default_factory=DecisionCfg)
    n_boot: int = 10000
    boot_seed: int = 0


def load_cb(path: str | Path | None = None, overrides: list[str] | None = None) -> CBConfig:
    with open(path or CONFIG, "r", encoding="utf-8") as fh:
        d = deep_merge(yaml.safe_load(fh) or {}, parse_overrides(overrides))
    return dataclass_from_dict(CBConfig, d)


# ---------------------------------------------------------------------- specs and schedules
def spec_a(cs: CalibSpec, cb: CBConfig, smoke_ov: list[str] | None = None, smoke_files=None) -> SgacSpec:
    spec = substrate_spec(cs, cb.stage, smoke_ov, smoke_files)
    if smoke_ov is None and smoke_files is None:
        require_e0_substrate(spec)
        if spec.loop.steps != cb.sgac_steps:
            raise RuntimeError("Strategy A must run the frozen 20-step curriculum")
    return spec


def spec_b(cs: CalibSpec, cb: CBConfig, smoke_ov: list[str] | None = None, smoke_files=None,
           cap: int | None = None) -> SgacSpec:
    ov = [f"loop.steps={cap or cb.random_max_steps}", f"loop.eval_every={cb.random_eval_every}",
          f"loop.math500_steps=[0,{cb.equal_steps_checkpoint}]", *(smoke_ov or [])]
    spec = substrate_spec(cs, cb.stage, ov, smoke_files)
    if smoke_ov is None and smoke_files is None:
        require_e0_except(spec, {"loop.steps": cb.random_max_steps, "loop.eval_every": cb.random_eval_every})
    return spec


def clean_rows(cs: CalibSpec) -> tuple[list[int], dict[int, str]]:
    roles = read_json(cs.manifest_dir / DATA_ROLES)
    uid = dict(zip(roles["curriculum_pool"]["rows"], roles["curriculum_pool"]["ids"]))
    return sorted(uid), uid


def write_schedules(cs: CalibSpec, cb: CBConfig) -> list[Path]:
    rows, uid = clean_rows(cs)
    return [write_schedule(cs.path(cb.schedule_dir), s, rows, cb.random_max_steps, 4, uid) for s in cb.seeds]


def check_schedules(cs: CalibSpec, cb: CBConfig) -> dict:
    """Every CB schedule reproduces, and its first steps equal the frozen calibration schedules (same RNG stream)."""
    rows, _ = clean_rows(cs)
    out = {}
    for s in cb.seeds:
        b = load_schedule(cs.path(cb.schedule_dir), s, rows, cb.random_max_steps, 4)
        frozen = read_json(schedule_path(cs.manifest_dir, s))
        if frozen is not None:
            n = len(frozen["batches"])
            if [x["rows"] for x in b[:n]] != [x["rows"] for x in frozen["batches"]]:
                raise RuntimeError(f"seed {s}: CB schedule prefix differs from the frozen calibration schedule")
        out[s] = len(b)
    return out


# ---------------------------------------------------------------------- the budget rule
def step_cost(history_entry: dict, keys) -> float:
    t = history_entry.get("timing") or {}
    return float(sum((t.get(k) or {}).get("wall_s", 0.0) for k in keys))


def training_seconds(history: list[dict], keys) -> float:
    return float(sum(step_cost(h, keys) for h in history))


def may_run_next(used_s: float, step_costs: list[float], budget_s: float) -> bool:
    """Run the next step only if, at the mean cost so far, it still fits inside the budget."""
    if budget_s <= 0:
        return False
    if not step_costs:
        return True
    return used_s + float(np.mean(step_costs)) <= budget_s


def sgac_budget(spec: SgacSpec, seed: int, steps: int) -> float:
    st = read_json(SgacRunDir(spec, f"sgac__seed{seed}").path / "state.json")
    if st is None or int(st.get("step_done", 0)) < steps:
        raise RuntimeError(f"Strategy A (sgac, seed {seed}) must finish its {steps} steps before Strategy B can be budgeted")
    return training_seconds(st["history"], A_COST_KEYS)


class RandomBudgetRun(SgacRun):
    """Strategy B: no screening; random pick per batch (the random arm's RNG stream); stops on the GPU-time budget."""

    def __init__(self, spec: SgacSpec, seed: int, budget_s: float, **kw):
        super().__init__(spec, "random", seed, **kw)
        self.arm = "random_budget"
        self.budget_s = float(budget_s)
        self.rd = SgacRunDir(spec, f"random_budget__seed{self.seed}")
        self.state_path = self.rd.path / "state.json"
        self.state = self._new_state()

    def _select(self, t: int, sd: Path, timings: dict):
        from ..artifacts import atomic_write_json as _w

        items = self._candidates(t)
        idx = int(random_arm_rng(self.seed, t).integers(len(items)))
        chosen = items[idx]
        selection = {"step": t, "arm": self.arm, "rule": "random", "selected_idx": idx, "selected_uid": chosen.unique_id,
                     "selected_row": chosen.row, "selected_level": chosen.level, "picks": {"random": idx}, "scores": {},
                     "candidates": [{"idx": i, "row": it.row, "unique_id": it.unique_id, "level": it.level}
                                    for i, it in enumerate(items)], "sieve": "skipped (no screening)"}
        _w(sd / "selection.json", selection)
        return chosen, selection

    def run(self) -> dict:
        if not self.rd.exists():
            self.rd.init(self.seed, extra={"arm": self.arm, "budget_s": self.budget_s,
                                           "rule": "run step t+1 while used + mean step cost <= budget"})
        self.state = self._load_state()
        self._reconcile()
        stop = self.state.get("budget_stop")
        if stop and self._final_done(stop["final_step"]) and self.rd.is_done():
            return self.state
        self._ensure_model()
        self.rd.set_status(state="running", stage="curriculum", step_done=self.state["step_done"])
        done = int(self.state["step_done"])
        if done > 0:
            self._evaluate_due(done)
        if not stop:
            for t in range(done + 1, int(self.spec.loop.steps) + 1):
                costs = [step_cost(h, B_COST_KEYS) for h in self.state["history"]]
                if not may_run_next(sum(costs), costs, self.budget_s):
                    break
                self._run_step(t)
                self._evaluate_due(t)
            used = training_seconds(self.state["history"], B_COST_KEYS)
            final = int(self.state["step_done"])
            stop = {"final_step": final, "budget_s": self.budget_s, "used_s": used,
                    "ratio": used / self.budget_s if self.budget_s else None,
                    "cap_hit": final == int(self.spec.loop.steps), "utc": utc_now()}
            self.state["budget_stop"] = stop
            self._save_state()
        final = int(stop["final_step"])
        if self.math500 and not self._has_eval(final, "math500"):
            self._evaluate(final, "math500", self.math500, int(self.spec.profile.eval_batch_size))
        if self.test50 and not self._has_eval(final, "test50"):
            self._evaluate(final, "test50", self.test50, test50_batch_size(self.spec))
        self.rd.mark_done(step_done=final)
        return self.state

    def _final_done(self, final: int) -> bool:
        need = [n for n, items in (("math500", self.math500), ("test50", self.test50)) if items]
        return all(self._has_eval(final, n) for n in need)


# ---------------------------------------------------------------------- jobs
def _ctx(cs: CalibSpec, spec: SgacSpec):
    from .curriculum import load_curriculum_context

    return load_curriculum_context(cs, spec)


def run_strategy(cs: CalibSpec, cb: CBConfig, strategy: str, seed: int, spec: SgacSpec | None = None,
                 ctx=None, a_spec: SgacSpec | None = None) -> dict:
    """One run. Strategy B's budget is the measured training-side time of Strategy A's run for the same seed
    (`a_spec` is A's spec; production uses `spec_a`)."""
    if strategy not in ("sgac", "random"):
        raise ValueError("strategy must be sgac or random")
    spec = spec or (spec_a(cs, cb) if strategy == "sgac" else spec_b(cs, cb))
    ctx = ctx or _ctx(cs, spec)
    rows = sorted(it.row for it in ctx.pool)
    schedule = load_schedule(cs.path(cb.schedule_dir), seed, rows, spec.loop.steps, spec.loop.batch_b)
    kw = {"pool": ctx.pool, "test50": ctx.test50, "math500": ctx.math500, "schedule": schedule, "grader": ctx.grader}
    if strategy == "sgac":
        run = SgacRun(spec, "sgac", seed, **kw)
    else:
        a_spec = a_spec or spec_a(cs, cb)
        run = RandomBudgetRun(spec, seed, sgac_budget(a_spec, seed, int(a_spec.loop.steps)), **kw)
    try:
        return run.run()
    except Exception as e:
        if run.rd.exists():
            run.rd.mark_failed(repr(e))
        raise


# ---------------------------------------------------------------------- analysis
def _per_item(path: Path, ids: list[str]) -> np.ndarray | None:
    if not path.exists():
        return None
    rows = {r["unique_id"]: bool(r["correct"]) for r in read_jsonl(path)}
    if not set(ids) <= set(rows):
        return None
    return np.array([rows[i] for i in ids], dtype=float)


def _eval_seconds(state: dict) -> float:
    return float(sum((x or {}).get("timing", {}).get("wall_s", 0.0) for v in (state.get("evals") or {}).values()
                     for x in v.values()))


def decide(test: dict, per_seed_pp: list[float], d: DecisionCfg) -> str:
    S = len(per_seed_pp)
    need = math.ceil(d.seed_consistency * S - 1e-9)
    m = 100 * test["mean"]
    if test["p"] < d.alpha and m >= d.min_effect_pp and sum(x > 0 for x in per_seed_pp) >= need:
        return "B better"
    if test["p"] < d.alpha and m <= -d.min_effect_pp and sum(x < 0 for x in per_seed_pp) >= need:
        return "A better"
    lo, hi = 100 * test["ci90"][0], 100 * test["ci90"][1]
    if -d.equivalence_pp <= lo and hi <= d.equivalence_pp:
        return "equivalent"
    return "inconclusive"


INTERPRETATION = {
    "B better": "Switch to Random: spending SGAC's screening compute on more training gives higher accuracy.",
    "equivalent": "SGAC's screening overhead is not justified: equal accuracy at equal compute with a simpler method "
                  "(prefer Random).",
    "A better": "Retain SGAC: its selection is worth more than the extra training the same compute buys.",
    "inconclusive": "Inconclusive at this power: neither retaining SGAC nor switching is supported by these data.",
}


def _test(e: np.ndarray, cb: CBConfig) -> dict:
    t = crossed_test(e, cb.decision.alpha)
    t["sensitivity"] = {"pigeonhole": pigeonhole_bootstrap(e, cb.n_boot, cb.boot_seed),
                        "item_bootstrap": item_bootstrap(e, cb.n_boot, cb.boot_seed)}
    return t


def _fmt_test(t: dict) -> dict:
    return {"mean_pp": 100 * t["mean"], "ci95_pp": [100 * v for v in t["ci"]], "ci90_pp": [100 * v for v in t["ci90"]],
            "p": t["p"], "df": t["df"], "per_seed_pp": [100 * v for v in t["per_seed"]],
            "sensitivity_ci95_pp": {k: [100 * v for v in s["ci"]] for k, s in t["sensitivity"].items()}}


def analyze(cs: CalibSpec, cb: CBConfig, eval_name: str = "math500", sa: SgacSpec | None = None,
            sb: SgacSpec | None = None, out_dir: Path | None = None, item_ids: list[str] | None = None) -> dict:
    sa, sb = sa or spec_a(cs, cb), sb or spec_b(cs, cb)
    ga = group_dir(sa)
    base_pi = ga / "_shared" / "base_eval" / eval_name / "per_item.jsonl"
    ids = item_ids or (read_json(cs.path("manifests/math500.json"))["ids"] if eval_name == "math500"
                       else sorted(r["unique_id"] for r in read_jsonl(base_pi)))
    base = _per_item(base_pi, ids)
    a_steps = int(sa.loop.steps)
    eq_step = min(int(cb.equal_steps_checkpoint), a_steps)
    rows, A, B20, BF = [], {}, {}, {}
    for s in cb.seeds:
        sta = read_json(SgacRunDir(sa, f"sgac__seed{s}").path / "state.json")
        stb = read_json(SgacRunDir(sb, f"random_budget__seed{s}").path / "state.json")
        if not sta or not stb or not stb.get("budget_stop"):
            rows.append({"seed": s, "status": "incomplete"})
            continue
        fin = int(stb["budget_stop"]["final_step"])

        def pi(sp, arm, step, s=s):
            return _per_item(SgacRunDir(sp, f"{arm}__seed{s}").path / "evals" / f"step-{step:03d}" / eval_name /
                             "per_item.jsonl", ids)

        ya, y20, yf = pi(sa, "sgac", a_steps), pi(sb, "random_budget", eq_step), pi(sb, "random_budget", fin)
        if ya is None or yf is None:
            rows.append({"seed": s, "status": "missing evaluation"})
            continue
        A[s], BF[s] = ya, yf
        if y20 is not None:
            B20[s] = y20
        a_train = training_seconds(sta["history"], A_COST_KEYS)
        b_train = training_seconds(stb["history"], B_COST_KEYS)
        a_sieve = training_seconds(sta["history"], ("sieve",))
        rows.append({"seed": s, "status": "ok", "A_acc": float(ya.mean()), "B_equal_steps_acc":
                     None if y20 is None else float(y20.mean()), "B_budget_acc": float(yf.mean()),
                     "A_steps": a_steps, "B_steps": fin, "A_train_gpu_h": a_train / 3600,
                     "A_screening_gpu_h": a_sieve / 3600, "B_train_gpu_h": b_train / 3600,
                     "budget_ratio": b_train / a_train if a_train else None, "B_cap_hit": stb["budget_stop"]["cap_hit"],
                     "A_eval_gpu_h": _eval_seconds(sta) / 3600, "B_eval_gpu_h": _eval_seconds(stb) / 3600,
                     "B_equal_steps_train_gpu_h": training_seconds(stb["history"][:eq_step], B_COST_KEYS) / 3600})
    ok = [s for s in cb.seeds if s in A and s in BF]
    res: dict = {"created": utc_now(), "eval": eval_name, "n_items": len(ids), "seeds_complete": ok,
                 "seeds_planned": list(cb.seeds), "rows": rows, "decision_rule": cb.decision.__dict__,
                 "base_acc": None if base is None else float(base.mean())}
    if len(ok) >= 2:
        prim = _test(np.stack([BF[s] - A[s] for s in ok]), cb)
        res["primary"] = {"comparison": "B (Random, budget-matched) - A (SGAC, 20 steps)", **_fmt_test(prim)}
        res["primary"]["verdict"] = decide(prim, res["primary"]["per_seed_pp"], cb.decision)
        res["interpretation"] = INTERPRETATION[res["primary"]["verdict"]]
        sec = {}
        both = [s for s in ok if s in B20]
        if len(both) >= 2:
            sec["B_equal_steps_minus_A"] = _fmt_test(_test(np.stack([B20[s] - A[s] for s in both]), cb))
            sec["B_budget_minus_B_equal_steps"] = _fmt_test(_test(np.stack([BF[s] - B20[s] for s in both]), cb))
        if base is not None:
            sec["A_minus_base"] = _fmt_test(_test(np.stack([A[s] - base for s in ok]), cb))
            sec["B_budget_minus_base"] = _fmt_test(_test(np.stack([BF[s] - base for s in ok]), cb))
        res["secondary"] = sec
        okr = [r for r in rows if r.get("status") == "ok"]
        res["compute"] = {k: float(sum(r[k] for r in okr)) for k in ("A_train_gpu_h", "A_screening_gpu_h",
                                                                         "B_train_gpu_h", "A_eval_gpu_h", "B_eval_gpu_h")}
        res["compute"]["mean_budget_ratio"] = float(np.mean([r["budget_ratio"] for r in okr]))
        res["compute"]["mean_B_steps"] = float(np.mean([r["B_steps"] for r in okr]))
        res["compute"]["screening_share_of_A"] = res["compute"]["A_screening_gpu_h"] / res["compute"]["A_train_gpu_h"]
    out_dir = out_dir or REPORT_DIR
    write_outputs(res, out_dir, sa, sb, cb, eval_name)
    return res


# ---------------------------------------------------------------------- outputs
def write_outputs(res: dict, out_dir: Path, sa: SgacSpec, sb: SgacSpec, cb: CBConfig, eval_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "compute_budget_results.json", res)
    cols = ["seed", "status", "A_acc", "B_equal_steps_acc", "B_budget_acc", "A_steps", "B_steps", "A_train_gpu_h",
            "A_screening_gpu_h", "B_train_gpu_h", "B_equal_steps_train_gpu_h", "budget_ratio", "B_cap_hit",
            "A_eval_gpu_h", "B_eval_gpu_h"]
    with open(out_dir / "compute_budget_seeds.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in res["rows"]:
            w.writerow({c: r.get(c) for c in cols})
    figs = _plots(res, out_dir / "figures", sa, sb, cb)
    (out_dir / "compute_budget_comparison.md").write_text(_markdown(res, figs, out_dir, eval_name), encoding="utf-8")


def _trajectory(spec: SgacSpec, arm: str, seed: int, keys) -> list[tuple[float, float]]:
    st = read_json(SgacRunDir(spec, f"{arm}__seed{seed}").path / "state.json") or {}
    cum, out = 0.0, []
    by_step = {int(h["step"]): step_cost(h, keys) for h in st.get("history", [])}
    for t in sorted(by_step):
        cum += by_step[t]
        ev = (st.get("evals") or {}).get(str(t), {}).get("test50")
        if ev:
            out.append((cum / 3600, 100 * ev["acc"]))
    return out


def _plots(res: dict, fig_dir: Path, sa: SgacSpec, sb: SgacSpec, cb: CBConfig) -> list[str]:
    ok = [r for r in res["rows"] if r.get("status") == "ok"]
    if not ok:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    names = []
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for r in ok:
        ax.plot([r["B_equal_steps_train_gpu_h"], r["B_train_gpu_h"]], [100 * (r["B_equal_steps_acc"] or np.nan),
                                                                        100 * r["B_budget_acc"]],
                color="#4c78a8", alpha=0.35, marker="o", ms=3, lw=1)
        ax.plot([r["A_train_gpu_h"]], [100 * r["A_acc"]], color="#e45756", alpha=0.6, marker="s", ms=4, ls="none")
    mx = lambda k: float(np.mean([r[k] for r in ok]))
    my = lambda k: 100 * float(np.mean([r[k] for r in ok if r[k] is not None]))
    ax.plot([mx("B_equal_steps_train_gpu_h"), mx("B_train_gpu_h")], [my("B_equal_steps_acc"), my("B_budget_acc")],
            color="#4c78a8", lw=2.5, marker="o", ms=7, label="B: Random, no screening (step 20 → budget-matched)")
    ax.plot([mx("A_train_gpu_h")], [my("A_acc")], color="#e45756", marker="s", ms=9, ls="none",
            label="A: SGAC, 20 steps (screening + training)")
    if res.get("base_acc") is not None:
        ax.axhline(100 * res["base_acc"], color="grey", ls=":", lw=1, label="base model")
    ax.set_xlabel("training-side GPU-hours per run (screening + training + save)")
    ax.set_ylabel(f"{res['eval']} accuracy (%)")
    ax.set_title("Accuracy vs actual compute (thin: seeds; thick: mean)")
    ax.legend(fontsize=8, loc="best", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(fig_dir / "compute_budget_accuracy_vs_compute.png", dpi=150)
    plt.close(fig)
    names.append("figures/compute_budget_accuracy_vs_compute.png")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for r in ok:
        for arm, sp, keys, col in (("sgac", sa, A_COST_KEYS, "#e45756"), ("random_budget", sb, B_COST_KEYS, "#4c78a8")):
            tr = _trajectory(sp, arm, r["seed"], keys)
            if tr:
                ax.plot(*zip(*tr), color=col, alpha=0.35, marker=".", lw=1)
    ax.plot([], [], color="#e45756", label="A: SGAC")
    ax.plot([], [], color="#4c78a8", label="B: Random, no screening")
    ax.set_xlabel("cumulative training-side GPU-hours")
    ax.set_ylabel("test50 accuracy (%)  [50 items: 1 item = 2 pp]")
    ax.set_title("test50 trajectories vs cumulative compute (per seed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "compute_budget_test50_trajectories.png", dpi=150)
    plt.close(fig)
    names.append("figures/compute_budget_test50_trajectories.png")
    return names


def _ci(c):
    return f"[{c[0]:+.2f}, {c[1]:+.2f}]"


def _markdown(res: dict, figs: list[str], out_dir: Path, eval_name: str) -> str:
    base_txt = "—" if res["base_acc"] is None else f"{100 * res['base_acc']:.1f}%"
    L = ["# SGAC vs Random under an equal GPU-compute budget", "",
         (f"_Generated {res['created']} by `python -m rlvr_v2.sgac_calib cb-analyze`. Design and analysis rules: "
         "`paper/sgac_calib_protocol.md` §13, amendment A6 (fixed before any run)._"), "",
         ("**Question.** Does spending compute on selecting training problems (SGAC) improve final accuracy more than "
         "spending the same compute on more training with Random selection?"), "",
         "- **A — SGAC:** the frozen 20-step curriculum (Eq. 10 as run); every step screens 4 candidates × 4 rollouts.",
         ("- **B — Random:** no screening; as many training steps as fit in A's *measured* training-side GPU time for "
         "the same seed (same batches, same seed, same substrate, same evaluation)."), "",
         (f"Seeds complete: {len(res['seeds_complete'])} of {len(res['seeds_planned'])} ({res['seeds_complete']}). "
         f"Evaluation: {eval_name}, {res['n_items']} items, greedy; base accuracy {base_txt}."), ""]
    if "primary" not in res:
        L += ["**Pending:** fewer than 2 complete seed pairs.", ""]
        return "\n".join(L)
    p, c = res["primary"], res["compute"]
    L += ["## Result", "",
          (f"**B − A = {p['mean_pp']:+.2f} pp, 95% CI {_ci(p['ci95_pp'])}, p = {p['p']:.3g} "
          f"(crossed seed × item test, {len(p['per_seed_pp'])} paired seeds) → verdict: {p['verdict']}.**"), "",
          f"{res['interpretation']}", "",
          f"Decision rule (fixed in advance): a winner needs p < {res['decision_rule']['alpha']}, a mean difference of "
          f"at least {res['decision_rule']['min_effect_pp']} pp and the same sign in ≥ "
          f"{math.ceil(res['decision_rule']['seed_consistency'] * len(p['per_seed_pp']) - 1e-9)} of "
          f"{len(p['per_seed_pp'])} seeds; 'equivalent' needs the 90% CI inside ±{res['decision_rule']['equivalence_pp']} "
          "pp. Sensitivity (bootstrap) CIs: "
          + ", ".join(f"{k} {_ci(v)}" for k, v in p["sensitivity_ci95_pp"].items()) + ".", "",
          "## Per seed", "",
          ("| Seed | A (SGAC, 20 steps) | B at 20 steps | B budget-matched | B steps | A GPU-h (of which screening) | "
          "B GPU-h | B/A compute |"), "|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        if r.get("status") != "ok":
            L.append(f"| {r['seed']} | {r.get('status')} | | | | | | |")
            continue
        b20 = "—" if r["B_equal_steps_acc"] is None else f"{100 * r['B_equal_steps_acc']:.1f}"
        L.append(f"| {r['seed']} | {100 * r['A_acc']:.1f} | {b20} | {100 * r['B_budget_acc']:.1f} | {r['B_steps']}"
                 f"{' (cap)' if r['B_cap_hit'] else ''} | {r['A_train_gpu_h']:.3f} ({r['A_screening_gpu_h']:.3f}) | "
                 f"{r['B_train_gpu_h']:.3f} | {r['budget_ratio']:.3f} |")
    L += ["", "## Compute (GPU-hours, measured)", "",
          "| | A: SGAC | B: Random |", "|---|---|---|",
          (f"| Training-side (budgeted) | {c['A_train_gpu_h']:.2f} (screening {c['A_screening_gpu_h']:.2f}, "
          f"{100 * c['screening_share_of_A']:.0f}%) | {c['B_train_gpu_h']:.2f} |"),
          f"| Evaluation (not budgeted) | {c['A_eval_gpu_h']:.2f} | {c['B_eval_gpu_h']:.2f} |",
          f"| Mean steps per run | {res['rows'][0].get('A_steps', 20)} | {c['mean_B_steps']:.1f} |", "",
          f"Mean realised B/A compute ratio: {c['mean_budget_ratio']:.3f} (B never knowingly exceeds A's budget).", "",
          "## Secondary comparisons", "", "| Comparison | Δ (pp) | 95% CI | p |", "|---|---|---|---|"]
    for k, v in (res.get("secondary") or {}).items():
        L.append(f"| {k} | {v['mean_pp']:+.2f} | {_ci(v['ci95_pp'])} | {v['p']:.3g} |")
    L += [""] + [f"![{Path(f).stem}]({f})" for f in figs] + ["",
          "## Limitations", "",
          ("- One model (Qwen2.5-Math-1.5B), one training recipe (SGAC's 5-update bursts), one pool; 8 seeds give a "
          "minimum detectable difference of roughly 1.6–1.8 pp, so smaller effects remain undetermined."),
          ("- Compute is wall-clock on one GPU; screening and training share the same hardware, so the ratio transfers to "
          "similar setups, not necessarily to batched or multi-GPU screening."),
          "- test50 trajectories use 50 items (1 item = 2 pp) and are descriptive only.", "",
          "## Reproduce", "", "```bash",
          "STAGES=compute_budget scripts/local/sgac_calib.sh      # timing check, base eval, 8 x (A then B), analysis",
          "python -m rlvr_v2.sgac_calib cb-analyze              # rebuild this report from the results", "```", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------- smoke / timing check
def run_cb_smoke(out_dir: Path, gpu: bool = False) -> dict:
    """Timing and plumbing check (never an accuracy claim). CPU: tiny model. GPU: the real model, A with 2 steps and
    64-item test50 subsets (MATH-500 off), B budgeted on A's measured time; projects B's step count for 20 A-steps."""
    from ..sgac.jobs import Context, run_base_eval
    from ..sgac.spec import CONFIG_DIR as SGAC_CONFIG_DIR

    cs = load_calib_for(out_dir)
    cb = load_cb()
    files = [] if gpu else [SGAC_CONFIG_DIR / "smoke_cpu.yaml"]
    ov = ["loop.steps=2", "loop.math500_steps=[]", "loop.eval_max_items=64"] if gpu else ["loop.steps=2"]
    sa = substrate_spec(cs, cb.stage, ov, files)
    cap = 12
    sb = spec_b(cs, cb, [f"loop.steps={cap}", "loop.eval_every=1" if not gpu else "loop.eval_every=4",
                         "loop.math500_steps=[]", *([] if not gpu else ["loop.eval_max_items=64"])], files, cap=cap)
    ctx_a = _ctx(cs, sa)
    run_base_eval(sa, Context(spec=sa, test50=ctx_a.test50, math500=ctx_a.math500, grader=ctx_a.grader))
    seed = int(cb.seeds[0])
    st_a = run_strategy(cs, cb, "sgac", seed, spec=sa, ctx=ctx_a)
    st_b = run_strategy(cs, cb, "random", seed, spec=sb, ctx=_ctx(cs, sb), a_spec=sa)
    again = run_strategy(cs, cb, "random", seed, spec=sb, ctx=_ctx(cs, sb), a_spec=sa)
    assert again["budget_stop"] == st_b["budget_stop"], "re-running a finished B run changed it"
    stop = st_b["budget_stop"]
    assert stop["used_s"] <= stop["budget_s"] * 1.5 or stop["final_step"] == 1, "B overshot its budget badly"
    a_per = training_seconds(st_a["history"], A_COST_KEYS) / max(1, len(st_a["history"]))
    b_per = training_seconds(st_b["history"], B_COST_KEYS) / max(1, len(st_b["history"]))
    summary = {"gpu": gpu, "A_steps": len(st_a["history"]), "A_per_step_s": a_per,
               "A_sieve_per_step_s": training_seconds(st_a["history"], ("sieve",)) / max(1, len(st_a["history"])),
               "B_per_step_s": b_per, "B_steps": stop["final_step"], "budget_ratio": stop["ratio"],
               "projected_B_steps_for_20_A_steps": 20 * a_per / b_per if b_per else None,
               "note": "timing/plumbing check only; accuracies here are not results"}
    res = analyze(cs, cb, eval_name="test50", sa=sa, sb=sb, out_dir=Path(out_dir) / "report")
    summary["analysis_rows"] = res["rows"]
    atomic_write_json(Path(out_dir) / ("cb_smoke_gpu.json" if gpu else "cb_smoke_cpu.json"), summary)
    return summary


def load_calib_for(out_dir: Path) -> CalibSpec:
    from .spec import load_calib

    return load_calib(None, [f"results_root={Path(out_dir).as_posix()}"])


def schedule_prefix_equal(cs: CalibSpec, cb: CBConfig, seed: int, steps: int) -> bool:
    rows, _ = clean_rows(cs)
    return [b["rows"] for b in load_schedule(cs.path(cb.schedule_dir), seed, rows, steps, 4)] == \
        candidate_schedule(rows, seed, steps, 4)
