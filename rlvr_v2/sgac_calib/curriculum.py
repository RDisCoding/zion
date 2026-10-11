"""Stage E (protocol section 6): the full 20-step curriculum on the clean pool with the frozen learned selector.

Every arm runs the frozen loop (`rlvr_v2.sgac.loop.SgacRun`) with the e0 substrate (hash 577f3319) on the clean
964-item pool and this track's schedules; only the results root differs from the reproduction. The `learned` arm
reuses the loop's own selection step (sieve, signals, every rule's counterfactual pick) and then replaces the pick with
the frozen selector's argmax (first maximum on ties, the loop's rule). `random_tm` (exploratory) is the Random arm
without the sieve, run for more steps to match the deployment GPU time of a sieving arm; its first 20 picks equal the
Random arm's (same RNG stream, same schedule prefix).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..artifacts import atomic_write_json, read_json
from ..sgac import data as sdata
from ..sgac import selection as sel
from ..sgac.loop import SgacRun
from ..sgac.runinfo import SgacRunDir
from ..sgac.schedule import load_schedule, random_arm_rng
from ..sgac.spec import SgacSpec
from .manifests import DATA_ROLES
from .selector import SELECTOR_JSON, LearnedSelector
from .spec import CalibSpec, require_e0_substrate, substrate_spec

log = logging.getLogger(__name__)

STAGE = "curriculum"
SELECTOR_PIN = "selector_pin.json"
SCHEDULE_STEPS = 40  # schedules are written longer than 20 steps; every arm reads its own prefix
ARMS = ("learned", "random", "sgac", "max_var", "max_d", "max_level", "random_tm")


@dataclass
class CurriculumContext:
    cs: CalibSpec
    spec: SgacSpec
    pool: list = field(default_factory=list)
    test50: list = field(default_factory=list)
    math500: list = field(default_factory=list)
    grader: object | None = None


def load_curriculum_context(cs: CalibSpec, spec: SgacSpec | None = None) -> CurriculumContext:
    from ..sgac.grading import DualGrader

    spec = spec or substrate_spec(cs, STAGE)
    roles = read_json(cs.manifest_dir / DATA_ROLES)
    manifest = sdata.load_data_manifest(spec)
    ds = sdata.load_split(spec)
    rows = set(roles["curriculum_pool"]["rows"])
    full = sdata.load_section(spec, "pool", ds, manifest)
    pool = [it for it in full if it.row in rows]
    if [it.unique_id for it in sorted(pool, key=lambda i: i.row)] != \
            [u for _, u in sorted(zip(roles["curriculum_pool"]["rows"], roles["curriculum_pool"]["ids"]))]:
        raise RuntimeError("clean curriculum pool does not match data_roles.json")
    ctx = CurriculumContext(cs=cs, spec=spec, pool=pool, grader=DualGrader(spec.profile.grader))
    ctx.test50 = sdata.load_section(spec, "test50", ds, manifest)
    if spec.loop.math500_steps:
        ctx.math500 = sdata.load_math500_items(spec)
    return ctx


def pinned_selector(cs: CalibSpec) -> LearnedSelector:
    pin = read_json(cs.manifest_dir / SELECTOR_PIN)
    if pin is None:
        raise FileNotFoundError(f"{cs.manifest_dir / SELECTOR_PIN} missing: freeze the selector (gate G-S) first")
    return LearnedSelector(cs.manifest_dir / SELECTOR_JSON, pin["sha256"])


class LearnedRun(SgacRun):
    def __init__(self, spec: SgacSpec, seed: int, selector: LearnedSelector, **kw):
        super().__init__(spec, "sgac", seed, **kw)
        self.selector = selector
        self.arm, self.rule = "learned", "learned"
        self.rd = SgacRunDir(spec, f"learned__seed{self.seed}")
        self.state_path = self.rd.path / "state.json"
        self.state = self._new_state()

    def _select(self, t: int, sd: Path, timings: dict):
        self.rule = "sgac_eq10"  # the base step names its own pick with a valid rule; replaced below
        try:
            _, selection = super()._select(t, sd, timings)
        finally:
            self.rule = "learned"
        signals = [c["signals"] for c in selection["candidates"]]
        scores = [self.selector.score(s) for s in signals]
        idx = sel.first_max(scores)
        items = self._candidates(t)
        chosen = items[idx]
        selection.update(rule="learned", selected_idx=int(idx), selected_uid=chosen.unique_id, selected_row=chosen.row,
                         selected_level=chosen.level, selected_signals=signals[idx])
        selection["picks"]["learned"] = int(idx)
        selection["scores"]["learned"] = scores
        atomic_write_json(sd / "selection.json", selection)
        log.info("[learned seed %d] step %d: picked #%d %s (scores %s)", self.seed, t, idx, chosen.unique_id, scores)
        return chosen, selection

    def run(self) -> dict:
        if not self.rd.exists():
            self.rd.init(self.seed, extra={"arm": "learned", "rule": "learned", "selector_sha256": self.selector.sha256,
                                           "selector": self.selector.doc})
        return super().run()


class RandomNoSieveRun(SgacRun):
    """Exploratory time-matched Random: the random arm's RNG stream without sieving (no counterfactual signals)."""

    def __init__(self, spec: SgacSpec, seed: int, **kw):
        super().__init__(spec, "random", seed, **kw)
        self.arm = "random_tm"
        self.rd = SgacRunDir(spec, f"random_tm__seed{self.seed}")
        self.state_path = self.rd.path / "state.json"
        self.state = self._new_state()

    def _select(self, t: int, sd: Path, timings: dict):
        items = self._candidates(t)
        idx = int(random_arm_rng(self.seed, t).integers(len(items)))
        chosen = items[idx]
        selection = {"step": t, "arm": self.arm, "rule": "random", "selected_idx": idx, "selected_uid": chosen.unique_id,
                     "selected_row": chosen.row, "selected_level": chosen.level, "candidates": [
                         {"idx": i, "row": it.row, "unique_id": it.unique_id, "level": it.level} for i, it in enumerate(items)],
                     "picks": {"random": idx}, "scores": {}, "sieve": "skipped (time-matched arm)"}
        atomic_write_json(sd / "selection.json", selection)
        return chosen, selection


def tm_steps(sieve_s: float, burst_s: float, base_steps: int = 20, cap: int = SCHEDULE_STEPS) -> int:
    """Pre-declared: round(20 * (t_sieve + t_burst) / t_burst), from T0-smoke timings, capped by the schedule length."""
    return int(min(cap, round(base_steps * (sieve_s + burst_s) / burst_s)))


def run_curriculum_arm(ctx: CurriculumContext, arm: str, seed: int, steps_override: int | None = None) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    spec = ctx.spec
    if arm == "random_tm":
        if steps_override is None:
            raise ValueError("random_tm needs its pre-declared step count")
        spec = substrate_spec(ctx.cs, STAGE, [f"loop.steps={steps_override}",
                                              f"loop.math500_steps=[0,{steps_override}]"])
    else:
        require_e0_substrate(spec)
    rows = sorted(it.row for it in ctx.pool)
    schedule = load_schedule(ctx.cs.manifest_dir, seed, rows, spec.loop.steps, spec.loop.batch_b)
    kw = {"pool": ctx.pool, "test50": ctx.test50, "math500": ctx.math500, "schedule": schedule, "grader": ctx.grader}
    if arm == "learned":
        run = LearnedRun(spec, seed, pinned_selector(ctx.cs), **kw)
    elif arm == "random_tm":
        run = RandomNoSieveRun(spec, seed, **kw)
    else:
        run = SgacRun(spec, arm, seed, **kw)
    try:
        return run.run()
    except Exception as e:
        if run.rd.exists():
            run.rd.mark_failed(repr(e))
        raise
