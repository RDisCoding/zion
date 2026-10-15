"""Exploratory Stage E (amendment A4): the frozen Stage-E design run with an EXPLORATORY selector.

Same loop (`rlvr_v2.sgac.loop.SgacRun`, e0 substrate 577f3319), clean pool, schedules (seeds 42-49), arms and
analysis (`analysis.confirmatory`: crossed seed x item test, Holm, the pre-declared decision rule) as the protocol's
Stage E. Differences: the learned arm reads `results_sgac_calib/explore/<tag>/selector.json`, pinned by the sha256
recorded in that tag's `equation.json`; runs live under `results_sgac_calib/curriculum_explore/`; the learned run is
named `learned_<tag>__seed<S>`; Random, SGAC and the heuristics are shared by every exploratory selector.
"""
from __future__ import annotations

import hashlib

from ..artifacts import atomic_write_json, read_json
from ..sgac.loop import SgacRun
from ..sgac.runinfo import SgacRunDir
from ..sgac.schedule import load_schedule
from .analysis import confirmatory, load_arm
from .curriculum import LearnedRun, load_curriculum_context
from .explore import EXPLORE
from .jobs import confirmatory_seeds
from .selector import SELECTOR_JSON, LearnedSelector
from .spec import CalibSpec, require_e0_substrate, substrate_spec

STAGE = "curriculum_explore"
SHARED_ARMS = ("random", "sgac", "max_var", "max_d", "max_level")


def explore_selector(cs: CalibSpec, tag: str) -> LearnedSelector:
    eq = read_json(cs.results_dir / EXPLORE / tag / "equation.json")
    if eq is None or "frozen" not in eq:
        raise FileNotFoundError(f"no frozen exploratory selector for tag {tag!r}; run `fit-explore --freeze` first")
    return LearnedSelector(cs.results_dir / EXPLORE / tag / SELECTOR_JSON, eq["frozen"]["sha256"])


class ExploreLearnedRun(LearnedRun):
    def __init__(self, spec, seed, selector: LearnedSelector, tag: str, **kw):
        super().__init__(spec, seed, selector, **kw)
        self.tag = tag
        self.rd = SgacRunDir(spec, f"learned_{tag}__seed{self.seed}")
        self.state_path = self.rd.path / "state.json"
        self.state = self._new_state()

    def run(self) -> dict:
        if not self.rd.exists():
            self.rd.init(self.seed, extra={"arm": "learned", "exploratory_tag": self.tag,
                                           "selector_sha256": self.selector.sha256, "selector": self.selector.doc})
        return super().run()


def explore_spec(cs: CalibSpec):
    spec = substrate_spec(cs, STAGE)
    require_e0_substrate(spec)
    return spec


def run_explore_arm(cs: CalibSpec, arm: str, seed: int, tag: str | None) -> dict:
    spec = explore_spec(cs)
    ctx = load_curriculum_context(cs, spec)
    rows = sorted(it.row for it in ctx.pool)
    schedule = load_schedule(cs.manifest_dir, seed, rows, spec.loop.steps, spec.loop.batch_b)
    kw = {"pool": ctx.pool, "test50": ctx.test50, "math500": ctx.math500, "schedule": schedule, "grader": ctx.grader}
    if arm == "learned":
        if not tag:
            raise ValueError("the learned arm needs --tag (which exploratory selector)")
        run = ExploreLearnedRun(spec, seed, explore_selector(cs, tag), tag, **kw)
    elif arm in SHARED_ARMS:
        run = SgacRun(spec, arm, seed, **kw)
    else:
        raise ValueError(f"unknown exploratory arm {arm!r}")
    try:
        return run.run()
    except Exception as e:
        if run.rd.exists():
            run.rd.mark_failed(repr(e))
        raise


def analyze_explore(cs: CalibSpec, tag: str) -> dict:
    from ..sgac.spec import group_dir

    spec = substrate_spec(cs, STAGE)
    ids = read_json(cs.path("manifests/math500.json"))["ids"]
    seeds = confirmatory_seeds(cs)
    g = group_dir(spec)
    arms = {"learned": load_arm(g, f"learned_{tag}", seeds, ids, step=int(spec.loop.steps)),
            "random": load_arm(g, "random", seeds, ids, step=int(spec.loop.steps)),
            "sgac": load_arm(g, "sgac", seeds, ids, step=int(spec.loop.steps))}
    res = confirmatory(cs, arms)
    eq = read_json(cs.results_dir / EXPLORE / tag / "equation.json") or {}
    res.update(exploratory=True, selector_tag=tag, selector_validated_by_gate_s=(eq.get("gate_s_thresholds_applied")
                                                                                or {}).get("passed"))
    atomic_write_json(cs.results_dir / STAGE / f"analysis_{tag}.json", res)
    return res


def file_sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
