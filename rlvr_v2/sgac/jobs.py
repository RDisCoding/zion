"""Job bodies behind the CLI: shared base / pi1 evaluations, curriculum runs, checkpoint re-evaluation and the
pre-declared tier queue (paper/sgac_repro_protocol.md section 5)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..artifacts import REPO_ROOT, atomic_write_json, read_json, read_jsonl
from . import data as sdata
from .runinfo import SgacRunDir, phase, run_manifest, shared_dir
from .schedule import load_schedule
from .spec import SgacSpec

log = logging.getLogger(__name__)

E0_G1_PER_ITEM = "results/e0/g1/oneshot_rlvr_chat/math500/per_item.jsonl"
E0_G1_META = "results/e0/g1/oneshot_rlvr_chat/math500/eval_meta.json"
E0_AGREEMENT_MIN = 0.98
TEST50_AGREEMENT_MIN = 0.98


@dataclass
class Context:
    spec: SgacSpec
    pool: list = field(default_factory=list)
    test50: list = field(default_factory=list)
    math500: list = field(default_factory=list)
    pi1: object | None = None
    grader: object | None = None


def load_context(spec: SgacSpec, need_pool: bool = True, need_math500: bool | None = None) -> Context:
    from .grading import DualGrader

    manifest = sdata.load_data_manifest(spec)
    ds = sdata.load_split(spec)
    ctx = Context(spec=spec)
    ctx.test50 = sdata.load_section(spec, "test50", ds, manifest)
    if need_pool:
        ctx.pool = sdata.load_section(spec, "pool", ds, manifest)
    need_math500 = bool(spec.loop.math500_steps) if need_math500 is None else need_math500
    if need_math500:
        ctx.math500 = sdata.load_math500_items(spec)
    ctx.pi1 = sdata.pi1_item()
    ctx.grader = DualGrader(spec.profile.grader)
    return ctx


# ---------------------------------------------------------------------- shared evaluations
def _eval_model(spec, model, tok, items, grader, out_dir, tag, policy, batch_size=None) -> dict:
    from .evaluation import evaluate_items
    from .model_io import stop_token_ids
    from .sampler import eval_sampler

    sampler = eval_sampler(model, tok, spec, stop_token_ids(spec, tok), batch_size)
    return evaluate_items(sampler, tok, items, spec, grader, out_dir, tag=tag, policy=policy)


def run_base_eval(spec: SgacSpec, ctx: Context, e0_g1: str | None = None) -> dict:
    """Base model on test50 (+ MATH-500). as_run: test50 at batch 1 and at the profile batch, the pre-declared batch
    rule is decided here. e0: per-item agreement with E0 G1 must be >= 0.98 (pipeline-validity gate before T2)."""
    from ..modeling import free_cuda
    from .evaluation import item_agreement
    from .model_io import load_policy_model, load_tokenizer

    out = shared_dir(spec, "base_eval")
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(spec)
    model, load_record = load_policy_model(spec)
    timings: dict = {}
    result: dict = {"manifest": run_manifest(spec, None, {"job": "base_eval", "load_record": load_record})}
    bs = int(spec.profile.eval_batch_size)
    with phase(timings, "test50"):
        result["test50"] = _eval_model(spec, model, tok, ctx.test50, ctx.grader, out / "test50", "base-test50", "base", bs)
    if spec.profile.test50_batch_check:
        with phase(timings, "test50_bs1"):
            result["test50_bs1"] = _eval_model(spec, model, tok, ctx.test50, ctx.grader, out / "test50_bs1",
                                               "base-test50-bs1", "base", 1)
        agree = item_agreement(out / "test50" / "per_item.jsonl", out / "test50_bs1" / "per_item.jsonl")
        decision = {"agreement": agree, "rule": f"batch {bs} if verdict agreement >= {TEST50_AGREEMENT_MIN} else batch 1",
                    "batch_size": bs if agree["verdict_agreement"] >= TEST50_AGREEMENT_MIN else 1}
        atomic_write_json(out / "test50_batch_decision.json", decision)
        result["test50_batch_decision"] = decision
    if 0 in tuple(spec.loop.math500_steps) and ctx.math500:
        with phase(timings, "math500"):
            result["math500"] = _eval_model(spec, model, tok, ctx.math500, ctx.grader, out / "math500", "base-math500",
                                            "base", bs)
        if spec.profile.name == "e0":
            ref = Path(e0_g1) if e0_g1 else REPO_ROOT / E0_G1_PER_ITEM
            if ref.exists():
                agree = item_agreement(ref, out / "math500" / "per_item.jsonl")
                ref_meta = read_json(ref.parent / "eval_meta.json", {}) or {}
                agree.update(reference=str(ref), reference_prompt_hash=ref_meta.get("prompt_hash"),
                             our_prompt_hash=result["math500"].get("prompt_hash_first"),
                             threshold=E0_AGREEMENT_MIN, passed=agree["verdict_agreement"] >= E0_AGREEMENT_MIN)
            else:
                agree = {"reference": str(ref), "passed": None, "error": "E0 G1 per_item.jsonl not found"}
            atomic_write_json(out / "e0_g1_agreement.json", agree)
            result["e0_g1_agreement"] = agree
    result["timings"] = timings
    atomic_write_json(out / "base_eval.json", result)
    del model
    free_cuda()
    return result


def run_pi1_eval(spec: SgacSpec, ctx: Context) -> dict:
    """Wang et al.'s pi1 checkpoint with the profile's evaluation protocol (NB-M cell 11 for as_run)."""
    from ..modeling import free_cuda
    from .loop import test50_batch_size
    from .model_io import load_pi1_model, load_tokenizer

    out = shared_dir(spec, "pi1_eval")
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(spec)  # NB-M reused the base Qwen tokenizer object
    model, load_record = load_pi1_model(spec)
    timings: dict = {}
    result: dict = {"manifest": run_manifest(spec, None, {"job": "pi1_eval", "load_record": load_record})}
    with phase(timings, "test50"):
        result["test50"] = _eval_model(spec, model, tok, ctx.test50, ctx.grader, out / "test50", "pi1-test50", "pi1",
                                       test50_batch_size(spec))
    if ctx.math500:
        with phase(timings, "math500"):
            result["math500"] = _eval_model(spec, model, tok, ctx.math500, ctx.grader, out / "math500", "pi1-math500",
                                            "pi1", int(spec.profile.eval_batch_size))
    result["timings"] = timings
    atomic_write_json(out / "pi1_eval.json", result)
    del model
    free_cuda()
    return result


# ---------------------------------------------------------------------- curriculum
def run_curriculum(spec: SgacSpec, ctx: Context, arm: str, seed: int, base_after_run: bool = False) -> dict:
    from .loop import SgacRun

    schedule = load_schedule(sdata.manifest_dir(spec), seed, [it.row for it in ctx.pool], spec.loop.steps,
                             spec.loop.batch_b) if arm != "fixed_pi1" else []
    run = SgacRun(spec, arm, seed, pool=ctx.pool, test50=ctx.test50, math500=ctx.math500, schedule=schedule,
                  grader=ctx.grader, pi1=ctx.pi1, base_after_run=base_after_run)
    try:
        return run.run()
    except Exception as e:
        if run.rd.exists():
            run.rd.mark_failed(repr(e))
        raise


def run_eval_checkpoints(spec: SgacSpec, ctx: Context, arm: str, seed: int, steps: list[int], sets: list[str]) -> dict:
    """Evaluate saved step adapters later (e.g. MATH-500 at steps 5/10/15) without retraining."""
    from ..modeling import free_cuda
    from .evaluation import evaluate_items
    from .loop import test50_batch_size
    from .model_io import load_policy_model, load_step_adapter, load_tokenizer, stop_token_ids
    from .sampler import eval_sampler

    rd = SgacRunDir(spec, f"{arm}__seed{seed}")
    state = read_json(rd.path / "state.json")
    if state is None:
        raise FileNotFoundError(f"no run at {rd.path}")
    tok = load_tokenizer(spec)
    base, _ = load_policy_model(spec)
    model = None
    done = {}
    for t in steps:
        adapter = rd.path / "steps" / f"step-{t:03d}" / "adapter"
        if not adapter.exists():
            log.warning("no saved adapter for step %d at %s; skipping", t, adapter)
            continue
        if model is not None:
            base = model.unload()  # drop the previous adapter, keep the base weights
        model = load_step_adapter(base, adapter, state.get("lora_dtype"))
        for name in sets:
            items = ctx.math500 if name == "math500" else ctx.test50
            bs = int(spec.profile.eval_batch_size) if name == "math500" else test50_batch_size(spec)
            out = rd.path / "evals" / f"step-{t:03d}" / name
            s = evaluate_items(eval_sampler(model, tok, spec, stop_token_ids(spec, tok), bs), tok, items, spec,
                               ctx.grader, out, tag=f"{arm}-seed{seed}-step{t}-{name}", policy=f"{arm}/seed{seed}/step{t}",
                               adapter_ref=str(adapter))
            done[f"{t}/{name}"] = s["acc"]
            free_cuda()
    atomic_write_json(rd.path / "evals" / "checkpoint_evals.json", done)
    return done


# ---------------------------------------------------------------------- tiers
CORE = ("sgac", "random", "max_var", "max_d", "max_level")


def tier_jobs(tier: int) -> list[str]:
    """Pre-declared queue order (never reordered on results). Each line is the argument list for `python -m rlvr_v2.sgac`."""
    run = "run --profile {p} --arm {a} --seed {s}"
    jobs: list[str] = []
    if tier == 0:
        jobs = ["probe-env --profile e0", "build-manifests --check --profile e0", "smoke --profile e0 --gpu",
                "smoke --profile as_run --gpu"]
    elif tier == 1:
        jobs = ["base-eval --profile e0", "pi1-eval --profile e0"]
    elif tier == 2:
        jobs = [run.format(p="e0", a=a, s=42) for a in CORE]
    elif tier == 3:
        jobs = ["base-eval --profile as_run", "pi1-eval --profile as_run",
                run.format(p="as_run", a="sgac", s=42) + " --base-after-run", run.format(p="as_run", a="random", s=42)]
    elif tier == 4:
        jobs = [run.format(p="e0", a=a, s=s) for a in ("sgac", "random") for s in (43, 44)]
        jobs += [run.format(p="e0", a=a, s=s) for a in ("max_var", "max_d", "max_level") for s in (43, 44)]
    elif tier == 5:
        jobs = [run.format(p="as_run", a=a, s=42) for a in ("max_var", "max_d", "max_level")]
        jobs += [run.format(p="as_run", a=a, s=s) for a in ("sgac", "random") for s in (43, 44)]
        jobs += [run.format(p="as_run", a=a, s=s) for a in ("max_var", "max_d", "max_level") for s in (43, 44)]
    elif tier == 6:
        jobs = [run.format(p="e0", a="sgac_label_corrected", s=s) for s in (42, 43, 44)]
        jobs += [run.format(p="e0", a="fixed_pi1", s=s) for s in (42, 43, 44)]
        jobs += ["phase1 --profile as_run"]
        jobs += [f"eval-checkpoints --profile e0 --arm {a} --seed 42 --steps 5,10,15 --sets math500" for a in CORE]
    else:
        raise ValueError(f"unknown tier {tier}")
    return jobs


def run_is_done(spec: SgacSpec, arm: str, seed: int) -> bool:
    return SgacRunDir(spec, f"{arm}__seed{seed}").is_done()


def read_per_item(path: str | Path) -> dict[str, dict]:
    return {r["unique_id"]: r for r in read_jsonl(path)}
