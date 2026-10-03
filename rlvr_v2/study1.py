"""Study 1: one 1-shot GRPO run per (candidate problem, seed), plus replicate seeds.

Jobs are expanded deterministically from the candidates manifest so a SLURM array index maps to exactly one
(candidate, seed) pair. `run_job` is resumable stage by stage: training is skipped when
`run_dir/train/done.flag` exists (see `train_grpo.run_one_shot_grpo`), each evaluation when its
`eval_summary.json` exists.

Heavy collaborators are reached through module-level lazy wrappers (patchable in tests) so that
`expand_jobs` / `write_jobs_csv` work without torch, as the CLI's `study1-jobs` command requires.
"""
from __future__ import annotations

import contextlib
import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from .artifacts import RunDir, atomic_write_json, read_json
from .config import Config
from .data import Manifest, Problem

log = logging.getLogger(__name__)

EVAL_SUMMARY = "eval_summary.json"


# ---------------------------------------------------------------------- lazy, patchable collaborators
def run_one_shot_grpo(*args, **kwargs):
    from .train_grpo import run_one_shot_grpo as impl

    return impl(*args, **kwargs)


def evaluate(*args, **kwargs):
    from .evaluate import evaluate as impl

    return impl(*args, **kwargs)


def make_sampler(*args, **kwargs):
    from .sampling import make_sampler as impl

    return impl(*args, **kwargs)


def generation_mode(model):
    from .modeling import generation_mode as impl

    return impl(model)


def load_tokenizer(cfg):
    from .modeling import load_tokenizer as impl

    return impl(cfg)


def load_base_model(cfg):
    from .modeling import load_base_model as impl

    return impl(cfg)


def load_adapter(base, adapter_dir):
    from .modeling import load_adapter as impl

    return impl(base, adapter_dir)


def free_cuda():
    from .modeling import free_cuda as impl

    return impl()


# ---------------------------------------------------------------------- jobs
@dataclass(frozen=True)
class Job:
    index: int
    unique_id: str
    seed: int
    replicate: bool

    def to_dict(self) -> dict:
        return asdict(self)


def slug(unique_id: str) -> str:
    """Filesystem-safe form of a MATH unique_id ('/' and '.' -> '-')."""
    return unique_id.replace("/", "-").replace(".", "-")


def run_name_for(job: Job) -> str:
    return f"{slug(job.unique_id)}__seed{job.seed}"


def expand_jobs(cfg: Config, candidates_manifest: Manifest) -> list[Job]:
    """One job per candidate (seed = study1.seed_base + position), followed by replicate jobs for the first
    `study1.replicate_count` ids listed in `manifest.meta['replicate_ids']` (default: the first candidates)
    with seed = base seed + study1.replicate_seed_offset. Ordering is deterministic."""
    s1 = cfg.study1
    ids = list(candidates_manifest.ids)
    if len(set(ids)) != len(ids):
        raise ValueError("candidates manifest contains duplicate unique_ids")
    jobs = [Job(i, uid, s1.seed_base + i, False) for i, uid in enumerate(ids)]
    seed_of = {j.unique_id: j.seed for j in jobs}
    meta = candidates_manifest.meta or {}
    rep_ids = list(meta.get("replicate_ids") or ids[: s1.replicate_count])[: s1.replicate_count]
    unknown = [u for u in rep_ids if u not in seed_of]
    if unknown:
        raise ValueError(f"replicate_ids not in the candidates manifest: {unknown}")
    for k, uid in enumerate(rep_ids):
        jobs.append(Job(len(ids) + k, uid, seed_of[uid] + s1.replicate_seed_offset, True))
    return jobs


def write_jobs_csv(jobs: list[Job], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "unique_id", "seed", "replicate", "run_name"])
        for j in jobs:
            w.writerow([j.index, j.unique_id, j.seed, int(j.replicate), run_name_for(j)])
    return path


# ---------------------------------------------------------------------- running one job
def _find_problem(problems: list[Problem], unique_id: str) -> Problem:
    for p in problems:
        if p.unique_id == unique_id:
            return p
    raise KeyError(f"candidate {unique_id!r} is not in the pool problems")


def _adapter_sampler(cfg: Config, tokenizer, adapter_dir: Path):
    """Sampler over `adapter_dir`: HF backend loads the adapter onto a fresh base; vLLM gets the path."""
    if cfg.gen.backend == "vllm":
        return make_sampler(cfg, tokenizer=tokenizer, adapter_path=str(adapter_dir)), None
    model = load_adapter(load_base_model(cfg), adapter_dir)
    return make_sampler(cfg, model=model, tokenizer=tokenizer, adapter_path=str(adapter_dir)), model


def run_job(
    cfg: Config,
    job: Job,
    pool_problems: list[Problem],
    eval_problems: list[Problem],
    heldout_problems: list[Problem] | None,
    grader,
    stages: tuple[str, ...] = ("train", "eval"),
) -> dict:
    """Train (1-shot GRPO) and evaluate one Study-1 job; writes `run_dir/result.json`. Resumable."""
    problem = _find_problem(pool_problems, job.unique_id)
    run_dir = RunDir(cfg, "study1", run_name_for(job))
    if not run_dir.exists():
        run_dir.init(job.seed, extra={"unique_id": job.unique_id, "replicate": job.replicate, "job_index": job.index})
    result_path = run_dir.path / "result.json"
    result: dict = read_json(result_path, {}) or {}
    result.update({"job_index": job.index, "unique_id": job.unique_id, "seed": job.seed, "replicate": job.replicate,
                   "run_id": run_dir.run_id, "run_dir": str(run_dir.path), "stages": list(stages)})
    adapter_dir = run_dir.path / "adapter"
    try:
        if "train" in stages:
            adapter_dir, burst = run_one_shot_grpo(problem, cfg, job.seed, run_dir, grader=grader)
            result["burst"] = burst.summary() if hasattr(burst, "summary") else burst.to_dict()
            result["adapter_dir"] = str(adapter_dir)
        elif "burst" not in result:
            persisted = read_json(run_dir.path / "train" / "burst_summary.json")
            if persisted is not None:
                result["burst"] = {k: v for k, v in persisted.items() if k != "logs"}

        if "eval" in stages:
            if not adapter_dir.exists() or not any(adapter_dir.iterdir()):
                raise RuntimeError(f"eval stage needs a trained adapter at {adapter_dir}; run the train stage first")
            evals: dict = result.setdefault("evals", {})
            targets = [("math500", eval_problems)]
            if heldout_problems:
                targets.append(("heldout", heldout_problems))
            pending = [(n, p) for n, p in targets if not (run_dir.path / "eval" / n / EVAL_SUMMARY).exists()]
            if pending:
                run_dir.set_status(stage="eval")
                tokenizer = load_tokenizer(cfg)
                sampler, model = _adapter_sampler(cfg, tokenizer, adapter_dir)
                for name, probs in pending:
                    out_dir = run_dir.path / "eval" / name
                    with (generation_mode(model) if model is not None else contextlib.nullcontext()):
                        summary = evaluate(sampler, tokenizer, probs, cfg, grader, out_dir,
                                           tag=f"{run_name_for(job)}-{name}", policy="adapter",
                                           adapter_path=str(adapter_dir))
                    d = summary.to_dict() if hasattr(summary, "to_dict") else dict(summary)
                    atomic_write_json(out_dir / EVAL_SUMMARY, d)
                    evals[name] = d
                    log.info("job %d %s: %s acc=%.4f", job.index, job.unique_id, name, float(summary.acc))
                del sampler, model
                free_cuda()
            for name, _ in targets:
                if name not in evals:
                    evals[name] = read_json(run_dir.path / "eval" / name / EVAL_SUMMARY)
            result["acc_math500"] = (evals.get("math500") or {}).get("acc")
            if "heldout" in evals:
                result["acc_heldout"] = (evals.get("heldout") or {}).get("acc")

        atomic_write_json(result_path, result)
        if "eval" in stages:
            run_dir.mark_done()
        else:
            run_dir.set_status(stage=f"{stages[-1]}_done" if stages else "noop")
    except Exception as e:
        run_dir.mark_failed(repr(e))
        raise
    return result
