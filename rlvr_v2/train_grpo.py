"""GRPO training bursts on a live PEFT model (TRL `GRPOTrainer`) with JSONL metric and rollout logging.

Structural guarantees against the old code's failure modes
1. Sequence lengths and sampling parameters come ONLY from `cfg.gen` through `Config.to_grpo_config`;
   `check_grpo_args` refuses to train if the resulting `GRPOConfig` disagrees with the config (e.g. TRL's
   256/512-token `max_completion_length` default) or if the tokenizer's EOS is not the primary stop token
   (TRL masks completions after the first `tokenizer.eos_token_id`).
2. One optimizer step == one generation round of `num_generations` completions:
   `steps_per_generation == gradient_accumulation_steps` and `generation_batch_size == num_generations`.
3. Every logged GRPO step metric is appended to `train_metrics.jsonl` (`JsonlMetricsCallback`) and every
   training rollout to `train_rollouts.jsonl` (via `rewards.make_reward_funcs`).
4. Rewards come from `MathVerifyGrader` only.
5. `TruncationGuardCallback` aborts the burst when `completions/clipped_ratio` is too high in the first
   logged steps.
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence

from transformers import TrainerCallback

from .artifacts import JsonlWriter, RunDir, atomic_write_json, read_json
from .config import Config
from .data import Problem
from .grader import MathVerifyGrader
from .prompts import build_prompt, configure_tokenizer, prompt_hash, resolve_stop_token_ids
from .rewards import make_reward_funcs

log = logging.getLogger(__name__)

CLIPPED_KEY = "completions/clipped_ratio"
CORRECT_KEY = "rewards/correctness/mean"
ZERO_STD_KEY = "frac_reward_zero_std"
LENGTH_KEY = "completions/mean_length"


# ---------------------------------------------------------------------- helpers
def _cuda_max_mem_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:  # pragma: no cover
        pass
    return None


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _mean_of(logs: Sequence[dict], key: str) -> float:
    vals = [f for f in (_finite(l.get(key)) for l in logs) if f is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def _first_last(logs: Sequence[dict], key: str) -> tuple[float, float]:
    vals = [f for f in (_finite(l.get(key)) for l in logs) if f is not None]
    return (vals[0], vals[-1]) if vals else (float("nan"), float("nan"))


def count_trainable_params(model) -> dict[str, float]:
    """Trainable/total parameter counts (PEFT-aware)."""
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        trainable = total = 0
        for p in model.parameters():
            n = p.numel()
            total += n
            trainable += n if p.requires_grad else 0
    return {"trainable_params": int(trainable), "total_params": int(total),
            "trainable_pct": (100.0 * trainable / total) if total else 0.0}


def latest_checkpoint(trainer_dir: str | Path) -> Path | None:
    """Most recent complete `checkpoint-N` (one that has a trainer_state.json) under `trainer_dir`."""
    d = Path(trainer_dir)
    if not d.exists():
        return None
    best: tuple[int, Path] | None = None
    for p in d.iterdir():
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and p.is_dir() and (p / "trainer_state.json").exists():
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, p)
    return best[1] if best else None


# ---------------------------------------------------------------------- callbacks
class JsonlMetricsCallback(TrainerCallback):
    """Append every `on_log` payload (all GRPO metrics) to a JSONL file and keep it in memory."""

    def __init__(self, path: str | Path, context: dict | None = None):
        self.path = Path(path)
        self.context = dict(context or {})
        self.logs: list[dict] = []
        self._t0 = time.time()
        self._writer: JsonlWriter | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        rec = {
            **logs,
            "global_step": int(getattr(state, "global_step", 0) or 0),
            "wall_s": round(time.time() - self._t0, 3),
            "cuda_max_mem_gb": _cuda_max_mem_gb(),
            **self.context,
        }
        self.logs.append(rec)
        if self._writer is None:
            self._writer = JsonlWriter(self.path)
        self._writer.write(rec)

    def on_train_end(self, args, state, control, **kwargs):
        self.close()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class TruncationGuardError(RuntimeError):
    """Raised when early GRPO steps show completions being cut off (the 256-token failure mode)."""


class TruncationGuardCallback(TrainerCallback):
    """Among the first `check_steps` logs carrying `completions/clipped_ratio`, abort if any exceeds
    `max_clipped_ratio`."""

    def __init__(self, max_clipped_ratio: float, check_steps: int):
        self.max_clipped_ratio = float(max_clipped_ratio)
        self.check_steps = int(check_steps)
        self.seen = 0
        self.values: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self.seen >= self.check_steps:
            return
        v = _finite(logs.get(CLIPPED_KEY))
        if v is None:
            return
        self.seen += 1
        self.values.append(v)
        if v > self.max_clipped_ratio:
            raise TruncationGuardError(
                f"GRPO truncation guard: {CLIPPED_KEY}={v:.3f} at global_step {getattr(state, 'global_step', '?')} "
                f"exceeds {self.max_clipped_ratio:.2f} within the first {self.check_steps} logged steps. Completions are "
                f"being cut off: check max_completion_length ({getattr(args, 'max_completion_length', '?')}) == "
                f"cfg.gen.max_new_tokens, that tokenizer.eos is the chat stop token (<|im_end|> for ChatML styles), "
                f"and that the prompt style matches the model."
            )


# ---------------------------------------------------------------------- results
@dataclass
class BurstResult:
    """Summary of one GRPO burst. Means are over the logged training steps; NaN when a metric is absent."""

    steps_done: int
    metrics_path: Path
    train_rollouts_path: Path | None
    reward_correct_mean_first: float
    reward_correct_mean_last: float
    clipped_ratio_mean: float
    frac_zero_std_mean: float
    mean_completion_length: float
    wall_s: float
    logs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metrics_path"] = str(self.metrics_path)
        d["train_rollouts_path"] = None if self.train_rollouts_path is None else str(self.train_rollouts_path)
        return d

    def summary(self) -> dict:
        """`to_dict()` without the per-step logs (for status files and curriculum history)."""
        return {k: v for k, v in self.to_dict().items() if k != "logs"}

    @classmethod
    def from_dict(cls, d: dict) -> "BurstResult":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        kw["metrics_path"] = Path(kw.get("metrics_path") or "")
        kw["train_rollouts_path"] = Path(kw["train_rollouts_path"]) if kw.get("train_rollouts_path") else None
        kw.setdefault("logs", [])
        for name in ("reward_correct_mean_first", "reward_correct_mean_last", "clipped_ratio_mean",
                     "frac_zero_std_mean", "mean_completion_length", "wall_s"):
            v = kw.get(name)
            kw[name] = float("nan") if v is None else float(v)
        return cls(**kw)

    @classmethod
    def from_logs(cls, logs: Sequence[dict], steps_done: int, metrics_path: str | Path,
                  train_rollouts_path: str | Path | None, wall_s: float) -> "BurstResult":
        first, last = _first_last(logs, CORRECT_KEY)
        return cls(
            steps_done=int(steps_done),
            metrics_path=Path(metrics_path),
            train_rollouts_path=Path(train_rollouts_path) if train_rollouts_path else None,
            reward_correct_mean_first=first,
            reward_correct_mean_last=last,
            clipped_ratio_mean=_mean_of(logs, CLIPPED_KEY),
            frac_zero_std_mean=_mean_of(logs, ZERO_STD_KEY),
            mean_completion_length=_mean_of(logs, LENGTH_KEY),
            wall_s=float(wall_s),
            logs=list(logs),
        )


# ---------------------------------------------------------------------- data / preflight
def count_prompt_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_train_dataset(problems: list[Problem], cfg: Config, tokenizer, n_rows: int):
    """`datasets.Dataset` with columns prompt (rendered by `prompts.build_prompt`), answer, unique_id.
    Rows = `problems` cycled to length max(n_rows, cfg.train.prompts_per_round). Raises if any rendered
    prompt exceeds `cfg.gen.max_prompt_tokens` (TRL 1.x has no max_prompt_length to truncate silently)."""
    from datasets import Dataset

    if not problems:
        raise ValueError("build_train_dataset: no problems given")
    rendered: list[tuple[str, Problem]] = []
    for p in problems:
        text = build_prompt(p.problem, cfg.prompt.style, tokenizer, cfg.prompt.system)
        n_tok = count_prompt_tokens(tokenizer, text)
        if n_tok > cfg.gen.max_prompt_tokens:
            raise ValueError(
                f"rendered prompt for {p.unique_id} has {n_tok} tokens > gen.max_prompt_tokens={cfg.gen.max_prompt_tokens}"
            )
        rendered.append((text, p))
    total = max(int(n_rows), int(cfg.train.prompts_per_round), 1)
    rows = [rendered[i % len(rendered)] for i in range(total)]
    return Dataset.from_dict({
        "prompt": [t for t, _ in rows],
        "answer": [p.answer for _, p in rows],
        "unique_id": [p.unique_id for _, p in rows],
    })


def check_grpo_args(args, cfg: Config, tokenizer=None, stop_token_ids: list[int] | None = None) -> dict:
    """Refuse to train unless the GRPOConfig matches `cfg` (lengths, group size, one step per generation
    round) and the tokenizer's EOS/pad are consistent with the stop tokens. Returns the audited values."""
    t, g = cfg.train, cfg.gen
    gen_kwargs = getattr(args, "generation_kwargs", None) or {}
    audited = {
        "max_completion_length": getattr(args, "max_completion_length", None),
        "num_generations": getattr(args, "num_generations", None),
        "per_device_train_batch_size": getattr(args, "per_device_train_batch_size", None),
        "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", None),
        "steps_per_generation": getattr(args, "steps_per_generation", None),
        "generation_batch_size": getattr(args, "generation_batch_size", None),
        "temperature": getattr(args, "temperature", None),
        "top_p": getattr(args, "top_p", None),
        "mask_truncated_completions": getattr(args, "mask_truncated_completions", None),
        "eos_token_id": gen_kwargs.get("eos_token_id"),
        "reward_weights": list(getattr(args, "reward_weights", None) or []),
        "learning_rate": getattr(args, "learning_rate", None),
        "loss_type": getattr(args, "loss_type", None),
        "beta": getattr(args, "beta", None),
        "max_steps": getattr(args, "max_steps", None),
    }
    problems: list[str] = []

    def expect(name: str, got, want) -> None:
        if got != want:
            problems.append(f"{name}: GRPOConfig has {got!r}, config requires {want!r}")

    expect("max_completion_length", audited["max_completion_length"], g.max_new_tokens)
    expect("num_generations", audited["num_generations"], t.num_generations)
    expect("per_device_train_batch_size", audited["per_device_train_batch_size"], t.per_device_train_batch_size)
    expect("gradient_accumulation_steps", audited["gradient_accumulation_steps"], t.gradient_accumulation_steps)
    expect("steps_per_generation", audited["steps_per_generation"], t.gradient_accumulation_steps)
    expect("generation_batch_size", audited["generation_batch_size"], t.generation_batch_size)
    expect("generation_batch_size == num_generations * prompts_per_round", audited["generation_batch_size"],
           t.num_generations * t.prompts_per_round)
    expect("temperature", audited["temperature"], g.sieve.temperature)
    expect("mask_truncated_completions", audited["mask_truncated_completions"], t.mask_truncated_completions)
    expect("reward_weights", audited["reward_weights"], [float(w) for _, w in t.reward_weights])
    mpl = getattr(args, "max_prompt_length", None)
    if mpl is not None and mpl < g.max_prompt_tokens:
        problems.append(f"max_prompt_length {mpl} < gen.max_prompt_tokens {g.max_prompt_tokens}")
    if stop_token_ids:
        expect("generation_kwargs.eos_token_id", audited["eos_token_id"], list(stop_token_ids))
        if tokenizer is not None:
            eos = getattr(tokenizer, "eos_token_id", None)
            pad = getattr(tokenizer, "pad_token_id", None)
            if eos != stop_token_ids[0]:
                problems.append(
                    f"tokenizer.eos_token_id={eos} is not the primary stop token {stop_token_ids[0]}; TRL masks "
                    f"completions after the first tokenizer EOS (run prompts.configure_tokenizer)"
                )
            if pad is None or pad == eos:
                problems.append(f"tokenizer.pad_token_id={pad} must exist and differ from eos_token_id={eos}")
    if problems:
        raise RuntimeError("GRPO preflight failed; refusing to train:\n  - " + "\n  - ".join(problems))
    return audited


# ---------------------------------------------------------------------- burst
def run_grpo_burst(
    model,
    tokenizer,
    problems: list[Problem],
    cfg: Config,
    out_dir: Path,
    max_steps: int,
    seed: int,
    grader: MathVerifyGrader,
    context: dict | None = None,
    resume_from_checkpoint: bool | str = False,
) -> BurstResult:
    """Train `model` (a PeftModel) for `max_steps` optimizer steps of GRPO on `problems`.

    Writes `out_dir/train_metrics.jsonl`, `out_dir/train_rollouts.jsonl`, `out_dir/burst_summary.json` and TRL
    checkpoints under `out_dir/trainer`. The model object is kept (not deleted) so callers can keep training
    or sample from it; the trainer and its accelerator state are released."""
    from trl import GRPOTrainer

    from . import modeling

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    context = dict(context or {})
    if getattr(model, "peft_config", None) is None:
        raise TypeError("run_grpo_burst expects a PeftModel (modeling.attach_fresh_lora / load_adapter); "
                        f"got {type(model).__name__}")

    style = cfg.prompt.style
    configure_tokenizer(tokenizer, style)  # idempotent: eos = primary stop, pad distinct, left padding
    stop_ids = resolve_stop_token_ids(tokenizer, style)
    args = cfg.to_grpo_config(out_dir / "trainer", max_steps, seed, stop_ids)
    audited = check_grpo_args(args, cfg, tokenizer, stop_ids)

    n_rows = int(max_steps) * cfg.train.prompts_per_round  # exactly one epoch of generation rounds
    dataset = build_train_dataset(problems, cfg, tokenizer, n_rows)

    metrics_path = out_dir / "train_metrics.jsonl"
    rollouts_path = out_dir / "train_rollouts.jsonl"
    rollout_writer = JsonlWriter(rollouts_path)
    reward_funcs = make_reward_funcs(grader, stop_token_ids=stop_ids, rollout_writer=rollout_writer, context=context)
    names = [f.__name__ for f in reward_funcs]
    expected = [n for n, _ in cfg.train.reward_weights]
    if names != expected:
        raise RuntimeError(f"reward function order {names} must match train.reward_weights {expected}")
    metrics_cb = JsonlMetricsCallback(metrics_path, context=context)
    guard_cb = TruncationGuardCallback(cfg.train.guard.max_clipped_ratio, cfg.train.guard.check_steps)

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover
        pass

    log.info("GRPO burst: %d steps x %d generations on %s (max_completion_length=%d, resume=%s) -> %s",
             max_steps, cfg.train.num_generations, [p.unique_id for p in problems], args.max_completion_length,
             resume_from_checkpoint, out_dir)
    t0 = time.time()
    trainer = None
    steps_done = 0
    try:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[metrics_cb, guard_cb],
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint if resume_from_checkpoint else None)
        steps_done = int(trainer.state.global_step)
    finally:
        rollout_writer.close()
        metrics_cb.close()
        if trainer is not None:
            try:
                trainer.accelerator.free_memory()
            except Exception as e:  # pragma: no cover
                log.warning("accelerator.free_memory failed: %r", e)
            del trainer
        try:
            modeling.free_cuda()
        except Exception as e:  # pragma: no cover
            log.warning("free_cuda failed: %r", e)

    result = BurstResult.from_logs(metrics_cb.logs, steps_done, metrics_path, rollouts_path, time.time() - t0)
    atomic_write_json(out_dir / "burst_summary.json", {
        **result.to_dict(),
        "context": context,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "unique_ids": [p.unique_id for p in problems],
        "grpo_args": audited,
    })
    log.info("GRPO burst done: %d steps, correctness %.3f -> %.3f, clipped %.3f, zero-std %.3f, %.0fs",
             result.steps_done, result.reward_correct_mean_first, result.reward_correct_mean_last,
             result.clipped_ratio_mean, result.frac_zero_std_mean, result.wall_s)
    return result


# ---------------------------------------------------------------------- Study 1 entry point
def run_one_shot_grpo(
    problem: Problem,
    cfg: Config,
    seed: int,
    run_dir: RunDir,
    rounds: int | None = None,
    model=None,
    tokenizer=None,
    grader: MathVerifyGrader | None = None,
) -> tuple[Path, BurstResult]:
    """Study-1 path: fresh LoRA on the base model, one GRPO burst of `rounds or cfg.train.rounds` steps on a
    single problem, adapter saved to `run_dir/adapter`. Resumable: a finished run returns the persisted
    summary; an interrupted run resumes from the latest TRL checkpoint under `run_dir/train/trainer`."""
    from . import modeling

    train_dir = run_dir.path / "train"
    adapter_dir = run_dir.path / "adapter"
    done_flag = train_dir / "done.flag"
    summary_path = train_dir / "burst_summary.json"

    if done_flag.exists():
        summary = read_json(summary_path)
        if summary is not None and adapter_dir.exists() and any(adapter_dir.iterdir()):
            log.info("train stage already done for %s (seed %d); reusing %s", problem.unique_id, seed, adapter_dir)
            return adapter_dir, BurstResult.from_dict(summary)
        log.warning("done.flag present but summary/adapter missing under %s; retraining", run_dir.path)
        done_flag.unlink()

    grader = grader or MathVerifyGrader()
    if tokenizer is None:
        tokenizer = modeling.load_tokenizer(cfg)
    if model is None:
        model = modeling.attach_fresh_lora(modeling.load_base_model(cfg), cfg)
    counts = count_trainable_params(model)
    run_dir.set_status(stage="train", unique_id=problem.unique_id, **counts)
    log.info("trainable params: %s", counts)

    ckpt = latest_checkpoint(train_dir / "trainer")
    if ckpt is not None:
        log.info("resuming GRPO from %s", ckpt)
    result = run_grpo_burst(
        model, tokenizer, [problem], cfg, train_dir,
        max_steps=int(rounds or cfg.train.rounds), seed=seed, grader=grader,
        context={"unique_id": problem.unique_id, "seed": int(seed)},
        resume_from_checkpoint=str(ckpt) if ckpt is not None else False,
    )

    rendered = build_prompt(problem.problem, cfg.prompt.style, tokenizer, cfg.prompt.system)
    meta = {
        "prompt_style": cfg.prompt.style,
        "prompt_hash": prompt_hash(rendered),
        "unique_id": problem.unique_id,
        "seed": int(seed),
        "steps_done": result.steps_done,
        "config_hash": cfg.config_hash(),
    }
    modeling.save_adapter(model, run_dir.stage("adapter"), meta=meta)
    done_flag.write_text("done\n", encoding="utf-8")
    run_dir.set_status(stage="train_done", steps_done=result.steps_done)
    return adapter_dir, result
