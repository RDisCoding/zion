"""Phase C (NB-M cell 7): one GRPO micro-burst on the selected problem, on the persistent PeftModel.

NB-M:
    GRPOConfig(output_dir, learning_rate=2e-5, per_device_train_batch_size=1, generation_batch_size=4,
               num_generations=4, max_steps=5, logging_steps=1, save_strategy="no", report_to="none")
    trainer = GRPOTrainer(model=active_model, reward_funcs=[binary_match_reward, format_reward], args=config,
                          train_dataset=Dataset.from_list([winning_cand]), processing_class=tokenizer)
    trainer.train(); del trainer; gc.collect(); torch.cuda.empty_cache()
Every field it left unset is pinned here to TRL 1.0.0's default (checked against the trl-1.0.0 wheel), because
TRL 1.14.1 changed some defaults (e.g. max_completion_length 256 -> 512). With per-device batch 1 and generation
batch 4, TRL derives steps_per_generation = 4 and gradient accumulation = 1: each burst is FIVE single-completion
optimizer updates drawn from two generation rounds of four completions, with a fresh AdamW state and a fresh linear
learning-rate schedule (2.0, 1.6, 1.2, 0.8, 0.4 e-5) -- exactly NB-M's mechanics.

TRL-version compensation (documented in the protocol): for this batch shape TRL 1.14.1's "dapo" normaliser is
TRL 1.0.0's multiplied by gradient_accumulation_steps / steps_per_generation = 1/4, so every loss and gradient is 4x
larger. AdamW is invariant to a constant gradient scale (eps aside) but gradient clipping is not, so max_grad_norm is
multiplied by the same factor; the clipped AdamW updates are then identical to TRL 1.0.0's. Logged losses stay 4x the
1.0.0 values; `loss_trl100` divides them back for comparison with the NB-M log.

Persistent-adapter hygiene: beta = 0 means no "ref" adapter (asserted); the input-require-grads hooks every trainer
registers are removed afterwards; sieve/eval run through `modeling.generation_mode`.
"""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import torch

from ..artifacts import JsonlWriter, atomic_write_json
from ..rewards import completion_text, is_truncated_ids, make_reward_funcs, per_completion
from ..train_grpo import JsonlMetricsCallback
from . import legacy
from .data import SgacItem
from .grading import DualGrader
from .model_io import (
    is_quantized,
    lora_delta_norms,
    lora_snapshot,
    lora_state,
    rendered_train_prompt,
    train_prompt,
)
from .spec import SgacSpec

log = logging.getLogger(__name__)

PINNED_KEYS_MIN = ("max_completion_length", "generation_batch_size", "num_generations", "max_steps", "beta",
                   "loss_type", "scale_rewards", "mask_truncated_completions", "temperature", "top_k", "top_p")


def steps_per_generation(spec: SgacSpec) -> int:
    return int(spec.grpo.generation_batch_size) // int(spec.grpo.per_device_train_batch_size)


def dapo_grad_scale(spec: SgacSpec) -> float:
    """Factor by which TRL 1.14.1's per-step dapo loss exceeds TRL 1.0.0's (gradient accumulation is 1)."""
    return float(steps_per_generation(spec)) if spec.grpo.loss_type == "dapo" else 1.0


def reward_weights(spec: SgacSpec) -> list[float]:
    # as_run: NB-M's [binary_match_reward, format_reward] (format already returns 0.5) with TRL's equal weights;
    # e0: rewards.make_reward_funcs' [correctness, format] (format returns 1.0) weighted [1, 0.5].
    return [1.0, 1.0] if spec.profile.grader == "legacy" else [1.0, float(spec.grpo.format_weight)]


def grpo_kwargs(spec: SgacSpec, out_dir: str | Path, seed: int, stop_ids: list[int], cuda: bool | None = None) -> dict:
    g, p = spec.grpo, spec.profile
    cuda = torch.cuda.is_available() if cuda is None else bool(cuda)
    return dict(
        output_dir=str(out_dir), learning_rate=float(g.learning_rate),
        per_device_train_batch_size=int(g.per_device_train_batch_size),
        generation_batch_size=int(g.generation_batch_size), num_generations=int(g.num_generations),
        max_steps=int(g.max_steps), logging_steps=int(g.logging_steps), save_strategy="no", report_to="none",
        max_completion_length=int(p.grpo_max_completion_length),
        temperature=float(g.temperature), top_p=float(g.top_p), top_k=int(g.top_k), min_p=None, repetition_penalty=1.0,
        generation_kwargs=({"eos_token_id": [int(s) for s in stop_ids]} if p.stops == "style" else None),
        beta=float(g.beta), num_iterations=int(g.num_iterations), epsilon=float(g.epsilon), loss_type=g.loss_type,
        scale_rewards=g.scale_rewards, multi_objective_aggregation="sum_then_normalize",
        importance_sampling_level="token", reward_weights=reward_weights(spec),
        mask_truncated_completions=bool(g.mask_truncated_completions), shuffle_dataset=True, disable_dropout=False,
        remove_unused_columns=False, log_completions=False, use_vllm=False,
        gradient_accumulation_steps=1, gradient_checkpointing=bool(g.gradient_checkpointing),
        bf16=cuda, fp16=False, use_cpu=not cuda,
        lr_scheduler_type=g.lr_scheduler_type, warmup_steps=int(g.warmup_steps),
        optim=(g.optim if cuda else "adamw_torch"), weight_decay=float(g.weight_decay),
        max_grad_norm=float(g.max_grad_norm) * dapo_grad_scale(spec), seed=int(seed), dataloader_num_workers=0,
    )


def build_grpo_config(kwargs: dict):
    """`trl.GRPOConfig(**kwargs)`, refusing (rather than silently dropping) any key this TRL does not know."""
    import dataclasses

    from trl import GRPOConfig

    known = {f.name for f in dataclasses.fields(GRPOConfig)}
    unknown = sorted(k for k in kwargs if k not in known)
    if unknown:
        raise RuntimeError(f"installed TRL GRPOConfig lacks pinned fields {unknown}; refusing to train")
    return GRPOConfig(**kwargs)


def _enum_value(v):
    return getattr(v, "value", v)


def audit_grpo_args(args, spec: SgacSpec) -> dict:
    """The derived values that make a burst NB-M-shaped; raises if any differs."""
    want = {
        "steps_per_generation": steps_per_generation(spec), "gradient_accumulation_steps": 1,
        "generation_batch_size": int(spec.grpo.generation_batch_size), "num_generations": int(spec.grpo.num_generations),
        "per_device_train_batch_size": int(spec.grpo.per_device_train_batch_size), "max_steps": int(spec.grpo.max_steps),
        "max_completion_length": int(spec.profile.grpo_max_completion_length), "beta": float(spec.grpo.beta),
        "loss_type": spec.grpo.loss_type, "scale_rewards": spec.grpo.scale_rewards,
        "lr_scheduler_type": spec.grpo.lr_scheduler_type, "learning_rate": float(spec.grpo.learning_rate),
        "max_grad_norm": float(spec.grpo.max_grad_norm) * dapo_grad_scale(spec), "num_iterations": int(spec.grpo.num_iterations),
        "mask_truncated_completions": bool(spec.grpo.mask_truncated_completions), "top_k": int(spec.grpo.top_k),
    }
    got = {k: _enum_value(getattr(args, k, None)) for k in want}
    bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    if bad:
        raise RuntimeError(f"GRPOConfig is not NB-M-shaped: {bad}")
    return {**got, "dapo_grad_scale_vs_trl100": dapo_grad_scale(spec), "reward_weights": list(args.reward_weights or []),
            "generation_kwargs": getattr(args, "generation_kwargs", None), "bf16": bool(args.bf16),
            "gradient_checkpointing": bool(args.gradient_checkpointing), "seed": int(args.seed)}


def train_dataset(item: SgacItem, spec: SgacSpec, tok):
    from datasets import Dataset

    return Dataset.from_list([{"prompt": train_prompt(item, spec, tok), "answer": item.answer,
                               "solution": item.legacy_solution, "unique_id": item.unique_id}])


# ---------------------------------------------------------------------- reward functions with logging
def make_burst_reward_funcs(spec: SgacSpec, grader: DualGrader, writer: JsonlWriter | None, context: dict,
                            stop_ids: list[int]):
    """[correct, format] reward callables named as TRL logs them, plus a per-completion log with both verdicts."""
    stops = [int(s) for s in stop_ids]
    ctx = dict(context)

    def _log(completions, completion_ids, answer, solution, trainer_state, correct_vals, format_vals):
        if writer is None:
            return
        n = len(completions)
        step = getattr(trainer_state, "global_step", None)
        for i, c in enumerate(completions):
            text = completion_text(c)
            ids = completion_ids[i] if completion_ids is not None and i < len(completion_ids) else None
            truncated = is_truncated_ids(ids, stops)
            gold = per_completion(answer, i, n)
            sol = per_completion(solution, i, n)
            mv = grader.mv.grade(text, str(gold), truncated)
            writer.write({
                "step": step, "optimizer_step": None if step is None else int(step) + 1,
                "reward_correct": float(correct_vals[i]), "reward_format": float(format_vals[i]),
                "mv_correct": bool(mv.correct), "mv_boxed": mv.boxed,
                "legacy_correct": bool(legacy.is_correct(legacy.extract_answer(text), str(sol))),
                "legacy_answer": legacy.extract_answer(text), "truncated": bool(truncated),
                "n_tokens": None if ids is None else len(ids), "last_id": None if not ids else int(ids[-1]),
                "text": text, **ctx,
            })

    if spec.profile.grader == "legacy":
        def binary_match_reward(prompts=None, completions=None, completion_ids=None, solution=None, answer=None,
                                trainer_state=None, **kwargs):
            vals = legacy.binary_match_reward(completions, solution=solution)
            _log(completions, completion_ids, answer, solution, trainer_state, vals, legacy.format_reward(completions))
            return vals

        def format_reward(prompts=None, completions=None, **kwargs):
            return legacy.format_reward(completions)

        binary_match_reward.__name__ = binary_match_reward.__qualname__ = "binary_match_reward"
        format_reward.__name__ = format_reward.__qualname__ = "format_reward"
        return [binary_match_reward, format_reward]

    inner_correct, inner_format = make_reward_funcs(grader.mv, stop_token_ids=stops)

    def correctness(prompts=None, completions=None, completion_ids=None, answer=None, solution=None, trainer_state=None,
                    **kwargs):
        vals = inner_correct(prompts, completions, completion_ids=completion_ids, answer=answer,
                             trainer_state=trainer_state, **kwargs)
        _log(completions, completion_ids, answer, solution, trainer_state, vals, inner_format(prompts, completions))
        return vals

    correctness.__name__ = correctness.__qualname__ = "correctness"
    return [correctness, inner_format]


# ---------------------------------------------------------------------- cleanup
def remove_input_require_grads_hooks(model) -> int:
    """Remove every `make_inputs_require_grads` forward hook (each GRPOTrainer init adds one; transformers'
    `disable_input_require_grads` only knows the most recent). Returns how many were removed."""
    n = 0
    for m in model.modules():
        hooks = getattr(m, "_forward_hooks", None)
        if hooks:
            for hid, fn in list(hooks.items()):
                if getattr(fn, "__name__", "") == "make_inputs_require_grads":
                    del hooks[hid]
                    for extra in ("_forward_hooks_with_kwargs", "_forward_hooks_always_called"):
                        d = getattr(m, extra, None)
                        if isinstance(d, dict):
                            d.pop(hid, None)
                    n += 1
        if getattr(m, "_require_grads_hooks", None):
            m._require_grads_hooks = []
        if "_require_grads_hook" in getattr(m, "__dict__", {}):
            del m.__dict__["_require_grads_hook"]
    return n


def count_input_require_grads_hooks(model) -> int:
    return sum(1 for m in model.modules() for fn in (getattr(m, "_forward_hooks", None) or {}).values()
               if getattr(fn, "__name__", "") == "make_inputs_require_grads")


def ensure_single_adapter(model) -> list[str]:
    """Delete any adapter other than "default" (TRL adds "ref" when beta != 0) and re-activate "default"."""
    extra = [n for n in list((getattr(model, "peft_config", None) or {}).keys()) if n != "default"]
    for name in extra:
        model.delete_adapter(name)
    if extra:
        model.set_adapter("default")
    return extra


def _cuda_peak_gb() -> float | None:
    return round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None


def _cuda_alloc_gb() -> float | None:
    return round(torch.cuda.memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None


# ---------------------------------------------------------------------- the burst
def summarize_log_history(history: list[dict], spec: SgacSpec) -> dict:
    steps = [h for h in history if "loss" in h and "train_runtime" not in h]
    losses = [float(h["loss"]) for h in steps]
    scale = dapo_grad_scale(spec)

    def series(key):
        return [float(h[key]) for h in history if key in h and h[key] is not None]

    reward_keys = sorted({k for h in history for k in h if k.startswith("rewards/") and k.endswith("/mean")})
    return {
        "losses": losses, "loss_trl100": [v / scale for v in losses], "all_zero_loss": bool(losses) and all(v == 0.0 for v in losses),
        "learning_rates": series("learning_rate"), "grad_norms": series("grad_norm"),
        "reward": series("reward"), "reward_std": series("reward_std"), "frac_reward_zero_std": series("frac_reward_zero_std"),
        "clipped_ratio": series("completions/clipped_ratio"), "mean_length": series("completions/mean_length"),
        "rewards_by_func": {k: series(k) for k in reward_keys}, "kl": series("kl"), "entropy": series("entropy"),
    }


def run_burst(model, tok, item: SgacItem, spec: SgacSpec, grader: DualGrader, out_dir: Path, seed: int, context: dict,
              stop_ids: list[int]) -> dict:
    from trl import GRPOTrainer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if getattr(model, "peft_config", None) is None:
        raise TypeError("run_burst expects the persistent PeftModel")
    pre_extra = ensure_single_adapter(model)
    args = build_grpo_config(grpo_kwargs(spec, out_dir / "trainer", seed, stop_ids))
    audit = audit_grpo_args(args, spec)
    rendered = rendered_train_prompt(item, spec, tok)
    atomic_write_json(out_dir / "grpo_args.json", {"audit": audit, "train_prompt": rendered,
                                                   "train_prompt_tokens": len(tok(rendered, add_special_tokens=False)["input_ids"]),
                                                   "all_args": args.to_dict()})
    dataset = train_dataset(item, spec, tok)
    before = lora_state(model)
    snap0 = lora_snapshot(model)
    writer = JsonlWriter(out_dir / "train_rollouts.jsonl", append=False)
    metrics_cb = JsonlMetricsCallback(out_dir / "train_metrics.jsonl", context=context)
    funcs = make_burst_reward_funcs(spec, grader, writer, context, stop_ids)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.train()
    t0 = time.time()
    trainer = None
    history: list[dict] = []
    steps_done = 0
    lora_dtype_in_burst = None
    try:
        trainer = GRPOTrainer(model=model, reward_funcs=funcs, args=args, train_dataset=dataset,
                              processing_class=tok, callbacks=[metrics_cb])
        lora_dtype_in_burst = lora_snapshot(model)["dtype"]  # TRL casts LoRA to bf16 here for a 4-bit base
        trainer.train()
        steps_done = int(trainer.state.global_step)
        history = list(trainer.state.log_history)
    finally:
        writer.close()
        metrics_cb.close()
        if trainer is not None:
            try:
                trainer.accelerator.free_memory()
            except Exception as e:  # pragma: no cover
                log.warning("accelerator.free_memory failed: %r", e)
            del trainer
        hooks_removed = remove_input_require_grads_hooks(model)
        post_extra = ensure_single_adapter(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    wall = time.time() - t0
    summary = {
        "unique_id": item.unique_id, "row": item.row, "seed": int(seed), "steps_done": steps_done,
        "expected_steps": int(spec.grpo.max_steps), "wall_s": round(wall, 2), "peak_mem_gb": _cuda_peak_gb(),
        "alloc_after_cleanup_gb": _cuda_alloc_gb(), "quantized_base": is_quantized(model),
        "lora_dtype_in_burst": lora_dtype_in_burst, "lora_before": snap0, "lora_after": lora_snapshot(model),
        **lora_delta_norms(before, model), "hooks_removed": hooks_removed,
        "extra_adapters_removed": pre_extra + post_extra, **summarize_log_history(history, spec), "context": context,
    }
    atomic_write_json(out_dir / "log_history.json", history)
    atomic_write_json(out_dir / "burst_summary.json", summary)
    if steps_done != int(spec.grpo.max_steps):
        raise RuntimeError(f"burst finished {steps_done} optimizer steps, expected {spec.grpo.max_steps}")
    log.info("burst %s: losses %s, reward %s, zero-std %s, clipped %s, |dA| %.3g |dB| %.3g, %.0fs", item.unique_id,
             [round(v, 4) for v in summary["losses"]], summary["reward"], summary["frac_reward_zero_std"],
             summary["clipped_ratio"], summary["delta_A"], summary["delta_B"], wall)
    return summary
