"""Configuration dataclasses, YAML loading, validation and TRL GRPOConfig construction.

Design rule: `GenerationCfg` is the single owner of sequence lengths and sampling
parameters for sieve, training and evaluation. `TrainCfg` deliberately has no length
fields; `Config.grpo_kwargs()` copies them from `GenerationCfg`.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1


@dataclass(frozen=True)
class GenerationCfg:
    """Single source of truth for lengths and decoding, shared by sieve, train and eval."""

    max_prompt_tokens: int = 1024
    max_new_tokens: int = 3072
    sieve: SamplingParams = field(default_factory=lambda: SamplingParams(temperature=1.0, n=32))
    eval: SamplingParams = field(default_factory=lambda: SamplingParams(temperature=0.0, n=1))
    backend: str = "hf"  # "hf" | "vllm"
    hf_batch_size: int = 32
    vllm_gpu_memory_utilization: float = 0.85


@dataclass(frozen=True)
class LoraCfg:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class ModelCfg:
    name: str = "Qwen/Qwen2.5-Math-1.5B"
    dtype: str = "bf16"  # bf16 | fp16 | fp32
    quant: str = "none"  # none | nf4
    attn_impl: str = "sdpa"
    lora: LoraCfg = field(default_factory=LoraCfg)


@dataclass(frozen=True)
class PromptCfg:
    style: str = "qwen_math_chat"  # qwen_math_chat | oneshot_rlvr_chat | raw
    system: str = DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True)
class GuardCfg:
    max_clipped_ratio: float = 0.20
    check_steps: int = 3
    preflight: bool = True


@dataclass(frozen=True)
class TrainCfg:
    """GRPO settings. One optimizer step == one generation round of `num_generations` completions."""

    rounds: int = 100
    num_generations: int = 64
    per_device_train_batch_size: int = 8
    learning_rate: float = 2.0e-5
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 0
    beta: float = 0.0
    loss_type: str = "dapo"
    scale_rewards: str = "group"
    epsilon: float = 0.2
    num_iterations: int = 1
    mask_truncated_completions: bool = True
    gradient_checkpointing: bool = True
    reward_weights: tuple[tuple[str, float], ...] = (("correctness", 1.0), ("format", 0.0))
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.35
    save_rounds: int = 25
    logging_steps: int = 1
    guard: GuardCfg = field(default_factory=GuardCfg)

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.num_generations // self.per_device_train_batch_size

    @property
    def generation_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @property
    def prompts_per_round(self) -> int:
        return self.generation_batch_size // self.num_generations


@dataclass(frozen=True)
class DataCfg:
    train_dataset: str = "nlile/hendrycks-MATH-benchmark"
    eval_dataset: str = "HuggingFaceH4/MATH-500"
    shuffle_seed: int = 20260101
    pool_size: int = 500
    heldout_size: int = 500
    near_dup_ratio: float = 0.9
    # The nlile "train" split also packages the non-MATH-500 part of the MATH test set (ids "test/...").
    # Keep only genuine MATH train problems so "candidates from the train split" is literally true.
    id_prefix: str | None = "train/"


@dataclass(frozen=True)
class EvalCfg:
    manifest: str = "manifests/math500.json"
    max_items: int | None = None
    bootstrap_samples: int = 2000
    store_text: bool = True


@dataclass(frozen=True)
class SelectorCfg:
    name: str = "random"  # random | variance | disagreement | ps_band | learned
    disagreement_metric: str = "d_simpson"  # d_simpson | entropy_bits | u_ratio
    ps_lo: float = 0.2
    ps_hi: float = 0.8
    ps_target: float = 0.5
    learned_path: str | None = None


@dataclass(frozen=True)
class Study1Cfg:
    k_sieve: int = 32
    pool_manifest: str = "manifests/pool.json"
    candidates_manifest: str = "manifests/study1_candidates.json"
    heldout_manifest: str = "manifests/heldout.json"
    pairs_per_bin: tuple[int, ...] = (4, 6, 4)
    bins: tuple[tuple[float, float], ...] = ((0.0, 0.25), (0.25, 0.75), (0.75, 0.9))
    n_null_anchors: int = 2  # P_s == 1
    n_zero_anchors: int = 2  # P_s == 0
    max_ps_diff: float = 2.0 / 32.0
    min_dwrong_gap: float = 0.4
    replicate_count: int = 8
    replicate_seed_offset: int = 100000
    seed_base: int = 1000


@dataclass(frozen=True)
class Study2Cfg:
    steps: int = 20
    batch_b: int = 16
    k: int = 32  # must equal gen.sieve.n; configs/study2.yaml sets both to 16
    rounds_per_burst: int = 10
    eval_every: int = 4
    arms: tuple[str, ...] = ("random", "variance", "disagreement")
    seeds: tuple[int, ...] = (1234, 2345, 3456)
    pool_manifest: str = "manifests/pool.json"
    repeat_one_uid: str | None = None


@dataclass(frozen=True)
class RunCfg:
    results_root: str = "results"
    tag: str = "dev"
    seed: int = 0
    require_gates: bool = True


@dataclass(frozen=True)
class Config:
    run: RunCfg = field(default_factory=RunCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    prompt: PromptCfg = field(default_factory=PromptCfg)
    gen: GenerationCfg = field(default_factory=GenerationCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    data: DataCfg = field(default_factory=DataCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    selector: SelectorCfg = field(default_factory=SelectorCfg)
    study1: Study1Cfg = field(default_factory=Study1Cfg)
    study2: Study2Cfg = field(default_factory=Study2Cfg)

    # TRL >= 1.x dropped `max_prompt_length`; prompt length is enforced by our own sampler/dataset code.
    REQUIRED_GRPO_KEYS: typing.ClassVar[tuple[str, ...]] = (
        "max_completion_length", "num_generations", "temperature", "mask_truncated_completions",
    )

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        errors: list[str] = []
        g, t, m = self.gen, self.train, self.model
        if g.max_new_tokens <= 0 or g.max_prompt_tokens <= 0:
            errors.append("gen lengths must be positive")
        if g.backend not in ("hf", "vllm"):
            errors.append(f"gen.backend must be hf|vllm, got {g.backend}")
        if g.eval.temperature != 0.0 or g.eval.n != 1:
            errors.append("gen.eval must be greedy (temperature 0, n 1)")
        if g.sieve.n < 2:
            errors.append("gen.sieve.n (K) must be >= 2")
        if t.num_generations < 2:
            errors.append("train.num_generations must be >= 2")
        if t.per_device_train_batch_size <= 0 or t.num_generations % t.per_device_train_batch_size != 0:
            errors.append(
                "train.num_generations must be a positive multiple of train.per_device_train_batch_size "
                f"({t.num_generations} vs {t.per_device_train_batch_size})"
            )
        if t.rounds <= 0:
            errors.append("train.rounds must be positive")
        names = [n for n, _ in t.reward_weights]
        if names != ["correctness", "format"]:
            errors.append(f"train.reward_weights must be (correctness, format) in that order, got {names}")
        if m.dtype not in ("bf16", "fp16", "fp32"):
            errors.append(f"model.dtype invalid: {m.dtype}")
        if m.quant not in ("none", "nf4"):
            errors.append(f"model.quant invalid: {m.quant}")
        if m.quant == "nf4" and (t.use_vllm or g.backend == "vllm"):
            errors.append("vLLM cannot be used with a 4-bit base model")
        if self.prompt.style not in ("qwen_math_chat", "oneshot_rlvr_chat", "raw"):
            errors.append(f"prompt.style invalid: {self.prompt.style}")
        if self.selector.name not in ("random", "variance", "disagreement", "ps_band", "learned"):
            errors.append(f"selector.name invalid: {self.selector.name}")
        if self.selector.name == "learned" and not self.selector.learned_path:
            errors.append("selector.learned_path is required for the learned selector")
        if self.study2.k != g.sieve.n:
            errors.append(
                f"study2.k ({self.study2.k}) must equal gen.sieve.n ({g.sieve.n}) so curriculum signals use the shared config"
            )
        if any(a not in ("random", "variance", "disagreement", "ps_band", "learned", "repeat_one") for a in self.study2.arms):
            errors.append(f"study2.arms contains an unknown arm: {self.study2.arms}")
        if len(self.study1.pairs_per_bin) != len(self.study1.bins):
            errors.append("study1.pairs_per_bin and study1.bins must have the same length")
        if errors:
            raise ValueError("Invalid config:\n  - " + "\n  - ".join(errors))

    # ------------------------------------------------------------------ TRL
    def grpo_kwargs(self, output_dir: str | Path, max_steps: int, seed: int,
                    stop_token_ids: list[int] | None = None) -> dict[str, Any]:
        """Keyword arguments for `trl.GRPOConfig`; lengths come from `gen`, never from `train`.
        `stop_token_ids` (primary first) are passed to HF generation so chat-style turns terminate."""
        t, g, m = self.train, self.gen, self.model
        gen_kwargs = {"eos_token_id": list(stop_token_ids)} if stop_token_ids else None
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover
            cuda = False
        return dict(
            output_dir=str(output_dir),
            max_steps=int(max_steps),
            seed=int(seed),
            learning_rate=t.learning_rate,
            lr_scheduler_type=t.lr_scheduler_type,
            warmup_steps=t.warmup_steps,
            per_device_train_batch_size=t.per_device_train_batch_size,
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            steps_per_generation=t.gradient_accumulation_steps,
            num_generations=t.num_generations,
            max_completion_length=g.max_new_tokens,
            temperature=g.sieve.temperature,
            top_p=g.sieve.top_p,
            generation_kwargs=gen_kwargs,
            beta=t.beta,
            loss_type=t.loss_type,
            scale_rewards=t.scale_rewards,
            epsilon=t.epsilon,
            num_iterations=t.num_iterations,
            mask_truncated_completions=t.mask_truncated_completions,
            gradient_checkpointing=t.gradient_checkpointing,
            bf16=(m.dtype == "bf16") and cuda,
            fp16=(m.dtype == "fp16") and cuda,
            use_cpu=not cuda,
            logging_steps=t.logging_steps,
            save_strategy="steps",
            save_steps=t.save_rounds,
            save_total_limit=1,
            report_to="none",
            use_vllm=t.use_vllm,
            vllm_mode=t.vllm_mode,
            vllm_gpu_memory_utilization=t.vllm_gpu_memory_utilization,
            reward_weights=[float(w) for _, w in t.reward_weights],
            disable_dropout=True,
            log_completions=False,
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )

    def to_grpo_config(self, output_dir: str | Path, max_steps: int, seed: int,
                       stop_token_ids: list[int] | None = None):
        """Build a `trl.GRPOConfig`, dropping kwargs the installed TRL does not know (with a warning),
        but refusing to drop the length/generation keys."""
        import warnings

        from trl import GRPOConfig  # imported lazily so CPU tests do not need TRL

        known = {f.name for f in dataclasses.fields(GRPOConfig)}
        kwargs = self.grpo_kwargs(output_dir, max_steps, seed, stop_token_ids)
        dropped = sorted(k for k in kwargs if k not in known)
        missing_required = [k for k in self.REQUIRED_GRPO_KEYS if k not in known]
        if missing_required:
            raise RuntimeError(f"Installed TRL GRPOConfig lacks required fields: {missing_required}")
        if dropped:
            warnings.warn(f"GRPOConfig: dropping unsupported kwargs for this TRL version: {dropped}")
        return GRPOConfig(**{k: v for k, v in kwargs.items() if k in known})

    # ------------------------------------------------------------------ hashing / io
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        d = self.to_dict()
        d["run"] = {k: v for k, v in d["run"].items() if k not in ("seed", "tag", "results_root")}
        blob = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:8]


# ---------------------------------------------------------------------- loading helpers
def _resolve_hint(hint: Any):
    """Return (dataclass_type or None, is_tuple) for a type hint, unwrapping Optional/Union."""
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        for arg in typing.get_args(hint):
            dc, is_tuple = _resolve_hint(arg)
            if dc is not None or is_tuple:
                return dc, is_tuple
        return None, False
    if origin is tuple:
        return None, True
    if is_dataclass(hint):
        return hint, False
    return None, False


def _to_tuple(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return tuple(_to_tuple(x) for x in v)
    return v


def dataclass_from_dict(cls, d: Mapping[str, Any]):
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - names)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {unknown}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        dc, is_tuple = _resolve_hint(hints[f.name])
        if dc is not None and isinstance(v, Mapping):
            kwargs[f.name] = dataclass_from_dict(dc, v)
        elif is_tuple:
            kwargs[f.name] = _to_tuple(v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


def deep_merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


_FLOAT_LIKE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)$")


def parse_overrides(overrides: list[str] | None) -> dict:
    """`a.b.c=value` strings -> nested dict with YAML-typed values (PyYAML reads `5e-5` as a string,
    so exponent-form numbers are coerced to float explicitly)."""
    out: dict = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        if isinstance(value, str) and _FLOAT_LIKE.match(value):
            value = float(value)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def load_config(paths: list[str | Path] | str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Merge YAML files left-to-right on top of dataclass defaults, then apply dotted overrides."""
    if paths is None:
        paths = []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    merged: dict = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            merged = deep_merge(merged, yaml.safe_load(fh) or {})
    merged = deep_merge(merged, parse_overrides(overrides))
    cfg = dataclass_from_dict(Config, merged)
    cfg.validate()
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(json.loads(json.dumps(cfg.to_dict(), default=str)), fh, sort_keys=False)
