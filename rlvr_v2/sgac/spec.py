"""SGAC reproduction configuration: its own dataclasses, YAML files (`configs/sgac/`) and hash.

Deliberately separate from `rlvr_v2.config.Config`: the shared schema (and therefore every prereg config hash) is
never touched. The generic helpers `deep_merge`, `parse_overrides` and `dataclass_from_dict` are reused unchanged, so
unknown YAML keys are rejected exactly as in the main config.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..artifacts import REPO_ROOT
from ..config import dataclass_from_dict, deep_merge, parse_overrides

PROFILES = ("e0", "as_run")


@dataclass(frozen=True)
class ModelSpec:
    name: str = "Qwen/Qwen2.5-Math-1.5B"
    revision: str | None = None  # None = the cached main revision, as in E0; the resolved commit is recorded per run
    attn_impl: str = "sdpa"
    pi1_name: str = "ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1"
    pi1_revision: str | None = None


@dataclass(frozen=True)
class LoraSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class ProfileSpec:
    name: str = "e0"
    quant: str = "none"  # none | nf4 (nf4 falls back to bf16 when bitsandbytes cannot load; recorded)
    dtype: str = "bf16"  # weights of a non-quantised model
    compute_dtype: str = "fp16"  # nf4 compute dtype (NB-M: float16)
    pi1_dtype: str = "bf16"  # NB-M evaluated pi1 in float16
    prompt: str = "oneshot_rlvr_chat"  # oneshot_rlvr_chat | legacy (raw NB-M instruction, no chat template)
    train_prompt: str = "rendered"  # rendered (string; TRL never re-templates) | conversational (TRL applies chat template)
    tokenizer_pad: str = "fim_pad"  # fim_pad (prompts.configure_tokenizer) | eos (NB-M: pad = eos)
    stops: str = "style"  # style (<|endoftext|> + <|im_end|>) | eos_only (NB-M generation_config)
    grader: str = "math_verify"  # math_verify | legacy  -- decides rewards, Ps and the primary eval verdict
    d_metric: str = "u_ratio"  # u_ratio (math-verify classes / K) | legacy_str (str(extract_answer) classes / K)
    max_prompt_tokens: int = 2048  # refuse-not-truncate guard; one original pool item has 1379 prompt tokens
    sieve_max_new_tokens: int = 3072
    eval_max_new_tokens: int = 3072
    grpo_max_completion_length: int = 3072
    sieve_top_k: int = 0
    sieve_batch_size: int = 16
    eval_batch_size: int = 32
    eval_prompts_per_call: int = 64  # mirrors evaluate.evaluate (E0 G1) so greedy batches are composed identically
    test50_batch_check: bool = False  # as_run: run base test50 at batch 1 and eval_batch_size, keep batch 1 if < 98% agree


@dataclass(frozen=True)
class DataSpec:
    dataset: str = "nlile/hendrycks-MATH-benchmark"
    revision: str = "465bcdb36f5962aa3512891498966df785fc3c18"
    split: str = "train"
    shuffle_seed: int = 42
    pool: tuple[int, int] = (0, 1000)
    test: tuple[int, int] = (1000, 1050)
    phase1_candidates: tuple[int, int] = (0, 4)
    phase1_test: tuple[int, int] = (4, 14)
    math500_manifest: str = "manifests/math500.json"
    manifest_dir: str = "configs/sgac/manifests"


@dataclass(frozen=True)
class LoopSpec:
    steps: int = 20
    batch_b: int = 4
    k: int = 4
    temperature: float = 1.0
    top_p: float = 1.0
    eval_every: int = 5
    math500_steps: tuple[int, ...] = (0, 20)  # () disables MATH-500 everywhere (CPU smoke)
    eval_max_items: int | None = None  # smoke only: truncate every eval set


@dataclass(frozen=True)
class GrpoSpec:
    """NB-M's GRPOConfig: the explicit values plus TRL 1.0.0's defaults for everything it left unset (verified against
    the trl-1.0.0 wheel; see paper/sgac_repro_protocol.md). `max_completion_length` lives in the profile."""

    learning_rate: float = 2.0e-5
    per_device_train_batch_size: int = 1
    generation_batch_size: int = 4
    num_generations: int = 4
    max_steps: int = 5
    logging_steps: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    beta: float = 0.0
    num_iterations: int = 1
    epsilon: float = 0.2
    loss_type: str = "dapo"
    scale_rewards: str = "group"
    mask_truncated_completions: bool = False
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 0
    optim: str = "adamw_torch_fused"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0  # TRL 1.0.0 semantics; burst.py rescales it for TRL 1.14.1's dapo normaliser
    gradient_checkpointing: bool = True
    format_weight: float = 0.5  # reward = correct + format_weight * boxed


@dataclass(frozen=True)
class RunSpec:
    results_root: str = "results_sgac"
    seeds: tuple[int, ...] = (42, 43, 44)
    arms: tuple[str, ...] = ("sgac", "random", "max_var", "max_d", "max_level")


@dataclass(frozen=True)
class SgacSpec:
    profile: ProfileSpec = field(default_factory=ProfileSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    lora: LoraSpec = field(default_factory=LoraSpec)
    data: DataSpec = field(default_factory=DataSpec)
    loop: LoopSpec = field(default_factory=LoopSpec)
    grpo: GrpoSpec = field(default_factory=GrpoSpec)
    run: RunSpec = field(default_factory=RunSpec)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def spec_hash(self) -> str:
        """Identity of everything that changes results (excludes where results go, and which seeds/arms are queued)."""
        d = self.to_dict()
        d["run"] = {}
        blob = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:8]

    def validate(self) -> None:
        p, lp, g = self.profile, self.loop, self.grpo
        errors = []
        if p.name not in PROFILES:
            errors.append(f"profile.name must be one of {PROFILES}")
        for key, allowed in (("quant", ("none", "nf4")), ("prompt", ("oneshot_rlvr_chat", "legacy")),
                             ("train_prompt", ("rendered", "conversational")), ("tokenizer_pad", ("fim_pad", "eos")),
                             ("stops", ("style", "eos_only")), ("grader", ("math_verify", "legacy")),
                             ("d_metric", ("u_ratio", "legacy_str"))):
            if getattr(p, key) not in allowed:
                errors.append(f"profile.{key}={getattr(p, key)!r} not in {allowed}")
        if g.generation_batch_size % g.num_generations or g.generation_batch_size % g.per_device_train_batch_size:
            errors.append("grpo.generation_batch_size must be a multiple of num_generations and the per-device batch")
        if lp.batch_b < 1 or lp.k < 2 or lp.steps < 1:
            errors.append("loop.batch_b >= 1, loop.k >= 2, loop.steps >= 1 required")
        if any(s < 0 or s > lp.steps for s in lp.math500_steps):
            errors.append("loop.math500_steps must lie in [0, steps]")
        from .selection import ARMS

        bad = [a for a in self.run.arms if a not in ARMS]
        if bad:
            errors.append(f"unknown arms {bad}; known {ARMS}")
        if errors:
            raise ValueError("Invalid SGAC spec:\n  - " + "\n  - ".join(errors))


CONFIG_DIR = REPO_ROOT / "configs" / "sgac"


def profile_files(profile: str) -> list[Path]:
    if profile not in PROFILES and not Path(profile).exists():
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES} or a YAML path")
    extra = Path(profile) if Path(profile).exists() else CONFIG_DIR / f"{profile}.yaml"
    return [CONFIG_DIR / "base.yaml", extra]


def load_spec(paths: list[str | Path], overrides: list[str] | None = None) -> SgacSpec:
    merged: dict = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            merged = deep_merge(merged, yaml.safe_load(fh) or {})
    merged = deep_merge(merged, parse_overrides(overrides))
    spec = dataclass_from_dict(SgacSpec, merged)
    spec.validate()
    return spec


def load_profile(profile: str, overrides: list[str] | None = None, extra_files: list[str | Path] | None = None) -> SgacSpec:
    return load_spec(profile_files(profile) + [Path(f) for f in (extra_files or [])], overrides)


def save_spec(spec: SgacSpec, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(json.loads(json.dumps(spec.to_dict(), default=str)), fh, sort_keys=False)


def results_root(spec: SgacSpec) -> Path:
    root = Path(spec.run.results_root)
    return root if root.is_absolute() else REPO_ROOT / root


def group_dir(spec: SgacSpec) -> Path:
    """results_sgac/<profile>/<spec_hash>/ -- every run and shared eval of one configuration."""
    return results_root(spec) / spec.profile.name / spec.spec_hash()
