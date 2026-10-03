"""Model and tokenizer loading, LoRA attachment, adapter IO and a safe generation context.

Rules
- The tokenizer is always configured through `prompts.configure_tokenizer` (eos = primary stop,
  distinct pad, left padding) so sieve, training and evaluation share stop tokens.
- transformers 5.x: `from_pretrained(..., dtype=...)` (the old `torch_dtype=` is deprecated).
- CPU-only machines load in float32 with `device_map=None` regardless of the config so smoke
  tests run anywhere; nf4 quantisation is skipped (with a warning) when CUDA is unavailable.
"""
from __future__ import annotations

import contextlib
import functools
import gc
import logging
from pathlib import Path
from typing import Any, Iterator

import torch

from . import prompts
from .artifacts import atomic_write_json, package_versions, read_json, utc_now
from .config import Config

log = logging.getLogger(__name__)

ADAPTER_META_FILE = "adapter_meta.json"

_DTYPES: dict[str, torch.dtype] = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def torch_dtype(name: str) -> torch.dtype:
    """Map the config dtype string (bf16 | fp16 | fp32) to a torch dtype."""
    try:
        return _DTYPES[name]
    except KeyError as e:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(_DTYPES)}") from e


# ---------------------------------------------------------------------- tokenizer / base model
def load_tokenizer(cfg: Config):
    """`AutoTokenizer` for `cfg.model.name`, configured for `cfg.prompt.style` (eos, pad, left padding)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    n_before = len(tok)
    prompts.configure_tokenizer(tok, cfg.prompt.style)
    if len(tok) != n_before:
        log.warning(
            "configure_tokenizer added %d token(s) to %s; the model embeddings must be resized before use",
            len(tok) - n_before, cfg.model.name,
        )
    log.info(
        "tokenizer %s: eos=%r(%s) pad=%r(%s) padding_side=%s stop_ids=%s",
        cfg.model.name, tok.eos_token, tok.eos_token_id, tok.pad_token, tok.pad_token_id, tok.padding_side,
        prompts.resolve_stop_token_ids(tok, cfg.prompt.style),
    )
    return tok


def load_base_model(cfg: Config, device_map: str | dict | None = "auto"):
    """Load the causal LM named in the config.

    - CUDA available: `dtype` from `cfg.model.dtype`, `attn_implementation=cfg.model.attn_impl`,
      the given `device_map`, and a bitsandbytes nf4 config when `cfg.model.quant == "nf4"`.
    - No CUDA: float32, `device_map=None`, no quantisation (smoke-test mode), with warnings.
    """
    from transformers import AutoModelForCausalLM

    cuda = torch.cuda.is_available()
    requested = torch_dtype(cfg.model.dtype)
    dtype = requested if cuda else torch.float32
    kwargs: dict[str, Any] = {"dtype": dtype, "attn_implementation": cfg.model.attn_impl}
    if cuda:
        kwargs["device_map"] = device_map
    else:
        kwargs["device_map"] = None
        if requested != torch.float32:
            log.warning("no CUDA device: loading %s in float32 instead of %s", cfg.model.name, cfg.model.dtype)

    if cfg.model.quant == "nf4":
        if not cuda:
            log.warning("no CUDA device: ignoring model.quant=nf4 and loading full-precision weights")
        else:
            try:
                import bitsandbytes  # noqa: F401  (import check only)
                from transformers import BitsAndBytesConfig
            except ImportError as e:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "model.quant='nf4' requires the `bitsandbytes` package (pip install bitsandbytes)"
                ) from e
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
    elif cfg.model.quant != "none":
        raise ValueError(f"unknown model.quant {cfg.model.quant!r}")

    log.info("loading %s (%s)", cfg.model.name, {k: v for k, v in kwargs.items() if k != "quantization_config"})
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **kwargs)
    model.config.use_cache = True
    log.info("loaded %s on %s with %s", cfg.model.name, device_of(model), next(model.parameters()).dtype)
    return model


# ---------------------------------------------------------------------- LoRA
def trainable_param_counts(model) -> dict[str, float]:
    """{"trainable", "total", "fraction"} parameter counts (4-bit packed weights are un-packed like PEFT does)."""
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        trainable = total = 0
        for p in model.parameters():
            n = p.numel()
            if n == 0 and hasattr(p, "ds_numel"):  # DeepSpeed zero-3 partitioned
                n = p.ds_numel
            if p.__class__.__name__ == "Params4bit":
                if hasattr(p, "element_size"):
                    n_bytes = p.element_size()
                elif hasattr(p, "quant_storage"):
                    n_bytes = p.quant_storage.itemsize
                else:
                    n_bytes = 1
                n = n * 2 * n_bytes
            total += n
            if p.requires_grad:
                trainable += n
    return {"trainable": int(trainable), "total": int(total), "fraction": (trainable / total) if total else 0.0}


def attach_fresh_lora(model, cfg: Config):
    """Wrap `model` in a fresh LoRA adapter built from `cfg.model.lora` and return the PeftModel.

    nf4 base models are first passed through `prepare_model_for_kbit_training`. Trainable-parameter
    counts are logged; call `trainable_param_counts(peft_model)` to get them as a dict.
    """
    from peft import LoraConfig, get_peft_model

    lora = cfg.model.lora
    if cfg.model.quant == "nf4" and getattr(model, "is_loaded_in_4bit", False):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.train.gradient_checkpointing)
    lcfg = LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lcfg)
    counts = trainable_param_counts(peft_model)
    log.info(
        "LoRA r=%d alpha=%d dropout=%s targets=%s: trainable %s / %s params (%.4f%%)",
        lora.r, lora.alpha, lora.dropout, list(lora.target_modules), f"{counts['trainable']:,}",
        f"{counts['total']:,}", 100.0 * counts["fraction"],
    )
    return peft_model


def load_adapter(base_model, adapter_dir: str | Path):
    """Attach a saved (frozen) LoRA adapter to `base_model`."""
    from peft import PeftModel

    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter directory not found: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    meta = load_adapter_meta(adapter_dir)
    log.info("loaded adapter %s (meta: %s)", adapter_dir, meta or "none")
    return model


def load_adapter_meta(adapter_dir: str | Path) -> dict | None:
    return read_json(Path(adapter_dir) / ADAPTER_META_FILE, None)


def save_adapter(model, out_dir: str | Path, meta: dict | None = None) -> Path:
    """`model.save_pretrained(out_dir)` plus `adapter_meta.json` (prompt style / prompt hash / anything in `meta`)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    doc = {"saved_utc": utc_now(), "versions": package_versions(), **(meta or {})}
    atomic_write_json(out / ADAPTER_META_FILE, doc)
    log.info("saved adapter to %s", out)
    return out


# ---------------------------------------------------------------------- generation context
def _gradient_checkpointing_kwargs(model) -> dict | None:
    """Best-effort recovery of the kwargs gradient checkpointing was enabled with (e.g. use_reentrant)."""
    try:
        for module in model.modules():
            fn = getattr(module, "_gradient_checkpointing_func", None)
            if isinstance(fn, functools.partial):
                return dict(fn.keywords or {})
    except Exception:  # pragma: no cover - purely defensive
        pass
    return None


@contextlib.contextmanager
def generation_mode(model) -> Iterator[Any]:
    """Eval mode + KV cache + no gradient checkpointing + `torch.inference_mode()`; everything is restored on exit."""
    was_training = bool(getattr(model, "training", False))
    gc_was_enabled = bool(getattr(model, "is_gradient_checkpointing", False))
    gc_kwargs = _gradient_checkpointing_kwargs(model) if gc_was_enabled else None
    config = getattr(model, "config", None)
    prev_use_cache = getattr(config, "use_cache", None) if config is not None else None

    model.eval()
    if gc_was_enabled:
        disable = getattr(model, "gradient_checkpointing_disable", None)
        if callable(disable):
            disable()
    if config is not None:
        config.use_cache = True
    try:
        with torch.inference_mode():
            yield model
    finally:
        if config is not None and prev_use_cache is not None:
            config.use_cache = prev_use_cache
        if gc_was_enabled:
            enable = getattr(model, "gradient_checkpointing_enable", None)
            if callable(enable):
                if gc_kwargs:
                    enable(gradient_checkpointing_kwargs=gc_kwargs)
                else:
                    enable()
        model.train(was_training)


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def device_of(model) -> torch.device:
    dev = getattr(model, "device", None)
    if isinstance(dev, torch.device):
        return dev
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - parameter-less module
        return torch.device("cpu")
