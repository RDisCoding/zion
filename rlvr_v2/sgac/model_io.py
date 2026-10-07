"""Tokenizer, prompts, model loading and adapter IO for the two profiles.

as_run follows NB-M cell 5 literally: `AutoTokenizer` with pad = eos (<|endoftext|>), a bitsandbytes NF4 base
(double quant, float16 compute, everything else at transformers' default dtype) and a plain `get_peft_model` LoRA
(no `prepare_model_for_kbit_training`, which `modeling.attach_fresh_lora` would call). e0 uses the E0 conventions
(`prompts.configure_tokenizer`, bf16 weights). TRL 1.0.0 and 1.14.1 both cast trainable LoRA weights of a 4-bit
model to bf16 at every trainer init; that behaviour is kept (faithful) and recorded per burst.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from .. import prompts
from ..curriculum import Curriculum
from ..modeling import device_of, save_adapter, trainable_param_counts
from . import legacy
from .data import SgacItem
from .spec import SgacSpec

log = logging.getLogger(__name__)

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# ---------------------------------------------------------------------- tokenizer / prompts
def load_tokenizer(spec: SgacSpec, name: str | None = None, revision: str | None = None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name or spec.model.name, revision=revision or spec.model.revision)
    if spec.profile.tokenizer_pad == "fim_pad":
        prompts.configure_tokenizer(tok, "oneshot_rlvr_chat")
    else:  # NB-M cell 5: `if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token`
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"  # NB-M generated at batch size 1; left padding makes batching equivalent
    return tok


def stop_token_ids(spec: SgacSpec, tok) -> list[int]:
    if spec.profile.stops == "style":
        return prompts.resolve_stop_token_ids(tok, "oneshot_rlvr_chat")
    return [int(tok.eos_token_id)]


def render_prompt(item: SgacItem, spec: SgacSpec, tok) -> str:
    """The prompt string used for sieve and evaluation (and, for e0, training)."""
    if spec.profile.prompt == "legacy":
        return legacy.legacy_prompt(item.problem)
    return prompts.build_prompt(item.problem, "oneshot_rlvr_chat", tok)


def train_prompt(item: SgacItem, spec: SgacSpec, tok) -> Any:
    """The dataset `prompt` for GRPO: a one-message conversation (as_run; TRL applies the chat template, as NB-M) or
    the pre-rendered string (e0; TRL never re-templates a string)."""
    if spec.profile.train_prompt == "conversational":
        text = legacy.legacy_prompt(item.problem) if spec.profile.prompt == "legacy" else item.problem
        return [{"role": "user", "content": text}]
    return render_prompt(item, spec, tok)


def rendered_train_prompt(item: SgacItem, spec: SgacSpec, tok) -> str:
    """What the policy actually sees during GRPO (for logging and prompt hashes)."""
    p = train_prompt(item, spec, tok)
    if isinstance(p, str):
        return p
    return tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------- models
def _bnb_config(spec: SgacSpec):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=_DTYPES[spec.profile.compute_dtype],
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")


def load_policy_model(spec: SgacSpec) -> tuple[Any, dict]:
    """Base model for the profile -> (model, load_record). nf4 falls back to bf16 (recorded) if bitsandbytes is
    missing or cannot load on this GPU; CPU (tests) always loads float32."""
    from transformers import AutoModelForCausalLM

    p = spec.profile
    record: dict = {"requested_quant": p.quant, "requested_dtype": p.dtype, "fallback": None}
    kwargs: dict = {"revision": spec.model.revision, "attn_implementation": spec.model.attn_impl}
    if not torch.cuda.is_available():
        kwargs.update(dtype=torch.float32, device_map=None)
        record["effective"] = "fp32-cpu"
    elif p.quant == "nf4":
        try:
            import bitsandbytes

            model = AutoModelForCausalLM.from_pretrained(spec.model.name, device_map="auto", dtype="auto",
                                                         quantization_config=_bnb_config(spec), **kwargs)
            record.update(effective="nf4", bitsandbytes=getattr(bitsandbytes, "__version__", "?"),
                          compute_dtype=p.compute_dtype)
            return _finish_load(model, record, spec.model.name, spec.model.revision)
        except Exception as e:  # noqa: BLE001 - any failure means the profile's documented fallback
            record["fallback"] = f"nf4 unavailable ({type(e).__name__}: {str(e)[:300]}); loaded bf16"
            log.warning("as_run: %s", record["fallback"])
            torch.cuda.empty_cache()
        kwargs.update(dtype=torch.bfloat16, device_map="auto")
        record["effective"] = "bf16"
    else:
        kwargs.update(dtype=_DTYPES[p.dtype], device_map="auto")
        record["effective"] = p.dtype
    model = AutoModelForCausalLM.from_pretrained(spec.model.name, **kwargs)
    return _finish_load(model, record, spec.model.name, spec.model.revision)


def cached_snapshot(repo_id: str, revision: str | None) -> str | None:
    """Commit hash of the locally cached snapshot that was loaded (model and tokenizer files come from it);
    transformers 5.x no longer stamps `_commit_hash` on the config when loading offline."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id, revision=revision, local_files_only=True)).name
    except Exception:  # noqa: BLE001 - provenance only, never fatal
        return None


def _finish_load(model, record: dict, name: str, revision: str | None):
    model.config.use_cache = True
    record["commit"] = getattr(model.config, "_commit_hash", None) or cached_snapshot(name, revision)
    record["param_dtype"] = str(next(model.parameters()).dtype)
    record["device"] = str(device_of(model))
    log.info("loaded policy base: %s", record)
    return model, record


def load_pi1_model(spec: SgacSpec):
    """Wang et al.'s pi1 checkpoint (full fine-tune). NB-M loaded it in float16; e0 uses bf16."""
    from transformers import AutoModelForCausalLM

    if torch.cuda.is_available():
        kw = {"dtype": _DTYPES[spec.profile.pi1_dtype], "device_map": "auto"}
    else:
        kw = {"dtype": torch.float32, "device_map": None}
    model = AutoModelForCausalLM.from_pretrained(spec.model.pi1_name, revision=spec.model.pi1_revision,
                                                 attn_implementation=spec.model.attn_impl, **kw)
    model.config.use_cache = True
    commit = getattr(model.config, "_commit_hash", None) or cached_snapshot(spec.model.pi1_name, spec.model.pi1_revision)
    return model, {"name": spec.model.pi1_name, "commit": commit, "param_dtype": str(next(model.parameters()).dtype)}


def attach_lora(model, spec: SgacSpec, init_seed: int):
    """NB-M cell 5: `get_peft_model(base, LoraConfig(r=16, lora_alpha=32, q/k/v/o, dropout 0, bias none))`."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(int(init_seed))
    lcfg = LoraConfig(r=spec.lora.r, lora_alpha=spec.lora.alpha, target_modules=list(spec.lora.target_modules),
                      lora_dropout=spec.lora.dropout, bias="none", task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, lcfg)
    peft_model.train()
    log.info("LoRA attached: %s", trainable_param_counts(peft_model))
    return peft_model


def is_quantized(model) -> bool:
    base = getattr(model, "base_model", model)
    inner = getattr(base, "model", base)
    return bool(getattr(inner, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_4bit", False))


def save_step_adapter(model, out_dir: str | Path, meta: dict) -> Path:
    return save_adapter(model, out_dir, meta=meta)


def load_step_adapter(base, adapter_dir: str | Path, lora_dtype: str | None):
    """Reload a step adapter as trainable and restore the dtype the live LoRA weights had when it was saved
    (PEFT reloads fp32; TRL had cast them to bf16 for a 4-bit base), so resumed and uninterrupted runs match."""
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
    Curriculum._ensure_trainable(model)
    if lora_dtype:
        target = getattr(torch, lora_dtype.replace("torch.", ""), None)
        if isinstance(target, torch.dtype):
            for _, p in model.named_parameters():
                if p.requires_grad and p.dtype != target:
                    p.data = p.data.to(target)
    model.train()
    return model


# ---------------------------------------------------------------------- LoRA diagnostics
def lora_snapshot(model) -> dict:
    """Squared norms of all lora_A / lora_B weights, their dtype and count (for per-burst update diagnostics)."""
    out = {"A_sq": 0.0, "B_sq": 0.0, "dtype": None, "n_tensors": 0}
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            out["n_tensors"] += 1
            out["dtype"] = str(p.dtype)
            v = float(p.detach().float().pow(2).sum().item())
            out["A_sq" if "lora_A" in name else "B_sq"] += v
    return out


def lora_state(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if "lora_" in n}


def lora_delta_norms(before: dict[str, torch.Tensor], model) -> dict:
    da = db = 0.0
    for n, p in model.named_parameters():
        if n in before:
            d = float((p.detach().float().cpu() - before[n]).pow(2).sum().item())
            if "lora_A" in n:
                da += d
            elif "lora_B" in n:
                db += d
    return {"delta_A": da ** 0.5, "delta_B": db ** 0.5}
