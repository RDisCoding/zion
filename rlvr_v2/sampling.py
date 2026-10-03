"""Batched completion sampling behind one small `Sampler` protocol.

Backends
- `HFSampler`:   transformers `generate` with left padding, length-sorted chunks of
                 `gen.hf_batch_size`, explicit stop ids and OOM back-off.
- `VLLMSampler`: optional vLLM backend (imports vLLM lazily; constructing it without vLLM raises).

Guarantees shared by both backends
- prompts are tokenised with `add_special_tokens=False` and NEVER truncated: a prompt longer than
  `gen.max_prompt_tokens` raises `ValueError` naming its index;
- completions stop at the first stop token (exclusive); `n_tokens` counts kept tokens only;
- `finish_reason` is "stop" or "length"; `truncated == (finish_reason == "length")`;
- results come back in the original prompt order, `n` ungraded `Rollout`s per prompt.

Reproducibility: `HFSampler` seeds torch with `seed + chunk_index` before each chunk, so CPU runs
are reproducible for a fixed chunking. GPU sampling is NOT bitwise reproducible (non-deterministic
kernels), and OOM back-off changes the chunking (and therefore the random streams).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Protocol, Sequence, runtime_checkable

import torch

from . import modeling
from . import prompts as prompts_mod
from .config import Config, GenerationCfg, SamplingParams
from .signals import Rollout

log = logging.getLogger(__name__)

_VLLM_DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


# ---------------------------------------------------------------------- pure helpers
def postprocess_sequence(generated_ids: Sequence[int], stop_ids: Sequence[int]) -> tuple[list[int], str]:
    """Cut `generated_ids` at the first stop id (exclusive).

    Returns `(kept_ids, finish_reason)` with `finish_reason` "stop" when a stop id was found and
    "length" otherwise (the sequence ran into `max_new_tokens`). Right padding after a stop id is
    discarded automatically because it comes after the cut.
    """
    if hasattr(generated_ids, "tolist"):
        generated_ids = generated_ids.tolist()
    stops = set(int(s) for s in stop_ids)
    for i, tok in enumerate(generated_ids):
        if int(tok) in stops:
            return [int(t) for t in generated_ids[:i]], "stop"
    return [int(t) for t in generated_ids], "length"


def is_oom_error(exc: BaseException) -> bool:
    """CUDA OOM, either the dedicated exception class or a RuntimeError mentioning it."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _empty_stats() -> dict[str, float]:
    return {"prompts": 0, "completions": 0, "generated_tokens": 0, "decode_steps": 0, "chunks": 0,
            "wall_s": 0.0, "tokens_per_s": 0.0, "s_per_step": 0.0}


# ---------------------------------------------------------------------- protocol
@runtime_checkable
class Sampler(Protocol):
    def generate(self, prompts: Sequence[str], n: int, params: SamplingParams, seed: int) -> list[list[Rollout]]:
        """Return, for each prompt (in order), `n` ungraded rollouts."""
        ...


# ---------------------------------------------------------------------- HF transformers
class HFSampler:
    """Batched `model.generate` sampler. `tokenizer` must already have a pad token distinct from eos
    (see `prompts.configure_tokenizer`); `padding_side` is forced to "left"."""

    def __init__(self, model, tokenizer, gen_cfg: GenerationCfg, stop_token_ids: Sequence[int]):
        if not stop_token_ids:
            raise ValueError("stop_token_ids must not be empty (use prompts.resolve_stop_token_ids)")
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer has no pad token; call prompts.configure_tokenizer first")
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            log.warning("tokenizer pad == eos (%s); padding will be indistinguishable from the stop token",
                        tokenizer.pad_token_id)
        self.model = model
        self.tokenizer = tokenizer
        self.gen_cfg = gen_cfg
        self.stop_token_ids = [int(s) for s in stop_token_ids]
        self.tokenizer.padding_side = "left"
        self.stats: dict[str, float] = _empty_stats()  # last call
        self.totals: dict[str, float] = _empty_stats()  # cumulative over calls

    # ------------------------------------------------------------- encoding
    def encode_prompts(self, prompts: Sequence[str]) -> list[list[int]]:
        """Token ids per prompt (`add_special_tokens=False`); raises instead of truncating long prompts."""
        out: list[list[int]] = []
        limit = int(self.gen_cfg.max_prompt_tokens)
        for i, p in enumerate(prompts):
            ids = list(self.tokenizer(p, add_special_tokens=False)["input_ids"])
            if len(ids) > limit:
                raise ValueError(
                    f"prompt[{i}] has {len(ids)} tokens > gen.max_prompt_tokens={limit}; "
                    "refusing to truncate a prompt silently"
                )
            out.append(ids)
        return out

    def generation_kwargs(self, params: SamplingParams) -> dict[str, Any]:
        """Decoding kwargs. Greedy passes `do_sample=False` only; sampling passes temperature/top_p and
        neutralises any top_k / repetition_penalty defaults baked into the model's generation_config."""
        kw: dict[str, Any] = {
            "max_new_tokens": int(self.gen_cfg.max_new_tokens),
            "eos_token_id": list(self.stop_token_ids),
            "pad_token_id": int(self.tokenizer.pad_token_id),
            "num_beams": 1,
            "num_return_sequences": 1,
            "repetition_penalty": 1.0,
            "use_cache": True,
        }
        if params.temperature > 0:
            kw.update(do_sample=True, temperature=float(params.temperature), top_p=float(params.top_p), top_k=0)
        else:
            kw["do_sample"] = False
        return kw

    # ------------------------------------------------------------- generation
    def _generate_chunk(self, prompt_ids: list[list[int]], gen_kwargs: dict[str, Any], device: torch.device) -> list[list[int]]:
        """Left-pad `prompt_ids`, run `generate`, return only the newly generated ids per row."""
        pad_id = int(self.tokenizer.pad_token_id)
        max_len = max(len(ids) for ids in prompt_ids)
        input_ids = torch.full((len(prompt_ids), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(prompt_ids), max_len), dtype=torch.long)
        for row, ids in enumerate(prompt_ids):
            if ids:
                input_ids[row, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
                attention_mask[row, max_len - len(ids):] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        out = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
        if hasattr(out, "sequences"):  # return_dict_in_generate=True in a model's generation_config
            out = out.sequences
        return out[:, max_len:].to("cpu").tolist()

    def generate(self, prompts: Sequence[str], n: int, params: SamplingParams, seed: int) -> list[list[Rollout]]:
        t0 = time.perf_counter()
        prompts = list(prompts)
        n = int(n)
        if not prompts or n <= 0:
            return [[] for _ in prompts]
        encoded = self.encode_prompts(prompts)
        results: list[list[Rollout | None]] = [[None] * n for _ in prompts]
        # expand each prompt n times; longest prompts first so padding waste per chunk is minimal
        work = [(pi, ri) for pi in range(len(prompts)) for ri in range(n)]
        work.sort(key=lambda w: -len(encoded[w[0]]))
        gen_kwargs = self.generation_kwargs(params)
        device = modeling.device_of(self.model)
        chunk_size = max(1, int(self.gen_cfg.hf_batch_size))
        pos = chunk_index = generated_tokens = decode_steps = 0
        with modeling.generation_mode(self.model):
            while pos < len(work):
                chunk = work[pos:pos + chunk_size]
                torch.manual_seed(int(seed) + chunk_index)
                try:
                    rows = self._generate_chunk([encoded[pi] for pi, _ in chunk], gen_kwargs, device)
                except Exception as exc:  # OOM back-off, everything else propagates
                    if not is_oom_error(exc) or chunk_size <= 1:
                        raise
                    new_size = max(1, chunk_size // 2)
                    log.warning("CUDA OOM with chunk size %d; retrying with %d", chunk_size, new_size)
                    chunk_size = new_size
                    modeling.free_cuda()
                    continue
                kept_lists, reasons = [], []
                for row in rows:
                    kept, reason = postprocess_sequence(row, self.stop_token_ids)
                    kept_lists.append(kept)
                    reasons.append(reason)
                texts = self.tokenizer.batch_decode(kept_lists, skip_special_tokens=True)
                chunk_steps = 0
                for (pi, ri), kept, reason, text in zip(chunk, kept_lists, reasons, texts):
                    results[pi][ri] = Rollout(text=text, n_tokens=len(kept), truncated=(reason == "length"),
                                              finish_reason=reason)
                    produced = len(kept) + (1 if reason == "stop" else 0)  # the stop token was generated too
                    generated_tokens += produced
                    chunk_steps = max(chunk_steps, produced)
                decode_steps += chunk_steps
                pos += len(chunk)
                chunk_index += 1
        wall = time.perf_counter() - t0
        self.stats = {
            "prompts": len(prompts), "completions": len(work), "generated_tokens": generated_tokens,
            "decode_steps": decode_steps, "chunks": chunk_index, "wall_s": wall,
            "tokens_per_s": generated_tokens / wall if wall > 0 else 0.0,
            "s_per_step": wall / decode_steps if decode_steps else 0.0,
        }
        for k in ("prompts", "completions", "generated_tokens", "decode_steps", "chunks", "wall_s"):
            self.totals[k] += self.stats[k]
        self.totals["tokens_per_s"] = self.totals["generated_tokens"] / self.totals["wall_s"] if self.totals["wall_s"] else 0.0
        self.totals["s_per_step"] = self.totals["wall_s"] / self.totals["decode_steps"] if self.totals["decode_steps"] else 0.0
        log.info("HF generate: %d prompts x %d -> %d tokens in %.1fs (%.1f tok/s, final chunk size %d)",
                 len(prompts), n, generated_tokens, wall, self.stats["tokens_per_s"], chunk_size)
        return [list(r) for r in results]  # type: ignore[arg-type]


# ---------------------------------------------------------------------- vLLM
class VLLMSampler:
    """vLLM backend. Imports vLLM only in `__init__`, so this module imports without vLLM installed.

    Prompts are tokenised with the engine tokenizer (`add_special_tokens=False`) and passed as token
    ids, so the prompt-length guard matches `HFSampler`. Each prompt gets `seed + prompt_index` so
    repeated prompts in one call do not yield identical samples.
    """

    def __init__(self, model_name_or_path: str, gen_cfg: GenerationCfg, stop_token_ids: Sequence[int],
                 adapter_path: str | None = None, dtype: str = "bfloat16", lora_rank: int = 32):
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("gen.backend='vllm' requires the `vllm` package (pip install 'rlvr_v2[vllm]')") from e
        if not stop_token_ids:
            raise ValueError("stop_token_ids must not be empty")
        self.gen_cfg = gen_cfg
        self.stop_token_ids = [int(s) for s in stop_token_ids]
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.model_name_or_path = str(model_name_or_path)
        self._llm = LLM(
            model=self.model_name_or_path,
            dtype=dtype,
            enable_lora=self.adapter_path is not None,
            max_lora_rank=int(lora_rank),
            gpu_memory_utilization=float(gen_cfg.vllm_gpu_memory_utilization),
            max_model_len=int(gen_cfg.max_prompt_tokens + gen_cfg.max_new_tokens),
            seed=0,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self.stats: dict[str, float] = _empty_stats()

    def _lora_request(self):
        if not self.adapter_path:
            return None
        try:
            from vllm.lora.request import LoRARequest
        except ImportError:  # pragma: no cover - older/newer vLLM layouts
            from vllm import LoRARequest  # type: ignore
        return LoRARequest("adapter", 1, self.adapter_path)

    def encode_prompts(self, prompts: Sequence[str]) -> list[list[int]]:
        out: list[list[int]] = []
        limit = int(self.gen_cfg.max_prompt_tokens)
        for i, p in enumerate(prompts):
            ids = list(self._tokenizer(p, add_special_tokens=False)["input_ids"])
            if len(ids) > limit:
                raise ValueError(f"prompt[{i}] has {len(ids)} tokens > gen.max_prompt_tokens={limit}; refusing to truncate")
            out.append(ids)
        return out

    def generate(self, prompts: Sequence[str], n: int, params: SamplingParams, seed: int) -> list[list[Rollout]]:
        from vllm import SamplingParams as VSP

        t0 = time.perf_counter()
        prompts = list(prompts)
        n = int(n)
        if not prompts or n <= 0:
            return [[] for _ in prompts]
        token_prompts = [{"prompt_token_ids": ids} for ids in self.encode_prompts(prompts)]
        sampling = [
            VSP(n=n, temperature=float(params.temperature), top_p=float(params.top_p),
                max_tokens=int(self.gen_cfg.max_new_tokens), stop_token_ids=list(self.stop_token_ids),
                seed=int(seed) + i, skip_special_tokens=True)
            for i in range(len(prompts))
        ]
        outputs = self._llm.generate(token_prompts, sampling, lora_request=self._lora_request(), use_tqdm=False)
        stops = set(self.stop_token_ids)
        results: list[list[Rollout]] = []
        generated_tokens = 0
        for req in outputs:
            rollouts = []
            for o in req.outputs:
                raw_reason = getattr(o, "finish_reason", None) or "stop"
                reason = "length" if raw_reason == "length" else ("stop" if raw_reason == "stop" else str(raw_reason))
                ids = list(o.token_ids)
                generated_tokens += len(ids)
                if reason == "stop" and ids and ids[-1] in stops:
                    ids = ids[:-1]  # count kept tokens only, like HFSampler
                rollouts.append(Rollout(text=o.text, n_tokens=len(ids), truncated=(reason != "stop"),
                                        finish_reason=reason))
            if len(rollouts) != n:
                raise RuntimeError(f"vLLM returned {len(rollouts)} completions for a prompt, expected {n}")
            results.append(rollouts)
        if len(results) != len(prompts):
            raise RuntimeError(f"vLLM returned {len(results)} results for {len(prompts)} prompts")
        wall = time.perf_counter() - t0
        self.stats = {**_empty_stats(), "prompts": len(prompts), "completions": len(prompts) * n,
                      "generated_tokens": generated_tokens, "wall_s": wall,
                      "tokens_per_s": generated_tokens / wall if wall > 0 else 0.0}
        log.info("vLLM generate: %d prompts x %d -> %d tokens in %.1fs (%.1f tok/s)",
                 len(prompts), n, generated_tokens, wall, self.stats["tokens_per_s"])
        return results


# ---------------------------------------------------------------------- factory
def make_sampler(cfg: Config, model=None, tokenizer=None, adapter_path: str | None = None,
                 stop_token_ids: Sequence[int] | None = None) -> Sampler:
    """Build the sampler selected by `cfg.gen.backend`.

    "hf" needs `model` and `tokenizer` (an adapter, if any, must already be attached to `model`);
    "vllm" loads `cfg.model.name` in a vLLM engine and applies `adapter_path` as a LoRA request.
    Stop ids default to `prompts.resolve_stop_token_ids(tokenizer, cfg.prompt.style)`.
    """
    backend = cfg.gen.backend
    if backend == "hf":
        if model is None or tokenizer is None:
            raise ValueError("gen.backend='hf' requires both model and tokenizer")
        if stop_token_ids is None:
            stop_token_ids = prompts_mod.resolve_stop_token_ids(tokenizer, cfg.prompt.style)
        return HFSampler(model, tokenizer, cfg.gen, stop_token_ids)
    if backend == "vllm":
        if stop_token_ids is None:
            if tokenizer is None:
                tokenizer = modeling.load_tokenizer(cfg)
            stop_token_ids = prompts_mod.resolve_stop_token_ids(tokenizer, cfg.prompt.style)
        return VLLMSampler(cfg.model.name, cfg.gen, stop_token_ids, adapter_path=adapter_path,
                           dtype=_VLLM_DTYPES.get(cfg.model.dtype, "bfloat16"), lora_rank=cfg.model.lora.r)
    raise ValueError(f"unknown gen.backend {backend!r}; expected 'hf' or 'vllm'")
