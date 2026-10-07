"""`HFSampler` with the profile's top_k (the shared sampler hardcodes top_k=0 for sampling).

NB-M's sieve called `generate(..., do_sample=True, temperature=1.0)` without top_k, and Qwen2.5-Math's
generation_config sets none, so transformers' default top_k=50 applied; E0 and TRL's GRPO generation use top_k=0.
Everything else (left padding, length-sorted chunks, explicit stop ids, OOM back-off, `generation_mode`) is the
shared, E0-validated implementation.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..config import GenerationCfg, SamplingParams
from ..sampling import HFSampler


class SgacSampler(HFSampler):
    def __init__(self, model, tokenizer, *, max_new_tokens: int, batch_size: int, stop_token_ids: Sequence[int],
                 top_k: int = 0, max_prompt_tokens: int = 1024):
        gen_cfg = GenerationCfg(max_prompt_tokens=int(max_prompt_tokens), max_new_tokens=int(max_new_tokens),
                                sieve=SamplingParams(1.0, 1.0, 2), eval=SamplingParams(0.0, 1.0, 1),
                                backend="hf", hf_batch_size=int(batch_size))
        super().__init__(model, tokenizer, gen_cfg, stop_token_ids)
        self.top_k = int(top_k)

    def generation_kwargs(self, params: SamplingParams) -> dict[str, Any]:
        kw = super().generation_kwargs(params)
        if kw.get("do_sample"):
            kw["top_k"] = self.top_k
        return kw

    def describe(self) -> dict:
        return {"max_new_tokens": self.gen_cfg.max_new_tokens, "batch_size": self.gen_cfg.hf_batch_size,
                "stop_token_ids": list(self.stop_token_ids), "top_k": self.top_k,
                "max_prompt_tokens": self.gen_cfg.max_prompt_tokens}


def sieve_sampler(model, tokenizer, spec, stop_ids) -> SgacSampler:
    p = spec.profile
    return SgacSampler(model, tokenizer, max_new_tokens=p.sieve_max_new_tokens, batch_size=p.sieve_batch_size,
                       stop_token_ids=stop_ids, top_k=p.sieve_top_k, max_prompt_tokens=p.max_prompt_tokens)


def eval_sampler(model, tokenizer, spec, stop_ids, batch_size: int | None = None) -> SgacSampler:
    p = spec.profile
    return SgacSampler(model, tokenizer, max_new_tokens=p.eval_max_new_tokens,
                       batch_size=int(batch_size or p.eval_batch_size), stop_token_ids=stop_ids, top_k=0,
                       max_prompt_tokens=p.max_prompt_tokens)
