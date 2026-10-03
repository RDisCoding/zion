import dataclasses
import importlib.util
import types

import pytest
import torch

from rlvr_v2 import modeling
from rlvr_v2.config import Config, GenerationCfg, SamplingParams
from rlvr_v2.sampling import HFSampler, Sampler, is_oom_error, make_sampler, postprocess_sequence

PAD, STOP_A, STOP_B = 0, 1, 2
STOP_IDS = [STOP_A, STOP_B]


def _valid_cfg(**gen_kwargs) -> Config:
    cfg = Config()
    cfg = dataclasses.replace(cfg, study2=dataclasses.replace(cfg.study2, k=cfg.gen.sieve.n))
    if gen_kwargs:
        cfg = dataclasses.replace(cfg, gen=dataclasses.replace(cfg.gen, **gen_kwargs))
    return cfg


# ---------------------------------------------------------------------- fakes
class FakeTokenizer:
    """One token per character (id = 100 + ord(c)); ids 0/1/2 are pad/stop/stop."""

    pad_token_id = PAD
    eos_token_id = STOP_A
    chat_template = None

    def __init__(self):
        self.padding_side = "right"

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [100 + ord(c) for c in text]}

    def batch_decode(self, seqs, skip_special_tokens=True):
        return ["".join(chr(t - 100) for t in s if t >= 100) for s in seqs]


class FakeModel:
    """Each row emits `min(real_prompt_len, max_new_tokens)` copies of its last prompt token, then a stop id
    (if room is left) and right padding. Records every generate() call."""

    def __init__(self, oom_once_above: int | None = None, oom_exc=None):
        self.training = True
        self.config = types.SimpleNamespace(use_cache=False)
        self.calls: list[dict] = []
        self.device = torch.device("cpu")
        self._oom_once_above = oom_once_above
        self._oom_exc = oom_exc or torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")
        self._oom_done = False
        self.training_seen_in_generate: list[bool] = []

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def parameters(self):
        yield torch.zeros(1)

    def generate(self, input_ids, attention_mask, **kw):
        self.calls.append({"batch": int(input_ids.shape[0]), "use_cache_cfg": self.config.use_cache, **kw})
        self.training_seen_in_generate.append(self.training)
        if self._oom_once_above is not None and input_ids.shape[0] > self._oom_once_above and not self._oom_done:
            self._oom_done = True
            raise self._oom_exc
        bsz, _ = input_ids.shape
        max_new = kw["max_new_tokens"]
        gen = torch.full((bsz, max_new), kw["pad_token_id"], dtype=torch.long)
        for r in range(bsz):
            real = int(attention_mask[r].sum())
            assert attention_mask[r, -1] == 1, "inputs must be left padded"
            m = min(real, max_new)
            gen[r, :m] = input_ids[r, -1]
            if m < max_new:
                gen[r, m] = STOP_A
        return torch.cat([input_ids, gen], dim=1)


# ---------------------------------------------------------------------- postprocess
def test_postprocess_stop_mid_sequence():
    assert postprocess_sequence([5, 6, 7, STOP_A, 8, PAD], STOP_IDS) == ([5, 6, 7], "stop")


def test_postprocess_no_stop_is_length():
    assert postprocess_sequence([5, 6, 7], STOP_IDS) == ([5, 6, 7], "length")


def test_postprocess_stop_at_zero_is_empty_stop():
    assert postprocess_sequence([STOP_B, 5, 6], STOP_IDS) == ([], "stop")


def test_postprocess_accepts_tensors_and_any_stop_id():
    kept, reason = postprocess_sequence(torch.tensor([9, STOP_B, STOP_A]), STOP_IDS)
    assert kept == [9] and reason == "stop"
    assert postprocess_sequence([], STOP_IDS) == ([], "length")


# ---------------------------------------------------------------------- HFSampler
def _sampler(model, **gen_kwargs):
    gen = GenerationCfg(max_prompt_tokens=gen_kwargs.pop("max_prompt_tokens", 32),
                        max_new_tokens=gen_kwargs.pop("max_new_tokens", 5),
                        hf_batch_size=gen_kwargs.pop("hf_batch_size", 2), **gen_kwargs)
    tok = FakeTokenizer()
    s = HFSampler(model, tok, gen, STOP_IDS)
    assert tok.padding_side == "left"
    assert isinstance(s, Sampler)
    return s


def test_hf_sampler_batches_sorts_and_restores_order():
    model = FakeModel()
    s = _sampler(model, max_new_tokens=5, hf_batch_size=2)
    out = s.generate(["ab", "abcdef", "abcd"], n=2, params=SamplingParams(temperature=0.0), seed=0)
    assert [len(g) for g in out] == [2, 2, 2]
    # prompt 0 ("ab"): 2 copies of "b" then stop -> stop/2 tokens
    for r in out[0]:
        assert (r.text, r.n_tokens, r.finish_reason, r.truncated) == ("bb", 2, "stop", False)
    # prompt 1 ("abcdef"): 6 > cap 5 -> runs into the cap
    for r in out[1]:
        assert (r.text, r.n_tokens, r.finish_reason, r.truncated) == ("fffff", 5, "length", True)
    for r in out[2]:
        assert (r.text, r.n_tokens, r.finish_reason, r.truncated) == ("dddd", 4, "stop", False)
    # 6 work items, longest first, chunks of 2
    assert [c["batch"] for c in model.calls] == [2, 2, 2]
    assert s.stats["prompts"] == 3 and s.stats["completions"] == 6 and s.stats["chunks"] == 3
    assert s.stats["generated_tokens"] == 2 * (5 + 5 + 3) and s.stats["decode_steps"] == 5 + 5 + 3
    assert s.stats["wall_s"] >= 0 and s.totals["completions"] == 6
    # generation_mode: eval during generate, state restored afterwards
    assert model.training_seen_in_generate == [False, False, False]
    assert model.training is True and model.config.use_cache is False
    assert all(c["use_cache_cfg"] is True for c in model.calls)


def test_hf_sampler_greedy_vs_sampling_kwargs():
    model = FakeModel()
    s = _sampler(model)
    s.generate(["ab"], 1, SamplingParams(temperature=0.0, top_p=1.0), seed=1)
    greedy = model.calls[-1]
    assert greedy["do_sample"] is False and "temperature" not in greedy and "top_p" not in greedy
    assert greedy["eos_token_id"] == STOP_IDS and greedy["pad_token_id"] == PAD and greedy["max_new_tokens"] == 5
    s.generate(["ab"], 1, SamplingParams(temperature=0.7, top_p=0.9), seed=1)
    sampled = model.calls[-1]
    assert sampled["do_sample"] is True and sampled["temperature"] == 0.7 and sampled["top_p"] == 0.9
    assert sampled["top_k"] == 0 and sampled["repetition_penalty"] == 1.0


def test_hf_sampler_refuses_long_prompts():
    s = _sampler(FakeModel(), max_prompt_tokens=3)
    with pytest.raises(ValueError, match=r"prompt\[1\]"):
        s.generate(["abc", "abcd"], 1, SamplingParams(temperature=0.0), seed=0)


@pytest.mark.parametrize("exc", [torch.cuda.OutOfMemoryError("CUDA out of memory"), RuntimeError("CUDA error: out of memory")])
def test_hf_sampler_oom_backoff(exc):
    model = FakeModel(oom_once_above=2, oom_exc=exc)
    s = _sampler(model, hf_batch_size=4)
    out = s.generate(["ab", "abc"], 2, SamplingParams(temperature=1.0), seed=3)
    assert [c["batch"] for c in model.calls] == [4, 2, 2]
    assert [r.text for r in out[0]] == ["bb", "bb"] and [r.text for r in out[1]] == ["ccc", "ccc"]


def test_hf_sampler_non_oom_errors_propagate():
    model = FakeModel(oom_once_above=0, oom_exc=RuntimeError("shape mismatch"))
    s = _sampler(model)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        s.generate(["ab"], 1, SamplingParams(temperature=0.0), seed=0)
    assert is_oom_error(torch.cuda.OutOfMemoryError("x")) and is_oom_error(RuntimeError("CUDA Out Of Memory"))
    assert not is_oom_error(RuntimeError("other")) and not is_oom_error(ValueError("out of memory"))


def test_hf_sampler_empty_inputs():
    s = _sampler(FakeModel())
    assert s.generate([], 4, SamplingParams(), seed=0) == []
    assert s.generate(["ab"], 0, SamplingParams(), seed=0) == [[]]


def test_hf_sampler_requires_pad_token():
    tok = FakeTokenizer()
    tok.pad_token_id = None
    with pytest.raises(ValueError):
        HFSampler(FakeModel(), tok, GenerationCfg(), STOP_IDS)
    with pytest.raises(ValueError):
        HFSampler(FakeModel(), FakeTokenizer(), GenerationCfg(), [])


# ---------------------------------------------------------------------- generation_mode
class GCModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.is_gradient_checkpointing = True
        self.events: list[str] = []

    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False
        self.events.append("disable")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.is_gradient_checkpointing = True
        self.events.append("enable")

    def modules(self):
        return []


def test_generation_mode_restores_state():
    m = GCModel()
    m.config.use_cache = False
    with modeling.generation_mode(m):
        assert m.training is False and m.config.use_cache is True and m.is_gradient_checkpointing is False
        assert torch.is_inference_mode_enabled()
    assert m.training is True and m.config.use_cache is False and m.is_gradient_checkpointing is True
    assert m.events == ["disable", "enable"]
    assert not torch.is_inference_mode_enabled()


def test_trainable_param_counts_plain_module():
    lin = torch.nn.Linear(4, 2)
    lin.bias.requires_grad_(False)
    counts = modeling.trainable_param_counts(lin)
    assert counts == {"trainable": 8, "total": 10, "fraction": 0.8}
    assert modeling.torch_dtype("bf16") is torch.bfloat16
    with pytest.raises(ValueError):
        modeling.torch_dtype("int8")


# ---------------------------------------------------------------------- factory
def test_make_sampler_hf_requires_model_and_tokenizer():
    cfg = _valid_cfg(backend="hf")
    with pytest.raises(ValueError):
        make_sampler(cfg)
    s = make_sampler(cfg, model=FakeModel(), tokenizer=FakeTokenizer(), stop_token_ids=STOP_IDS)
    assert isinstance(s, HFSampler) and s.stop_token_ids == STOP_IDS


@pytest.mark.skipif(importlib.util.find_spec("vllm") is not None, reason="vllm is installed")
def test_make_sampler_vllm_without_vllm_raises_import_error():
    cfg = _valid_cfg(backend="vllm")
    with pytest.raises(ImportError):
        make_sampler(cfg, tokenizer=FakeTokenizer(), stop_token_ids=STOP_IDS)


def test_make_sampler_unknown_backend():
    cfg = _valid_cfg(backend="nope")
    with pytest.raises(ValueError):
        make_sampler(cfg, model=FakeModel(), tokenizer=FakeTokenizer(), stop_token_ids=STOP_IDS)


# ---------------------------------------------------------------------- smoke (real tiny model, downloads)
@pytest.mark.smoke
def test_smoke_tiny_model_generation():
    from rlvr_v2 import prompts

    cfg = _valid_cfg(max_new_tokens=8, max_prompt_tokens=256, hf_batch_size=4)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, name="HuggingFaceTB/SmolLM2-135M-Instruct"))
    tok = modeling.load_tokenizer(cfg)
    model = modeling.load_base_model(cfg)
    stop_ids = prompts.resolve_stop_token_ids(tok, cfg.prompt.style)
    sampler = make_sampler(cfg, model=model, tokenizer=tok)
    rendered = [prompts.build_prompt(q, cfg.prompt.style, tok, cfg.prompt.system) for q in ("What is 2+2?", "Name a prime.")]
    out = sampler.generate(rendered, n=2, params=SamplingParams(temperature=0.0), seed=0)
    assert [len(g) for g in out] == [2, 2]
    for group in out:
        assert group[0].text == group[1].text  # greedy is deterministic across the two copies
        for r in group:
            assert r.finish_reason in ("stop", "length") and 0 <= r.n_tokens <= 8
            assert r.truncated == (r.finish_reason == "length")
    assert sampler.stats["generated_tokens"] > 0 and stop_ids
