"""The pinned GRPOConfig is NB-M-shaped under the installed TRL (5 single-completion updates per burst)."""
import dataclasses

import pytest

from rlvr_v2.sgac import burst
from rlvr_v2.sgac.spec import load_profile

trl = pytest.importorskip("trl")


@pytest.mark.parametrize("profile, cap", [("e0", 3072), ("as_run", 256)])
def test_config_is_nbm_shaped(tmp_path, profile, cap):
    spec = load_profile(profile)
    kw = burst.grpo_kwargs(spec, tmp_path, seed=42, stop_ids=[151643, 151645], cuda=False)
    known = {f.name for f in dataclasses.fields(trl.GRPOConfig)}
    assert not [k for k in kw if k not in known]
    args = burst.build_grpo_config(kw)
    audit = burst.audit_grpo_args(args, spec)
    assert args.steps_per_generation == 4 and args.gradient_accumulation_steps == 1
    assert args.generation_batch_size == 4 and args.num_generations == 4 and args.max_steps == 5
    assert args.max_completion_length == cap and args.beta == 0.0 and args.loss_type == "dapo"
    assert args.mask_truncated_completions is False and args.top_k == 0 and args.temperature == 1.0
    assert args.max_grad_norm == pytest.approx(4.0)  # TRL 1.14.1 dapo normaliser compensation
    assert audit["dapo_grad_scale_vs_trl100"] == 4.0
    if profile == "e0":
        assert args.reward_weights == [1.0, 0.5] and args.generation_kwargs == {"eos_token_id": [151643, 151645]}
    else:
        assert args.reward_weights == [1.0, 1.0] and args.generation_kwargs is None


def test_unknown_key_is_refused(tmp_path):
    spec = load_profile("e0")
    kw = burst.grpo_kwargs(spec, tmp_path, seed=0, stop_ids=[1], cuda=False)
    with pytest.raises(RuntimeError):
        burst.build_grpo_config({**kw, "max_prompt_length": 512})


def test_linear_schedule_matches_nbm():
    import torch
    from transformers import get_scheduler

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=2e-5)
    sch = get_scheduler("linear", opt, num_warmup_steps=0, num_training_steps=5)
    lrs = []
    for _ in range(5):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    assert lrs == pytest.approx([2.0e-5, 1.6e-5, 1.2e-5, 0.8e-5, 0.4e-5])


def test_installed_trl_is_the_audited_version():
    # the TRL 1.0.0 -> 1.14.1 differences were audited for this exact version (protocol section 3)
    assert trl.__version__ == "1.14.1"
