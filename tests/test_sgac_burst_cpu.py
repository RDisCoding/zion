"""Real GRPO bursts (TRL 1.14.1) on a tiny model on CPU: one persistent adapter, NB-M's batch shape, clean state
between bursts, and a sieve on the same live model in between."""
import pytest

from rlvr_v2.artifacts import read_jsonl

pytestmark = pytest.mark.smoke


def _setup(profile, overrides=()):
    from rlvr_v2.sgac import model_io
    from rlvr_v2.sgac.grading import DualGrader
    from rlvr_v2.sgac.spec import CONFIG_DIR, load_profile

    spec = load_profile(profile, list(overrides), [CONFIG_DIR / "smoke_cpu.yaml"])
    tok = model_io.load_tokenizer(spec)
    base, _ = model_io.load_policy_model(spec)
    model = model_io.attach_lora(base, spec, 0)
    return spec, tok, model, model_io.stop_token_ids(spec, tok), DualGrader(spec.profile.grader)


def _item():
    from rlvr_v2.sgac.data import SgacItem

    return SgacItem(0, 0, "u0", "What is 2+3?", "So \\boxed{5}.", "5", 1, "Algebra")


def test_three_bursts_on_one_adapter_e0(tmp_path, monkeypatch):
    from rlvr_v2.sgac import burst
    from rlvr_v2.sgac import sampler as ssampler
    from rlvr_v2.sgac import sieve as ssieve

    def varying_rewards(mv, stop_token_ids=None, **kw):  # force reward variance so every burst updates the adapter
        def correctness(prompts, completions, completion_ids=None, answer=None, trainer_state=None, **k):
            return [float(i % 2) for i in range(len(completions))]

        def fmt(prompts, completions, **k):
            return [0.0] * len(completions)

        correctness.__name__, fmt.__name__ = "correctness", "format"
        return [correctness, fmt]

    monkeypatch.setattr(burst, "make_reward_funcs", varying_rewards)
    spec, tok, model, stops, grader = _setup("e0", ["grpo.gradient_checkpointing=true"])
    item = _item()
    for b in range(3):
        s = burst.run_burst(model, tok, item, spec, grader, tmp_path / f"b{b}", seed=0, context={"b": b}, stop_ids=stops)
        assert s["steps_done"] == 5 and len(s["losses"]) == 5
        assert s["delta_B"] > 0 and s["lora_after"]["B_sq"] > 0
        assert set(model.peft_config) == {"default"} and burst.count_input_require_grads_hooks(model) == 0
        assert len(read_jsonl(tmp_path / f"b{b}" / "train_rollouts.jsonl")) == 8  # 2 generation rounds x 4
        lrs = s["learning_rates"]
        assert len(lrs) == 5 and all(a > b for a, b in zip(lrs, lrs[1:]))
        sampler = ssampler.sieve_sampler(model, tok, spec, stops)
        cands = ssieve.sieve_candidates(sampler, tok, [item], spec, grader, seed=b, out_dir=tmp_path / f"s{b}",
                                        policy_tag="t")
        assert len(cands) == 1 and set(cands[0]["selection_signals"]) == {"Ps", "Var", "D", "L"}
        assert model.training  # generation_mode restored the training state


def test_as_run_burst_uses_conversational_prompt_and_legacy_rewards(tmp_path):
    from rlvr_v2.artifacts import read_json
    from rlvr_v2.sgac import burst

    spec, tok, model, stops, grader = _setup("as_run")
    s = burst.run_burst(model, tok, _item(), spec, grader, tmp_path, seed=0, context={}, stop_ids=stops)
    assert s["steps_done"] == 5 and set(model.peft_config) == {"default"}
    args = read_json(tmp_path / "grpo_args.json")
    assert args["audit"]["max_completion_length"] == 12 and args["audit"]["reward_weights"] == [1.0, 1.0]
    assert "What is 2+3?" in args["train_prompt"] and "Solve the following math problem" in args["train_prompt"]
    rows = read_jsonl(tmp_path / "train_rollouts.jsonl")
    assert len(rows) == 8 and {"legacy_correct", "mv_correct", "reward_format"} <= set(rows[0])
