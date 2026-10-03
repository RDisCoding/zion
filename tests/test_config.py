import dataclasses

import pytest
import yaml

from rlvr_v2.config import Config, GenerationCfg, ModelCfg, TrainCfg, load_config, parse_overrides


def test_defaults_validate_and_derive():
    cfg = Config()
    cfg.validate()
    assert cfg.train.gradient_accumulation_steps == 8
    assert cfg.train.generation_batch_size == 64 and cfg.train.prompts_per_round == 1
    kw = cfg.grpo_kwargs("out", 10, 3, stop_token_ids=[151645, 151643])
    assert kw["max_completion_length"] == cfg.gen.max_new_tokens
    assert kw["num_generations"] == 64 and kw["steps_per_generation"] == 8
    assert kw["generation_kwargs"] == {"eos_token_id": [151645, 151643]}
    assert kw["reward_weights"] == [1.0, 0.0] and kw["mask_truncated_completions"] is True
    assert "max_prompt_length" not in kw


def test_divisibility_and_vllm_checks():
    with pytest.raises(ValueError):
        dataclasses.replace(Config(), train=TrainCfg(num_generations=64, per_device_train_batch_size=3)).validate()
    with pytest.raises(ValueError):
        dataclasses.replace(Config(), model=ModelCfg(quant="nf4"), gen=GenerationCfg(backend="vllm")).validate()
    with pytest.raises(ValueError):
        dataclasses.replace(Config(), gen=GenerationCfg(eval=GenerationCfg().sieve)).validate()


def test_overrides_and_yaml(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text(yaml.safe_dump({"train": {"learning_rate": 5e-5, "rounds": 50}, "gen": {"max_new_tokens": 2048}}))
    cfg = load_config([y], ["gen.backend=vllm", "study2.arms=[random,variance]", "train.per_device_train_batch_size=4"])
    assert cfg.train.learning_rate == 5e-5 and cfg.train.rounds == 50 and cfg.gen.max_new_tokens == 2048
    assert cfg.gen.backend == "vllm" and cfg.study2.arms == ("random", "variance")
    assert cfg.train.gradient_accumulation_steps == 16
    assert parse_overrides(["a.b=1", "a.c=x"]) == {"a": {"b": 1, "c": "x"}}
    with pytest.raises(ValueError):
        load_config([], ["train.not_a_field=1"])


def test_hash_ignores_seed_and_tag_but_not_lr():
    base = Config()
    seeded = dataclasses.replace(base, run=dataclasses.replace(base.run, seed=99, tag="other"))
    lr = dataclasses.replace(base, train=dataclasses.replace(base.train, learning_rate=1e-6))
    assert base.config_hash() == seeded.config_hash() != lr.config_hash()


def test_study2_k_must_match_sieve_n():
    cfg = Config()
    bad = dataclasses.replace(cfg, study2=dataclasses.replace(cfg.study2, k=8))
    with pytest.raises(ValueError):
        bad.validate()


def test_to_grpo_config_with_installed_trl(tmp_path):
    pytest.importorskip("trl")
    cfg = Config()
    args = cfg.to_grpo_config(tmp_path, max_steps=5, seed=1, stop_token_ids=[1, 2])
    assert args.max_completion_length == cfg.gen.max_new_tokens
    assert args.num_generations == 64 and args.per_device_train_batch_size == 8
    assert args.max_steps == 5 and args.mask_truncated_completions is True
    assert args.generation_kwargs == {"eos_token_id": [1, 2]}
