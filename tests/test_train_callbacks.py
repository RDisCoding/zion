import dataclasses
import json
import math

import pytest
from transformers import TrainerControl, TrainerState, TrainingArguments

from rlvr_v2.artifacts import read_jsonl
from rlvr_v2.config import Config
from rlvr_v2.train_grpo import (BurstResult, JsonlMetricsCallback, TruncationGuardCallback, TruncationGuardError,
                                build_train_dataset, check_grpo_args, latest_checkpoint)


class DummyTok:
    chat_template = None
    unk_token_id = None

    def __init__(self, eos=151643, pad=151662):  # Qwen2.5-Math base: eos <|endoftext|>, pad <|fim_pad|>
        self.eos_token_id = eos
        self.pad_token_id = pad

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}

    def convert_tokens_to_ids(self, s):
        return {"<|im_end|>": 151645, "<|endoftext|>": 151643}.get(s)


def small_cfg(tmp_path):
    cfg = Config()
    cfg = dataclasses.replace(
        cfg,
        run=dataclasses.replace(cfg.run, results_root=str(tmp_path), require_gates=False),
        gen=dataclasses.replace(cfg.gen, sieve=dataclasses.replace(cfg.gen.sieve, n=4), max_new_tokens=64),
        train=dataclasses.replace(cfg.train, num_generations=4, per_device_train_batch_size=2, rounds=2),
        study2=dataclasses.replace(cfg.study2, k=4),
    )
    cfg.validate()
    return cfg


@pytest.fixture
def targs(tmp_path):
    return TrainingArguments(output_dir=str(tmp_path / "out"), report_to="none")


def test_truncation_guard(targs):
    control = TrainerControl()
    cb = TruncationGuardCallback(max_clipped_ratio=0.2, check_steps=3)
    cb.on_log(targs, TrainerState(global_step=1), control, logs={"completions/clipped_ratio": 0.1, "loss": 0.0})
    cb.on_log(targs, TrainerState(global_step=1), control, logs={"loss": 0.0})  # no key: ignored, not counted
    assert cb.seen == 1
    with pytest.raises(TruncationGuardError, match="clipped_ratio"):
        cb.on_log(targs, TrainerState(global_step=2), control, logs={"completions/clipped_ratio": 0.5})
    # once check_steps logs have been inspected the guard is inert
    cb2 = TruncationGuardCallback(0.2, 2)
    for _ in range(2):
        cb2.on_log(targs, TrainerState(global_step=1), control, logs={"completions/clipped_ratio": 0.0})
    cb2.on_log(targs, TrainerState(global_step=3), control, logs={"completions/clipped_ratio": 0.9})
    assert cb2.seen == 2 and cb2.values == [0.0, 0.0]


def test_jsonl_metrics_callback(targs, tmp_path):
    path = tmp_path / "m" / "train_metrics.jsonl"
    cb = JsonlMetricsCallback(path, context={"arm": "random"})
    control = TrainerControl()
    cb.on_log(targs, TrainerState(global_step=1), control, logs={"loss": 0.1, "rewards/correctness/mean": 0.25})
    cb.on_log(targs, TrainerState(global_step=2), control, logs={"loss": 0.2, "rewards/correctness/mean": 0.5})
    cb.on_log(targs, TrainerState(global_step=2), control, logs=None)  # ignored
    cb.on_train_end(targs, TrainerState(global_step=2), control)
    recs = read_jsonl(path)
    assert [r["global_step"] for r in recs] == [1, 2]
    assert all(r["arm"] == "random" and "wall_s" in r and "cuda_max_mem_gb" in r for r in recs)
    assert [r["rewards/correctness/mean"] for r in recs] == [0.25, 0.5]
    assert cb.logs == recs


def test_burst_result_from_logs_and_roundtrip(tmp_path):
    logs = [
        {"rewards/correctness/mean": 0.25, "completions/clipped_ratio": 0.0, "frac_reward_zero_std": 1.0,
         "completions/mean_length": 100.0, "global_step": 1},
        {"rewards/correctness/mean": 0.75, "completions/clipped_ratio": 0.5, "frac_reward_zero_std": 0.0,
         "completions/mean_length": 300.0, "global_step": 2},
        {"train_runtime": 12.0, "global_step": 2},  # TRL's final summary log carries no step metrics
    ]
    r = BurstResult.from_logs(logs, steps_done=2, metrics_path=tmp_path / "m.jsonl",
                              train_rollouts_path=tmp_path / "r.jsonl", wall_s=12.0)
    assert (r.reward_correct_mean_first, r.reward_correct_mean_last) == (0.25, 0.75)
    assert (r.clipped_ratio_mean, r.frac_zero_std_mean, r.mean_completion_length) == (0.25, 0.5, 200.0)
    back = BurstResult.from_dict(json.loads(json.dumps(r.to_dict())))
    assert back.steps_done == 2 and back.metrics_path == tmp_path / "m.jsonl" and len(back.logs) == 3
    assert back.train_rollouts_path == tmp_path / "r.jsonl"
    empty = BurstResult.from_logs([], 0, "m", None, 0.0)
    assert math.isnan(empty.clipped_ratio_mean) and math.isnan(empty.reward_correct_mean_first)
    assert "logs" not in empty.summary() and empty.to_dict()["train_rollouts_path"] is None


def test_build_train_dataset(tmp_path, problem):
    cfg = small_cfg(tmp_path)
    ds = build_train_dataset([problem], cfg, DummyTok(), n_rows=5)
    assert ds.column_names == ["prompt", "answer", "unique_id"] and len(ds) == 5
    assert ds[0]["answer"] == "7" and ds[4]["unique_id"] == problem.unique_id
    assert ds[0]["prompt"].endswith("<|im_start|>assistant\n") and "What is 3+4?" in ds[0]["prompt"]
    assert len(build_train_dataset([problem], cfg, DummyTok(), n_rows=0)) == cfg.train.prompts_per_round
    tiny = dataclasses.replace(cfg, gen=dataclasses.replace(cfg.gen, max_prompt_tokens=3))
    with pytest.raises(ValueError, match="max_prompt_tokens"):
        build_train_dataset([problem], tiny, DummyTok(), n_rows=1)
    with pytest.raises(ValueError):
        build_train_dataset([], cfg, DummyTok(), n_rows=1)


def test_check_grpo_args_catches_length_and_eos_mismatch(tmp_path):
    pytest.importorskip("trl")
    cfg = small_cfg(tmp_path)
    stop = [151643, 151645]  # primary = the model's EOS <|endoftext|>, secondary <|im_end|>
    args = cfg.to_grpo_config(tmp_path / "trainer", max_steps=2, seed=1, stop_token_ids=stop)
    audited = check_grpo_args(args, cfg, DummyTok(), stop)
    assert audited["max_completion_length"] == 64 and audited["num_generations"] == 4
    assert audited["generation_batch_size"] == 4  # == num_generations: one prompt per generation round
    assert audited["steps_per_generation"] == audited["gradient_accumulation_steps"] == 2
    assert audited["eos_token_id"] == stop and audited["reward_weights"] == [1.0, 0.0]
    args.max_completion_length = 256  # TRL's old silent default
    with pytest.raises(RuntimeError, match="max_completion_length"):
        check_grpo_args(args, cfg, DummyTok(), stop)
    args.max_completion_length = 64
    with pytest.raises(RuntimeError, match="eos_token_id"):
        check_grpo_args(args, cfg, DummyTok(eos=151643, pad=151643), stop)
    with pytest.raises(RuntimeError, match="is not the primary stop"):
        check_grpo_args(args, cfg, DummyTok(eos=151645, pad=151662), stop)
    assert check_grpo_args(args, cfg, DummyTok(), stop)["max_completion_length"] == 64


def test_check_grpo_args_rejects_pad_that_is_a_stop_token(tmp_path):
    """The E0 G3 configuration (eos <|im_end|>, pad <|endoftext|>, both stops) must be refused: TRL would keep the
    padding after an <|endoftext|> stop inside the completion mask while still counting it as finished."""
    pytest.importorskip("trl")
    cfg = small_cfg(tmp_path)
    old_stop = [151645, 151643]
    args = cfg.to_grpo_config(tmp_path / "trainer", max_steps=2, seed=1, stop_token_ids=old_stop)
    with pytest.raises(RuntimeError, match="is a stop token"):
        check_grpo_args(args, cfg, DummyTok(eos=151645, pad=151643), old_stop)


def test_latest_checkpoint(tmp_path):
    assert latest_checkpoint(tmp_path / "nope") is None
    for n in (5, 25, 30):
        (tmp_path / f"checkpoint-{n}").mkdir()
    (tmp_path / "checkpoint-5" / "trainer_state.json").write_text("{}")
    (tmp_path / "checkpoint-25" / "trainer_state.json").write_text("{}")
    assert latest_checkpoint(tmp_path).name == "checkpoint-25"  # checkpoint-30 is incomplete
