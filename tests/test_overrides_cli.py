"""Repeated --override flags must accumulate: the local scripts prepend the gate-frozen overrides
(results/e0/frozen_args.txt) to any user EXTRA_ARGS, so a last-wins parser would silently drop the pinned style."""
from rlvr_v2.cli import _cfg, build_parser


def test_repeated_override_flags_accumulate_and_frozen_values_survive():
    frozen = ["--override", "prompt.style=oneshot_rlvr_chat", "train.rounds=100", "train.learning_rate=2e-05"]
    extra = ["--override", "eval.max_items=7"]
    args = build_parser().parse_args(["show-config", "--config", "configs/study1.yaml", *frozen, *extra])
    assert args.override == ["prompt.style=oneshot_rlvr_chat", "train.rounds=100", "train.learning_rate=2e-05",
                             "eval.max_items=7"]
    cfg = _cfg(args)
    assert cfg.prompt.style == "oneshot_rlvr_chat" and cfg.train.rounds == 100
    assert cfg.train.learning_rate == 2e-05 and cfg.eval.max_items == 7


def test_frozen_training_batch_size_is_four():
    args = build_parser().parse_args(["show-config", "--config", "configs/study1.yaml"])
    cfg = _cfg(args)
    assert cfg.train.per_device_train_batch_size == 4 and cfg.train.gradient_accumulation_steps == 16
    assert cfg.train.num_generations == 64 and cfg.train.prompts_per_round == 1
