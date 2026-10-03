"""End-to-end CPU smoke test with a tiny model: manifests -> sieve -> every selector -> 2 GRPO rounds
-> eval -> artifact/schema checks. Verifies plumbing, not accuracy."""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np

from .artifacts import RunDir, read_jsonl
from .config import Config
from .data import Problem, build_manifest, assert_disjoint

log = logging.getLogger(__name__)

REQUIRED_SIGNAL_KEYS = ("unique_id", "k", "p_s", "v_bin", "d_simpson", "entropy_bits", "d_wrong", "trunc_rate")
REQUIRED_ROLLOUT_KEYS = ("unique_id", "rollout_idx", "n_tokens", "finish_reason", "correct", "text")


def synthetic_problems(n: int, seed: int, prefix: str) -> list[Problem]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a, b = rng.randint(2, 40), rng.randint(2, 40)
        out.append(Problem(f"{prefix}/{i}", f"What is {a} + {b}?", str(a + b), None, 1 + (i % 5), "Prealgebra", "math_train"))
    return out


def run_smoke(cfg: Config, out_dir: Path) -> dict:
    from .evaluate import evaluate
    from .grader import MathVerifyGrader
    from .modeling import attach_fresh_lora, load_base_model, load_tokenizer, save_adapter, trainable_param_counts
    from .sampling import make_sampler
    from .selectors import ARM_NAMES, LearnedLinearSelector, build_selector
    from .sieve import measure_signals
    from .train_grpo import run_grpo_burst

    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"model": cfg.model.name, "steps": {}}
    grader = MathVerifyGrader()

    # 1) data + manifests
    pool = synthetic_problems(cfg.data.pool_size, 1, "smoke-pool")
    eval_items = synthetic_problems(cfg.eval.max_items or 5, 2, "smoke-eval")
    m_pool = build_manifest(pool, "pool", "math_train")
    m_eval = build_manifest(eval_items, "math500", "math500")
    assert_disjoint(m_pool, m_eval)
    m_pool.save(out_dir / "manifests" / "pool.json")
    m_eval.save(out_dir / "manifests" / "math500.json")
    report["steps"]["manifests"] = {"pool": len(m_pool), "eval": len(m_eval)}

    # 2) model + sieve
    tok = load_tokenizer(cfg)
    model = load_base_model(cfg)
    sampler = make_sampler(cfg, model=model, tokenizer=tok)
    cands = pool[:3]
    signals = measure_signals(sampler, tok, cands, cfg, grader, policy_tag="base", k=cfg.gen.sieve.n,
                              seed=cfg.run.seed, out_dir=out_dir / "sieve")
    rows = read_jsonl(out_dir / "sieve" / "signals.jsonl")
    assert len(rows) == len(cands), f"expected {len(cands)} signal rows, got {len(rows)}"
    for r in rows:
        missing = [k for k in REQUIRED_SIGNAL_KEYS if k not in r]
        assert not missing, f"signals.jsonl missing keys {missing}"
    roll = read_jsonl(out_dir / "sieve" / "rollouts.jsonl")
    assert len(roll) == len(cands) * cfg.gen.sieve.n, "rollouts.jsonl row count mismatch"
    for r in roll[:3]:
        missing = [k for k in REQUIRED_ROLLOUT_KEYS if k not in r]
        assert not missing, f"rollouts.jsonl missing keys {missing}"
    report["steps"]["sieve"] = {"n": len(rows), "k": cfg.gen.sieve.n,
                                "p_s": [s.p_s for s in signals], "trunc_rate": [s.trunc_rate for s in signals]}

    # 3) every selector
    picks = {}
    learned_path = out_dir / "learned_smoke.json"
    LearnedLinearSelector(features=("p_s", "d_simpson"), coef={"p_s": -1.0, "d_simpson": 1.0}).to_json(learned_path)
    sel_cfg = cfg.selector.__class__(name="random", learned_path=str(learned_path))
    for name in ARM_NAMES:
        idx, scores = build_selector(sel_cfg, name=name).select(signals, np.random.default_rng(0))
        picks[name] = {"idx": idx, "uid": cands[idx].unique_id, "scores": scores}
    report["steps"]["selectors"] = picks
    chosen = cands[picks["variance"]["idx"]]

    # 4) GRPO burst on the chosen problem (fresh LoRA)
    model = attach_fresh_lora(model, cfg)
    report["steps"]["lora"] = trainable_param_counts(model)
    run_dir = RunDir(cfg, "smoke", "burst", root=out_dir / "results")
    run_dir.init(seed=cfg.run.seed)
    burst = run_grpo_burst(model, tok, [chosen], cfg, run_dir.stage("train"), max_steps=cfg.train.rounds,
                           seed=cfg.run.seed, grader=grader, context={"smoke": True})
    metrics = read_jsonl(run_dir.path / "train" / "train_metrics.jsonl")
    assert metrics, "train_metrics.jsonl is empty"
    keys = set().union(*(m.keys() for m in metrics))
    report["steps"]["train"] = {"steps_logged": len(metrics), "has_clipped_ratio": "completions/clipped_ratio" in keys,
                               "has_reward_correct": any(k.startswith("rewards/correctness") for k in keys),
                               "burst": burst.to_dict()}
    save_adapter(model, run_dir.stage("adapter"), meta={"smoke": True})
    assert (run_dir.path / "adapter").exists()

    # 5) eval with the trained adapter (in-memory PEFT model)
    sampler2 = make_sampler(cfg, model=model, tokenizer=tok)
    summary = evaluate(sampler2, tok, eval_items, cfg, grader, run_dir.stage("eval") / "math500", tag="smoke")
    per_item = read_jsonl(run_dir.path / "eval" / "math500" / "per_item.jsonl")
    assert len(per_item) == len(eval_items), "per_item.jsonl row count mismatch"
    report["steps"]["eval"] = {"acc": summary.acc, "n": summary.n, "ci": [summary.ci_lo, summary.ci_hi]}
    run_dir.mark_done()
    report["wall_s"] = round(time.time() - t0, 1)
    report["ok"] = True
    log.info("smoke report: %s", json.dumps(report, default=str)[:1200])
    return report
