"""E0 infrastructure gates. Every study job refuses to start unless `results/e0/gates.json` exists
with `all_passed: true` (or `run.require_gates` is disabled for development).

G1 base MATH-500 accuracy under two prompt styles (truncation/format rates reported; best style pinned)
G2 grader unit tests
G3 stop-token / truncation check on pool problems with the sieve sampler
G4 positive control: 1-shot GRPO on Wang et al.'s pi1 through the budget ladder, MATH-500 gain vs base
G5 evaluation determinism: two identical evals agree
"""
from __future__ import annotations

import dataclasses
import logging
import subprocess
import sys
import time
from pathlib import Path

from .artifacts import REPO_ROOT, RunDir, atomic_write_json, read_json
from .config import Config

log = logging.getLogger(__name__)

THRESHOLDS = {
    "g1_trunc_rate_max": 0.10,
    "g1_acc_min": 0.15,
    "g1_acc_max": 0.95,
    "g3_trunc_rate_max": 0.10,
    "g3_format_rate_min": 0.90,
    "g4_min_gain_points": 3.0,
    "g4_train_reward_min": 0.8,
    "g5_min_item_agreement": 0.98,
    "g5_max_acc_diff": 0.005,
}
LADDER = ((100, 2.0e-5), (100, 5.0e-5), (200, 5.0e-5))
STYLES = ("qwen_math_chat", "oneshot_rlvr_chat")


class _Ctx:
    """Lazy holder for model, tokenizer, problems and the grader shared by the gates."""

    def __init__(self, cfg: Config, n_items: int | None, out_root: Path):
        self.cfg = cfg
        self.n_items = n_items
        self.out_root = out_root
        self._model = None
        self._tok_style: str | None = None
        self._tok = None
        self._splits: dict | None = None
        from .grader import MathVerifyGrader

        self.grader = MathVerifyGrader()

    def model(self):
        if self._model is None:
            from .modeling import load_base_model

            self._model = load_base_model(self.cfg)
        return self._model

    def drop_model(self) -> None:
        from .modeling import free_cuda

        self._model = None
        free_cuda()

    def tokenizer(self, style: str):
        if self._tok is None or self._tok_style != style:
            from .modeling import load_tokenizer

            self._tok = load_tokenizer(self.cfg_for_style(style))
            self._tok_style = style
        return self._tok

    def cfg_for_style(self, style: str) -> Config:
        return dataclasses.replace(self.cfg, prompt=dataclasses.replace(self.cfg.prompt, style=style))

    def splits(self) -> dict:
        if self._splits is None:
            from .cli import load_split_problems

            self._splits = load_split_problems(self.cfg, ("pool", "math500"))
        return self._splits

    def eval_items(self) -> list:
        items = self.splits()["math500"]
        return items[: self.n_items] if self.n_items else items


def _eval(ctx: _Ctx, style: str, out_dir: Path, tag: str, adapter_path: Path | None = None, items=None):
    from .evaluate import evaluate
    from .modeling import load_adapter, load_base_model
    from .sampling import make_sampler

    cfg_s = ctx.cfg_for_style(style)
    tok = ctx.tokenizer(style)
    if adapter_path is not None:
        model = load_adapter(load_base_model(cfg_s), adapter_path)
    else:
        model = ctx.model()
    sampler = make_sampler(cfg_s, model=model, tokenizer=tok, adapter_path=str(adapter_path) if adapter_path else None)
    return evaluate(sampler, tok, items or ctx.eval_items(), cfg_s, ctx.grader, out_dir, tag=tag,
                    policy="adapter" if adapter_path else "base", adapter_path=str(adapter_path) if adapter_path else None)


# ---------------------------------------------------------------------- gates
def gate_g1(ctx: _Ctx) -> dict:
    results = {}
    for style in STYLES:
        s = _eval(ctx, style, ctx.out_root / "g1" / style / "math500", tag=f"base-{style}")
        results[style] = {"acc": s.acc, "ci": [s.ci_lo, s.ci_hi], "trunc_rate": s.trunc_rate,
                          "format_rate": s.format_rate, "n": s.n, "prompt_hash": s.prompt_hash}
    best = max(STYLES, key=lambda st: (results[st]["acc"] - 0.5 * results[st]["trunc_rate"]))
    r = results[best]
    passed = (r["trunc_rate"] <= THRESHOLDS["g1_trunc_rate_max"]
              and THRESHOLDS["g1_acc_min"] <= r["acc"] <= THRESHOLDS["g1_acc_max"])
    return {"passed": bool(passed), "styles": results, "pinned_style": best,
            "base_per_item": str(ctx.out_root / "g1" / best / "math500" / "per_item.jsonl")}


def gate_g2(ctx: _Ctx) -> dict:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_grader.py", "-p", "no:cacheprovider"],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)
    tail = "\n".join(proc.stdout.strip().splitlines()[-3:])
    return {"passed": proc.returncode == 0, "returncode": proc.returncode, "summary": tail,
            "note": "cross-check against the Qwen2.5-Math grader is manual (see prereg G2)"}


def gate_g3(ctx: _Ctx) -> dict:
    from .sampling import make_sampler
    from .sieve import measure_signals

    style = _pinned_style(ctx)
    cfg_s = ctx.cfg_for_style(style)
    tok = ctx.tokenizer(style)
    probs = ctx.splits()["pool"][:32]
    sampler = make_sampler(cfg_s, model=ctx.model(), tokenizer=tok)
    sig = measure_signals(sampler, tok, probs, cfg_s, ctx.grader, policy_tag="base", k=cfg_s.gen.sieve.n,
                          seed=cfg_s.run.seed, out_dir=ctx.out_root / "g3")
    trunc = sum(s.trunc_rate for s in sig) / len(sig)
    fmt = sum(s.format_rate for s in sig) / len(sig)
    none_rate = sum(s.none_rate for s in sig) / len(sig)
    len_mean = sum(s.len_mean for s in sig) / len(sig)
    passed = trunc <= THRESHOLDS["g3_trunc_rate_max"] and fmt >= THRESHOLDS["g3_format_rate_min"]
    return {"passed": bool(passed), "style": style, "n_problems": len(sig), "k": cfg_s.gen.sieve.n,
            "trunc_rate": trunc, "format_rate": fmt, "none_rate": none_rate, "mean_completion_tokens": len_mean,
            "p_s_mean": sum(s.p_s for s in sig) / len(sig)}


def gate_g5(ctx: _Ctx) -> dict:
    from .evaluate import load_per_item

    style = _pinned_style(ctx)
    items = ctx.eval_items()[:100]
    a = _eval(ctx, style, ctx.out_root / "g5" / "run_a", tag="det-a", items=items)
    b = _eval(ctx, style, ctx.out_root / "g5" / "run_b", tag="det-b", items=items)
    pa = {r["unique_id"]: r["correct"] for r in load_per_item(ctx.out_root / "g5" / "run_a" / "per_item.jsonl")}
    pb = {r["unique_id"]: r["correct"] for r in load_per_item(ctx.out_root / "g5" / "run_b" / "per_item.jsonl")}
    shared = [u for u in pa if u in pb]
    agreement = sum(pa[u] == pb[u] for u in shared) / max(1, len(shared))
    diff = abs(a.acc - b.acc)
    passed = agreement >= THRESHOLDS["g5_min_item_agreement"] and diff <= max(THRESHOLDS["g5_max_acc_diff"], 1 / max(1, len(shared)))
    return {"passed": bool(passed), "acc_a": a.acc, "acc_b": b.acc, "item_agreement": agreement, "n": len(shared)}


def gate_g4(ctx: _Ctx) -> dict:
    """Positive control: pi1 through the ladder; stop at the first passing rung."""
    from .data import load_pi1
    from .evaluate import paired_delta
    from .train_grpo import run_one_shot_grpo

    style = _pinned_style(ctx)
    base_per_item = _base_per_item(ctx, style)
    ctx.drop_model()  # training loads its own copy
    rungs = []
    passed_rung = None
    for rounds, lr in LADDER:
        cfg_r = dataclasses.replace(
            ctx.cfg_for_style(style),
            train=dataclasses.replace(ctx.cfg.train, rounds=rounds, learning_rate=lr),
            run=dataclasses.replace(ctx.cfg.run, tag="e0"),
        )
        run_dir = RunDir(cfg_r, "e0", f"g4_pi1_r{rounds}_lr{lr:g}")
        run_dir.init(seed=cfg_r.run.seed, extra={"gate": "g4", "rounds": rounds, "lr": lr})
        t0 = time.time()
        adapter_dir, burst = run_one_shot_grpo(load_pi1(), cfg_r, seed=cfg_r.run.seed, run_dir=run_dir, grader=ctx.grader)
        summary = _eval(ctx, style, run_dir.path / "eval" / "math500", tag=f"pi1-r{rounds}", adapter_path=Path(adapter_dir))
        delta = paired_delta(Path(base_per_item), run_dir.path / "eval" / "math500" / "per_item.jsonl")
        gain_pts = 100.0 * delta["delta"]
        ok = (gain_pts >= THRESHOLDS["g4_min_gain_points"] and delta["ci_lo"] > 0
              and burst.reward_correct_mean_last >= THRESHOLDS["g4_train_reward_min"])
        rung = {"rounds": rounds, "lr": lr, "acc": summary.acc, "gain_points": gain_pts,
                "delta_ci": [100 * delta["ci_lo"], 100 * delta["ci_hi"]],
                "train_reward_first": burst.reward_correct_mean_first, "train_reward_last": burst.reward_correct_mean_last,
                "clipped_ratio_mean": burst.clipped_ratio_mean, "frac_zero_std_mean": burst.frac_zero_std_mean,
                "wall_s": round(time.time() - t0, 1), "passed": bool(ok), "run_dir": str(run_dir.path)}
        rungs.append(rung)
        run_dir.mark_done(gate_passed=bool(ok))
        ctx.drop_model()
        if ok:
            passed_rung = rung
            break
    return {"passed": passed_rung is not None, "rungs": rungs, "frozen_budget": passed_rung and
            {"rounds": passed_rung["rounds"], "learning_rate": passed_rung["lr"]}, "style": style}


# ---------------------------------------------------------------------- helpers
def _pinned_style(ctx: _Ctx) -> str:
    rep = read_json(ctx.out_root / "gates.json", {}) or {}
    g1 = rep.get("gates", {}).get("g1", {})
    return g1.get("pinned_style") or ctx.cfg.prompt.style


def _base_per_item(ctx: _Ctx, style: str) -> Path:
    p = ctx.out_root / "g1" / style / "math500" / "per_item.jsonl"
    if not p.exists():
        _eval(ctx, style, p.parent, tag=f"base-{style}")
    return p


GATES = {"g1": gate_g1, "g2": gate_g2, "g3": gate_g3, "g4": gate_g4, "g5": gate_g5}


def run_gates(cfg: Config, gates: tuple[str, ...] = ("g1", "g2", "g3", "g5", "g4"), out: Path | None = None,
              extra: dict | None = None) -> dict:
    out = Path(out or REPO_ROOT / "results" / "e0" / "gates.json")
    out_root = out.parent
    report = read_json(out, {}) or {}
    report.setdefault("gates", {})
    report["config_hash"] = cfg.config_hash()
    report["thresholds"] = THRESHOLDS
    ctx = _Ctx(cfg, (extra or {}).get("n_items"), out_root)
    for g in gates:
        if g not in GATES:
            raise ValueError(f"unknown gate {g}; known: {sorted(GATES)}")
        t0 = time.time()
        log.info("running gate %s", g)
        try:
            res = GATES[g](ctx)
        except Exception as e:  # keep going so the report shows every failure
            log.exception("gate %s crashed", g)
            res = {"passed": False, "error": repr(e)}
        res["wall_s"] = round(time.time() - t0, 1)
        report["gates"][g] = res
        report["all_passed"] = all(bool(r.get("passed")) for r in report["gates"].values())
        atomic_write_json(out, report)
        log.info("gate %s -> passed=%s", g, res.get("passed"))
    return report
