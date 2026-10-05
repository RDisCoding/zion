"""The ONE curriculum implementation for Study 2, parameterised by `arm` (no per-arm scripts).

Every arm performs exactly `study2.steps x study2.rounds_per_burst` GRPO optimizer steps on the same live
PEFT model; arms differ only in how the next training problem is chosen:

- random / variance / disagreement / ps_band / learned: at step t, draw `batch_b` unused pool problems
  (without replacement across steps), sieve them with the CURRENT policy (K = study2.k rollouts each) and
  pick one with `selectors.build_selector(cfg.selector, name=arm)`.
- repeat_one: always train the given `repeat_problem` (pi1-style control); no sieve.

Layout under `run_dir.path`
    state.json                       {arm, seed, step_done, used_ids, history[], evals[]}   (atomic)
    adapter/                         the live adapter after the last completed step (overwritten every step)
    steps/step-NNN/sieve/            signals from `measure_signals`
    steps/step-NNN/selection.json    {step, arm, selected_uid, selected_idx, scores{uid: score}, signals_of_selected}
    steps/step-NNN/train/            train_metrics.jsonl, train_rollouts.jsonl, burst_summary.json
    steps/step-NNN/done.flag
    evals/step-NNN/                  evaluation outputs (layout matches `aggregate.py`)

Resume: completed steps (done.flag AND a state entry) are skipped; a partial step directory is deleted and
redone from the persisted adapter. The adapter is written to `adapter.tmp` together with a marker and then
swapped into place, so a crash can never leave a half-written adapter; a crash between the swap and the
state update is detected through the marker and the step's bookkeeping is completed without retraining.

Heavy collaborators (`sieve.measure_signals`, `evaluate.evaluate`, `sampling.make_sampler`, `modeling.*`,
`train_grpo.run_grpo_burst`) are reached through module-level lazy wrappers so this module imports without
torch and tests can monkeypatch `rlvr_v2.curriculum.<name>`.
"""
from __future__ import annotations

import dataclasses
import itertools
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .artifacts import RunDir, atomic_write_json, read_json, utc_now
from .config import Config
from .data import Problem
from .prompts import resolve_stop_token_ids
from .selectors import build_selector
from .signals import Signals

log = logging.getLogger(__name__)

ARMS: tuple[str, ...] = ("random", "variance", "disagreement", "ps_band", "learned", "repeat_one")
ADAPTER_MARKER = "curriculum_state.json"


# ---------------------------------------------------------------------- lazy, patchable collaborators
def measure_signals(*args, **kwargs):
    from .sieve import measure_signals as impl

    return impl(*args, **kwargs)


def evaluate(*args, **kwargs):
    from .evaluate import evaluate as impl

    return impl(*args, **kwargs)


def make_sampler(*args, **kwargs):
    from .sampling import make_sampler as impl

    return impl(*args, **kwargs)


def run_grpo_burst(*args, **kwargs):
    from .train_grpo import run_grpo_burst as impl

    return impl(*args, **kwargs)


def generation_mode(model):
    from .modeling import generation_mode as impl

    return impl(model)


def save_adapter(model, out_dir, meta=None):
    from .modeling import save_adapter as impl

    return impl(model, out_dir, meta=meta)


def load_adapter(base, adapter_dir):
    from .modeling import load_adapter as impl

    return impl(base, adapter_dir)


def load_tokenizer(cfg):
    from .modeling import load_tokenizer as impl

    return impl(cfg)


def load_base_model(cfg):
    from .modeling import load_base_model as impl

    return impl(cfg)


def attach_fresh_lora(model, cfg):
    from .modeling import attach_fresh_lora as impl

    return impl(model, cfg)


def free_cuda():
    from .modeling import free_cuda as impl

    return impl()


# ---------------------------------------------------------------------- adapter marker
def write_adapter_marker(adapter_dir: str | Path, step: int, arm: str, seed: int, extra: dict | None = None) -> None:
    atomic_write_json(Path(adapter_dir) / ADAPTER_MARKER,
                      {"curriculum_step": int(step), "arm": arm, "seed": int(seed), "saved": utc_now(), **(extra or {})})


def read_adapter_marker(adapter_dir: str | Path) -> int | None:
    """Curriculum step stored with the adapter; None when there is no adapter directory or no marker."""
    d = read_json(Path(adapter_dir) / ADAPTER_MARKER)
    return None if d is None else int(d["curriculum_step"])


def _ci_of(summary, d: dict) -> list | None:
    lo = getattr(summary, "ci_lo", None)
    hi = getattr(summary, "ci_hi", None)
    if lo is None or hi is None:
        lo, hi = d.get("ci_lo"), d.get("ci_hi")
    if lo is not None and hi is not None:
        return [float(lo), float(hi)]
    ci = d.get("ci")
    return list(ci) if isinstance(ci, (list, tuple)) else None


# ---------------------------------------------------------------------- curriculum
class Curriculum:
    """Run one Study-2 arm for one seed; see the module docstring for the on-disk layout and resume rules."""

    def __init__(
        self,
        cfg: Config,
        arm: str,
        seed: int,
        run_dir: RunDir,
        pool: list[Problem],
        eval_problems: list[Problem],
        grader,
        model=None,
        tokenizer=None,
        repeat_problem: Problem | None = None,
        ineligible: Sequence[str] = (),
    ):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        if arm == "repeat_one" and repeat_problem is None:
            raise ValueError("arm 'repeat_one' needs repeat_problem (cfg.study2.repeat_one_uid)")
        uids = [p.unique_id for p in pool]
        if len(set(uids)) != len(uids):
            raise ValueError("pool contains duplicate unique_ids")
        self.cfg = cfg
        self.arm = arm
        self.seed = int(seed)
        self.run_dir = run_dir
        # prereg §5/§6 eligibility rule: items with unparseable gold answers are never drawn (fixed before any run)
        self.ineligible = sorted(set(ineligible))
        self.pool = [p for p in pool if p.unique_id not in set(self.ineligible)]
        self.eval_problems = list(eval_problems)
        self.grader = grader
        self.model = model
        self.tokenizer = tokenizer
        self.repeat_problem = repeat_problem
        self.selector = None if arm == "repeat_one" else build_selector(cfg.selector, name=arm)  # fail fast
        self.state_path = run_dir.path / "state.json"
        self.adapter_dir = run_dir.path / "adapter"
        self.evals_dir = run_dir.path / "evals"
        # Sieve/eval over the live PEFT model always go through the HF backend.
        self._hf_cfg = dataclasses.replace(cfg, gen=dataclasses.replace(cfg.gen, backend="hf"))
        self.state: dict[str, Any] = self._new_state()

    # ------------------------------------------------------------------ paths & state
    def step_dir(self, t: int) -> Path:
        return self.run_dir.path / "steps" / f"step-{t:03d}"

    def _new_state(self) -> dict[str, Any]:
        return {"arm": self.arm, "seed": self.seed, "step_done": 0, "used_ids": [], "history": [], "evals": []}

    def _load_state(self) -> dict[str, Any]:
        st = read_json(self.state_path)
        if st is None:
            return self._new_state()
        if st.get("arm") != self.arm or int(st.get("seed", -1)) != self.seed:
            raise RuntimeError(f"{self.state_path} belongs to arm={st.get('arm')} seed={st.get('seed')}, "
                               f"not {self.arm}/{self.seed}")
        for key, default in self._new_state().items():
            st.setdefault(key, default)
        return st

    def _save_state(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def _history_entry(self, t: int) -> dict | None:
        return next((h for h in self.state["history"] if h["step"] == t), None)

    def _step_complete(self, t: int) -> bool:
        return (self.step_dir(t) / "done.flag").exists() and self._history_entry(t) is not None

    def _has_eval(self, t: int) -> bool:
        return any(e["step"] == t for e in self.state["evals"])

    def _eval_due(self, t: int) -> bool:
        every = self.cfg.study2.eval_every
        return t == self.cfg.study2.steps or (every > 0 and t % every == 0)

    # ------------------------------------------------------------------ model
    def _ensure_model(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(self.cfg)
        done = self.state["step_done"]
        if self.model is not None:
            if done > 0:
                log.warning("resuming at step %d with a caller-provided model; it must already carry %s",
                            done, self.adapter_dir)
            return
        base = load_base_model(self.cfg)
        if done > 0:
            if read_adapter_marker(self.adapter_dir) is None:
                raise RuntimeError(f"state says {done} steps done but no persisted adapter in {self.adapter_dir}")
            log.info("resuming %s/%d at step %d from %s", self.arm, self.seed, done, self.adapter_dir)
            self.model = load_adapter(base, self.adapter_dir)
            self._ensure_trainable(self.model)
        else:
            self.model = attach_fresh_lora(base, self.cfg)

    @staticmethod
    def _ensure_trainable(model) -> None:
        """A reloaded adapter must be trainable; re-enable LoRA grads if the loader froze them."""
        if not hasattr(model, "named_parameters"):
            return
        try:
            for pc in (getattr(model, "peft_config", None) or {}).values():
                pc.inference_mode = False  # modeling.load_adapter loads with is_trainable=False
            if any(p.requires_grad for _, p in model.named_parameters()):
                return
            n = 0
            for name, p in model.named_parameters():
                if "lora_" in name:
                    p.requires_grad_(True)
                    n += 1
            log.warning("reloaded adapter had no trainable parameters; re-enabled grads on %d LoRA tensors", n)
        except Exception as e:  # pragma: no cover
            log.warning("could not inspect trainable parameters: %r", e)

    def _sampler(self):
        return make_sampler(self._hf_cfg, model=self.model, tokenizer=self.tokenizer,
                            stop_token_ids=resolve_stop_token_ids(self.tokenizer, self.cfg.prompt.style))

    # ------------------------------------------------------------------ adapter persistence
    def _recover_adapter_dirs(self) -> None:
        tmp, bak = self.adapter_dir.with_name("adapter.tmp"), self.adapter_dir.with_name("adapter.bak")
        if not self.adapter_dir.exists() and bak.exists():
            log.warning("restoring %s from %s (crash during adapter swap)", self.adapter_dir, bak)
            os.rename(bak, self.adapter_dir)
        for d in (tmp, bak):
            if d.exists():
                shutil.rmtree(d)

    def _persist_adapter(self, t: int, selected: Problem) -> None:
        """Save the live adapter to adapter.tmp (with marker) and swap it into `adapter/`."""
        tmp, bak = self.adapter_dir.with_name("adapter.tmp"), self.adapter_dir.with_name("adapter.bak")
        for d in (tmp, bak):
            if d.exists():
                shutil.rmtree(d)
        tmp.mkdir(parents=True)
        meta = {"arm": self.arm, "seed": self.seed, "curriculum_step": t, "unique_id": selected.unique_id,
                "prompt_style": self.cfg.prompt.style, "config_hash": self.cfg.config_hash()}
        save_adapter(self.model, tmp, meta=meta)
        write_adapter_marker(tmp, t, self.arm, self.seed, {"unique_id": selected.unique_id})
        if self.adapter_dir.exists():
            try:
                os.rename(self.adapter_dir, bak)
            except OSError as e:  # pragma: no cover - e.g. open handles on Windows
                log.warning("could not rename old adapter dir (%r); removing it instead", e)
                shutil.rmtree(self.adapter_dir)
        os.rename(tmp, self.adapter_dir)
        if bak.exists():
            shutil.rmtree(bak)

    # ------------------------------------------------------------------ reconcile on (re)start
    def _reconcile(self) -> None:
        """Make disk and state consistent before running: recover adapter swaps, finish bookkeeping for a
        step whose adapter was saved but not recorded, and delete partial step directories."""
        self._recover_adapter_dirs()
        done = self.state["step_done"]
        marker = read_adapter_marker(self.adapter_dir)
        if self.adapter_dir.exists() and marker is None:
            if done == 0:
                log.warning("removing stale adapter dir without marker: %s", self.adapter_dir)
                shutil.rmtree(self.adapter_dir)
            else:
                raise RuntimeError(f"{self.adapter_dir} has no {ADAPTER_MARKER}; cannot verify it matches step {done}")
        if marker is not None and marker != done:
            if marker == done + 1 and self._complete_bookkeeping(done + 1):
                done = self.state["step_done"]
            else:
                raise RuntimeError(
                    f"persisted adapter is at curriculum step {marker} but state.json says {done} steps are done; "
                    f"refusing to continue (fix state.json or delete {self.run_dir.path} to restart)"
                )
        for t in range(1, done + 1):
            if not self._step_complete(t):
                raise RuntimeError(f"state says step {t} is done but {self.step_dir(t)}/done.flag or its history "
                                   f"entry is missing")
        for t in range(done + 1, self.cfg.study2.steps + 1):
            sd = self.step_dir(t)
            if sd.exists():
                log.warning("removing partial step directory %s", sd)
                shutil.rmtree(sd)

    def _complete_bookkeeping(self, t: int) -> bool:
        """Finish a step whose burst and adapter save completed but whose state update did not."""
        sd = self.step_dir(t)
        selection = read_json(sd / "selection.json")
        burst = read_json(sd / "train" / "burst_summary.json")
        if selection is None or burst is None:
            return False
        log.warning("completing bookkeeping for step %d (adapter saved, state not updated)", t)
        candidates = [uid for uid in selection.get("candidate_uids", [])]
        self._commit_step(t, selection, {k: v for k, v in burst.items() if k != "logs"}, candidates)
        return True

    # ------------------------------------------------------------------ one step
    def _draw_candidates(self, rng: np.random.Generator, t: int) -> list[Problem]:
        used = set(self.state["used_ids"])
        unused = [p for p in self.pool if p.unique_id not in used]
        b = self.cfg.study2.batch_b
        if len(unused) < b:
            raise RuntimeError(f"pool exhausted at step {t}: {len(unused)} unused problems < batch_b={b}")
        pick = rng.choice(len(unused), size=b, replace=False)
        return [unused[int(i)] for i in pick]

    def _sieve(self, candidates: list[Problem], t: int) -> list[Signals]:
        sd = self.step_dir(t)
        with generation_mode(self.model):
            signals = measure_signals(self._sampler(), self.tokenizer, candidates, self.cfg, self.grader,
                                      policy_tag=f"step{t}", k=self.cfg.study2.k, seed=self.seed * 1000 + t,
                                      out_dir=sd / "sieve")
        free_cuda()
        by_uid = {s.unique_id: s for s in signals}
        missing = [c.unique_id for c in candidates if c.unique_id not in by_uid]
        if missing:
            raise RuntimeError(f"measure_signals returned no signals for {missing}")
        return [by_uid[c.unique_id] for c in candidates]

    def _run_step(self, t: int) -> None:
        sd = self.step_dir(t)
        sd.mkdir(parents=True, exist_ok=True)
        rng_t = np.random.default_rng([self.seed, t])
        if self.arm == "repeat_one":
            selected = self.repeat_problem
            candidates = [selected]
            idx, scores, sig = 0, {selected.unique_id: 1.0}, None
        else:
            candidates = self._draw_candidates(rng_t, t)
            signals = self._sieve(candidates, t)
            idx, raw_scores = self.selector.select(signals, rng_t)
            selected, sig = candidates[idx], signals[idx]
            scores = {c.unique_id: float(s) for c, s in zip(candidates, raw_scores)}
        selection = {
            "step": t, "arm": self.arm, "seed": self.seed, "policy_tag": f"step{t}",
            "selected_uid": selected.unique_id, "selected_idx": int(idx),
            "candidate_uids": [c.unique_id for c in candidates], "scores": scores,
            "signals_of_selected": sig.to_dict() if sig is not None else None,
            "signals": sig.to_dict() if sig is not None else None,  # alias read by aggregate.py
        }
        atomic_write_json(sd / "selection.json", selection)
        log.info("[%s seed %d] step %d/%d: selected %s (p_s=%s) from %d candidates", self.arm, self.seed, t,
                 self.cfg.study2.steps, selected.unique_id, getattr(sig, "p_s", None), len(candidates))
        burst = run_grpo_burst(
            self.model, self.tokenizer, [selected], self.cfg, sd / "train",
            max_steps=self.cfg.study2.rounds_per_burst, seed=self.seed * 1000 + t, grader=self.grader,
            context={"curriculum_step": t, "arm": self.arm},
        )
        burst_summary = burst.summary() if hasattr(burst, "summary") else {k: v for k, v in burst.to_dict().items()
                                                                            if k != "logs"}
        self._persist_adapter(t, selected)
        self._commit_step(t, selection, burst_summary, [c.unique_id for c in candidates])

    def _commit_step(self, t: int, selection: dict, burst_summary: dict, candidate_uids: list[str]) -> None:
        (self.step_dir(t) / "done.flag").write_text(utc_now() + "\n", encoding="utf-8")
        if self.arm != "repeat_one":
            used = set(self.state["used_ids"])
            self.state["used_ids"].extend(u for u in candidate_uids if u not in used)
        self.state["history"] = [h for h in self.state["history"] if h["step"] != t] + [{
            "step": t,
            "selected_uid": selection["selected_uid"],
            "selected_idx": selection["selected_idx"],
            "candidate_uids": candidate_uids,
            "scores": selection["scores"],
            "signals_of_selected": selection.get("signals_of_selected"),
            "burst_summary": burst_summary,
        }]
        self.state["step_done"] = t
        self._save_state()
        self.run_dir.set_status(stage="curriculum", step_done=t)

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self, t: int) -> dict:
        tag = f"step-{t:03d}"
        out_dir = self.evals_dir / tag
        with generation_mode(self.model):
            summary = evaluate(self._sampler(), self.tokenizer, self.eval_problems, self.cfg, self.grader, out_dir,
                               tag=tag, policy="base" if t == 0 else f"step{t}",
                               adapter_path=str(self.adapter_dir) if t > 0 else None)
        free_cuda()
        d = summary.to_dict() if hasattr(summary, "to_dict") else dict(summary)
        entry = {"step": t, "tag": tag, "acc": float(summary.acc), "ci": _ci_of(summary, d),
                 "n": d.get("n", getattr(summary, "n", None)), "summary": d}
        self.state["evals"] = sorted([e for e in self.state["evals"] if e["step"] != t] + [entry],
                                     key=lambda e: e["step"])
        self._save_state()
        log.info("[%s seed %d] eval %s: acc=%.4f ci=%s", self.arm, self.seed, tag, entry["acc"], entry["ci"])
        return entry

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        s2 = self.cfg.study2
        if not self.run_dir.exists():
            self.run_dir.init(self.seed, extra={"arm": self.arm, "study2": dataclasses.asdict(s2)})
        self.state = self._load_state()
        self._reconcile()
        self._ensure_model()
        self.run_dir.set_status(state="running", stage="curriculum", arm=self.arm, step_done=self.state["step_done"])
        log.info("curriculum %s seed %d: %d steps x %d rounds, batch_b=%d, K=%d (resuming at step %d)", self.arm,
                 self.seed, s2.steps, s2.rounds_per_burst, s2.batch_b, s2.k, self.state["step_done"])

        if not self._has_eval(0):
            if self.state["step_done"] == 0:
                self._evaluate(0)
            else:
                log.warning("step-000 evaluation is missing but the policy has already been trained; skipping")

        for t in range(1, s2.steps + 1):
            if self._step_complete(t):
                if self._eval_due(t) and not self._has_eval(t):
                    if t == self.state["step_done"]:
                        self._evaluate(t)
                    else:
                        log.warning("evaluation for completed step %d is missing and cannot be reproduced "
                                    "(adapter has moved on)", t)
                continue
            self._run_step(t)
            if self._eval_due(t):
                self._evaluate(t)

        self.run_dir.mark_done(step_done=self.state["step_done"])
        return self.state


# ---------------------------------------------------------------------- jobs
def expand_study2_jobs(cfg: Config) -> list[dict]:
    """[{index, arm, seed}] over cfg.study2.arms x cfg.study2.seeds (arms outer, seeds inner)."""
    return [{"index": i, "arm": arm, "seed": int(seed)}
            for i, (arm, seed) in enumerate(itertools.product(cfg.study2.arms, cfg.study2.seeds))]


def resolve_repeat_problem(cfg: Config, pool: list[Problem]) -> Problem | None:
    uid = cfg.study2.repeat_one_uid
    if not uid:
        return None
    if uid == "oneshot/pi1":
        from .data import load_pi1

        return load_pi1()
    return next((p for p in pool if p.unique_id == uid), None)


def run_study2_job(cfg: Config, index: int, pool: list[Problem], eval_problems: list[Problem], grader,
                   repeat_problem: Problem | None = None, ineligible: Sequence[str] = ()) -> dict:
    """Run job `index` of `expand_study2_jobs(cfg)` (resumable) and return a small summary dict."""
    jobs = expand_study2_jobs(cfg)
    if index < 0 or index >= len(jobs):
        raise ValueError(f"job index {index} out of range (0..{len(jobs) - 1})")
    arm, seed = jobs[index]["arm"], jobs[index]["seed"]
    if arm == "repeat_one" and repeat_problem is None:
        repeat_problem = resolve_repeat_problem(cfg, pool)
        if repeat_problem is None:
            raise ValueError("arm 'repeat_one' requires cfg.study2.repeat_one_uid (or repeat_problem)")
    run_dir = RunDir(cfg, "study2", f"{arm}__seed{seed}")
    cur = Curriculum(cfg, arm, seed, run_dir, pool, eval_problems, grader, repeat_problem=repeat_problem,
                     ineligible=ineligible)
    try:
        state = cur.run()
    except Exception as e:
        if run_dir.exists():
            run_dir.mark_failed(repr(e))
        raise
    evals = state.get("evals", [])
    return {"index": index, "arm": arm, "seed": seed, "run_dir": str(run_dir.path), "step_done": state["step_done"],
            "final_acc": evals[-1]["acc"] if evals else None, "n_evals": len(evals)}
