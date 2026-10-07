"""The 20-step SGAC curriculum for one (profile, arm, seed), resumable at step granularity.

Per step t (NB-M cell 7): read the precomputed batch of 4 pool rows -> sieve them with the CURRENT policy (K=4) ->
every rule's pick is logged, the arm's rule chooses -> one GRPO micro-burst on the persistent PeftModel -> adapter
saved -> evaluation on test50 every `eval_every` steps (and MATH-500 at `loop.math500_steps`). Step 0 (the base model)
is evaluated once per profile under `_shared/base_eval`; a freshly attached LoRA has B = 0, so it is identical.

Layout under results_sgac/<profile>/<spec_hash>/<arm>__seed<S>/
    run.json, spec.yaml, status.json, state.json
    steps/step-NNN/{batch.json, sieve/, selection.json, train/, adapter/, timing.json, done.flag}
    evals/step-NNN/{test50,math500}/  and  evals/base_after_run/test50/ (optional, NB-M cell 11 style)
Adapters are kept for evaluation steps and for the latest step (resume point); others are deleted once the next step
is committed. Resume: completed steps are skipped, a step whose adapter and done.flag exist but whose state update was
lost is committed from its files, any other partial step directory is deleted and redone from the last adapter.

Heavy collaborators are module-level lazy wrappers so tests can monkeypatch `rlvr_v2.sgac.loop.<name>`.
"""
from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from ..artifacts import atomic_write_json, read_json, utc_now
from . import selection as sel
from .data import SgacItem
from .runinfo import SgacRunDir, phase, shared_dir
from .schedule import derived_seeds, random_arm_rng
from .spec import SgacSpec

log = logging.getLogger(__name__)

TEST50_DECISION = "test50_batch_decision.json"


# ---------------------------------------------------------------------- lazy, patchable collaborators
def load_tokenizer(spec):
    from .model_io import load_tokenizer as impl

    return impl(spec)


def load_policy_model(spec):
    from .model_io import load_policy_model as impl

    return impl(spec)


def attach_lora(model, spec, init_seed):
    from .model_io import attach_lora as impl

    return impl(model, spec, init_seed)


def load_step_adapter(base, adapter_dir, lora_dtype):
    from .model_io import load_step_adapter as impl

    return impl(base, adapter_dir, lora_dtype)


def save_step_adapter(model, out_dir, meta):
    from .model_io import save_step_adapter as impl

    return impl(model, out_dir, meta)


def stop_token_ids(spec, tok):
    from .model_io import stop_token_ids as impl

    return impl(spec, tok)


def make_sieve_sampler(model, tok, spec, stop_ids):
    from .sampler import sieve_sampler as impl

    return impl(model, tok, spec, stop_ids)


def make_eval_sampler(model, tok, spec, stop_ids, batch_size=None):
    from .sampler import eval_sampler as impl

    return impl(model, tok, spec, stop_ids, batch_size)


def sieve_candidates(*args, **kwargs):
    from .sieve import sieve_candidates as impl

    return impl(*args, **kwargs)


def run_burst(*args, **kwargs):
    from .burst import run_burst as impl

    return impl(*args, **kwargs)


def evaluate_items(*args, **kwargs):
    from .evaluation import evaluate_items as impl

    return impl(*args, **kwargs)


def free_cuda():
    from ..modeling import free_cuda as impl

    return impl()


def test50_batch_size(spec: SgacSpec) -> int:
    """e0: the profile's eval batch size. as_run: the pre-declared batch-1 agreement rule, decided by base-eval."""
    if not spec.profile.test50_batch_check:
        return int(spec.profile.eval_batch_size)
    d = read_json(shared_dir(spec, "base_eval") / TEST50_DECISION)
    if d is None:
        raise RuntimeError(f"{TEST50_DECISION} missing: run `python -m rlvr_v2.sgac base-eval --profile "
                           f"{spec.profile.name}` first (it decides the test50 batch size)")
    return int(d["batch_size"])


# ---------------------------------------------------------------------- the run
class SgacRun:
    def __init__(self, spec: SgacSpec, arm: str, seed: int, *, pool: Sequence[SgacItem], test50: Sequence[SgacItem],
                 math500: Sequence[SgacItem], schedule: Sequence[dict], grader, pi1: SgacItem | None = None,
                 base_after_run: bool = False):
        if arm not in sel.ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {sel.ARMS}")
        if arm == "fixed_pi1" and pi1 is None:
            raise ValueError("arm fixed_pi1 needs the pi1 item")
        self.spec, self.arm, self.seed = spec, arm, int(seed)
        self.rule = sel.ARM_RULE[arm]
        self.pool_by_row = {it.row: it for it in pool}
        self.test50, self.math500 = list(test50), list(math500)
        self.schedule = list(schedule)
        if len(self.schedule) < spec.loop.steps and arm != "fixed_pi1":
            raise ValueError(f"schedule has {len(self.schedule)} batches < {spec.loop.steps} steps")
        self.grader, self.pi1, self.base_after_run = grader, pi1, bool(base_after_run)
        self.rd = SgacRunDir(spec, f"{arm}__seed{self.seed}")
        self.state_path = self.rd.path / "state.json"
        self.model = self.tok = self.stop_ids = None
        self.load_record: dict | None = None
        self.state: dict = self._new_state()

    # ------------------------------------------------------------------ paths & state
    def step_dir(self, t: int) -> Path:
        return self.rd.path / "steps" / f"step-{t:03d}"

    def eval_dir(self, t: int | str, name: str) -> Path:
        tag = t if isinstance(t, str) else f"step-{t:03d}"
        return self.rd.path / "evals" / tag / name

    def eval_due(self, t: int) -> bool:
        every = int(self.spec.loop.eval_every)
        return t == self.spec.loop.steps or (every > 0 and t % every == 0)

    def math500_due(self, t: int) -> bool:
        return t > 0 and t in tuple(self.spec.loop.math500_steps) and bool(self.math500)

    def keep_adapter(self, t: int) -> bool:
        return self.eval_due(t) or t == self.state["step_done"]

    def _new_state(self) -> dict:
        return {"arm": self.arm, "seed": self.seed, "spec_hash": self.spec.spec_hash(), "profile": self.spec.profile.name,
                "step_done": 0, "history": [], "evals": {}, "lora_dtype": None}

    def _load_state(self) -> dict:
        st = read_json(self.state_path)
        if st is None:
            return self._new_state()
        mine = self._new_state()
        for key in ("arm", "seed", "spec_hash", "profile"):
            if st.get(key) != mine[key]:
                raise RuntimeError(f"{self.state_path} belongs to {key}={st.get(key)!r}, not {mine[key]!r}; refusing")
        for k, v in mine.items():
            st.setdefault(k, v)
        return st

    def _save_state(self) -> None:
        atomic_write_json(self.state_path, self.state)

    # ------------------------------------------------------------------ reconcile
    def _step_files_complete(self, t: int) -> bool:
        sd = self.step_dir(t)
        return (sd / "done.flag").exists() and (sd / "adapter").exists() and (sd / "train" / "burst_summary.json").exists()

    def _reconcile(self) -> None:
        done = int(self.state["step_done"])
        for t in range(1, done + 1):
            if not (self.step_dir(t) / "done.flag").exists():
                raise RuntimeError(f"state says step {t} is done but {self.step_dir(t)}/done.flag is missing")
        nxt = done + 1
        entry = self._history_from_files(nxt) if nxt <= self.spec.loop.steps and self._step_files_complete(nxt) else None
        if entry is not None:  # finished and written (done.flag last), only the state update was lost
            log.warning("step %d finished but its state update was lost; committing it from its files", nxt)
            self._commit(nxt, entry)
            done = nxt
        if done > 0 and not (self.step_dir(done) / "adapter").exists():
            raise RuntimeError(f"resume point {self.step_dir(done)}/adapter is missing")
        for t in range(done + 1, self.spec.loop.steps + 1):
            sd = self.step_dir(t)
            if sd.exists():
                log.warning("removing partial step directory %s", sd)
                shutil.rmtree(sd)

    def _history_from_files(self, t: int) -> dict | None:
        sd = self.step_dir(t)
        selection = read_json(sd / "selection.json")
        burst = read_json(sd / "train" / "burst_summary.json")
        if selection is None or burst is None:
            return None
        return {"step": t, "selection": selection, "burst": _burst_brief(burst), "timing": read_json(sd / "timing.json", {})}

    # ------------------------------------------------------------------ model
    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        self.tok = load_tokenizer(self.spec)
        base, self.load_record = load_policy_model(self.spec)
        done = int(self.state["step_done"])
        if done == 0:
            self.model = attach_lora(base, self.spec, derived_seeds(self.seed, 0)["lora_init"])
        else:
            log.info("resuming %s seed %d at step %d", self.arm, self.seed, done)
            self.model = load_step_adapter(base, self.step_dir(done) / "adapter", self.state.get("lora_dtype"))
        self.stop_ids = stop_token_ids(self.spec, self.tok)
        self.state.setdefault("load_records", []).append({"at_step": done, "utc": utc_now(), **(self.load_record or {})})
        self._save_state()

    # ------------------------------------------------------------------ one step
    def _candidates(self, t: int) -> list[SgacItem]:
        batch = self.schedule[t - 1]
        items = [self.pool_by_row[int(r)] for r in batch["rows"]]
        uids = [it.unique_id for it in items]
        if batch.get("unique_ids") and list(batch["unique_ids"]) != uids:
            raise RuntimeError(f"schedule step {t} unique_ids {batch['unique_ids']} != pool rows {uids}")
        return items

    def _select(self, t: int, sd: Path, timings: dict) -> tuple[SgacItem, dict]:
        if self.arm == "fixed_pi1":
            selection = {"step": t, "arm": self.arm, "rule": None, "selected_idx": 0, "selected_uid": self.pi1.unique_id,
                         "selected_row": self.pi1.row, "candidates": [], "picks": {}, "scores": {}}
            atomic_write_json(sd / "selection.json", selection)
            return self.pi1, selection
        items = self._candidates(t)
        atomic_write_json(sd / "batch.json", {"step": t, "rows": [it.row for it in items],
                                              "unique_ids": [it.unique_id for it in items]})
        seeds = derived_seeds(self.seed, t)
        with phase(timings, "sieve"):
            sampler = make_sieve_sampler(self.model, self.tok, self.spec, self.stop_ids)
            cands = sieve_candidates(sampler, self.tok, items, self.spec, self.grader, seed=seeds["sieve"],
                                     out_dir=sd / "sieve", policy_tag=f"step{t}")
        free_cuda()
        signals = [c["selection_signals"] for c in cands]
        picks = sel.all_picks(signals, random_arm_rng(self.seed, t))
        idx = picks[self.rule]["idx"]
        chosen = items[idx]
        selection = {
            "step": t, "arm": self.arm, "rule": self.rule, "selected_idx": int(idx), "selected_uid": chosen.unique_id,
            "selected_row": chosen.row, "selected_level": chosen.level, "selected_signals": signals[idx],
            "candidates": [{"idx": c["cand_idx"], "row": c["row"], "unique_id": c["unique_id"], "level": c["level"],
                            "signals": c["selection_signals"],
                            "legacy": {k: c["legacy"][k] for k in ("Ps", "Var", "D")},
                            "mv": {k: c["mv"][k] for k in ("p_s", "v_legacy", "u_ratio", "d_simpson", "trunc_rate",
                                                           "format_rate", "len_mean", "none_rate")}} for c in cands],
            "picks": {r: p["idx"] for r, p in picks.items()},
            "scores": {r: p["scores"] for r, p in picks.items()},
            "eq10_picked_max_level": sel.picked_max_level(signals, picks["sgac_eq10"]["idx"]),
        }
        atomic_write_json(sd / "selection.json", selection)
        log.info("[%s/%s seed %d] step %d: picked #%d %s (L%s, signals %s); all rules %s", self.spec.profile.name,
                 self.arm, self.seed, t, idx, chosen.unique_id, chosen.level, signals[idx], selection["picks"])
        return chosen, selection

    def _persist_adapter(self, t: int, chosen: SgacItem, burst: dict) -> None:
        sd = self.step_dir(t)
        tmp, final = sd / "adapter.tmp", sd / "adapter"
        for d in (tmp, final):
            if d.exists():
                shutil.rmtree(d)
        save_step_adapter(self.model, tmp, {"arm": self.arm, "seed": self.seed, "curriculum_step": t,
                                            "unique_id": chosen.unique_id, "spec_hash": self.spec.spec_hash(),
                                            "profile": self.spec.profile.name,
                                            "lora_dtype": (burst.get("lora_after") or {}).get("dtype")})
        os.replace(tmp, final)

    def _run_step(self, t: int) -> None:
        sd = self.step_dir(t)
        sd.mkdir(parents=True, exist_ok=True)
        timings: dict = {}
        chosen, selection = self._select(t, sd, timings)
        with phase(timings, "burst"):
            burst = run_burst(self.model, self.tok, chosen, self.spec, self.grader, sd / "train",
                              seed=derived_seeds(self.seed, t)["grpo"],
                              context={"curriculum_step": t, "arm": self.arm, "seed": self.seed}, stop_ids=self.stop_ids)
        with phase(timings, "save"):
            self._persist_adapter(t, chosen, burst)
        atomic_write_json(sd / "timing.json", timings)
        (sd / "done.flag").write_text(utc_now() + "\n", encoding="utf-8")
        self._commit(t, {"step": t, "selection": selection, "burst": _burst_brief(burst), "timing": timings})

    def _commit(self, t: int, entry: dict) -> None:
        self.state["history"] = [h for h in self.state["history"] if h["step"] != t] + [entry]
        self.state["history"].sort(key=lambda h: h["step"])
        self.state["step_done"] = t
        dtype = (entry.get("burst") or {}).get("lora_dtype_after")
        if dtype:
            self.state["lora_dtype"] = dtype.replace("torch.", "")
        self._save_state()
        self.rd.set_status(stage="curriculum", step_done=t)
        prev = t - 1
        if prev >= 1 and not self.keep_adapter(prev):
            ad = self.step_dir(prev) / "adapter"
            if ad.exists():
                shutil.rmtree(ad)

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self, t: int, name: str, items: Sequence[SgacItem], batch_size: int, tag: str | None = None) -> dict:
        timings: dict = {}
        with phase(timings, f"eval_{name}"):
            sampler = make_eval_sampler(self.model, self.tok, self.spec, self.stop_ids, batch_size)
            summary = evaluate_items(sampler, self.tok, items, self.spec, self.grader, self.eval_dir(tag or t, name),
                                     tag=f"{self.arm}-seed{self.seed}-{tag or f'step{t}'}-{name}",
                                     policy=f"{self.arm}/seed{self.seed}/{tag or f'step{t}'}",
                                     adapter_ref=None if tag else str(self.step_dir(t) / "adapter"))
        free_cuda()
        key = str(tag or t)
        self.state["evals"].setdefault(key, {})[name] = {
            "acc": summary["acc"], "ci": [summary["ci_lo"], summary["ci_hi"]], "acc_mv": summary["acc_mv"],
            "acc_legacy": summary["acc_legacy"], "n": summary["n"], "trunc_rate": summary["trunc_rate"],
            "timing": timings[f"eval_{name}"]}
        self._save_state()
        log.info("[%s/%s seed %d] eval %s %s: acc %.4f", self.spec.profile.name, self.arm, self.seed, key, name,
                 summary["acc"])
        return summary

    def _has_eval(self, key: str, name: str) -> bool:
        return name in self.state["evals"].get(str(key), {})

    def _evaluate_due(self, t: int) -> None:
        if self.eval_due(t) and not self._has_eval(t, "test50"):
            self._evaluate(t, "test50", self.test50, test50_batch_size(self.spec))
        if self.math500_due(t) and not self._has_eval(t, "math500"):
            self._evaluate(t, "math500", self.math500, int(self.spec.profile.eval_batch_size))

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        if not self.rd.exists():
            self.rd.init(self.seed, extra={"arm": self.arm, "rule": self.rule})
        self.state = self._load_state()
        self._reconcile()
        steps = int(self.spec.loop.steps)
        if self.state["step_done"] >= steps and self._all_evals_present() and self.rd.is_done():
            log.info("%s already complete", self.rd.path)
            return self.state
        self._ensure_model()
        self.rd.set_status(state="running", stage="curriculum", step_done=self.state["step_done"])
        done = int(self.state["step_done"])
        if done > 0:
            self._evaluate_due(done)  # an evaluation interrupted after its step was committed
        for t in range(done + 1, steps + 1):
            self._run_step(t)
            self._evaluate_due(t)
        if self.base_after_run and not self._has_eval("base_after_run", "test50"):
            with self.model.disable_adapter():  # NB-M cell 11: base measured AFTER training, adapter disabled
                self._evaluate(steps, "test50", self.test50, test50_batch_size(self.spec), tag="base_after_run")
        self.rd.mark_done(step_done=self.state["step_done"])
        return self.state

    def _all_evals_present(self) -> bool:
        steps = int(self.spec.loop.steps)
        need = [(t, "test50") for t in range(1, steps + 1) if self.eval_due(t)]
        need += [(t, "math500") for t in range(1, steps + 1) if self.math500_due(t)]
        if self.base_after_run:
            need.append(("base_after_run", "test50"))
        return all(self._has_eval(k, n) for k, n in need)


def _burst_brief(b: dict) -> dict:
    keys = ("unique_id", "steps_done", "losses", "loss_trl100", "all_zero_loss", "reward", "reward_std",
            "frac_reward_zero_std", "clipped_ratio", "mean_length", "learning_rates", "grad_norms", "delta_A", "delta_B",
            "peak_mem_gb", "wall_s", "lora_dtype_in_burst", "hooks_removed", "extra_adapters_removed")
    out = {k: b.get(k) for k in keys}
    out["lora_dtype_after"] = (b.get("lora_after") or {}).get("dtype")
    return out
