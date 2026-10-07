"""SGAC loop semantics with every heavy collaborator monkeypatched on `rlvr_v2.sgac.loop`."""
import contextlib
import json
from pathlib import Path

import pytest

import rlvr_v2.sgac.loop as lp
from rlvr_v2.artifacts import atomic_write_json, read_json
from rlvr_v2.sgac import selection as sel
from rlvr_v2.sgac.data import SgacItem
from rlvr_v2.sgac.schedule import candidate_schedule
from rlvr_v2.sgac.spec import load_profile

SEED = 42


class DummyModel:
    def __init__(self, name):
        self.name = name
        self.trained_on: list[str] = []
        self.adapter_disabled = False

    @contextlib.contextmanager
    def disable_adapter(self):
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.adapter_disabled = False


def make_pool(n=60):
    return [SgacItem(row=i, orig_index=i, unique_id=f"p{i}", problem=f"Problem {i}", solution=f"\\boxed{{{i}}}",
                     answer=str(i), level=(i % 5) + 1, subject="Algebra") for i in range(n)]


def make_test():
    return [SgacItem(row=1000 + i, orig_index=i, unique_id=f"t{i}", problem=f"Test {i}", solution=None, answer="1",
                     level=1, subject="Algebra") for i in range(3)]


def signals_of(uid: str) -> dict:
    i = int(uid[1:])
    return {"Ps": (i % 5) / 4, "Var": ((i * 7) % 10) / 20, "D": ((i % 4) + 1) / 4, "L": (i % 5) + 1}


class Fakes:
    def __init__(self):
        self.sieve_calls, self.burst_calls, self.eval_calls, self.saves = [], [], [], []
        self.loads = self.fresh = 0
        self.crash_burst_at: int | None = None

    def sieve_candidates(self, sampler, tok, items, spec, grader, seed, out_dir, policy_tag):
        self.sieve_calls.append({"uids": [it.unique_id for it in items], "seed": seed, "trained_on": list(sampler.trained_on)})
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return [{"cand_idx": i, "row": it.row, "unique_id": it.unique_id, "level": it.level,
                 "selection_signals": signals_of(it.unique_id), "legacy": {"Ps": 0, "Var": 0, "D": 0},
                 "mv": {k: 0 for k in ("p_s", "v_legacy", "u_ratio", "d_simpson", "trunc_rate", "format_rate",
                                       "len_mean", "none_rate")}} for i, it in enumerate(items)]

    def run_burst(self, model, tok, item, spec, grader, out_dir, seed, context, stop_ids):
        t = context["curriculum_step"]
        if self.crash_burst_at == t:
            self.crash_burst_at = None
            raise RuntimeError("simulated crash mid-burst")
        self.burst_calls.append({"uid": item.unique_id, "step": t, "seed": seed})
        model.trained_on.append(item.unique_id)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        summary = {"unique_id": item.unique_id, "steps_done": 5, "losses": [0.0] * 5, "all_zero_loss": True,
                   "lora_after": {"dtype": "torch.float32"}, "reward": [0.5], "peak_mem_gb": None, "wall_s": 0.1}
        atomic_write_json(Path(out_dir) / "burst_summary.json", summary)
        return summary

    def evaluate_items(self, sampler, tok, items, spec, grader, out_dir, tag, policy, adapter_ref=None):
        self.eval_calls.append({"tag": tag, "n": len(items), "trained_on": list(sampler.trained_on),
                                "disabled": sampler.adapter_disabled})
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return {"acc": 0.5, "ci_lo": 0.2, "ci_hi": 0.8, "acc_mv": 0.5, "acc_legacy": 0.6, "n": len(items),
                "trunc_rate": 0.0}

    def save_step_adapter(self, model, out_dir, meta):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter.json").write_text(json.dumps({"trained_on": model.trained_on, "meta": meta}))
        self.saves.append(meta["curriculum_step"])

    def load_step_adapter(self, base, adapter_dir, lora_dtype):
        self.loads += 1
        m = DummyModel("loaded")
        m.trained_on = json.loads((Path(adapter_dir) / "adapter.json").read_text())["trained_on"]
        return m

    def attach_lora(self, base, spec, seed):
        self.fresh += 1
        return DummyModel("fresh")


def install(monkeypatch, f: Fakes):
    for name in ("sieve_candidates", "run_burst", "evaluate_items", "save_step_adapter", "load_step_adapter",
                 "attach_lora"):
        monkeypatch.setattr(lp, name, getattr(f, name))
    monkeypatch.setattr(lp, "load_tokenizer", lambda spec: object())
    monkeypatch.setattr(lp, "load_policy_model", lambda spec: (object(), {"effective": "fake"}))
    monkeypatch.setattr(lp, "stop_token_ids", lambda spec, tok: [1])
    monkeypatch.setattr(lp, "make_sieve_sampler", lambda model, tok, spec, stop: model)
    monkeypatch.setattr(lp, "make_eval_sampler", lambda model, tok, spec, stop, bs=None: model)
    monkeypatch.setattr(lp, "free_cuda", lambda: None)


def make_run(tmp_path, arm, base_after_run=False, steps=6):
    spec = load_profile("e0", [f"run.results_root={tmp_path.as_posix()}", f"loop.steps={steps}", "loop.eval_every=3",
                               f"loop.math500_steps=[0,{steps}]"])
    pool = make_pool()
    sched = [{"step": t + 1, "rows": b, "unique_ids": [f"p{r}" for r in b]}
             for t, b in enumerate(candidate_schedule([p.row for p in pool], SEED, steps, 4))]
    return lp.SgacRun(spec, arm, SEED, pool=pool, test50=make_test(), math500=make_test(), schedule=sched,
                      grader=None, pi1=SgacItem(-1, -1, "oneshot/pi1", "pi1?", None, "12.8", None, "Algebra", "oneshot"),
                      base_after_run=base_after_run)


def test_arms_see_identical_batches_and_pick_by_their_rule(tmp_path, monkeypatch):
    batches = {}
    for arm in ("sgac", "random", "max_var"):
        f = Fakes()
        install(monkeypatch, f)
        run = make_run(tmp_path, arm)
        state = run.run()
        batches[arm] = [c["uids"] for c in f.sieve_calls]
        for h in state["history"]:
            s = h["selection"]
            cand_sigs = [c["signals"] for c in s["candidates"]]
            assert set(s["picks"]) == set(sel.RULES)
            assert s["selected_idx"] == s["picks"][sel.ARM_RULE[arm]]
            if arm != "random":
                assert s["selected_idx"] == sel.pick(sel.ARM_RULE[arm], cand_sigs)[0]
            assert s["eq10_picked_max_level"] is True
        assert [c["uid"] for c in f.burst_calls] == [h["selection"]["selected_uid"] for h in state["history"]]
    assert batches["sgac"] == batches["random"] == batches["max_var"]


def test_eval_schedule_adapter_retention_and_base_after_run(tmp_path, monkeypatch):
    f = Fakes()
    install(monkeypatch, f)
    run = make_run(tmp_path, "sgac", base_after_run=True)
    state = run.run()
    assert state["step_done"] == 6
    assert sorted(state["evals"]) == ["3", "6", "base_after_run"]
    assert set(state["evals"]["6"]) == {"test50", "math500"} and set(state["evals"]["3"]) == {"test50"}
    assert f.eval_calls[-1]["disabled"] is True  # base_after_run evaluates with the adapter disabled
    kept = sorted(p.parent.name for p in (run.rd.path / "steps").glob("step-*/adapter"))
    assert kept == ["step-003", "step-006"]
    assert run.rd.is_done()


def test_crash_mid_burst_redoes_only_that_step(tmp_path, monkeypatch):
    ref = Fakes()
    install(monkeypatch, ref)
    reference = make_run(tmp_path / "ref", "sgac").run()
    f = Fakes()
    f.crash_burst_at = 4
    install(monkeypatch, f)
    with pytest.raises(RuntimeError):
        make_run(tmp_path / "x", "sgac").run()
    st = make_run(tmp_path / "x", "sgac").run()
    assert f.loads == 1 and [c["step"] for c in f.burst_calls] == [1, 2, 3, 4, 5, 6]
    assert [h["selection"]["selected_uid"] for h in st["history"]] == \
           [h["selection"]["selected_uid"] for h in reference["history"]]
    calls = len(f.burst_calls)
    make_run(tmp_path / "x", "sgac").run()  # third invocation: no-op
    assert len(f.burst_calls) == calls


def test_lost_state_update_is_committed_without_retraining(tmp_path, monkeypatch):
    f = Fakes()
    install(monkeypatch, f)
    run = make_run(tmp_path, "max_d")
    run.run()
    st = read_json(run.state_path)
    st["step_done"] = 5  # pretend the process died right after step 6 wrote done.flag
    st["history"] = [h for h in st["history"] if h["step"] < 6]
    st["evals"] = {k: v for k, v in st["evals"].items() if k != "6"}
    atomic_write_json(run.state_path, st)
    run.rd.set_status(state="running")
    bursts = len(f.burst_calls)
    st2 = make_run(tmp_path, "max_d").run()
    assert len(f.burst_calls) == bursts and st2["step_done"] == 6 and "6" in st2["evals"]


def test_state_of_another_arm_is_refused(tmp_path, monkeypatch):
    f = Fakes()
    install(monkeypatch, f)
    run = make_run(tmp_path, "sgac", steps=3)
    run.run()
    st = read_json(run.state_path)
    st["arm"] = "random"
    atomic_write_json(run.state_path, st)
    with pytest.raises(RuntimeError):
        make_run(tmp_path, "sgac", steps=3).run()


def test_fixed_pi1_trains_pi1_every_step_without_sieve(tmp_path, monkeypatch):
    f = Fakes()
    install(monkeypatch, f)
    st = make_run(tmp_path, "fixed_pi1", steps=3).run()
    assert not f.sieve_calls and [c["uid"] for c in f.burst_calls] == ["oneshot/pi1"] * 3
    assert st["step_done"] == 3
