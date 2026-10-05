"""Curriculum resume/arm semantics with every heavy collaborator monkeypatched on `rlvr_v2.curriculum`."""
import contextlib
import dataclasses
import json
from pathlib import Path

import pytest

import rlvr_v2.curriculum as cur
from rlvr_v2.artifacts import RunDir, atomic_write_json, read_json
from rlvr_v2.config import Config
from rlvr_v2.data import Problem
from rlvr_v2.train_grpo import BurstResult

SEED = 7


# ---------------------------------------------------------------------- fakes
class DummyTok:
    chat_template = None
    eos_token_id = 151645
    pad_token_id = 151643
    unk_token_id = None

    def convert_tokens_to_ids(self, s):
        return {"<|im_end|>": 151645, "<|endoftext|>": 151643}.get(s)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [0] * len(text.split())}


class DummyModel:
    def __init__(self, name):
        self.name = name
        self.trained_on: list[str] = []


class FakeSummary:
    def __init__(self, acc, n=3):
        self.acc, self.ci_lo, self.ci_hi, self.n = acc, acc - 0.1, acc + 0.1, n

    def to_dict(self):
        return {"acc": self.acc, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi, "n": self.n}


def _idx(uid: str) -> int:
    return int(uid.rsplit("p", 1)[1])


def p_s_of(uid: str) -> float:
    return (_idx(uid) % 9) / 8.0


def d_simpson_of(uid: str) -> float:
    return ((_idx(uid) * 7) % 11) / 10.0


def make_pool(n=40):
    return [Problem(f"math_train/p{i}", f"Problem number {i}?", str(i), None, (i % 5) + 1, "Algebra", "math_train")
            for i in range(n)]


def eval_problems():
    return [Problem(f"math500/e{i}", f"Eval {i}", "1", None, 1, "Algebra", "math500") for i in range(3)]


class Fakes:
    def __init__(self, mk_signals):
        self.mk = mk_signals
        self.sieve_calls, self.burst_calls, self.eval_calls, self.saves = [], [], [], []
        self.loads = self.fresh = 0

    def measure_signals(self, sampler, tokenizer, problems, cfg, grader, policy_tag, k, seed, out_dir=None):
        self.sieve_calls.append({"uids": [p.unique_id for p in problems], "policy_tag": policy_tag, "k": k,
                                 "seed": seed, "out_dir": Path(out_dir), "trained_on": list(sampler.trained_on)})
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return [self.mk(p.unique_id, p_s=p_s_of(p.unique_id), d_simpson=d_simpson_of(p.unique_id), k=k)
                for p in problems]

    def run_grpo_burst(self, model, tokenizer, problems, cfg, out_dir, max_steps, seed, grader, context=None,
                       resume_from_checkpoint=False):
        self.burst_calls.append({"uids": [p.unique_id for p in problems], "out_dir": Path(out_dir),
                                 "max_steps": max_steps, "seed": seed, "context": dict(context or {}), "model": model})
        model.trained_on.append(problems[0].unique_id)
        return BurstResult(steps_done=max_steps, metrics_path=Path(out_dir) / "train_metrics.jsonl",
                           train_rollouts_path=None, reward_correct_mean_first=0.1, reward_correct_mean_last=0.2,
                           clipped_ratio_mean=0.0, frac_zero_std_mean=0.0, mean_completion_length=100.0, wall_s=1.0,
                           logs=[{"global_step": 1}])

    def evaluate(self, sampler, tokenizer, problems, cfg, grader, out_dir, tag, policy="base", adapter_path=None):
        self.eval_calls.append({"tag": tag, "policy": policy, "adapter_path": adapter_path, "n": len(problems),
                                "trained_on": list(sampler.trained_on)})
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return FakeSummary(0.3 + 0.01 * len(self.eval_calls))

    def save_adapter(self, model, out_dir, meta=None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_model.json").write_text(json.dumps({"meta": meta, "trained_on": model.trained_on}))
        self.saves.append(dict(meta or {}))

    def load_adapter(self, base, adapter_dir):
        self.loads += 1
        d = json.loads((Path(adapter_dir) / "adapter_model.json").read_text())
        m = DummyModel("loaded")
        m.trained_on = list(d["trained_on"])
        return m

    def attach_fresh_lora(self, base, cfg):
        self.fresh += 1
        return DummyModel("fresh")

    def make_sampler(self, cfg, model=None, tokenizer=None, adapter_path=None, stop_token_ids=None):
        assert cfg.gen.backend == "hf" and model is not None and stop_token_ids == [151645, 151643]
        return model  # the "sampler" stands in for the live policy

    @contextlib.contextmanager
    def generation_mode(self, model):
        yield model


def install(monkeypatch, fakes: Fakes) -> None:
    for name in ("measure_signals", "run_grpo_burst", "evaluate", "save_adapter", "load_adapter",
                 "attach_fresh_lora", "make_sampler", "generation_mode"):
        monkeypatch.setattr(cur, name, getattr(fakes, name))
    monkeypatch.setattr(cur, "load_tokenizer", lambda cfg: DummyTok())
    monkeypatch.setattr(cur, "load_base_model", lambda cfg: object())
    monkeypatch.setattr(cur, "free_cuda", lambda: None)


def small_cfg(tmp_path, arms=("random", "variance", "disagreement", "repeat_one"), seeds=(SEED,)):
    cfg = Config()
    cfg = dataclasses.replace(
        cfg,
        run=dataclasses.replace(cfg.run, results_root=str(tmp_path), require_gates=False),
        gen=dataclasses.replace(cfg.gen, sieve=dataclasses.replace(cfg.gen.sieve, n=4), max_new_tokens=64),
        train=dataclasses.replace(cfg.train, num_generations=4, per_device_train_batch_size=2, rounds=2),
        study2=dataclasses.replace(cfg.study2, steps=3, batch_b=4, k=4, eval_every=2, rounds_per_burst=2,
                                   arms=arms, seeds=seeds),
    )
    cfg.validate()
    return cfg


def run_arm(cfg, arm, grader, repeat=None, seed=SEED):
    run_dir = RunDir(cfg, "study2", f"{arm}__seed{seed}")
    c = cur.Curriculum(cfg, arm, seed, run_dir, make_pool(), eval_problems(), grader, repeat_problem=repeat)
    return run_dir, c.run()


def _history(state):
    return {h["step"]: h for h in state["history"]}


# ---------------------------------------------------------------------- tests
def test_three_step_curriculum_per_arm(tmp_path, mk_signals, grader, monkeypatch):
    cfg = small_cfg(tmp_path)
    picks, batches = {}, {}
    for arm in ("random", "variance", "disagreement"):
        fakes = Fakes(mk_signals)
        install(monkeypatch, fakes)
        run_dir, state = run_arm(cfg, arm, grader)
        hist = _history(state)
        assert state["arm"] == arm and state["seed"] == SEED and state["step_done"] == 3 and len(hist) == 3
        assert len(state["used_ids"]) == 12 == len(set(state["used_ids"]))
        assert [e["step"] for e in state["evals"]] == [0, 2, 3]
        assert all(e["ci"] == [pytest.approx(e["acc"] - 0.1), pytest.approx(e["acc"] + 0.1)] for e in state["evals"])
        # equal budget: 3 bursts x rounds_per_burst, each tagged with its curriculum step and arm
        assert [c["max_steps"] for c in fakes.burst_calls] == [2, 2, 2]
        assert [c["context"] for c in fakes.burst_calls] == [{"curriculum_step": t, "arm": arm} for t in (1, 2, 3)]
        assert [c["seed"] for c in fakes.burst_calls] == [SEED * 1000 + t for t in (1, 2, 3)]
        # sieve of the CURRENT policy at every step: K = study2.k, per-step seed, per-step output dir
        assert [c["policy_tag"] for c in fakes.sieve_calls] == ["step1", "step2", "step3"]
        assert all(c["k"] == 4 and len(c["uids"]) == 4 for c in fakes.sieve_calls)
        assert [c["seed"] for c in fakes.sieve_calls] == [SEED * 1000 + t for t in (1, 2, 3)]
        assert [c["out_dir"] for c in fakes.sieve_calls] == [run_dir.path / "steps" / f"step-{t:03d}" / "sieve"
                                                            for t in (1, 2, 3)]
        # step 3's sieve saw the policy trained on steps 1-2
        assert fakes.sieve_calls[2]["trained_on"] == [hist[1]["selected_uid"], hist[2]["selected_uid"]]
        assert fakes.sieve_calls[0]["trained_on"] == []
        # per-step artefacts
        for t in (1, 2, 3):
            sd = run_dir.path / "steps" / f"step-{t:03d}"
            assert (sd / "done.flag").exists()
            sel = read_json(sd / "selection.json")
            assert sel["selected_uid"] == hist[t]["selected_uid"] == fakes.burst_calls[t - 1]["uids"][0]
            assert set(sel["scores"]) == set(sel["candidate_uids"]) == set(fakes.sieve_calls[t - 1]["uids"])
            assert sel["signals_of_selected"]["unique_id"] == sel["selected_uid"]
            assert hist[t]["burst_summary"]["steps_done"] == 2 and "logs" not in hist[t]["burst_summary"]
        # the selection rule of the arm
        for call, t in zip(fakes.sieve_calls, (1, 2, 3)):
            chosen = hist[t]["selected_uid"]
            if arm == "variance":
                v = {u: p_s_of(u) * (1 - p_s_of(u)) for u in call["uids"]}
                assert v[chosen] == max(v.values())
            elif arm == "disagreement":
                assert d_simpson_of(chosen) == max(d_simpson_of(u) for u in call["uids"])
        assert cur.read_adapter_marker(run_dir.path / "adapter") == 3
        assert fakes.fresh == 1 and fakes.loads == 0 and len(fakes.saves) == 3
        assert run_dir.status()["state"] == "done"
        picks[arm] = [hist[t]["selected_uid"] for t in (1, 2, 3)]
        batches[arm] = [c["uids"] for c in fakes.sieve_calls]
    # same seed -> identical candidate batches for every arm; the arms differ only in what they select
    assert batches["random"] == batches["variance"] == batches["disagreement"]
    assert len({tuple(p) for p in picks.values()}) >= 2
    assert picks["variance"] != picks["disagreement"]


def test_resume_redoes_only_the_crashed_step(tmp_path, mk_signals, grader, monkeypatch):
    cfg = small_cfg(tmp_path)
    fakes = Fakes(mk_signals)
    install(monkeypatch, fakes)
    run_dir, state = run_arm(cfg, "variance", grader)
    first_picks = [h["selected_uid"] for h in state["history"]]
    adapter = run_dir.path / "adapter"

    # Simulate a crash during step 3's burst: no done.flag, no state entry, adapter still at step 2.
    (run_dir.path / "steps" / "step-003" / "done.flag").unlink()
    st = read_json(run_dir.path / "state.json")
    st["history"], st["step_done"], st["used_ids"] = st["history"][:2], 2, st["used_ids"][:8]
    st["evals"] = [e for e in st["evals"] if e["step"] != 3]
    atomic_write_json(run_dir.path / "state.json", st)
    cur.write_adapter_marker(adapter, 2, "variance", SEED)
    (adapter / "adapter_model.json").write_text(json.dumps({"meta": {}, "trained_on": first_picks[:2]}))

    fakes2 = Fakes(mk_signals)
    install(monkeypatch, fakes2)
    run_dir2, state2 = run_arm(cfg, "variance", grader)
    assert run_dir2.path == run_dir.path
    assert fakes2.loads == 1 and fakes2.fresh == 0  # resumed from the persisted adapter
    assert (run_dir.path / "steps" / "step-003" / "done.flag").exists()
    assert [c["policy_tag"] for c in fakes2.sieve_calls] == ["step3"]  # steps 1-2 were not re-sieved
    assert [c["context"]["curriculum_step"] for c in fakes2.burst_calls] == [3]
    assert [c["tag"] for c in fakes2.eval_calls] == ["step-003"]  # earlier evals kept
    assert [h["selected_uid"] for h in state2["history"]] == first_picks  # deterministic redo
    assert state2["step_done"] == 3 and len(set(state2["used_ids"])) == 12
    assert [e["step"] for e in state2["evals"]] == [0, 2, 3]
    assert json.loads((adapter / "adapter_model.json").read_text())["trained_on"] == first_picks
    assert cur.read_adapter_marker(adapter) == 3

    # A third invocation is a no-op apart from reloading the adapter.
    fakes3 = Fakes(mk_signals)
    install(monkeypatch, fakes3)
    _, state3 = run_arm(cfg, "variance", grader)
    assert fakes3.loads == 1 and not fakes3.sieve_calls and not fakes3.burst_calls and not fakes3.eval_calls
    assert state3["step_done"] == 3 and len(state3["history"]) == 3


def test_resume_completes_bookkeeping_when_adapter_is_ahead(tmp_path, mk_signals, grader, monkeypatch):
    """Crash after the adapter swap but before state.json was updated: no retraining, state is repaired."""
    cfg = small_cfg(tmp_path)
    fakes = Fakes(mk_signals)
    install(monkeypatch, fakes)
    run_dir, state = run_arm(cfg, "random", grader)
    picks = [h["selected_uid"] for h in state["history"]]
    sd3 = run_dir.path / "steps" / "step-003"
    (sd3 / "done.flag").unlink()
    atomic_write_json(sd3 / "train" / "burst_summary.json", {"steps_done": 2, "logs": [1, 2]})
    st = read_json(run_dir.path / "state.json")
    st["history"], st["step_done"], st["used_ids"] = st["history"][:2], 2, st["used_ids"][:8]
    st["evals"] = [e for e in st["evals"] if e["step"] != 3]
    atomic_write_json(run_dir.path / "state.json", st)  # adapter marker stays at 3

    fakes2 = Fakes(mk_signals)
    install(monkeypatch, fakes2)
    _, state2 = run_arm(cfg, "random", grader)
    assert not fakes2.sieve_calls and not fakes2.burst_calls
    assert [h["selected_uid"] for h in state2["history"]] == picks
    assert state2["step_done"] == 3 and len(set(state2["used_ids"])) == 12
    assert _history(state2)[3]["burst_summary"] == {"steps_done": 2}
    assert [c["tag"] for c in fakes2.eval_calls] == ["step-003"] and (sd3 / "done.flag").exists()

    # an adapter further ahead than one step is an inconsistency we refuse to paper over
    cur.write_adapter_marker(run_dir.path / "adapter", 5, "random", SEED)
    with pytest.raises(RuntimeError, match="curriculum step 5"):
        run_arm(cfg, "random", grader)


def test_repeat_one_never_sieves(tmp_path, mk_signals, grader, monkeypatch):
    cfg = small_cfg(tmp_path)
    fakes = Fakes(mk_signals)
    install(monkeypatch, fakes)
    pi1 = Problem("oneshot/pi1", "pi1 text", "12.8", None, None, "Algebra", "oneshot")
    run_dir, state = run_arm(cfg, "repeat_one", grader, repeat=pi1)
    assert fakes.sieve_calls == []
    assert [c["uids"] for c in fakes.burst_calls] == [["oneshot/pi1"]] * 3
    assert [c["max_steps"] for c in fakes.burst_calls] == [2, 2, 2]  # same budget as the sieving arms
    assert state["used_ids"] == [] and [h["selected_uid"] for h in state["history"]] == ["oneshot/pi1"] * 3
    assert [e["step"] for e in state["evals"]] == [0, 2, 3]
    assert read_json(run_dir.path / "steps" / "step-002" / "selection.json")["signals_of_selected"] is None
    with pytest.raises(ValueError, match="repeat_problem"):
        cur.Curriculum(cfg, "repeat_one", SEED, run_dir, make_pool(), eval_problems(), grader)
    with pytest.raises(ValueError, match="unknown arm"):
        cur.Curriculum(cfg, "bogus", SEED, run_dir, make_pool(), eval_problems(), grader)
    with pytest.raises(ValueError, match="learned_path"):
        cur.Curriculum(cfg, "learned", SEED, run_dir, make_pool(), eval_problems(), grader)


def test_expand_study2_jobs(tmp_path):
    cfg = small_cfg(tmp_path, arms=("random", "variance"), seeds=(1, 2, 3))
    jobs = cur.expand_study2_jobs(cfg)
    assert len(jobs) == 6 and [j["index"] for j in jobs] == list(range(6))
    assert jobs[0] == {"index": 0, "arm": "random", "seed": 1} and jobs[3] == {"index": 3, "arm": "variance", "seed": 1}


def test_ineligible_pool_items_are_never_drawn(tmp_path, mk_signals, grader, monkeypatch):
    """prereg §6: the pool-eligibility rule applies to Study 2 as well."""
    cfg = small_cfg(tmp_path)
    install(monkeypatch, Fakes(mk_signals))
    banned = [f"math_train/p{i}" for i in range(0, 40, 2)]  # half the pool
    run_dir = RunDir(cfg, "study2", f"random__seed{SEED}")
    c = cur.Curriculum(cfg, "random", SEED, run_dir, make_pool(), eval_problems(), grader, ineligible=banned)
    assert len(c.pool) == 20 and not {p.unique_id for p in c.pool} & set(banned)
    state = c.run()
    drawn = set(state["used_ids"]) | {h["selected_uid"] for h in state["history"]}
    assert drawn and not drawn & set(banned)
