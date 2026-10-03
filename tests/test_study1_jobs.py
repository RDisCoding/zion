import contextlib
import csv
import dataclasses
from pathlib import Path

import pytest

import rlvr_v2.study1 as s1
from rlvr_v2.artifacts import RunDir, read_json
from rlvr_v2.config import Config
from rlvr_v2.data import Manifest, Problem
from rlvr_v2.study1 import Job, expand_jobs, run_name_for, slug, write_jobs_csv
from rlvr_v2.train_grpo import BurstResult


def _manifest(n=5, **meta):
    ids = [f"math_train/algebra/{i}.json" for i in range(n)]
    return Manifest(name="cands", source="math_train", ids=ids, hashes={u: "0" * 16 for u in ids}, meta=meta)


def _cfg(tmp_path, replicate_count=2):
    cfg = Config()
    cfg = dataclasses.replace(
        cfg,
        run=dataclasses.replace(cfg.run, results_root=str(tmp_path), require_gates=False),
        gen=dataclasses.replace(cfg.gen, sieve=dataclasses.replace(cfg.gen.sieve, n=4)),
        study2=dataclasses.replace(cfg.study2, k=4),
        study1=dataclasses.replace(cfg.study1, replicate_count=replicate_count),
    )
    cfg.validate()
    return cfg


def test_expand_jobs_counts_and_seeds(tmp_path):
    cfg = _cfg(tmp_path)
    m = _manifest()
    jobs = expand_jobs(cfg, m)
    assert len(jobs) == 5 + 2 and [j.index for j in jobs] == list(range(7))
    assert [j.seed for j in jobs[:5]] == [1000 + i for i in range(5)] and not any(j.replicate for j in jobs[:5])
    assert [(j.unique_id, j.seed, j.replicate) for j in jobs[5:]] == [(m.ids[0], 101000, True), (m.ids[1], 101001, True)]
    assert expand_jobs(cfg, m) == jobs  # deterministic
    m2 = _manifest(replicate_ids=[m.ids[3], m.ids[1], m.ids[4]])  # explicit list, capped at replicate_count
    assert [(j.unique_id, j.seed) for j in expand_jobs(cfg, m2) if j.replicate] == [(m.ids[3], 101003), (m.ids[1], 101001)]
    assert len(expand_jobs(_cfg(tmp_path, replicate_count=0), m)) == 5
    with pytest.raises(ValueError):
        expand_jobs(cfg, _manifest(replicate_ids=["nope"]))


def test_write_jobs_csv_and_slug(tmp_path):
    jobs = expand_jobs(_cfg(tmp_path), _manifest(3))
    path = write_jobs_csv(jobs, tmp_path / "jobs" / "study1_jobs.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5 and rows[0]["unique_id"] == "math_train/algebra/0.json" and rows[0]["seed"] == "1000"
    assert rows[4]["replicate"] == "1" and rows[4]["run_name"] == "math_train-algebra-1-json__seed101001"
    assert slug("math_train/algebra/0.json") == "math_train-algebra-0-json"
    assert run_name_for(Job(0, "a/b.c", 5, False)) == "a-b-c__seed5"


def test_run_job_writes_result_and_skips_finished_evals(tmp_path, monkeypatch, grader):
    cfg = _cfg(tmp_path)
    m = _manifest(2)
    job = expand_jobs(cfg, m)[1]
    pool = [Problem(u, f"problem {u}", "7", None, 1, "Algebra", "math_train") for u in m.ids]
    evalp = [Problem(f"math500/{i}", f"eval {i}", "1", None, 1, "Algebra", "math500") for i in range(3)]
    calls = {"train": 0, "eval": []}

    def fake_one_shot(problem, cfg_, seed, run_dir, rounds=None, model=None, tokenizer=None, grader=None):
        calls["train"] += 1
        assert problem.unique_id == job.unique_id and seed == job.seed
        adapter = run_dir.stage("adapter")
        (adapter / "adapter_model.safetensors").write_bytes(b"x")
        return adapter, BurstResult(2, run_dir.path / "train" / "train_metrics.jsonl", None, 0.1, 0.5, 0.0, 0.2,
                                    100.0, 3.0, logs=[{"global_step": 1}])

    class Summary:
        acc, ci_lo, ci_hi, n = 0.4, 0.3, 0.5, 3

        def to_dict(self):
            return {"acc": 0.4, "ci_lo": 0.3, "ci_hi": 0.5, "n": 3}

    def fake_evaluate(sampler, tokenizer, problems, cfg_, grader_, out_dir, tag, policy="base", adapter_path=None):
        calls["eval"].append((tag, len(problems), policy, adapter_path, sampler))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return Summary()

    monkeypatch.setattr(s1, "run_one_shot_grpo", fake_one_shot)
    monkeypatch.setattr(s1, "evaluate", fake_evaluate)
    monkeypatch.setattr(s1, "load_tokenizer", lambda cfg_: "tok")
    monkeypatch.setattr(s1, "load_base_model", lambda cfg_: "base")
    monkeypatch.setattr(s1, "load_adapter", lambda base, d: ("adapter-model", str(d)))
    monkeypatch.setattr(s1, "make_sampler",
                        lambda cfg_, model=None, tokenizer=None, adapter_path=None, stop_token_ids=None: model)
    monkeypatch.setattr(s1, "generation_mode", lambda model: contextlib.nullcontext(model))
    monkeypatch.setattr(s1, "free_cuda", lambda: None)

    result = s1.run_job(cfg, job, pool, evalp, None, grader)
    run_dir = RunDir(cfg, "study1", run_name_for(job))
    assert run_dir.path.name == "math_train-algebra-1-json__seed1001"
    assert result["burst"]["steps_done"] == 2 and "logs" not in result["burst"]
    assert result["acc_math500"] == 0.4 and set(result["evals"]) == {"math500"}
    assert read_json(run_dir.path / "result.json")["unique_id"] == job.unique_id
    assert read_json(run_dir.path / "eval" / "math500" / "eval_summary.json")["acc"] == 0.4
    assert run_dir.status()["state"] == "done"
    assert calls["train"] == 1
    assert [(c[0], c[1], c[2]) for c in calls["eval"]] == [(f"{run_name_for(job)}-math500", 3, "adapter")]
    assert calls["eval"][0][3] == str(run_dir.path / "adapter")
    assert calls["eval"][0][4] == ("adapter-model", str(run_dir.path / "adapter"))  # adapter loaded on a fresh base

    # second invocation with a held-out set: math500 is skipped, only heldout is evaluated
    result2 = s1.run_job(cfg, job, pool, evalp, evalp[:2], grader)
    assert len(calls["eval"]) == 2 and calls["eval"][1][0].endswith("-heldout") and calls["eval"][1][1] == 2
    assert set(result2["evals"]) == {"math500", "heldout"} and result2["acc_heldout"] == 0.4

    # eval-only stage without an adapter fails loudly and marks the run failed
    other = expand_jobs(cfg, m)[0]
    with pytest.raises(RuntimeError, match="adapter"):
        s1.run_job(cfg, other, pool, evalp, None, grader, stages=("eval",))
    assert RunDir(cfg, "study1", run_name_for(other)).status()["state"] == "failed"
    with pytest.raises(KeyError):
        s1.run_job(cfg, Job(9, "missing", 1, False), pool, evalp, None, grader)
