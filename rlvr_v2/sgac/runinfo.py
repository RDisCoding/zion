"""Run directories and provenance for SGAC runs (`results_sgac/<profile>/<spec_hash>/<run_name>/`)."""
from __future__ import annotations

import contextlib
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from ..artifacts import REPO_ROOT, atomic_write_json, git_hash, gpu_info, package_versions, read_json, utc_now
from .spec import SgacSpec, group_dir, save_spec


def git_dirty() -> bool | None:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        return bool(out.stdout.strip())
    except Exception:
        return None


def extra_versions() -> dict:
    import importlib.metadata as md

    out = {}
    for name in ("bitsandbytes", "sympy", "antlr4-python3-runtime", "math-verify", "latex2sympy2_extended", "safetensors"):
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = "absent"
    return out


def gpu_details() -> dict:
    info = gpu_info()
    try:
        import torch

        if torch.cuda.is_available():
            info["capability"] = list(torch.cuda.get_device_capability(0))
            info["cuda_runtime"] = torch.version.cuda
    except Exception:  # pragma: no cover
        pass
    return info


def run_manifest(spec: SgacSpec, seed: int | None, extra: dict | None = None) -> dict:
    return {
        "created": utc_now(), "spec_hash": spec.spec_hash(), "profile": spec.profile.name, "seed": seed,
        "git_hash": git_hash(), "git_dirty": git_dirty(), "versions": {**package_versions(), **extra_versions()},
        "gpu": gpu_details(), "host": socket.gethostname(), "platform": platform.platform(), "argv": sys.argv,
        "dataset": {"name": spec.data.dataset, "revision": spec.data.revision}, "model": spec.to_dict()["model"],
        **(extra or {}),
    }


class SgacRunDir:
    def __init__(self, spec: SgacSpec, run_name: str):
        self.spec = spec
        self.run_name = run_name
        self.path = group_dir(spec) / run_name

    def exists(self) -> bool:
        return (self.path / "run.json").exists()

    def init(self, seed: int | None, extra: dict | None = None) -> dict:
        self.path.mkdir(parents=True, exist_ok=True)
        meta = run_manifest(self.spec, seed, {"run_name": self.run_name, **(extra or {})})
        atomic_write_json(self.path / "run.json", meta)
        save_spec(self.spec, self.path / "spec.yaml")
        self.set_status(state="running", stage="init")
        return meta

    def status(self) -> dict:
        return read_json(self.path / "status.json", {"state": "new"})

    def set_status(self, **kw) -> None:
        st = self.status()
        st.update(kw)
        st["updated"] = utc_now()
        atomic_write_json(self.path / "status.json", st)

    def mark_done(self, **kw) -> None:
        self.set_status(state="done", **kw)

    def mark_failed(self, error: str) -> None:
        self.set_status(state="failed", error=error[-4000:])

    def is_done(self) -> bool:
        return self.status().get("state") == "done"


@contextlib.contextmanager
def phase(timings: dict, name: str) -> Iterator[None]:
    """Record wall time and CUDA peak memory of a phase into `timings[name]`."""
    cuda = False
    try:
        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover
        pass
    t0 = time.perf_counter()
    try:
        yield
    finally:
        rec = {"wall_s": round(time.perf_counter() - t0, 3)}
        if cuda:
            import torch

            rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        timings[name] = rec


def shared_dir(spec: SgacSpec, name: str) -> Path:
    return group_dir(spec) / "_shared" / name
