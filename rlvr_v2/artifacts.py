"""Run directories, run metadata, JSONL writers and atomic status files."""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import Config, save_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_hash() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def package_versions() -> dict[str, str]:
    import importlib

    out: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "math_verify", "vllm", "numpy"):
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = "absent"
    return out


def gpu_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        free, total = torch.cuda.mem_get_info()
        return {"cuda": True, "name": torch.cuda.get_device_name(0), "free_gb": round(free / 1e9, 2),
                "total_gb": round(total / 1e9, 2)}
    except Exception as e:  # pragma: no cover
        return {"cuda": False, "error": repr(e)}


def slurm_info() -> dict[str, str]:
    keys = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_NAME")
    return {k: os.environ[k] for k in keys if k in os.environ}


def atomic_write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


class JsonlWriter:
    def __init__(self, path: str | Path, append: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a" if append else "w", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class RunDir:
    """results/<study>/<tag>-<cfghash>/<run_name>/ with run.json, config.yaml, status.json."""

    def __init__(self, cfg: Config, study: str, run_name: str, root: str | Path | None = None):
        self.cfg = cfg
        self.study = study
        self.run_name = run_name
        base = Path(root or cfg.run.results_root)
        if not base.is_absolute():
            base = REPO_ROOT / base
        self.group = f"{cfg.run.tag}-{cfg.config_hash()}"
        self.path = base / study / self.group / run_name
        self.run_id = f"{study}/{self.group}/{run_name}"

    def stage(self, name: str) -> Path:
        p = self.path / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def exists(self) -> bool:
        return (self.path / "run.json").exists()

    def init(self, seed: int, extra: dict | None = None) -> dict:
        self.path.mkdir(parents=True, exist_ok=True)
        meta = {
            "run_id": self.run_id,
            "study": self.study,
            "run_name": self.run_name,
            "created": utc_now(),
            "seed": seed,
            "config_hash": self.cfg.config_hash(),
            "git_hash": git_hash(),
            "versions": package_versions(),
            "gpu": gpu_info(),
            "slurm": slurm_info(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "argv": sys.argv,
            **(extra or {}),
        }
        atomic_write_json(self.path / "run.json", meta)
        save_config(self.cfg, self.path / "config.yaml")
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
