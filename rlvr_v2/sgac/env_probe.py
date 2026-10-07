"""T0 environment probe: GPU, bitsandbytes on this GPU, the legacy sympy path, the math-verify self-check, versions and
the cached model/dataset revisions. Written to results_sgac/env_probe.json; decides nothing by itself (the as_run nf4
fallback is applied, and recorded, by `model_io.load_policy_model`)."""
from __future__ import annotations

import json
import logging

from ..artifacts import atomic_write_json, utc_now
from . import legacy
from .runinfo import extra_versions, gpu_details
from .spec import SgacSpec, results_root

log = logging.getLogger(__name__)


def _bnb_probe() -> dict:
    try:
        import bitsandbytes as bnb
        import torch
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"import: {type(e).__name__}: {e}"}
    if not torch.cuda.is_available():
        return {"ok": False, "error": "no CUDA device", "version": getattr(bnb, "__version__", "?")}
    try:
        layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.float16, quant_type="nf4",
                                  compress_statistics=True).to("cuda")
        x = torch.randn(2, 64, device="cuda", dtype=torch.float16)
        y = layer(x)
        ok = bool(torch.isfinite(y).all().item())
        return {"ok": ok, "version": getattr(bnb, "__version__", "?"), "out_dtype": str(y.dtype)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "version": getattr(bnb, "__version__", "?"), "error": f"{type(e).__name__}: {str(e)[:400]}"}


def _cached_revision(repo_id: str, repo_type: str, revision: str | None) -> dict:
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id, repo_type=repo_type, revision=revision, local_files_only=True)
        return {"cached": True, "snapshot": path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]}
    except Exception as e:  # noqa: BLE001
        return {"cached": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def probe(spec: SgacSpec) -> dict:
    out: dict = {"utc": utc_now(), "gpu": gpu_details(), "versions": extra_versions(), "bitsandbytes": _bnb_probe(),
                 "legacy_symbolic": legacy.symbolic_status()}
    try:
        from ..grader import MathVerifyGrader

        MathVerifyGrader()
        out["math_verify_self_check"] = "ok"
    except Exception as e:  # noqa: BLE001
        out["math_verify_self_check"] = f"FAILED {type(e).__name__}: {e}"
    out["hf_cache"] = {
        "model": _cached_revision(spec.model.name, "model", spec.model.revision),
        "pi1": _cached_revision(spec.model.pi1_name, "model", spec.model.pi1_revision),
        "dataset": _cached_revision(spec.data.dataset, "dataset", spec.data.revision),
        "math500": _cached_revision("HuggingFaceH4/MATH-500", "dataset", None),
    }
    path = results_root(spec) / "env_probe.json"
    atomic_write_json(path, out)
    log.info("env probe -> %s\n%s", path, json.dumps(out, indent=1, default=str))
    return out
