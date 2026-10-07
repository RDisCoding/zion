"""The SGAC track leaves the pre-registered study untouched: frozen config hashes, shared callables, frozen files."""
import shutil
import subprocess
import sys

import pytest

from rlvr_v2.artifacts import REPO_ROOT
from rlvr_v2.config import load_config

FROZEN_PATHS = ["rlvr_v2/config.py", "configs/base.yaml", "configs/e0.yaml", "configs/study1.yaml",
                "configs/study2.yaml", "manifests", "paper/prereg.md"]


def test_prereg_config_hashes_unchanged():
    c = REPO_ROOT / "configs"
    study1 = load_config([c / "base.yaml", c / "study1.yaml"],
                         ["prompt.style=oneshot_rlvr_chat", "train.rounds=100", "train.learning_rate=2e-05"])
    assert study1.config_hash() == "84e100e2"  # recorded in paper/prereg.md
    assert load_config([c / "base.yaml", c / "e0.yaml"]).config_hash() == "924c03f2"


def test_importing_sgac_patches_nothing():
    code = (
        "import rlvr_v2.sampling as s, rlvr_v2.train_grpo as t, rlvr_v2.config as c, rlvr_v2.prompts as p\n"
        "before = (s.HFSampler.generation_kwargs, t.check_grpo_args, c.Config.validate, p.configure_tokenizer,"
        " t.run_grpo_burst, s.HFSampler.generate)\n"
        "import importlib\n"
        "for m in ('legacy','nbm_original','selection','schedule','spec','data','grading','sampler','model_io','sieve',"
        "'evaluation','burst','runinfo','loop','env_probe','jobs','smoke','report','cli'):\n"
        "    importlib.import_module('rlvr_v2.sgac.' + m)\n"
        "after = (s.HFSampler.generation_kwargs, t.check_grpo_args, c.Config.validate, p.configure_tokenizer,"
        " t.run_grpo_burst, s.HFSampler.generate)\n"
        "assert all(a is b for a, b in zip(before, after)), 'an SGAC import patched a shared callable'\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)


def test_frozen_files_identical_to_prereg_tag():
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not available")
    has_tag = subprocess.run([git, "rev-parse", "--verify", "--quiet", "prereg-v1"], cwd=REPO_ROOT, capture_output=True)
    if has_tag.returncode != 0:
        pytest.skip("tag prereg-v1 not present in this clone")
    diff = subprocess.run([git, "diff", "--name-only", "prereg-v1", "--", *FROZEN_PATHS], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    assert diff.returncode == 0 and diff.stdout.strip() == "", f"frozen files changed since prereg-v1: {diff.stdout}"
