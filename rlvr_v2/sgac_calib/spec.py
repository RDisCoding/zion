"""Calibration-track configuration (`configs/sgac_calib/calib.yaml`) and the training substrate it runs on.

The substrate is the frozen e0 SGAC spec, loaded unchanged from `configs/sgac/`. Only `run.results_root` is redirected
to this track's own results directory, and `SgacSpec.spec_hash()` excludes `run`, so every calibration and curriculum
run records the historical e0 hash (577f3319): proof that the training recipe is the reproduction's. The track's own
identity (thresholds, seeds, manifests' rules) is `CalibSpec.calib_hash()`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..artifacts import REPO_ROOT
from ..config import dataclass_from_dict, deep_merge, parse_overrides
from ..sgac.spec import SgacSpec, load_profile

CONFIG_DIR = REPO_ROOT / "configs" / "sgac_calib"
E0_SPEC_HASH = "577f3319"  # the historical reproduction's primary profile (paper/sgac_repro_protocol.md)


@dataclass(frozen=True)
class DataCfg:
    manifest_dir: str = "configs/sgac_calib/manifests"
    near_dup_ratio: float = 0.8
    prereg_pool_manifest: str = "manifests/pool.json"
    prereg_heldout_manifest: str = "manifests/heldout.json"
    prereg_ineligible_manifest: str = "manifests/pool_ineligible.json"
    sieve_rollouts: str = "results/pool/rollouts.jsonl"
    sieve_signals: str = "results/pool/signals.jsonl"
    study1_candidates: str = "manifests/study1_candidates.json"


@dataclass(frozen=True)
class SamplingCfg:
    rng_seed: int = 20261010
    ps_bins: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.0, 0.4), (0.4, 0.8), (0.8, 1.0))
    pilot_per_bin: int = 4
    expansion_per_bin: int = 8


@dataclass(frozen=True)
class FeatureCfg:
    k_deploy: int = 4
    n_subsamples: int = 16
    subsample_seed: int = 20261011


@dataclass(frozen=True)
class CalibrationCfg:
    seed_offsets: tuple[int, ...] = (7000, 17000)
    pi1_seed_offsets: tuple[int, ...] = (9000, 19000)
    dev_eval_repeats: int = 2
    gate_p: dict = field(default_factory=lambda: {"icc_min": 0.30, "sd_min_pp": 1.0})


@dataclass(frozen=True)
class SelectorCfg:
    features: tuple[str, ...] = ("Ps", "Var", "D", "L")
    cv_folds: int = 6
    cv_repeats: int = 100
    inner_folds: int = 5
    ridge_alphas: tuple[float, ...] = (1e-3, 3.16e-3, 1e-2, 3.16e-2, 0.1, 0.316, 1.0, 3.16, 10.0, 31.6, 100.0, 316.0, 1e3)
    gbm: dict = field(default_factory=lambda: {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.05,
                                               "subsample": 0.8, "min_leaf_candidates": 5})
    n_bootstrap: int = 2000
    n_permutations: int = 1000
    perm_cv_repeats: int = 10
    n_mc_batches: int = 20000
    alpha_cv_repeats: int = 10
    seed: int = 0
    gate_s: dict = field(default_factory=lambda: {"rho_min": 0.30, "perm_p_max": 0.05})


@dataclass(frozen=True)
class CurriculumCfg:
    seeds_min: int = 6
    seeds_max: int = 10
    first_seed: int = 42
    confirmatory_arms: tuple[str, ...] = ("learned", "random", "sgac")
    secondary_arms: tuple[str, ...] = ("max_var", "max_d", "max_level")
    secondary_seeds: tuple[int, ...] = (42, 43, 44)


@dataclass(frozen=True)
class AnalysisCfg:
    alpha: float = 0.05
    min_effect_pp: float = 1.0
    seed_consistency: float = 5 / 6
    equivalence_pp: float = 2.0
    mde_alpha: float = 0.025
    mde_power: float = 0.80
    mde_max_pp: float = 2.0
    n_boot: int = 10000
    boot_seed: int = 0


@dataclass(frozen=True)
class CalibSpec:
    substrate_profile: str = "e0"
    results_root: str = "results_sgac_calib"
    data: DataCfg = field(default_factory=DataCfg)
    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    features: FeatureCfg = field(default_factory=FeatureCfg)
    calibration: CalibrationCfg = field(default_factory=CalibrationCfg)
    selector: SelectorCfg = field(default_factory=SelectorCfg)
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
    analysis: AnalysisCfg = field(default_factory=AnalysisCfg)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def calib_hash(self) -> str:
        """Identity of the track's settings. Excludes where results go and where input files live: their content is
        pinned by sha256 in the committed manifests, so a different path on another machine is not a different track."""
        d = self.to_dict()
        d.pop("results_root")
        for k in ("sieve_rollouts", "sieve_signals", "study1_candidates"):
            d["data"].pop(k)
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]

    def validate(self) -> None:
        errors = []
        if self.substrate_profile != "e0":
            errors.append("substrate_profile must be e0 (the validated substrate; protocol section 6)")
        lo = [b[0] for b in self.sampling.ps_bins]
        if lo != sorted(lo):
            errors.append("sampling.ps_bins must be ordered")
        if self.features.k_deploy < 2:
            errors.append("features.k_deploy >= 2 required")
        if not (2 <= len(self.calibration.seed_offsets)):
            errors.append("calibration needs >= 2 training seeds per candidate (label reliability, gate G-P)")
        if not (self.curriculum.seeds_min <= self.curriculum.seeds_max):
            errors.append("curriculum.seeds_min <= seeds_max required")
        if errors:
            raise ValueError("Invalid calibration spec:\n  - " + "\n  - ".join(errors))

    # ------------------------------------------------------------------ paths
    def path(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def manifest_dir(self) -> Path:
        return self.path(self.data.manifest_dir)

    @property
    def results_dir(self) -> Path:
        return self.path(self.results_root)


def load_calib(path: str | Path | None = None, overrides: list[str] | None = None) -> CalibSpec:
    with open(path or CONFIG_DIR / "calib.yaml", "r", encoding="utf-8") as fh:
        merged = yaml.safe_load(fh) or {}
    merged = deep_merge(merged, parse_overrides(overrides))
    spec = dataclass_from_dict(CalibSpec, merged)
    spec.validate()
    return spec


def substrate_spec(cs: CalibSpec, stage: str, overrides: list[str] | None = None,
                   extra_files: list[str | Path] | None = None) -> SgacSpec:
    """The frozen e0 SGAC spec with its results root moved under this track (`<results_root>/<stage>`).

    `overrides`/`extra_files` exist for the CPU smoke only; production runs must keep the e0 hash, which
    `require_e0_substrate` enforces."""
    root = (cs.results_dir / stage).as_posix()
    return load_profile(cs.substrate_profile, [f"run.results_root={root}", *(overrides or [])], extra_files)


def require_e0_substrate(spec: SgacSpec) -> None:
    if spec.spec_hash() != E0_SPEC_HASH:
        raise RuntimeError(f"substrate spec hash {spec.spec_hash()} != historical e0 {E0_SPEC_HASH}: the training recipe "
                           "would differ from the reproduction's; refusing (protocol section 6)")
