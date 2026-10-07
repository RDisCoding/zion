"""Reproduction of the ORIGINAL Selector-Guided Autonomous Curriculum (SGAC) experiment.

A separate track from the pre-registered study: nothing here changes the shared modules, the `Config` schema,
`configs/{base,e0,study1,study2}.yaml`, `manifests/*` or `paper/prereg.md` (tests/test_sgac_shared_frozen.py).
Protocol, verdict rule and deviation log: `paper/sgac_repro_protocol.md`. Entry point: `python -m rlvr_v2.sgac`.

Profiles (configs/sgac/*.yaml)
- e0      PRIMARY: the E0-validated substrate (bf16, oneshot_rlvr_chat, math-verify grader, E0 stop/pad tokens,
          3072-token cap for sieve, GRPO and eval) running the original SGAC recipe.
- as_run  secondary fidelity check: the historical Kaggle configuration (legacy prompt/grader, 4-bit if
          available, GRPO capped at TRL 1.0.0's default 256 tokens).

Kept import-light: `legacy` must import with only re/numpy/sympy so it also runs in the isolated grader env.
"""
