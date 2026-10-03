#!/usr/bin/env python
"""Benchmark batched HF generation throughput and extrapolate eval / GRPO round times.

Examples
    python scripts/bench_throughput.py --config configs/base.yaml --n-prompts 64 --batch-sizes 1,8,32,64 \
        --max-new-tokens 512 --out results/bench/bench.json
    # CPU smoke check with a tiny model (downloads ~270 MB):
    python scripts/bench_throughput.py --model HuggingFaceTB/SmolLM2-135M-Instruct --n-prompts 4 \
        --batch-sizes 1,4 --max-new-tokens 16

Prompts come from the first N problems of `manifests/pool.json` (requires the MATH train dataset) when
it exists, otherwise from synthetic arithmetic questions; both are rendered with the configured prompt
style. Decoding is greedy. Extrapolations assume the per-step cost measured here (it grows with context
length, so long-generation estimates are optimistic):
  eval500   : 500 MATH-500 items at a 3072-token cap, lower bound = 500*mean_tokens/tok_s,
              upper bound = ceil(500/bs)*3072*s_per_step (every batch runs to the cap)
  grpo_round: 64 completions of ~600 tokens, nominal = ceil(64/bs)*600*s_per_step
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from rlvr_v2 import modeling, prompts  # noqa: E402
from rlvr_v2.artifacts import atomic_write_json, gpu_info, package_versions, utc_now  # noqa: E402
from rlvr_v2.config import (  # noqa: E402
    Config,
    SamplingParams,
    dataclass_from_dict,
    deep_merge,
    load_config,
    parse_overrides,
)
from rlvr_v2.sampling import HFSampler  # noqa: E402

log = logging.getLogger("bench_throughput")

EVAL_N, EVAL_CAP, GRPO_N, GRPO_MEAN_TOKENS = 500, 3072, 64, 600


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml", help="YAML config (defaults are used if missing)")
    ap.add_argument("--overrides", nargs="*", default=[], help="dotted overrides, e.g. gen.hf_batch_size=16")
    ap.add_argument("--model", default=None, help="shortcut for --overrides model.name=...")
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--batch-sizes", default="1,8,32,64")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--assumed-mean-tokens", type=int, default=600, help="mean completion length for the eval lower bound")
    ap.add_argument("--manifest", default="manifests/pool.json")
    ap.add_argument("--synthetic", action="store_true", help="force synthetic prompts even if the manifest exists")
    ap.add_argument("--out", default="results/bench/bench.json")
    return ap.parse_args(argv)


def load_cfg(args: argparse.Namespace) -> Config:
    path = Path(args.config)
    paths = [path] if path.exists() else []
    if not paths:
        log.warning("config %s not found; using dataclass defaults", path)
    overrides = list(args.overrides)
    if args.model:
        overrides.append(f"model.name={args.model}")
    try:
        return load_config(paths, overrides)
    except ValueError as e:
        log.warning("config does not validate (%s); the benchmark only needs model/prompt/gen fields, continuing", e)
        merged: dict = {}
        for p in paths:
            with open(p, "r", encoding="utf-8") as fh:
                merged = deep_merge(merged, yaml.safe_load(fh) or {})
        merged = deep_merge(merged, parse_overrides(overrides))
        return dataclass_from_dict(Config, merged)


def problem_texts(n: int, manifest: Path, synthetic: bool, cfg: Config) -> tuple[list[str], str]:
    if not synthetic and manifest.exists():
        try:
            from rlvr_v2.data import Manifest, load_math_train, select_by_manifest

            man = Manifest.load(manifest)
            head = dataclasses.replace(man, ids=man.ids[:n], hashes={u: man.hashes[u] for u in man.ids[:n]})
            probs = select_by_manifest(head, load_math_train(man.meta.get("dataset", cfg.data.train_dataset)))
            return [p.problem for p in probs], str(manifest)
        except Exception as e:  # noqa: BLE001 - dataset download / manifest problems must not stop a benchmark
            log.warning("could not build prompts from %s (%s); using synthetic prompts", manifest, e)
    rng = random.Random(0)
    templates = (
        "Compute {a} + {b}.",
        "What is {a} times {b}?",
        "Find the remainder when {a} is divided by {c}.",
        "Simplify the fraction {a}/{c}.",
        "How many positive divisors does {b} have?",
    )
    texts = [templates[i % len(templates)].format(a=rng.randint(100, 999), b=rng.randint(10, 99), c=rng.randint(2, 19))
             for i in range(n)]
    return texts, "synthetic"


def bench_one(model, tok, cfg: Config, stop_ids: list[int], rendered: list[str], bs: int, max_new: int,
              assumed_mean: int, cuda: bool) -> dict:
    gen_cfg = dataclasses.replace(cfg.gen, hf_batch_size=bs, max_new_tokens=max_new)
    sampler = HFSampler(model, tok, gen_cfg, stop_ids)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = sampler.generate(rendered, 1, SamplingParams(temperature=0.0, n=1), seed=0)
    if cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    st = sampler.stats
    n = len(rendered)
    tok_s = st["generated_tokens"] / wall if wall > 0 else float("nan")
    s_step = wall / st["decode_steps"] if st["decode_steps"] else float("nan")
    lens = [g[0].n_tokens for g in out]
    row = {
        "batch_size": bs, "effective_batch": min(bs, n), "n_prompts": n, "chunks": st["chunks"], "wall_s": wall,
        "generated_tokens": st["generated_tokens"], "decode_steps": st["decode_steps"], "tokens_per_s": tok_s,
        "s_per_prompt": wall / n, "s_per_step": s_step,
        "mean_completion_tokens": sum(lens) / n, "max_completion_tokens": max(lens),
        "trunc_rate": sum(1 for g in out if g[0].truncated) / n,
        "peak_cuda_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if cuda else None,
        "est_eval500_s_lower": EVAL_N * assumed_mean / tok_s if tok_s else float("nan"),
        "est_eval500_s_upper": math.ceil(EVAL_N / bs) * EVAL_CAP * s_step,
        "est_grpo_round_s_lower": GRPO_N * GRPO_MEAN_TOKENS / tok_s if tok_s else float("nan"),
        "est_grpo_round_s_nominal": math.ceil(GRPO_N / bs) * GRPO_MEAN_TOKENS * s_step,
        "est_grpo_round_s_upper": math.ceil(GRPO_N / bs) * EVAL_CAP * s_step,
        "sample_text": out[0][0].text[:120],
    }
    modeling.free_cuda()
    return row


def fmt_time(s: float) -> str:
    if math.isnan(s):
        return "n/a"
    return f"{s / 60:.1f}m" if s >= 90 else f"{s:.1f}s"


def print_table(rows: list[dict]) -> None:
    head = f"{'bs':>4} {'wall_s':>8} {'tokens':>8} {'tok/s':>8} {'s/prompt':>9} {'s/step':>8} {'peak_GB':>8} " \
           f"{'eval500 lo..hi':>18} {'grpo_round nominal':>19}"
    print(head)
    print("-" * len(head))
    for r in rows:
        peak = f"{r['peak_cuda_mem_gb']:.2f}" if r["peak_cuda_mem_gb"] is not None else "cpu"
        print(f"{r['batch_size']:>4} {r['wall_s']:>8.1f} {r['generated_tokens']:>8} {r['tokens_per_s']:>8.1f} "
              f"{r['s_per_prompt']:>9.2f} {r['s_per_step']:>8.4f} {peak:>8} "
              f"{fmt_time(r['est_eval500_s_lower']) + '..' + fmt_time(r['est_eval500_s_upper']):>18} "
              f"{fmt_time(r['est_grpo_round_s_nominal']):>19}")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = load_cfg(args)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    cuda = torch.cuda.is_available()
    texts, source = problem_texts(args.n_prompts, Path(args.manifest), args.synthetic, cfg)

    tok = modeling.load_tokenizer(cfg)
    model = modeling.load_base_model(cfg)
    stop_ids = prompts.resolve_stop_token_ids(tok, cfg.prompt.style)
    rendered = [prompts.build_prompt(t, cfg.prompt.style, tok, cfg.prompt.system) for t in texts]
    prompt_lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in rendered]
    log.info("%d prompts from %s; prompt tokens mean %.0f max %d; stop ids %s; device %s", len(rendered), source,
             sum(prompt_lens) / len(prompt_lens), max(prompt_lens), stop_ids, modeling.device_of(model))

    # warm-up (kernel selection / allocator) so the first measured batch size is not penalised
    warm = HFSampler(model, tok, dataclasses.replace(cfg.gen, hf_batch_size=1, max_new_tokens=4), stop_ids)
    warm.generate(rendered[:1], 1, SamplingParams(temperature=0.0, n=1), seed=0)

    rows = []
    for bs in batch_sizes:
        log.info("benchmarking batch size %d ...", bs)
        rows.append(bench_one(model, tok, cfg, stop_ids, rendered, bs, args.max_new_tokens, args.assumed_mean_tokens, cuda))
        log.info("bs=%d: %.1f tok/s, %.4f s/step, sample: %r", bs, rows[-1]["tokens_per_s"], rows[-1]["s_per_step"],
                 rows[-1]["sample_text"][:60])

    print_table(rows)
    gpu = gpu_info()
    report = {
        "created": utc_now(), "model": cfg.model.name, "dtype": cfg.model.dtype if cuda else "fp32 (cpu)",
        "attn_impl": cfg.model.attn_impl, "prompt_style": cfg.prompt.style, "prompt_source": source,
        "n_prompts": len(rendered), "max_new_tokens": args.max_new_tokens, "assumed_mean_tokens": args.assumed_mean_tokens,
        "prompt_tokens_mean": sum(prompt_lens) / len(prompt_lens), "prompt_tokens_max": max(prompt_lens),
        "stop_token_ids": stop_ids, "gpu": gpu, "gpu_name": gpu.get("name", "cpu"), "versions": package_versions(),
        "assumptions": {"eval_n": EVAL_N, "eval_cap": EVAL_CAP, "grpo_n": GRPO_N, "grpo_mean_tokens": GRPO_MEAN_TOKENS},
        "results": rows,
    }
    out = Path(args.out)
    atomic_write_json(out, report)
    print(f"\nwrote {out} ({json.dumps({'gpu': report['gpu_name'], 'model': cfg.model.name})})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
