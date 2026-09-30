from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def worker_python() -> str:
    explicit = os.environ.get("AFMOUN_WORKER_PYTHON")
    if explicit:
        return explicit
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return sys.executable


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_under_root(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidates = [ROOT / p, ROOT.parent / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_data_path(meta_dir: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    candidates = [meta_dir / p]
    for parent in meta_dir.parents:
        candidates.append(parent / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_data_cache(data_dir: Path, requested_tokens: int, block_size: int) -> None:
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        train_path = resolve_data_path(data_dir, meta["train_path"])
        dtype = str(meta.get("dtype", "uint32"))
    else:
        train_path = data_dir / "train_tokens_uint32.bin"
        dtype = "uint32"
    if not train_path.exists():
        raise FileNotFoundError(f"missing train token file: {train_path}")
    bytes_per_token = 4 if dtype == "uint32" else 2
    stored_tokens = train_path.stat().st_size // bytes_per_token
    usable_tokens = max(0, (stored_tokens // block_size) * block_size)
    if usable_tokens < requested_tokens:
        raise RuntimeError(
            f"Llama cache too small: usable={usable_tokens:,} requested={requested_tokens:,} path={train_path}"
        )
    print(
        f"data ok: {data_dir} dtype={dtype} stored_tokens={stored_tokens:,} usable_tokens={usable_tokens:,}",
        flush=True,
    )


def validate_model_dir(model_dir: Path) -> None:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing model config: {config_path}")
    cfg = read_json(config_path)
    print(
        "model ok: "
        f"{model_dir} hidden={cfg.get('hidden_size')} layers={cfg.get('num_hidden_layers')} "
        f"heads={cfg.get('num_attention_heads')} vocab={cfg.get('vocab_size')}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run paper Llama-3.2-1B NB98-matched V2 jobs.")
    p.add_argument("--gpus", nargs="+", type=int, default=[5, 6, 7])
    p.add_argument("--run-root", default="runs/01_paper_main_results_llama32_1b_nb98_v2_fixed")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--seeds", default="43,44,45")
    p.add_argument("--arms", default="muon,sign,afmoun")
    p.add_argument("--train-tokens", type=int, default=2_000_000_000)
    p.add_argument("--eval-tokens", type=int, default=1_000_000)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=3_814)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--vector-lr", type=float, default=3e-4)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--matrix-weight-decay", type=float, default=0.1)
    p.add_argument("--hybrid-aux-weight-decay", type=float, default=0.01)
    p.add_argument("--rho-hidden", type=float, default=50.0)
    p.add_argument("--rho-output", type=float, default=3000.0)
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--retry-incomplete", action="store_true")
    return p.parse_args()


def arm_label(arm: str) -> str:
    return {
        "muon": "hybrid_muon",
        "sign": "scion_c1_s1",
        "afmoun": "afmoun_c3_s0p5",
    }[arm]


def optimizer_for_arm(arm: str) -> str:
    return "muon" if arm == "muon" else "afmoun_v2"


def build_job_command(args: argparse.Namespace, arm: str, seed: int, out_dir: Path) -> list[str]:
    cmd = [
        worker_python(),
        str(ROOT / "scripts" / "train_01_paper_llama32_nb98_v2.py"),
        "--model-dir",
        str(resolve_under_root(args.model_dir)),
        "--data-dir",
        str(resolve_under_root(args.data_dir)),
        "--output-dir",
        str(out_dir),
        "--arm",
        arm,
        "--max-steps",
        str(args.max_steps),
        "--train-tokens",
        str(args.train_tokens),
        "--eval-tokens",
        str(args.eval_tokens),
        "--eval-every",
        str(args.eval_every),
        "--log-every-steps",
        str(args.log_every_steps),
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--block-size",
        str(args.block_size),
        "--muon-lr",
        str(args.muon_lr),
        "--vector-lr",
        str(args.vector_lr),
        "--momentum",
        str(args.momentum),
        "--matrix-weight-decay",
        str(args.matrix_weight_decay),
        "--aux-weight-decay",
        str(args.hybrid_aux_weight_decay if arm == "muon" else 0.0),
        "--rho-hidden",
        str(args.rho_hidden),
        "--rho-output",
        str(args.rho_output),
        "--chunk-rows",
        str(args.chunk_rows),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--bf16",
    ]
    return cmd


def make_jobs(args: argparse.Namespace) -> list[dict]:
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    allowed = {"muon", "sign", "afmoun"}
    unknown = set(arms) - allowed
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    jobs: list[dict] = []
    for seed in seeds:
        for arm in arms:
            name = f"llama32_1b_{arm_label(arm)}_seed{seed}_nb98_v2"
            out_dir = Path(args.run_root) / name
            metrics = out_dir / "metrics.jsonl"
            done = metrics.exists() and not args.retry_incomplete
            jobs.append({"seed": seed, "arm": arm, "name": name, "out_dir": out_dir, "done": done})
    return jobs


def launch(args: argparse.Namespace, job: dict, gpu: int) -> dict:
    run_root = Path(args.run_root)
    log_dir = run_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job['name']}.log"
    job["out_dir"].mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        build_job_command(args, job["arm"], job["seed"], job["out_dir"]),
        cwd=str(ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    handle.close()
    print(f"[GPU {gpu}] launched {job['name']} pid={proc.pid} log={log_path}", flush=True)
    return {"job": job, "gpu": gpu, "proc": proc, "start": time.time(), "log": log_path}


def main() -> None:
    args = parse_args()
    model_dir = resolve_under_root(args.model_dir)
    data_dir = resolve_under_root(args.data_dir)
    validate_model_dir(model_dir)
    validate_data_cache(data_dir, args.train_tokens, args.block_size)
    tokens_per_step = args.micro_batch_size * args.gradient_accumulation_steps * args.block_size
    expected_steps = args.train_tokens // tokens_per_step
    if expected_steps != args.max_steps:
        raise ValueError(
            f"NB98 step mismatch: train_tokens/tokens_per_step={expected_steps}, max_steps={args.max_steps}"
        )

    jobs = make_jobs(args)
    pending = [j for j in jobs if not j["done"]]
    skipped = [j for j in jobs if j["done"]]
    print("=== 01 paper Llama-3.2-1B NB98-matched V2 launcher ===", flush=True)
    print(f"run_root={args.run_root}", flush=True)
    print(f"gpus={args.gpus}", flush=True)
    print(
        f"tokens={args.train_tokens:,} steps={args.max_steps:,} tokens_per_step={tokens_per_step:,} "
        f"eval_every={args.eval_every} log_every={args.log_every_steps}",
        flush=True,
    )
    print(
        f"seeds={args.seeds} arms={args.arms} skipped_existing={len(skipped)} pending={len(pending)}",
        flush=True,
    )
    for job in jobs:
        status = "skip-existing" if job["done"] else "pending"
        print(
            f"{status}: {job['name']} arm={job['arm']} optimizer={optimizer_for_arm(job['arm'])} "
            f"seed={job['seed']}",
            flush=True,
        )
    if args.dry_run:
        return

    free_gpus = list(args.gpus)
    running: list[dict] = []
    completed: list[dict] = []
    failures: list[tuple[str, int, str]] = []
    while pending or running:
        while pending and free_gpus:
            running.append(launch(args, pending.pop(0), free_gpus.pop(0)))
        time.sleep(max(1, int(args.poll_seconds)))
        still_running: list[dict] = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                still_running.append(item)
                continue
            elapsed = time.time() - item["start"]
            name = item["job"]["name"]
            print(f"finished {name}: exit={rc} seconds={elapsed:.1f} gpu={item['gpu']}", flush=True)
            if rc != 0:
                failures.append((name, int(rc), str(item["log"])))
            completed.append(item)
            free_gpus.append(item["gpu"])
        running = still_running
        if running:
            print(", ".join(f"{x['job']['name']}=running(GPU{x['gpu']})" for x in running), flush=True)
    print(f"Completed {len(completed)} launched jobs; skipped {len(skipped)} existing jobs.", flush=True)
    if failures:
        raise RuntimeError(f"paper Llama jobs failed: {failures}")
    print("All paper Llama NB98-matched V2 jobs completed.", flush=True)


if __name__ == "__main__":
    main()
