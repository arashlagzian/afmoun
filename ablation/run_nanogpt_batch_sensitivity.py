from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCK_SIZE = 1024
MICRO_BATCH_SIZE = 16
TARGET_TOKENS = 500_000_000


def worker_python() -> str:
    return os.environ.get("AFMOUN_WORKER_PYTHON") or sys.executable


def arm_label(arm: str) -> str:
    return {
        "muon": "hybrid_muon",
        "sign": "scion_c1_s1",
        "afmoun": "afmoun_c3_s0p5",
    }[arm]


def paper_name(arm: str) -> str:
    return {
        "muon": "Hybrid Muon",
        "sign": "SCION-style Sign",
        "afmoun": "AF-Muon",
    }[arm]


def aux_weight_decay_for_arm(arm: str) -> float:
    return 0.01 if arm == "muon" else 0.0


def parse_csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_csv_text(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def iterations_for_batch(batch_size: int, target_tokens: int) -> int:
    tokens_per_step = batch_size * BLOCK_SIZE
    return max(1, int(round(target_tokens / tokens_per_step)))


def batch_tag(batch_size: int) -> str:
    tokens_per_step = batch_size * BLOCK_SIZE
    if tokens_per_step % 1024 == 0:
        return f"{tokens_per_step // 1024}k"
    return str(tokens_per_step)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch NanoGPT global-batch-size sensitivity runs."
    )
    p.add_argument("--gpus", nargs="+", type=int, required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument(
        "--run-root",
        default="runs/nanogpt_batch_sensitivity_seed43_500m",
    )
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--arms", default="muon,sign,afmoun")
    p.add_argument("--batch-sizes", default="32,64,128,256,512")
    p.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    p.add_argument("--eval-every-steps", type=int, default=125)
    p.add_argument("--diag-every-steps", type=int, default=125)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--eval-tokens", type=int, default=5_000_000)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--check-config", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_name(arm: str, seed: int, batch_size: int) -> str:
    return (
        f"nanogpt_{arm_label(arm)}_seed{seed}_batch{batch_size}"
        f"_tok{batch_tag(batch_size)}_500m_fp32"
    )


def command(args: argparse.Namespace, arm: str, batch_size: int, out_dir: Path) -> list[str]:
    iterations = iterations_for_batch(batch_size, int(args.target_tokens))
    gradient_accumulation = batch_size // MICRO_BATCH_SIZE
    if batch_size % MICRO_BATCH_SIZE != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by micro_batch_size={MICRO_BATCH_SIZE}"
        )

    worker = ROOT / "nanogpt" / "train_nanogpt.py"
    cmd = [
        worker_python(),
        str(worker),
        "--data-dir", str(args.data_dir),
        "--output-dir", str(out_dir),
        "--arm", arm,
        "--seed", str(args.seed),
        "--device", "cuda",
        "--autocast", "bf16",
        "--layers", "12",
        "--heads", "6",
        "--head-dim", "128",
        "--mlp-hidden", "3072",
        "--vocab-size", "50304",
        "--block-size", str(BLOCK_SIZE),
        "--iterations", str(iterations),
        "--batch-size", str(batch_size),
        "--micro-batch-size", str(MICRO_BATCH_SIZE),
        "--eval-tokens", str(args.eval_tokens),
        "--eval-every-steps", str(args.eval_every_steps),
        "--log-every-steps", str(args.log_every_steps),
        "--diag-every-steps", str(args.diag_every_steps),
        "--checkpoint-every-steps", "0",
        "--full-checkpoint-every-steps", "0",
        "--muon-lr", "0.02",
        "--muon-lr-multiplier", "1.0",
        "--multiplier-mode", "muon_only",
        "--vector-lr", "0.0003",
        "--momentum", "0.95",
        "--matrix-weight-decay", "0.1",
        "--aux-weight-decay", str(aux_weight_decay_for_arm(arm)),
        "--rho-hidden", "50.0",
        "--rho-output", "3000.0",
        "--tied-cap", "3.0",
        "--tied-scale", "0.5",
        "--chunk-rows", "2048",
        "--warmdown-frac", "0.0",
        "--activation-mode", "all_scaled",
        "--max-grad-norm", "1.0",
    ]
    if args.check_config:
        cmd.append("--check-config")
        cmd[cmd.index("--device") + 1] = "cpu"
    return cmd


def main() -> None:
    args = parse_args()
    arms = parse_csv_text(args.arms)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    valid_arms = {"muon", "sign", "afmoun"}
    bad_arms = [arm for arm in arms if arm not in valid_arms]
    if bad_arms:
        raise ValueError(f"unknown arms: {bad_arms}")

    jobs = []
    for batch_size in batch_sizes:
        if batch_size % MICRO_BATCH_SIZE != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by micro_batch_size={MICRO_BATCH_SIZE}"
            )
        for arm in arms:
            name = run_name(arm, args.seed, batch_size)
            jobs.append(
                {
                    "arm": arm,
                    "batch_size": batch_size,
                    "iterations": iterations_for_batch(batch_size, int(args.target_tokens)),
                    "name": name,
                    "out_dir": Path(args.run_root) / name,
                }
            )

    print("=== NanoGPT batch-size sensitivity launcher ===", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(f"batch_sizes: {batch_sizes}", flush=True)
    print("jobs:", [j["name"] for j in jobs], flush=True)
    print(
        "fixed settings: micro_batch=16, block=1024, vector_lr=0.0003, "
        "matrix_lr=0.02, fp32 params/state, bf16 autocast",
        flush=True,
    )
    for batch_size in batch_sizes:
        steps = iterations_for_batch(batch_size, int(args.target_tokens))
        print(
            f"batch={batch_size}: ga={batch_size // MICRO_BATCH_SIZE}, "
            f"tokens/update={batch_size * BLOCK_SIZE}, steps={steps}, "
            f"tokens={steps * batch_size * BLOCK_SIZE}",
            flush=True,
        )

    if args.dry_run:
        for job in jobs:
            print()
            print(f"# {paper_name(job['arm'])}, batch={job['batch_size']}")
            print(" ".join(command(args, job["arm"], job["batch_size"], job["out_dir"])))
        return

    if args.check_config:
        for job in jobs:
            subprocess.run(
                command(args, job["arm"], job["batch_size"], job["out_dir"]),
                cwd=str(ROOT),
                check=True,
            )
        return

    pending = list(jobs)
    running = []
    failures = []
    log_dir = Path(args.run_root) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    while pending or running:
        for gpu in args.gpus:
            if not pending:
                break
            if any(item["gpu"] == gpu for item in running):
                continue
            job = pending.pop(0)
            job["out_dir"].mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{job['name']}.log"
            handle = log_path.open("w", encoding="utf-8")
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
            }
            proc = subprocess.Popen(
                command(args, job["arm"], job["batch_size"], job["out_dir"]),
                cwd=str(ROOT),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            handle.close()
            running.append({"job": job, "gpu": gpu, "proc": proc, "log": log_path})
            print(
                f"[GPU {gpu}] launched {job['name']} "
                f"pid={proc.pid} log={log_path}",
                flush=True,
            )

        time.sleep(max(1, int(args.poll_seconds)))
        alive = []
        for item in running:
            code = item["proc"].poll()
            if code is None:
                alive.append(item)
                continue
            print(
                f"[GPU {item['gpu']}] finished {item['job']['name']} code={code}",
                flush=True,
            )
            if code != 0:
                failures.append((item["job"]["name"], code, str(item["log"])))
        running = alive
        if running or pending:
            print(
                "heartbeat running="
                + str([f"{r['job']['name']}@GPU{r['gpu']}" for r in running])
                + f" pending={len(pending)}",
                flush=True,
            )

    if failures:
        raise RuntimeError(f"NanoGPT batch-size jobs failed: {failures}")


if __name__ == "__main__":
    main()
