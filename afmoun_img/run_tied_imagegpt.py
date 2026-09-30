from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def resolve_repo_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def worker_python() -> str:
    return os.environ.get("AFMOUN_WORKER_PYTHON") or sys.executable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch tied ImageGPT-style RGB554 jobs.")
    p.add_argument("--gpus", nargs="+", type=int, required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--seeds", default="43,44,45")
    p.add_argument("--arms", default="muon,sign,afmoun")
    p.add_argument("--autocast", choices=["fp16", "bf16", "none"], default="bf16")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--mlp-hidden", type=int, default=2048)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--micro-batch-size", type=int, default=256)
    p.add_argument("--iterations", type=int, default=2500)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--eval-every-steps", type=int, default=250)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--diag-every-steps", type=int, default=250)
    p.add_argument("--checkpoint-every-steps", type=int, default=0)
    p.add_argument("--full-checkpoint-every-steps", type=int, default=2500)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--vector-lr", type=float, default=3e-4)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--matrix-weight-decay", type=float, default=0.1)
    p.add_argument("--hybrid-aux-weight-decay", type=float, default=0.01)
    p.add_argument("--aux-weight-decay", type=float, default=0.0)
    p.add_argument("--rho-hidden", type=float, default=50.0)
    p.add_argument("--rho-output", type=float, default=3000.0)
    p.add_argument("--tied-cap", type=float, default=3.0)
    p.add_argument("--tied-scale", type=float, default=0.5)
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--check-config", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=30)
    return p.parse_args()


def arm_label(arm: str) -> str:
    return {"muon": "hybrid_muon", "sign": "scion_c1_s1", "afmoun": "afmoun_c3_s0p5"}[arm]


def command(args: argparse.Namespace, arm: str, seed: int, out_dir: Path) -> list[str]:
    aux_wd = args.hybrid_aux_weight_decay if arm == "muon" else args.aux_weight_decay
    cmd = [
        worker_python(),
        str(SCRIPT_DIR / "train_tied_imagegpt.py"),
        "--data-dir", args.data_dir,
        "--output-dir", str(out_dir),
        "--arm", arm,
        "--seed", str(seed),
        "--autocast", args.autocast,
        "--layers", str(args.layers),
        "--heads", str(args.heads),
        "--head-dim", str(args.head_dim),
        "--mlp-hidden", str(args.mlp_hidden),
        "--block-size", str(args.block_size),
        "--batch-size", str(args.batch_size),
        "--micro-batch-size", str(args.micro_batch_size),
        "--iterations", str(args.iterations),
        "--eval-batches", str(args.eval_batches),
        "--eval-every-steps", str(args.eval_every_steps),
        "--log-every-steps", str(args.log_every_steps),
        "--diag-every-steps", str(args.diag_every_steps),
        "--checkpoint-every-steps", str(args.checkpoint_every_steps),
        "--full-checkpoint-every-steps", str(args.full_checkpoint_every_steps),
        "--muon-lr", str(args.muon_lr),
        "--vector-lr", str(args.vector_lr),
        "--momentum", str(args.momentum),
        "--matrix-weight-decay", str(args.matrix_weight_decay),
        "--aux-weight-decay", str(aux_wd),
        "--rho-hidden", str(args.rho_hidden),
        "--rho-output", str(args.rho_output),
        "--tied-cap", str(args.tied_cap),
        "--tied-scale", str(args.tied_scale),
        "--chunk-rows", str(args.chunk_rows),
        "--max-grad-norm", str(args.max_grad_norm),
    ]
    if args.check_config:
        cmd.extend(["--check-config", "--device", "cpu"])
    return cmd


def main() -> None:
    args = parse_args()
    args.data_dir = str(resolve_repo_path(args.data_dir))
    run_root = resolve_repo_path(args.run_root)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    jobs = []
    for seed in seeds:
        for arm in arms:
            name = f"tied_imagegpt_rgb554_16k_8l512_{arm_label(arm)}_seed{seed}"
            jobs.append({"seed": seed, "arm": arm, "name": name, "out_dir": run_root / name})

    print("=== Tied ImageGPT RGB554 launcher ===", flush=True)
    print("jobs:", [j["name"] for j in jobs], flush=True)
    if args.dry_run:
        for job in jobs:
            print(" ".join(command(args, job["arm"], job["seed"], job["out_dir"])))
        return
    if args.check_config:
        for job in jobs:
            subprocess.run(command(args, job["arm"], job["seed"], job["out_dir"]), cwd=str(REPO_ROOT), check=True)
        return

    pending = list(jobs)
    running = []
    failures = []
    log_dir = run_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    while pending or running:
        for gpu in args.gpus:
            if not pending:
                break
            if any(r["gpu"] == gpu for r in running):
                continue
            job = pending.pop(0)
            job["out_dir"].mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{job['name']}.log"
            handle = log_path.open("w", encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            proc = subprocess.Popen(command(args, job["arm"], job["seed"], job["out_dir"]), cwd=str(REPO_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
            handle.close()
            running.append({"job": job, "gpu": gpu, "proc": proc, "log": log_path})
            print(f"[GPU {gpu}] launched {job['name']} pid={proc.pid} log={log_path}", flush=True)

        time.sleep(max(1, int(args.poll_seconds)))
        alive = []
        for item in running:
            code = item["proc"].poll()
            if code is None:
                alive.append(item)
            else:
                print(f"[GPU {item['gpu']}] finished {item['job']['name']} code={code}", flush=True)
                if code != 0:
                    failures.append((item["job"]["name"], code, str(item["log"])))
        running = alive
        if running or pending:
            print("heartbeat running=" + str([f"{r['job']['name']}@GPU{r['gpu']}" for r in running]) + f" pending={len(pending)}", flush=True)

    if failures:
        raise RuntimeError(f"Tied ImageGPT jobs failed: {failures}")


if __name__ == "__main__":
    main()
