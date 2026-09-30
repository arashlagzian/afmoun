from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def worker_python() -> str:
    return os.environ.get("AFMOUN_WORKER_PYTHON") or sys.executable


def parse_csv_caps(text: str) -> list[float]:
    caps = []
    for raw in text.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        caps.append(float("inf") if item in {"inf", "infinity"} else float(item))
    return caps


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def num_tag(value: float) -> str:
    if math.isinf(float(value)):
        return "inf"
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch NanoGPT AF-Muon tied cap/scale sensitivity runs."
    )
    p.add_argument("--gpus", nargs="+", type=int, required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument(
        "--run-root",
        default="runs/nanogpt_afmoun_cap_scale_sensitivity_3seeds_step750_matchbatch524k",
    )
    p.add_argument("--seeds", default="43,44,45")
    p.add_argument("--caps", default="1,2,3,4,6,10,inf")
    p.add_argument("--scales", default="0.5,1.0")
    p.add_argument("--iterations", type=int, default=750)
    p.add_argument("--eval-every-steps", type=int, default=125)
    p.add_argument("--diag-every-steps", type=int, default=125)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--eval-tokens", type=int, default=5_000_000)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--check-config", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_name(seed: int, cap: float, scale: float, iterations: int) -> str:
    return (
        f"nanogpt_afmoun_seed{seed}_cap{num_tag(cap)}_scale{num_tag(scale)}"
        f"_step{int(iterations)}_fp32_matchbatch524k"
    )


def command(args: argparse.Namespace, cap: float, scale: float, out_dir: Path) -> list[str]:
    worker = ROOT / "ablation" / "train_nanogpt_vocab_effect.py"
    cmd = [
        worker_python(),
        str(worker),
        "--data-dir", str(args.data_dir),
        "--output-dir", str(out_dir),
        "--arm", "afmoun",
        "--seed", str(args.seed),
        "--device", "cuda",
        "--autocast", "bf16",
        "--layers", "12",
        "--heads", "6",
        "--head-dim", "128",
        "--mlp-hidden", "3072",
        "--vocab-size", "50304",
        "--block-size", "1024",
        "--iterations", str(args.iterations),
        "--batch-size", "512",
        "--micro-batch-size", "16",
        "--eval-tokens", str(args.eval_tokens),
        "--eval-every-steps", str(args.eval_every_steps),
        "--log-every-steps", str(args.log_every_steps),
        "--diag-every-steps", str(args.diag_every_steps),
        "--checkpoint-every-steps", "0",
        "--full-checkpoint-every-steps", str(args.iterations),
        "--muon-lr", "0.02",
        "--vector-lr", "0.0003",
        "--momentum", "0.95",
        "--matrix-weight-decay", "0.1",
        "--rho-hidden", "50.0",
        "--rho-output", "3000.0",
        "--tied-cap", "inf" if math.isinf(cap) else str(cap),
        "--tied-scale", str(scale),
        "--chunk-rows", "2048",
        "--max-grad-norm", "1.0",
    ]
    if args.check_config:
        cmd.append("--check-config")
        cmd[cmd.index("--device") + 1] = "cpu"
    return cmd


def main() -> None:
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    caps = parse_csv_caps(args.caps)
    scales = parse_csv_floats(args.scales)
    invalid = [c for c in caps if not math.isinf(c) and c < 1.0]
    if invalid:
        raise ValueError(f"AF-Muon tied cap must be >= 1 or inf; invalid caps: {invalid}")

    jobs = []
    for seed in seeds:
        for scale in scales:
            for cap in caps:
                name = run_name(seed, cap, scale, args.iterations)
                jobs.append(
                    {
                        "seed": seed,
                        "cap": cap,
                        "scale": scale,
                        "name": name,
                        "out_dir": Path(args.run_root) / name,
                    }
                )

    print("=== NanoGPT AF-Muon cap/scale sensitivity launcher ===", flush=True)
    print(f"seeds: {seeds}", flush=True)
    print("caps:", ["inf" if math.isinf(c) else c for c in caps], flush=True)
    print(f"scales: {scales}", flush=True)
    print("jobs:", [j["name"] for j in jobs], flush=True)
    print(
        "matched settings: AF-Muon only, batch=512, micro_batch=16, "
        "tokens/update=524288, matrix_lr=0.02, vector_lr=0.0003, "
        "matrix_wd=0.1, aux/tied_wd=0, fp32 params/state, bf16 autocast, "
        "full checkpoint saved at the final matched-token audit",
        flush=True,
    )

    if args.dry_run:
        for job in jobs:
            print()
            print(f"# seed={job['seed']}, cap={job['cap']}, scale={job['scale']}")
            args.seed = job["seed"]
            print(" ".join(command(args, job["cap"], job["scale"], job["out_dir"])))
        return

    if args.check_config:
        for job in jobs:
            args.seed = job["seed"]
            subprocess.run(
                command(args, job["cap"], job["scale"], job["out_dir"]),
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
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            args.seed = job["seed"]
            proc = subprocess.Popen(
                command(args, job["cap"], job["scale"], job["out_dir"]),
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
        raise RuntimeError(f"NanoGPT cap/scale jobs failed: {failures}")


if __name__ == "__main__":
    main()
