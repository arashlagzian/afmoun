from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def worker_python() -> str:
    return os.environ.get("AFMOUN_WORKER_PYTHON") or sys.executable


def lr_tag(value: float) -> str:
    text = f"{value:.8g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


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


def parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_csv_text(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch NanoGPT auxiliary/fallback learning-rate sensitivity runs."
    )
    p.add_argument("--gpus", nargs="+", type=int, required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument(
        "--run-root",
        default="runs/nanogpt_auxlr_sensitivity_seed43_500m_matchbatch524k",
    )
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--arms", default="muon,sign,afmoun")
    p.add_argument("--vector-lrs", default="7.5e-5,1.5e-4,3e-4,6e-4,1.2e-3,2.4e-3,4.8e-3,9.6e-3")
    p.add_argument("--iterations", type=int, default=954)
    p.add_argument("--eval-every-steps", type=int, default=125)
    p.add_argument("--diag-every-steps", type=int, default=125)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--eval-tokens", type=int, default=5_000_000)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--check-config", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_name(arm: str, seed: int, vector_lr: float) -> str:
    return (
        f"nanogpt_{arm_label(arm)}_seed{seed}_auxlr{lr_tag(vector_lr)}"
        "_500m_fp32_matchbatch524k"
    )


def command(args: argparse.Namespace, arm: str, vector_lr: float, out_dir: Path) -> list[str]:
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
        "--block-size", "1024",
        "--iterations", str(args.iterations),
        "--batch-size", "512",
        "--micro-batch-size", "16",
        "--eval-tokens", str(args.eval_tokens),
        "--eval-every-steps", str(args.eval_every_steps),
        "--log-every-steps", str(args.log_every_steps),
        "--diag-every-steps", str(args.diag_every_steps),
        "--checkpoint-every-steps", "0",
        "--full-checkpoint-every-steps", "0",
        "--muon-lr", "0.02",
        "--muon-lr-multiplier", "1.0",
        "--multiplier-mode", "muon_only",
        "--vector-lr", str(vector_lr),
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
    vector_lrs = parse_csv_floats(args.vector_lrs)
    valid_arms = {"muon", "sign", "afmoun"}
    bad_arms = [arm for arm in arms if arm not in valid_arms]
    if bad_arms:
        raise ValueError(f"unknown arms: {bad_arms}")

    jobs = []
    for vector_lr in vector_lrs:
        for arm in arms:
            name = run_name(arm, args.seed, vector_lr)
            jobs.append(
                {
                    "arm": arm,
                    "vector_lr": vector_lr,
                    "name": name,
                    "out_dir": Path(args.run_root) / name,
                }
            )

    print("=== NanoGPT auxiliary/fallback LR sensitivity launcher ===", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(f"vector_lrs: {vector_lrs}", flush=True)
    print("jobs:", [j["name"] for j in jobs], flush=True)
    print("matched settings: batch=512, micro_batch=16, tokens/update=524288, fp32 params/state, bf16 autocast", flush=True)

    if args.dry_run:
        for job in jobs:
            print()
            print(f"# {paper_name(job['arm'])}, vector_lr={job['vector_lr']}")
            print(" ".join(command(args, job["arm"], job["vector_lr"], job["out_dir"])))
        return

    if args.check_config:
        for job in jobs:
            subprocess.run(
                command(args, job["arm"], job["vector_lr"], job["out_dir"]),
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
                command(args, job["arm"], job["vector_lr"], job["out_dir"]),
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
        raise RuntimeError(f"NanoGPT aux LR jobs failed: {failures}")


if __name__ == "__main__":
    main()
