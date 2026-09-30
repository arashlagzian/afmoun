from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def lr_tag(lr: float) -> str:
    return str(lr).replace(".", "p").replace("-", "m")


def parse_lrs(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_gpus(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Queue AdamW protein LR sweep over a fixed set of GPUs.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--gpus", default="1,2")
    p.add_argument("--lrs", default="1e-4,3e-4,6e-4,1e-3")
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--run-prefix", default="protein_adamw_lr")
    p.add_argument("--data-dir", default="data/protgpt2_bpe_uniref50_500m_local")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--eval-every-steps", type=int, default=100)
    p.add_argument("--log-every-steps", type=int, default=50)
    args = p.parse_args()

    gpus = parse_gpus(args.gpus)
    jobs = parse_lrs(args.lrs)
    if not gpus:
        raise ValueError("at least one GPU is required")
    if not jobs:
        raise ValueError("at least one LR is required")

    pending = list(jobs)
    running: dict[str, tuple[float, subprocess.Popen]] = {}

    def launch(gpu: str, lr: float) -> subprocess.Popen:
        run_dir = ROOT / "runs" / f"{args.run_prefix}{lr_tag(lr)}_363m_500step"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"
        cmd = [
            args.python,
            "scripts/train_protein_causal_lm.py",
            "--data-dir",
            args.data_dir,
            "--output-dir",
            str(run_dir),
            "--optimizer",
            "adamw",
            "--adamw-lr",
            str(lr),
            "--adamw-weight-decay",
            "0.1",
            "--adamw-beta1",
            "0.9",
            "--adamw-beta2",
            "0.95",
            "--lr-schedule",
            "constant",
            "--warmup-steps",
            "0",
            "--train-tokens",
            "500000000",
            "--eval-tokens",
            "5000000",
            "--max-steps",
            str(args.max_steps),
            "--eval-every-steps",
            str(args.eval_every_steps),
            "--log-every-steps",
            str(args.log_every_steps),
            "--micro-batch-size",
            "4",
            "--gradient-accumulation-steps",
            "32",
            "--block-size",
            "512",
            "--hidden-size",
            "960",
            "--intermediate-size",
            "2560",
            "--num-hidden-layers",
            "32",
            "--num-attention-heads",
            "15",
            "--num-key-value-heads",
            "5",
            "--rope-theta",
            "100000.0",
            "--max-grad-norm",
            "1.0",
            "--bf16",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"launched lr={lr:g} on gpu={gpu} pid={proc.pid} log={log_path}", flush=True)
        return proc

    while pending or running:
        for gpu in gpus:
            if gpu not in running and pending:
                lr = pending.pop(0)
                running[gpu] = (lr, launch(gpu, lr))

        finished = []
        for gpu, (lr, proc) in running.items():
            code = proc.poll()
            if code is not None:
                print(f"finished lr={lr:g} on gpu={gpu} returncode={code}", flush=True)
                finished.append(gpu)
        for gpu in finished:
            running.pop(gpu, None)

        if pending or running:
            time.sleep(args.poll_seconds)

    print("all AdamW LR sweep jobs finished", flush=True)


if __name__ == "__main__":
    main()
