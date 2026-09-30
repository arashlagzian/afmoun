from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch protein UniRef50 three-arm sentinel jobs.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run-root", default="runs/protein_363m_500m_matchbatch524k")
    p.add_argument("--gpus", default="0,1,2")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--train-tokens", type=int, default=500_000_000)
    p.add_argument("--eval-tokens", type=int, default=5_000_000)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--gradient-accumulation-steps", type=int, default=32)
    p.add_argument("--eval-every-steps", type=int, default=125)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--hybrid-aux-weight-decay", type=float, default=0.01)
    p.add_argument("--seeds", default="43,44,45")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    log_dir = root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    arms = [
        ("muon", "hybrid_muon"),
        ("sign", "sign_endpoint"),
        ("afmoun", "afmoun_c3_s0p5"),
    ]
    if not gpus:
        raise ValueError("provide at least one GPU id")

    def build_job(seed: int, opt: str, tag: str) -> dict:
        out_dir = root / f"protein_{tag}_seed{seed}_matchbatch524k_mbs32"
        log_path = log_dir / f"protein_{tag}_seed{seed}.log"
        cmd = [
            args.python,
            "scripts/train_protein_causal_lm.py",
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(out_dir),
            "--optimizer",
            opt,
            "--train-tokens",
            str(args.train_tokens),
            "--eval-tokens",
            str(args.eval_tokens),
            "--block-size",
            str(args.block_size),
            "--micro-batch-size",
            str(args.micro_batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--eval-every-steps",
            str(args.eval_every_steps),
            "--log-every-steps",
            str(args.log_every_steps),
            "--max-grad-norm",
            str(args.max_grad_norm),
            "--aux-weight-decay",
            str(args.hybrid_aux_weight_decay if opt == "muon" else 0.0),
            "--seed",
            str(seed),
        ]
        if args.bf16:
            cmd.append("--bf16")
        return {"seed": seed, "tag": tag, "cmd": cmd, "log_path": log_path}

    pending = [build_job(seed, opt, tag) for seed in seeds for opt, tag in arms]
    if args.dry_run:
        for idx, job in enumerate(pending):
            gpu = gpus[idx % len(gpus)]
            print(f"[GPU {gpu}] {' '.join(job['cmd'])} > {job['log_path']}", flush=True)
        return

    running = []
    free_gpus = list(gpus)
    failures = []
    while pending or running:
        while pending and free_gpus:
            job = pending.pop(0)
            gpu = free_gpus.pop(0)
            env = dict(**__import__("os").environ, CUDA_VISIBLE_DEVICES=gpu)
            print(f"[GPU {gpu}] {' '.join(job['cmd'])} > {job['log_path']}", flush=True)
            f = job["log_path"].open("w", encoding="utf-8")
            proc = subprocess.Popen(job["cmd"], stdout=f, stderr=subprocess.STDOUT, env=env)
            running.append({"job": job, "gpu": gpu, "proc": proc, "handle": f})
            time.sleep(2.0)
        time.sleep(5.0)
        still_running = []
        for item in running:
            rc = item["proc"].poll()
            if rc is None:
                still_running.append(item)
                continue
            item["handle"].close()
            tag = item["job"]["tag"]
            seed = item["job"]["seed"]
            print(f"protein_{tag}_seed{seed} exit={rc}", flush=True)
            free_gpus.append(item["gpu"])
            if rc != 0:
                failures.append((tag, seed, rc))
        running = still_running
    if failures:
        raise RuntimeError(f"protein jobs failed: {failures}")


if __name__ == "__main__":
    main()
