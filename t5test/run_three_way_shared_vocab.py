from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def arm_label(arm: str) -> str:
    if arm == "muon":
        return "hybrid_muon"
    if arm == "sign":
        return "scion_c1_s1"
    if arm == "afmoun":
        return "afmoun_c3_s0p5"
    raise ValueError(arm)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch three-way shared-vocab sentinel arms.")
    p.add_argument("--gpus", nargs="+", type=int, required=True)
    p.add_argument("--run-root", default="runs/t5test_three_way_shared_vocab")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    p.add_argument("--arms", default="muon,sign,afmoun")
    p.add_argument("--sharing", choices=["full", "untied"], default="full")
    p.add_argument("--rho-output-values", nargs="+", type=float, default=[50.0])
    p.add_argument(
        "--rho-sweep-arms",
        default="sign,afmoun",
        help="Comma-separated arms that receive --rho-output-values. Other arms run once.",
    )
    p.add_argument("--train-tokens", type=int, default=275_000_000)
    p.add_argument("--eval-tokens", type=int, default=1_000_000)
    p.add_argument("--vocab-size", type=int, default=49152)
    p.add_argument("--dtype", default="")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--encoder-layers", type=int, default=4)
    p.add_argument("--decoder-layers", type=int, default=4)
    p.add_argument("--source-len", type=int, default=256)
    p.add_argument("--target-len", type=int, default=256)
    p.add_argument("--micro-batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--eval-every-steps", type=int, default=5000)
    p.add_argument("--log-every-steps", type=int, default=500)
    p.add_argument("--diag-every-steps", type=int, default=5000)
    p.add_argument("--encoder-ablation-every-steps", type=int, default=5000)
    p.add_argument("--checkpoint-every-steps", type=int, default=0)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--vector-lr", type=float, default=3e-4)
    p.add_argument("--matrix-weight-decay", type=float, default=0.1)
    p.add_argument("--hybrid-aux-weight-decay", type=float, default=0.01)
    p.add_argument("--sign-aux-weight-decay", type=float, default=0.0)
    p.add_argument("--afmoun-aux-weight-decay", type=float, default=0.0)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    log_dir = root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rho_sweep_arms = {a.strip() for a in args.rho_sweep_arms.split(",") if a.strip()}
    rho_values = args.rho_output_values or [None]
    pending = []
    for seed in args.seeds:
        for arm in arms:
            arm_rhos = rho_values if arm in rho_sweep_arms else [None]
            pending.extend((seed, arm, rho) for rho in arm_rhos)
    print("=== three-way shared-vocab sentinel launcher ===", flush=True)
    print(f"run_root={root}", flush=True)
    print(f"gpus={args.gpus}", flush=True)
    print(
        f"seeds={args.seeds} arms={arms} rho_output_values={rho_values} "
        f"rho_sweep_arms={sorted(rho_sweep_arms)} sharing={args.sharing} pending={len(pending)}",
        flush=True,
    )

    worker = Path(__file__).with_name("train_three_way_shared_vocab.py")
    python = os.environ.get("AFMOUN_WORKER_PYTHON", sys.executable)
    def launch(seed: int, arm: str, rho_output: float | None, gpu: int):
        suffix = "" if rho_output is None else f"_rho{rho_output:g}"
        sharing_suffix = "" if args.sharing == "full" else f"_{args.sharing}"
        name = f"t5test_{arm_label(arm)}_seed{seed}{suffix}_three_way_shared_vocab{sharing_suffix}"
        out_dir = root / name
        log_path = log_dir / f"{name}.log"
        cmd = [
            python,
            str(worker),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(out_dir),
            "--arm",
            arm,
            "--sharing",
            str(args.sharing),
            "--seed",
            str(seed),
            "--train-tokens",
            str(args.train_tokens),
            "--eval-tokens",
            str(args.eval_tokens),
            "--vocab-size",
            str(args.vocab_size),
            "--dtype",
            str(args.dtype),
            "--dim",
            str(args.dim),
            "--heads",
            str(args.heads),
            "--encoder-layers",
            str(args.encoder_layers),
            "--decoder-layers",
            str(args.decoder_layers),
            "--source-len",
            str(args.source_len),
            "--target-len",
            str(args.target_len),
            "--micro-batch-size",
            str(args.micro_batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--eval-every-steps",
            str(args.eval_every_steps),
            "--log-every-steps",
            str(args.log_every_steps),
            "--diag-every-steps",
            str(args.diag_every_steps),
            "--encoder-ablation-every-steps",
            str(args.encoder_ablation_every_steps),
            "--checkpoint-every-steps",
            str(args.checkpoint_every_steps),
            "--muon-lr",
            str(args.muon_lr),
            "--vector-lr",
            str(args.vector_lr),
            "--matrix-weight-decay",
            str(args.matrix_weight_decay),
            "--hybrid-aux-weight-decay",
            str(args.hybrid_aux_weight_decay),
            "--sign-aux-weight-decay",
            str(args.sign_aux_weight_decay),
            "--afmoun-aux-weight-decay",
            str(args.afmoun_aux_weight_decay),
        ]
        if rho_output is not None:
            cmd.extend(["--rho-output", str(rho_output)])
        if args.bf16:
            cmd.append("--bf16")
        print(f"pending: {name} arm={arm} seed={seed} rho_output={rho_output} gpu={gpu}", flush=True)
        if args.dry_run:
            print(" ".join(cmd), flush=True)
            return None
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        f = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        print(f"[GPU {gpu}] launched {name} pid={proc.pid} log={log_path}", flush=True)
        return (name, proc, f, log_path, gpu)

    queue = list(pending)
    if args.dry_run:
        for idx, (seed, arm, rho) in enumerate(queue):
            launch(seed, arm, rho, int(args.gpus[idx % len(args.gpus)]))
        return

    launched = []
    free_gpus = [int(g) for g in args.gpus]
    while queue and free_gpus:
        seed, arm, rho = queue.pop(0)
        launched.append(launch(seed, arm, rho, free_gpus.pop(0)))

    failures = []
    while launched or queue:
        alive = []
        status = []
        for name, proc, f, log_path, gpu in launched:
            rc = proc.poll()
            if rc is None:
                alive.append((name, proc, f, log_path, gpu))
                status.append(f"{name}=running(GPU{gpu})")
            else:
                f.close()
                print(f"finished {name}: exit={rc} gpu={gpu} log={log_path}", flush=True)
                if rc != 0:
                    failures.append((name, rc, str(log_path)))
                elif queue:
                    seed, arm, rho = queue.pop(0)
                    new_job = launch(seed, arm, rho, gpu)
                    if new_job is not None:
                        alive.append(new_job)
                    continue
                free_gpus.append(gpu)
        if status:
            print(", ".join(status), flush=True)
            time.sleep(max(1, int(args.poll_seconds)))
        elif queue and free_gpus:
            seed, arm, rho = queue.pop(0)
            new_job = launch(seed, arm, rho, free_gpus.pop(0))
            if new_job is not None:
                alive.append(new_job)
        launched = alive

    if failures:
        raise RuntimeError(f"three-way shared-vocab jobs failed: {failures}")


if __name__ == "__main__":
    main()
