from __future__ import annotations

import argparse
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
    p = argparse.ArgumentParser(description="Launch random-init T5Gemma2-style shared-vocab arms.")
    p.add_argument("--gpus", nargs="+", type=int, required=True)
    p.add_argument("--run-root", default="runs/t5gemma2_style_270m270m_750m")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-config", default="google/t5gemma-2-270m-270m")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=[43])
    p.add_argument("--arms", default="muon,sign,afmoun")
    p.add_argument("--source-len", type=int, default=256)
    p.add_argument("--target-len", type=int, default=256)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--train-tokens", type=int, default=750_000_000)
    p.add_argument("--eval-tokens", type=int, default=1_000_000)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--vocab-size", type=int, default=262208)
    p.add_argument("--dtype", default="")
    p.add_argument("--eval-every-steps", type=int, default=250)
    p.add_argument("--log-every-steps", type=int, default=25)
    p.add_argument("--checkpoint-every-steps", type=int, default=0)
    p.add_argument("--skip-initial-eval", action="store_true")
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--vector-lr", type=float, default=3e-4)
    p.add_argument("--matrix-weight-decay", type=float, default=0.1)
    p.add_argument("--hybrid-aux-weight-decay", type=float, default=0.01)
    p.add_argument("--sign-aux-weight-decay", type=float, default=0.0)
    p.add_argument("--afmoun-aux-weight-decay", type=float, default=0.0)
    p.add_argument("--rho-hidden", type=float, default=50.0)
    p.add_argument("--rho-output", type=float, default=3000.0)
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--bos-token-id", type=int, default=2)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--estimate-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    log_dir = root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    pending = [(seed, arm) for seed in args.seeds for arm in arms]
    print("=== T5Gemma2-style random-init launcher ===", flush=True)
    print(f"run_root={root}", flush=True)
    print(f"model_config={args.model_config}", flush=True)
    print(f"gpus={args.gpus}", flush=True)
    print(f"seeds={args.seeds} arms={arms} pending={len(pending)}", flush=True)

    worker = Path(__file__).with_name("train_t5gemma2_style.py")
    python = os.environ.get("AFMOUN_WORKER_PYTHON", sys.executable)

    def launch(seed: int, arm: str, gpu: int):
        budget = f"{int(args.train_tokens / 1_000_000)}m"
        name = f"t5gemma2_style_{arm_label(arm)}_seed{seed}_{budget}_vocab{args.vocab_size}"
        out_dir = root / name
        log_path = log_dir / f"{name}.log"
        cmd = [
            python,
            str(worker),
            "--model-config",
            str(args.model_config),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(out_dir),
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--source-len",
            str(args.source_len),
            "--target-len",
            str(args.target_len),
            "--micro-batch-size",
            str(args.micro_batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--train-tokens",
            str(args.train_tokens),
            "--eval-tokens",
            str(args.eval_tokens),
            "--max-steps",
            str(args.max_steps),
            "--vocab-size",
            str(args.vocab_size),
            "--dtype",
            str(args.dtype),
            "--eval-every-steps",
            str(args.eval_every_steps),
            "--log-every-steps",
            str(args.log_every_steps),
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
            "--rho-hidden",
            str(args.rho_hidden),
            "--rho-output",
            str(args.rho_output),
            "--chunk-rows",
            str(args.chunk_rows),
            "--max-grad-norm",
            str(args.max_grad_norm),
            "--bos-token-id",
            str(args.bos_token_id),
        ]
        if args.local_files_only:
            cmd.append("--local-files-only")
        if args.trust_remote_code:
            cmd.append("--trust-remote-code")
        if args.bf16:
            cmd.append("--bf16")
        else:
            cmd.append("--no-bf16")
        if args.gradient_checkpointing:
            cmd.append("--gradient-checkpointing")
        else:
            cmd.append("--no-gradient-checkpointing")
        if args.estimate_only:
            cmd.append("--estimate-only")
        if args.skip_initial_eval:
            cmd.append("--skip-initial-eval")
        print(f"pending: {name} arm={arm} seed={seed} gpu={gpu}", flush=True)
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
        for idx, (seed, arm) in enumerate(queue):
            launch(seed, arm, int(args.gpus[idx % len(args.gpus)]))
        return

    launched = []
    free_gpus = [int(g) for g in args.gpus]
    while queue and free_gpus:
        seed, arm = queue.pop(0)
        launched.append(launch(seed, arm, free_gpus.pop(0)))

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
                    seed, arm = queue.pop(0)
                    new_job = launch(seed, arm, gpu)
                    if new_job is not None:
                        alive.append(new_job)
                    continue
                free_gpus.append(gpu)
        if status:
            print(", ".join(status), flush=True)
            time.sleep(max(1, int(args.poll_seconds)))
        elif queue and free_gpus:
            seed, arm = queue.pop(0)
            new_job = launch(seed, arm, free_gpus.pop(0))
            if new_job is not None:
                alive.append(new_job)
        launched = alive

    if failures:
        raise RuntimeError(f"T5Gemma2-style jobs failed: {failures}")


if __name__ == "__main__":
    main()
