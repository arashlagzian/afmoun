from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


SOURCE_LABELS = {
    "input_only": "sparse_input_only",
    "output_only": "dense_output_only",
    "both": "input_plus_output",
}


def worker_python() -> str:
    return os.environ.get("AFMOUN_WORKER_PYTHON") or sys.executable


def parse_csv_text(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch AF-Muon tied input/output gradient-source ablation runs."
    )
    p.add_argument("--gpus", nargs="+", type=int, required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument(
        "--run-root",
        default="runs/nanogpt_tied_source_ablation_seed43_500m_matchbatch524k",
    )
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--sources", default="input_only,output_only,both")
    p.add_argument("--iterations", type=int, default=954)
    p.add_argument("--eval-every-steps", type=int, default=125)
    p.add_argument("--diag-every-steps", type=int, default=125)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--eval-tokens", type=int, default=5_000_000)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--check-config", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_name(source: str, seed: int) -> str:
    return (
        f"nanogpt_afmoun_c3_s0p5_seed{seed}_tied_{SOURCE_LABELS[source]}"
        "_500m_fp32_matchbatch524k"
    )


def command(args: argparse.Namespace, source: str, out_dir: Path) -> list[str]:
    worker = ROOT / "ablation" / "train_nanogpt_tied_source_ablation.py"
    cmd = [
        worker_python(),
        str(worker),
        "--data-dir", str(args.data_dir),
        "--output-dir", str(out_dir),
        "--arm", "afmoun",
        "--tied-grad-source", source,
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
        "--skip-final-checkpoint",
        "--muon-lr", "0.02",
        "--muon-lr-multiplier", "1.0",
        "--multiplier-mode", "muon_only",
        "--vector-lr", "0.0003",
        "--momentum", "0.95",
        "--matrix-weight-decay", "0.1",
        "--aux-weight-decay", "0.0",
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
    sources = parse_csv_text(args.sources)
    bad_sources = [source for source in sources if source not in SOURCE_LABELS]
    if bad_sources:
        raise ValueError(f"unknown sources: {bad_sources}")

    jobs = []
    for source in sources:
        name = run_name(source, args.seed)
        jobs.append(
            {
                "source": source,
                "name": name,
                "out_dir": Path(args.run_root) / name,
            }
        )

    print("=== NanoGPT AF-Muon tied source ablation launcher ===", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(f"sources: {sources}", flush=True)
    print("jobs:", [j["name"] for j in jobs], flush=True)
    print(
        "matched settings: AF-Muon c=3 s=0.5, batch=512, micro_batch=16, "
        "tokens/update=524288, vector_lr=0.0003, matrix_lr=0.02, "
        "fp32 params/state, bf16 autocast",
        flush=True,
    )

    if args.dry_run:
        for job in jobs:
            print()
            print(f"# tied source: {job['source']}")
            print(" ".join(command(args, job["source"], job["out_dir"])))
        return

    if args.check_config:
        for job in jobs:
            subprocess.run(
                command(args, job["source"], job["out_dir"]),
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
                command(args, job["source"], job["out_dir"]),
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
        raise RuntimeError(f"NanoGPT tied source ablation jobs failed: {failures}")


if __name__ == "__main__":
    main()
