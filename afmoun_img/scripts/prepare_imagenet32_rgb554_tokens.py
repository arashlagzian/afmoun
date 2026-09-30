from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare ImageNet-32 RGB554 uint16 token cache.")
    p.add_argument("--dataset-name", default="benjamin-paine/imagenet-1k-32x32")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hf-home", default="")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="validation")
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument("--limit-train-images", type=int, default=0)
    p.add_argument("--limit-eval-images", type=int, default=0)
    return p.parse_args()


def rgb554_tokens(image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint16)
    if arr.shape != (32, 32, 3):
        raise ValueError(f"expected 32x32 RGB image, got {arr.shape}")
    r = arr[..., 0] >> 3
    g = arr[..., 1] >> 3
    b = arr[..., 2] >> 4
    return (r * 512 + g * 16 + b).astype(np.uint16).reshape(-1)


def write_split(ds, out_path: Path, *, limit_images: int, log_prefix: str, log_every: int) -> dict:
    n = len(ds)
    if limit_images > 0:
        n = min(n, int(limit_images))
    tokens_per_image = 1024
    total_tokens = n * tokens_per_image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))

    start = time.time()
    for i in range(n):
        mmap[i * tokens_per_image : (i + 1) * tokens_per_image] = rgb554_tokens(ds[i]["image"])
        if (i + 1) % int(log_every) == 0 or (i + 1) == n:
            elapsed = max(1e-6, time.time() - start)
            print(
                f"[{log_prefix}] images={i + 1:,}/{n:,} tokens={(i + 1) * tokens_per_image:,}/{total_tokens:,} "
                f"img/s={(i + 1) / elapsed:,.1f}",
                flush=True,
            )

    mmap.flush()
    return {"images": n, "tokens": total_tokens, "path": out_path.name}


def main() -> None:
    args = parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
        os.environ["HF_HUB_CACHE"] = str(Path(args.hf_home) / "hub")
        os.environ["HF_DATASETS_CACHE"] = str(Path(args.hf_home) / "datasets")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.dataset_name} from HF cache/download...", flush=True)
    train = load_dataset(args.dataset_name, split=args.train_split, streaming=False)
    eval_ds = load_dataset(args.dataset_name, split=args.eval_split, streaming=False)
    print(train, flush=True)
    print(eval_ds, flush=True)

    train_meta = write_split(
        train,
        out / "train_tokens_uint16.bin",
        limit_images=args.limit_train_images,
        log_prefix="train",
        log_every=args.log_every,
    )
    eval_meta = write_split(
        eval_ds,
        out / "eval_tokens_uint16.bin",
        limit_images=args.limit_eval_images,
        log_prefix="eval",
        log_every=max(1_000, args.log_every // 10),
    )

    metadata = {
        "dataset_name": args.dataset_name,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "tokenizer": "deterministic_rgb554",
        "color_vocab_size": 16_384,
        "sos_token": 16_384,
        "vocab_size": 16_385,
        "dtype": "uint16",
        "image_size": 32,
        "channels": 3,
        "sequence_length": 1024,
        "train_images": train_meta["images"],
        "eval_images": eval_meta["images"],
        "stored_train_tokens": train_meta["tokens"],
        "stored_eval_tokens": eval_meta["tokens"],
        "train_blocks": train_meta["images"],
        "eval_blocks": eval_meta["images"],
        "train_path": train_meta["path"],
        "eval_path": eval_meta["path"],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    print("ImageNet-32 RGB554 token cache prepared.", flush=True)


if __name__ == "__main__":
    main()

