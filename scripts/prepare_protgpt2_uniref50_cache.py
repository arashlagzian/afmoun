from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


TEXT_COLUMNS = ("sequence", "Sequence", "seq", "text", "protein_sequence")
TOKEN_COLUMNS = ("input_ids", "ids", "tokens")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokenize UniRef50 with the ProtGPT2 BPE tokenizer.")
    p.add_argument("--dataset-name", default="nferruz/UR50_2021_04")
    p.add_argument("--tokenizer-name", default="nferruz/ProtGPT2")
    p.add_argument("--output-dir", default="data/protgpt2_bpe_uniref50_500m")
    p.add_argument("--train-tokens", type=int, default=500_000_000)
    p.add_argument("--eval-tokens", type=int, default=5_000_000)
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="")
    p.add_argument("--sequence-column", default="")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--no-streaming", action="store_false", dest="streaming")
    p.add_argument("--append-eos", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-sequences", type=int, default=0)
    return p.parse_args()


def choose_column(example: dict, requested: str) -> str:
    if requested:
        if requested not in example:
            raise KeyError(f"requested sequence column {requested!r} not found; keys={list(example)}")
        return requested
    for col in TOKEN_COLUMNS:
        if col in example:
            return col
    for col in TEXT_COLUMNS:
        if col in example:
            return col
    for key, value in example.items():
        if isinstance(value, str):
            return key
    raise KeyError(f"could not infer sequence column from keys={list(example)}")


def write_tokens(
    iterable,
    *,
    tokenizer,
    output_path: Path,
    target_tokens: int,
    sequence_column: str,
    append_eos: bool,
    max_sequences: int,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eos_id = tokenizer.eos_token_id
    if append_eos and eos_id is None:
        raise ValueError("tokenizer has no eos_token_id; disable --append-eos or choose another tokenizer")

    n_tokens = 0
    n_sequences = 0
    with output_path.open("wb") as f:
        for ex in iterable:
            value = ex[sequence_column]
            if isinstance(value, str):
                seq = value.strip()
                if not seq:
                    continue
                ids = tokenizer.encode(seq, add_special_tokens=False)
            else:
                ids = value.tolist() if hasattr(value, "tolist") else list(value)
                ids = [int(x) for x in ids if int(x) >= 0]
            if append_eos and (not ids or ids[-1] != int(eos_id)):
                ids.append(int(eos_id))
            if not ids:
                continue
            remaining = int(target_tokens) - n_tokens
            if remaining <= 0:
                break
            if len(ids) > remaining:
                ids = ids[:remaining]
            arr = np.asarray(ids, dtype=np.uint32)
            f.write(arr.tobytes())
            n_tokens += int(arr.size)
            n_sequences += 1
            if max_sequences > 0 and n_sequences >= max_sequences:
                break
            if n_tokens >= int(target_tokens):
                break

    return {"path": str(output_path), "tokens": n_tokens, "sequences": n_sequences}


def get_split(dataset_name: str, split: str, streaming: bool, seed: int):
    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    if streaming:
        return ds.shuffle(seed=seed, buffer_size=10_000)
    return ds.shuffle(seed=seed)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    train_split = args.train_split.strip()
    train_iter = get_split(args.dataset_name, train_split, args.streaming, args.seed)
    first = next(iter(train_iter))
    sequence_column = choose_column(first, args.sequence_column)
    train_iter = get_split(args.dataset_name, train_split, args.streaming, args.seed)

    eval_split = args.eval_split.strip()
    if not eval_split:
        raise ValueError(
            "A held-out --eval-split is required for reproducible validation. "
            "For the paper recipe use --eval-split validation."
        )
    if eval_split == train_split:
        raise ValueError(
            "--eval-split must differ from --train-split for reproducible validation. "
            "For the paper recipe use --train-split train --eval-split validation."
        )
    eval_iter = get_split(args.dataset_name, eval_split, args.streaming, args.seed + 1)

    train_info = write_tokens(
        train_iter,
        tokenizer=tokenizer,
        output_path=out_dir / "train_tokens_uint32.bin",
        target_tokens=args.train_tokens,
        sequence_column=sequence_column,
        append_eos=args.append_eos,
        max_sequences=args.max_sequences,
    )
    eval_info = write_tokens(
        eval_iter,
        tokenizer=tokenizer,
        output_path=out_dir / "eval_tokens_uint32.bin",
        target_tokens=args.eval_tokens,
        sequence_column=sequence_column,
        append_eos=args.append_eos,
        max_sequences=0,
    )

    meta = {
        "dataset_name": args.dataset_name,
        "tokenizer_name": args.tokenizer_name,
        "sequence_column": sequence_column,
        "train_split": train_split,
        "eval_split": eval_split,
        "train_path": "train_tokens_uint32.bin",
        "eval_path": "eval_tokens_uint32.bin",
        "dtype": "uint32",
        "vocab_size": int(len(tokenizer)),
        "eos_token_id": tokenizer.eos_token_id,
        "append_eos": bool(args.append_eos),
        "train_tokens": train_info["tokens"],
        "eval_tokens": eval_info["tokens"],
        "train_sequences": train_info["sequences"],
        "eval_sequences": eval_info["sequences"],
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
