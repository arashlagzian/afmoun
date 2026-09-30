from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from optimizers.afmoun_v2 import AFMuonV2, chunked_tied_clipped_row_update_v2, vector_rms_update_v2
from optimizers.muon import HybridMuon, adamw_aux_update, muon_matrix_update


class Seq2SeqTokenDataset(Dataset):
    def __init__(
        self,
        path: Path,
        *,
        seq_len: int,
        num_examples: int,
        dtype: str,
    ) -> None:
        np_dtype = np.uint32 if dtype == "uint32" else np.uint16
        self.tokens = np.memmap(path, dtype=np_dtype, mode="r")
        self.seq_len = int(seq_len)
        self.num_examples = min(int(num_examples), max(0, (len(self.tokens) - 1) // self.seq_len))

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = int(idx) * self.seq_len
        end = start + self.seq_len
        arr = np.asarray(self.tokens[start:end], dtype=np.int64)
        return torch.from_numpy(arr.copy())


class ExplicitAFMuonV2(AFMuonV2):
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            role = group["role"]
            lr = float(group["lr"])
            beta = float(group.get("momentum", 0.95))
            weight_decay = float(group.get("weight_decay", 0.0))
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if role == "matrix":
                    momentum = state.setdefault("momentum_buffer", torch.zeros_like(param))
                    update = muon_matrix_update(
                        param.grad,
                        momentum,
                        beta=beta,
                        ns_steps=int(group.get("ns_steps", 5)),
                    )
                    param.mul_(1.0 - lr * weight_decay)
                    param.add_(update.reshape_as(param), alpha=-lr)
                elif role == "tied":
                    d_model = int(param.shape[1])
                    tied_lr = (
                        lr
                        * float(group.get("rho_output", 3000.0))
                        / float(group.get("rho_hidden", 50.0))
                        * float(group.get("tied_scale", 0.5))
                        / float(d_model)
                    )
                    decay_lr = float(group.get("decay_lr", lr))
                    if weight_decay:
                        param.mul_(1.0 - decay_lr * weight_decay)
                    update = chunked_tied_clipped_row_update_v2(
                        param,
                        param.grad,
                        state,
                        beta=beta,
                        cap=float(group.get("tied_cap", 3.0)),
                        chunk_rows=int(group.get("chunk_rows", 2048)),
                        bisection_steps=int(group.get("bisection_steps", 32)),
                        max_bracket_steps=int(group.get("max_bracket_steps", 128)),
                    )
                    param.add_(update, alpha=-tied_lr)
                elif role == "vector":
                    if weight_decay:
                        param.mul_(1.0 - lr * weight_decay)
                    update = vector_rms_update_v2(
                        param,
                        param.grad,
                        state,
                        beta=beta,
                        eps=float(group.get("eps", 1e-8)),
                    )
                    param.add_(update, alpha=-lr)
                else:
                    raise ValueError(f"unknown role: {role}")
        return loss


class ExplicitHybridMuon(HybridMuon):
    pass


def resolve_data_path(data_dir: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    candidates = [data_dir / p]
    for parent in data_dir.parents:
        candidates.append(parent / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def unique_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    seen: set[int] = set()
    out: list[tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        out.append((name, p))
    return out


def load_data_metadata(data_dir: Path) -> dict:
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def infer_dtype(metadata: dict, explicit: str) -> str:
    if explicit:
        return explicit
    dtype = str(metadata.get("dtype", "")).lower()
    if dtype in {"uint16", "uint32"}:
        return dtype
    return "uint32"


def build_t5gemma2_model(args: argparse.Namespace):
    try:
        from transformers import AutoConfig, AutoModelForSeq2SeqLM
    except ImportError as exc:
        raise RuntimeError("This runner requires transformers with T5Gemma2 support.") from exc

    config = AutoConfig.from_pretrained(
        args.model_config,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=bool(args.trust_remote_code),
    )
    if args.vocab_size > 0:
        config.vocab_size = int(args.vocab_size)
    if hasattr(config, "text_config") and args.vocab_size > 0:
        config.text_config.vocab_size = int(args.vocab_size)
    if hasattr(config, "tie_word_embeddings"):
        config.tie_word_embeddings = True
    if hasattr(config, "text_config") and hasattr(config.text_config, "tie_word_embeddings"):
        config.text_config.tie_word_embeddings = True
    if hasattr(config, "use_cache"):
        config.use_cache = False
    if hasattr(config, "text_config") and hasattr(config.text_config, "use_cache"):
        config.text_config.use_cache = False

    model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=bool(args.trust_remote_code))
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, config


def _module_weight(module) -> nn.Parameter | None:
    if module is None:
        return None
    weight = getattr(module, "weight", None)
    return weight if isinstance(weight, nn.Parameter) else None


def find_tied_vocab_parameter(model: nn.Module, vocab_size: int) -> nn.Parameter:
    inp = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    inp_weight = _module_weight(inp)
    out_weight = _module_weight(out)
    if inp_weight is not None and inp_weight is out_weight:
        return inp_weight

    candidates: list[tuple[str, nn.Parameter]] = []
    if inp_weight is not None:
        for name, p in unique_named_parameters(model):
            if p is inp_weight:
                candidates.append((name, p))
                break
    if not candidates:
        for name, p in unique_named_parameters(model):
            if p.ndim == 2 and (vocab_size <= 0 or int(p.shape[0]) == int(vocab_size)):
                candidates.append((name, p))
    if len(candidates) != 1:
        detail = [(n, tuple(p.shape)) for n, p in candidates[:20]]
        in_shape = None if inp_weight is None else tuple(inp_weight.shape)
        out_shape = None if out_weight is None else tuple(out_weight.shape)
        raise RuntimeError(
            "expected exactly one tied text vocabulary parameter. "
            f"input_embedding_shape={in_shape} output_embedding_shape={out_shape} "
            f"vocab_size_arg={vocab_size} candidates={len(candidates)}: {detail}"
        )
    return candidates[0][1]


def partition(model: nn.Module, vocab_size: int) -> dict:
    tied = find_tied_vocab_parameter(model, vocab_size)
    matrix, tied_params, vector = [], [], []
    names = {"matrix": [], "tied": [], "vector": []}
    for name, p in unique_named_parameters(model):
        if p is tied:
            tied_params.append(p)
            names["tied"].append(name)
        elif p.ndim >= 2:
            matrix.append(p)
            names["matrix"].append(name)
        elif p.ndim <= 1:
            vector.append(p)
            names["vector"].append(name)
        else:
            raise ValueError(f"unassigned parameter {name} shape={tuple(p.shape)}")
    if tied_params != [tied]:
        raise RuntimeError("tied vocabulary table was not uniquely assigned")
    return {"matrix": matrix, "tied": tied_params, "vector": vector, "names": names}


def build_optimizer(model: nn.Module, args: argparse.Namespace, vocab_size: int):
    parts = partition(model, vocab_size)
    if args.arm == "muon":
        opt = ExplicitHybridMuon(
            [
                {
                    "params": parts["matrix"],
                    "role": "matrix",
                    "lr": float(args.muon_lr),
                    "momentum": float(args.momentum),
                    "weight_decay": float(args.matrix_weight_decay),
                    "ns_steps": int(args.ns_steps),
                },
                {
                    "params": parts["tied"] + parts["vector"],
                    "role": "aux",
                    "lr": float(args.vector_lr),
                    "weight_decay": float(args.hybrid_aux_weight_decay),
                    "beta1": 0.9,
                    "beta2": 0.95,
                    "eps": 1e-10,
                },
            ]
        )
    else:
        tied_cap = 1.0 if args.arm == "sign" else 3.0
        tied_scale = 1.0 if args.arm == "sign" else 0.5
        aux_weight_decay = float(args.sign_aux_weight_decay if args.arm == "sign" else args.afmoun_aux_weight_decay)
        opt = ExplicitAFMuonV2(
            [
                {
                    "params": parts["matrix"],
                    "role": "matrix",
                    "lr": float(args.muon_lr),
                    "momentum": float(args.momentum),
                    "weight_decay": float(args.matrix_weight_decay),
                    "ns_steps": int(args.ns_steps),
                },
                {
                    "params": parts["tied"],
                    "role": "tied",
                    "lr": float(args.muon_lr),
                    "decay_lr": float(args.vector_lr),
                    "momentum": float(args.momentum),
                    "weight_decay": aux_weight_decay,
                    "tied_cap": tied_cap,
                    "tied_scale": tied_scale,
                    "rho_hidden": float(args.rho_hidden),
                    "rho_output": float(args.rho_output),
                    "chunk_rows": int(args.chunk_rows),
                    "bisection_steps": int(args.bisection_steps),
                    "max_bracket_steps": int(args.max_bracket_steps),
                },
                {
                    "params": parts["vector"],
                    "role": "vector",
                    "lr": float(args.vector_lr),
                    "momentum": float(args.momentum),
                    "weight_decay": aux_weight_decay,
                    "eps": 1e-8,
                },
            ]
        )
    opt.afmoun_role_names = parts["names"]
    return opt, parts


def split_batch(batch: torch.Tensor, *, source_len: int, target_len: int, bos_token_id: int):
    source = batch[:, :source_len].long().contiguous()
    target = batch[:, source_len : source_len + target_len].long().contiguous()
    bos = torch.full((target.shape[0], 1), int(bos_token_id), dtype=torch.long, device=target.device)
    decoder_input = torch.cat([bos, target[:, :-1]], dim=1).contiguous()
    labels = target.contiguous()
    return source, decoder_input, labels


@torch.no_grad()
def evaluate(model, loader, args, *, device: torch.device) -> tuple[float, float, int]:
    model.eval()
    loss_sum = 0.0
    target_count = 0
    max_batches = max(1, math.ceil(int(args.eval_tokens) / (int(args.micro_batch_size) * int(args.target_len))))
    autocast_enabled = bool(args.bf16 and device.type == "cuda")
    for i, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        source, decoder_input, labels = split_batch(
            batch,
            source_len=int(args.source_len),
            target_len=int(args.target_len),
            bos_token_id=int(args.bos_token_id),
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            out = model(input_ids=source, decoder_input_ids=decoder_input, labels=labels)
        batch_targets = int(labels.numel())
        loss_sum += float(out.loss.detach().cpu()) * batch_targets
        target_count += batch_targets
        if i + 1 >= max_batches:
            break
    model.train()
    loss = loss_sum / target_count if target_count > 0 else float("nan")
    return loss, math.exp(min(20.0, loss)), target_count


def write_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def count_params(params) -> int:
    return sum(int(p.numel()) for p in params)


def parameter_shape(param: nn.Parameter | None) -> list[int] | None:
    if param is None:
        return None
    return [int(dim) for dim in param.shape]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random-init T5Gemma2-style shared-vocab encoder-decoder runner.")
    p.add_argument("--model-config", default="google/t5gemma-2-270m-270m")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--arm", choices=["muon", "sign", "afmoun"], default="afmoun")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--source-len", type=int, default=256)
    p.add_argument("--target-len", type=int, default=256)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--train-tokens", type=int, default=750_000_000)
    p.add_argument("--eval-tokens", type=int, default=1_000_000)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--eval-every-steps", type=int, default=250)
    p.add_argument("--log-every-steps", type=int, default=25)
    p.add_argument("--checkpoint-every-steps", type=int, default=0)
    p.add_argument("--skip-initial-eval", action="store_true")
    p.add_argument("--vocab-size", type=int, default=262208)
    p.add_argument("--dtype", default="")
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--vector-lr", type=float, default=3e-4)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--matrix-weight-decay", type=float, default=0.1)
    p.add_argument("--hybrid-aux-weight-decay", type=float, default=0.01)
    p.add_argument("--sign-aux-weight-decay", type=float, default=0.0)
    p.add_argument("--afmoun-aux-weight-decay", type=float, default=0.0)
    p.add_argument("--rho-hidden", type=float, default=50.0)
    p.add_argument("--rho-output", type=float, default=3000.0)
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--ns-steps", type=int, default=5)
    p.add_argument("--bisection-steps", type=int, default=32)
    p.add_argument("--max-bracket-steps", type=int, default=128)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--bos-token-id", type=int, default=2)
    p.add_argument("--train-path", default="")
    p.add_argument("--eval-path", default="")
    p.add_argument("--estimate-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    metadata = load_data_metadata(data_dir)
    dtype = infer_dtype(metadata, args.dtype)
    train_rel = args.train_path or metadata.get("train_path", "train_tokens_uint32.bin")
    eval_rel = args.eval_path or metadata.get("eval_path", "eval_tokens_uint32.bin")
    train_path = resolve_data_path(data_dir, str(train_rel))
    eval_path = resolve_data_path(data_dir, str(eval_rel))
    seq_len = int(args.source_len) + int(args.target_len)
    target_tokens_per_step = int(args.micro_batch_size) * int(args.gradient_accumulation_steps) * int(args.target_len)
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else math.ceil(int(args.train_tokens) / target_tokens_per_step)
    train_examples = max_steps * int(args.gradient_accumulation_steps) * int(args.micro_batch_size) + 16
    eval_examples = math.ceil(int(args.eval_tokens) / int(args.target_len)) + int(args.micro_batch_size)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, config = build_t5gemma2_model(args)
    model.to(device=device, dtype=torch.float32)
    bad_dtypes = sorted({str(p.dtype) for p in model.parameters() if p.dtype != torch.float32})
    if bad_dtypes:
        raise AssertionError(
            "T5Gemma2-style runner expects FP32 trainable parameters; "
            f"BF16 is autocast-only. Found non-FP32 dtypes: {bad_dtypes}"
        )
    opt, parts = build_optimizer(model, args, int(args.vocab_size))

    input_weight = _module_weight(model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None)
    output_weight = _module_weight(model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None)
    tied_weight = parts["tied"][0] if parts["tied"] else None
    param_count = count_params(model.parameters())
    role_counts = {k: count_params(v) for k, v in parts.items() if k in {"matrix", "tied", "vector"}}
    config_payload = {
        **vars(args),
        "data_metadata": metadata,
        "train_path_resolved": str(train_path),
        "eval_path_resolved": str(eval_path),
        "seq_len": seq_len,
        "target_tokens_per_step": target_tokens_per_step,
        "raw_tokens_per_step": target_tokens_per_step * 2,
        "max_steps": max_steps,
        "actual_target_train_tokens": max_steps * target_tokens_per_step,
        "param_count": param_count,
        "role_param_counts": role_counts,
        "role_names": parts["names"],
        "input_embedding_shape": parameter_shape(input_weight),
        "output_embedding_shape": parameter_shape(output_weight),
        "tied_embedding_shape": parameter_shape(tied_weight),
        "input_output_weights_tied": bool(input_weight is not None and input_weight is output_weight),
        "model_config_class": config.__class__.__name__,
        "master_params": "fp32",
        "autocast_only": "bf16" if args.bf16 else "none",
        "random_init": True,
        "architecture_note": "T5Gemma2-style Hugging Face seq2seq config from random initialization.",
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"phase": "config", **config_payload}, sort_keys=True), flush=True)

    if args.estimate_only:
        return

    train_ds = Seq2SeqTokenDataset(train_path, seq_len=seq_len, num_examples=train_examples, dtype=dtype)
    eval_ds = Seq2SeqTokenDataset(eval_path, seq_len=seq_len, num_examples=eval_examples, dtype=dtype)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.micro_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(args.micro_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model.train()
    autocast_enabled = bool(args.bf16 and device.type == "cuda")
    step = 0
    micro_step = 0
    tokens_seen = 0
    running_loss = 0.0
    running_count = 0
    start_time = time.time()
    opt.zero_grad(set_to_none=True)

    train_iter = iter(train_loader)
    while step < max_steps:
        accum_loss = 0.0
        accum_targets = 0
        for _ in range(int(args.gradient_accumulation_steps)):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = batch.to(device, non_blocking=True)
            source, decoder_input, labels = split_batch(
                batch,
                source_len=int(args.source_len),
                target_len=int(args.target_len),
                bos_token_id=int(args.bos_token_id),
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                out = model(input_ids=source, decoder_input_ids=decoder_input, labels=labels)
                loss = out.loss / int(args.gradient_accumulation_steps)
            loss.backward()
            batch_targets = int(labels.numel())
            accum_loss += float(out.loss.detach().cpu()) * batch_targets
            accum_targets += batch_targets
            micro_step += 1
        if float(args.max_grad_norm) > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            grad_norm_value = float(grad_norm.detach().cpu())
        else:
            grad_norm_value = float("nan")
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += accum_targets
        running_loss += accum_loss
        running_count += accum_targets

        if step == 1 or step % int(args.log_every_steps) == 0:
            loss_value = running_loss / max(1, running_count)
            payload = {
                "phase": "train",
                "step": step,
                "micro_step": micro_step,
                "target_tokens_seen": tokens_seen,
                "loss": loss_value,
                "ppl": math.exp(min(20.0, loss_value)),
                "grad_norm_preclip": grad_norm_value,
                "seconds": time.time() - start_time,
            }
            print(json.dumps(payload, sort_keys=True), flush=True)
            write_jsonl(metrics_path, payload)
            running_loss = 0.0
            running_count = 0

        do_eval = step % int(args.eval_every_steps) == 0 or step == max_steps
        if step == 1 and not args.skip_initial_eval:
            do_eval = True
        if do_eval:
            ev_loss, ev_ppl, ev_tokens = evaluate(model, eval_loader, args, device=device)
            payload = {
                "phase": "eval",
                "step": step,
                "target_tokens_seen": tokens_seen,
                "eval_loss": ev_loss,
                "eval_ppl": ev_ppl,
                "eval_tokens": ev_tokens,
                "seconds": time.time() - start_time,
            }
            print(json.dumps(payload, sort_keys=True), flush=True)
            write_jsonl(metrics_path, payload)

        if int(args.checkpoint_every_steps) > 0 and (step % int(args.checkpoint_every_steps) == 0 or step == max_steps):
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / f"model_step{step}.pt"
            torch.save({"model": model.state_dict(), "step": step, "args": vars(args)}, path)
            payload = {"phase": "checkpoint", "step": step, "path": str(path), "seconds": time.time() - start_time}
            print(json.dumps(payload, sort_keys=True), flush=True)
            write_jsonl(metrics_path, payload)


if __name__ == "__main__":
    main()
