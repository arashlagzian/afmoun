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
from torch.nn import functional as F
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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = RMSNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim, bias=False),
            nn.GELU(),
            nn.Linear(ff_mult * dim, dim, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(a)
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = RMSNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln3 = RMSNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim, bias=False),
            nn.GELU(),
            nn.Linear(ff_mult * dim, dim, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.dropout(a)
        h = self.ln2(x)
        a, _ = self.cross_attn(h, memory, memory, need_weights=False)
        x = x + self.dropout(a)
        x = x + self.dropout(self.ff(self.ln3(x)))
        return x


class ThreeWaySharedVocabSeq2Seq(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int = 512,
        heads: int = 8,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        ff_mult: int = 4,
        source_len: int = 256,
        target_len: int = 256,
        max_positions: int = 512,
        dropout: float = 0.0,
        sharing: str = "full",
    ) -> None:
        super().__init__()
        if sharing not in {"full", "untied"}:
            raise ValueError(f"sharing must be 'full' or 'untied', got {sharing}")
        self.sharing = sharing
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.source_len = int(source_len)
        self.target_len = int(target_len)
        self.encoder_embed = nn.Embedding(vocab_size, dim)
        if sharing == "full":
            self.decoder_embed = self.encoder_embed
            self.output_weight = self.encoder_embed.weight
        else:
            self.decoder_embed = nn.Embedding(vocab_size, dim)
            self.output_weight = nn.Parameter(torch.empty(vocab_size, dim))
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(dim, heads, ff_mult, dropout) for _ in range(int(encoder_layers))]
        )
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(dim, heads, ff_mult, dropout) for _ in range(int(decoder_layers))]
        )
        self.encoder_norm = RMSNorm(dim)
        self.decoder_norm = RMSNorm(dim)
        self.register_buffer("pos", self._sinusoidal_positions(max_positions, dim), persistent=False)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((target_len, target_len), float("-inf")), diagonal=1),
            persistent=False,
        )
        nn.init.normal_(self.encoder_embed.weight, mean=0.0, std=0.02)
        if sharing == "untied":
            nn.init.normal_(self.decoder_embed.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.output_weight, mean=0.0, std=0.02)

    @staticmethod
    def _sinusoidal_positions(max_positions: int, dim: int) -> torch.Tensor:
        pos = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_positions, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def get_input_embeddings(self):
        return self.decoder_embed

    def get_output_embeddings(self):
        if self.output_weight is self.decoder_embed.weight:
            return self.decoder_embed
        return None

    def vocab_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "encoder_embed": self.encoder_embed.weight,
            "decoder_embed": self.decoder_embed.weight,
            "output_weight": self.output_weight,
        }

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.encoder_embed(input_ids) * math.sqrt(float(self.dim))
        pos = self.pos[: input_ids.shape[1]].to(device=x.device, dtype=x.dtype)
        x = x + pos
        return self.encode_embedded(x)

    def encode_embedded(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.encoder_blocks:
            x = block(x)
        return self.encoder_norm(x)

    def decode(self, decoder_input_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(decoder_input_ids) * math.sqrt(float(self.dim))
        pos = self.pos[: decoder_input_ids.shape[1]].to(device=x.device, dtype=x.dtype)
        x = x + pos
        return self.decode_embedded(x, memory)

    def decode_embedded(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        mask = self.causal_mask[: x.shape[1], : x.shape[1]].to(x.device)
        for block in self.decoder_blocks:
            x = block(x, memory, mask)
        return self.decoder_norm(x)

    def forward_with_embedding_weights(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        *,
        encoder_weight: torch.Tensor,
        decoder_weight: torch.Tensor,
        output_weight: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        enc = F.embedding(input_ids, encoder_weight) * math.sqrt(float(self.dim))
        enc = enc + self.pos[: input_ids.shape[1]].to(device=enc.device, dtype=enc.dtype)
        memory = self.encode_embedded(enc)
        dec = F.embedding(decoder_input_ids, decoder_weight) * math.sqrt(float(self.dim))
        dec = dec + self.pos[: decoder_input_ids.shape[1]].to(device=dec.device, dtype=dec.dtype)
        hidden = self.decode_embedded(dec, memory)
        logits = F.linear(hidden, output_weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return {"loss": loss, "logits": logits}

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        encoder_mode: str = "normal",
    ):
        memory = self.encode(input_ids)
        if encoder_mode == "zero":
            memory = torch.zeros_like(memory)
        elif encoder_mode == "shuffle":
            memory = memory[torch.randperm(memory.shape[0], device=memory.device)]
        elif encoder_mode != "normal":
            raise ValueError(f"unknown encoder_mode: {encoder_mode}")
        hidden = self.decode(decoder_input_ids, memory)
        logits = F.linear(hidden, self.output_weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return {"loss": loss, "logits": logits}


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
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"momentum must lie in [0, 1), got {beta}")
            if lr < 0.0 or weight_decay < 0.0:
                raise ValueError("learning rate and weight decay must be nonnegative")

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
                elif role == "aux":
                    if weight_decay:
                        param.mul_(1.0 - lr * weight_decay)
                    update = adamw_aux_update(
                        param.grad,
                        state,
                        beta1=float(group.get("beta1", 0.9)),
                        beta2=float(group.get("beta2", 0.95)),
                        eps=float(group.get("eps", 1e-10)),
                    )
                    param.add_(update, alpha=-lr)
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
                    raise ValueError(f"unknown AF-Muon role: {role}")

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


def partition(model: ThreeWaySharedVocabSeq2Seq) -> dict:
    vocab_param_ids = {id(p) for p in model.vocab_tensors().values()}
    tied = model.encoder_embed.weight if model.sharing == "full" else None
    matrix_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
            matrix_ids.add(id(module.weight))
        if isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                matrix_ids.add(id(module.in_proj_weight))
            if module.out_proj.weight is not None:
                matrix_ids.add(id(module.out_proj.weight))

    matrix, tied_params, vocab_aux, vector = [], [], [], []
    names = {"matrix": [], "tied": [], "vocab_aux": [], "vector": []}
    for name, p in unique_named_parameters(model):
        if p is tied:
            tied_params.append(p)
            names["tied"].append(name)
        elif id(p) in vocab_param_ids:
            vocab_aux.append(p)
            names["vocab_aux"].append(name)
        elif id(p) in matrix_ids:
            matrix.append(p)
            names["matrix"].append(name)
        elif p.ndim <= 1:
            vector.append(p)
            names["vector"].append(name)
        else:
            raise ValueError(f"unassigned parameter {name} shape={tuple(p.shape)}")

    if model.sharing == "full" and tied_params != [tied]:
        raise ValueError("shared vocabulary table must occur once after identity deduplication")
    if model.sharing == "untied" and tied_params:
        raise ValueError("untied sharing should not produce tied parameters")
    return {"matrix": matrix, "tied": tied_params, "vocab_aux": vocab_aux, "vector": vector, "names": names}


def build_optimizer(model: ThreeWaySharedVocabSeq2Seq, args: argparse.Namespace):
    parts = partition(model)
    if model.sharing != "full":
        raise ValueError("factorial tied-table ablations require --sharing full")

    arm_to_modes = {
        "adamw_adamw": ("adamw", "adamw"),
        "cap_adamw": ("cap", "adamw"),
        "adamw_rms": ("adamw", "rms"),
        "cap_rms": ("cap", "rms"),
        # Compatibility aliases for older T5 sentinel runs.
        "muon": ("adamw", "adamw"),
        "afmoun": ("cap", "rms"),
        "sign": ("sign", "rms"),
    }
    tied_mode, vector_mode = arm_to_modes[args.arm]
    tied_weight_decay = float(args.tied_weight_decay)
    vector_weight_decay = float(args.vector_weight_decay)

    groups = [
        {
            "params": parts["matrix"],
            "role": "matrix",
            "lr": float(args.muon_lr),
            "momentum": float(args.momentum),
            "weight_decay": float(args.matrix_weight_decay),
            "ns_steps": int(args.ns_steps),
        }
    ]
    if tied_mode == "adamw":
        groups.append(
            {
                "params": parts["tied"] + parts["vocab_aux"],
                "role": "aux",
                "lr": float(args.vector_lr),
                "weight_decay": tied_weight_decay,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-10,
            }
        )
    else:
        tied_cap = 1.0 if tied_mode == "sign" else float(args.tied_cap)
        tied_scale = 1.0 if tied_mode == "sign" else float(args.tied_scale)
        groups.append(
            {
                "params": parts["tied"],
                "role": "tied",
                "lr": float(args.muon_lr),
                "decay_lr": float(args.vector_lr),
                "momentum": float(args.momentum),
                "weight_decay": tied_weight_decay,
                "tied_cap": tied_cap,
                "tied_scale": tied_scale,
                "rho_hidden": float(args.rho_hidden),
                "rho_output": float(args.rho_output),
                "chunk_rows": int(args.chunk_rows),
                "bisection_steps": int(args.bisection_steps),
                "max_bracket_steps": int(args.max_bracket_steps),
            }
        )

    if vector_mode == "adamw":
        groups.append(
            {
                "params": parts["vector"],
                "role": "aux",
                "lr": float(args.vector_lr),
                "weight_decay": vector_weight_decay,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-10,
            }
        )
    else:
        groups.append(
            {
                "params": parts["vector"],
                "role": "vector",
                "lr": float(args.vector_lr),
                "momentum": float(args.momentum),
                "weight_decay": vector_weight_decay,
                "eps": 1e-8,
            }
        )

    opt = ExplicitAFMuonV2(groups)
    opt.afmoun_factorial_modes = {"tied": tied_mode, "vector": vector_mode}
    opt.afmoun_role_names = parts["names"]
    return opt, parts


def assert_three_way_tie(model: ThreeWaySharedVocabSeq2Seq) -> None:
    enc = model.encoder_embed.weight
    dec = model.get_input_embeddings().weight
    out = model.output_weight
    if not (enc is dec and dec is out):
        raise AssertionError("encoder, decoder, and output table are not the same Python parameter")
    ptrs = {enc.data_ptr(), dec.data_ptr(), out.data_ptr()}
    ids = {id(enc), id(dec), id(out)}
    if len(ptrs) != 1 or len(ids) != 1:
        raise AssertionError("three-way shared vocabulary table identity/data_ptr check failed")


def assert_fully_untied(model: ThreeWaySharedVocabSeq2Seq) -> None:
    tensors = model.vocab_tensors()
    ids = {id(t) for t in tensors.values()}
    ptrs = {int(t.data_ptr()) for t in tensors.values()}
    if len(ids) != 3 or len(ptrs) != 3:
        raise AssertionError("fully untied topology expects three distinct vocab tensors")


def split_batch(batch: torch.Tensor, *, source_len: int, target_len: int, bos_token_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = batch[:, :source_len]
    target = batch[:, source_len : source_len + target_len]
    bos = torch.full((batch.shape[0], 1), int(bos_token_id), dtype=batch.dtype, device=batch.device)
    decoder_input = torch.cat([bos, target[:, :-1]], dim=1)
    return source, decoder_input, target


@torch.no_grad()
def evaluate(model, loader, args, *, device: torch.device, encoder_mode: str = "normal") -> tuple[float, float, int]:
    model.eval()
    losses = []
    tokens_seen = 0
    max_batches = max(1, math.ceil(int(args.eval_tokens) / (int(args.micro_batch_size) * int(args.target_len))))
    autocast_enabled = bool(args.bf16 and device.type == "cuda")
    for i, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        source, decoder_input, labels = split_batch(
            batch, source_len=args.source_len, target_len=args.target_len, bos_token_id=args.bos_token_id
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            out = model(source, decoder_input, labels, encoder_mode=encoder_mode)
        losses.append(float(out["loss"].detach().cpu()))
        tokens_seen += int(labels.numel())
        if i + 1 >= max_batches:
            break
    model.train()
    loss = float(np.mean(losses)) if losses else float("nan")
    return loss, math.exp(min(20.0, loss)), tokens_seen


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(row, flush=True)


def atomic_torch_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def optimizer_state_dtypes(opt) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                out.setdefault(key, set()).add(str(value.dtype))
    return {key: sorted(values) for key, values in out.items()}


def validate_token_range(path: Path, *, dtype: str, needed_tokens: int, vocab_size: int, label: str) -> dict:
    arr = np.memmap(path, dtype=np.uint32 if dtype == "uint32" else np.uint16, mode="r")
    n = min(int(needed_tokens), int(arr.shape[0]))
    if n <= 0:
        raise ValueError(f"{label} token file is empty")
    chunk = np.asarray(arr[:n])
    mn = int(chunk.min())
    mx = int(chunk.max())
    if mn < 0 or mx >= int(vocab_size):
        raise ValueError(f"{label} token ids out of range for vocab_size={vocab_size}: min={mn} max={mx}")
    return {f"{label}_checked_tokens": n, f"{label}_min_token_id": mn, f"{label}_max_token_id": mx}


def role_gradient_probe(model, batch: torch.Tensor, args, *, device: torch.device) -> dict:
    if model.sharing != "full":
        return {}
    model.zero_grad(set_to_none=True)
    batch = batch.to(device, non_blocking=True)
    source, decoder_input, labels = split_batch(
        batch, source_len=args.source_len, target_len=args.target_len, bos_token_id=args.bos_token_id
    )
    autocast_enabled = bool(args.bf16 and device.type == "cuda")

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        shared_out = model(source, decoder_input, labels)
    shared_out["loss"].backward()
    shared_grad = model.encoder_embed.weight.grad.detach().float().clone()
    model.zero_grad(set_to_none=True)

    shared = model.encoder_embed.weight.detach()
    e_enc = shared.detach().clone().requires_grad_(True)
    e_dec = shared.detach().clone().requires_grad_(True)
    e_out = shared.detach().clone().requires_grad_(True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        split_out = model.forward_with_embedding_weights(
            source,
            decoder_input,
            encoder_weight=e_enc,
            decoder_weight=e_dec,
            output_weight=e_out,
            labels=labels,
        )
    split_out["loss"].backward()
    roles = {
        "enc_lookup_grad": e_enc.grad.detach().float().clone(),
        "dec_lookup_grad": e_dec.grad.detach().float().clone(),
        "out_head_grad": e_out.grad.detach().float().clone(),
    }
    model.zero_grad(set_to_none=True)

    row_counts = {}
    norms = {}
    for name, grad in roles.items():
        row_nonzero = grad.abs().sum(dim=1) > 0
        row_counts[f"{name}_nonzero_row_frac"] = float(row_nonzero.float().mean().item())
        norms[f"{name}_fro"] = float(grad.norm().item())

    def cosine(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
        if mask is not None:
            if not bool(mask.any().item()):
                return float("nan")
            a = a[mask]
            b = b[mask]
        denom = float(a.norm().item() * b.norm().item())
        if denom == 0.0:
            return float("nan")
        return float((a.flatten() @ b.flatten()).item() / denom)

    enc_rows = roles["enc_lookup_grad"].abs().sum(dim=1) > 0
    dec_rows = roles["dec_lookup_grad"].abs().sum(dim=1) > 0
    union_rows = enc_rows | dec_rows
    intersect_rows = enc_rows & dec_rows
    split_sum = roles["enc_lookup_grad"] + roles["dec_lookup_grad"] + roles["out_head_grad"]
    diff = shared_grad - split_sum
    rel = diff.norm() / shared_grad.norm().clamp_min(1e-30)
    return {
        **row_counts,
        **norms,
        "shared_grad_fro": float(shared_grad.norm().item()),
        "split_grad_sum_fro": float(split_sum.norm().item()),
        "shared_split_grad_abs_error": float(diff.norm().item()),
        "shared_split_grad_rel_error": float(rel.item()),
        "enc_dec_support_intersection_frac": float(intersect_rows.float().mean().item()),
        "enc_dec_support_union_frac": float(union_rows.float().mean().item()),
        "enc_dec_grad_cos_global": cosine(roles["enc_lookup_grad"], roles["dec_lookup_grad"]),
        "enc_out_grad_cos_active_rows": cosine(roles["enc_lookup_grad"], roles["out_head_grad"], enc_rows),
        "dec_out_grad_cos_active_rows": cosine(roles["dec_lookup_grad"], roles["out_head_grad"], dec_rows),
        "enc_dec_grad_cos_intersection": cosine(roles["enc_lookup_grad"], roles["dec_lookup_grad"], intersect_rows),
        "enc_dec_grad_cos_union": cosine(roles["enc_lookup_grad"], roles["dec_lookup_grad"], union_rows),
    }


def vocab_stats(model: ThreeWaySharedVocabSeq2Seq) -> dict:
    out = {}
    for name, tensor in model.vocab_tensors().items():
        t = tensor.detach().float()
        out[f"{name}_rms"] = float(t.square().mean().sqrt().item())
        out[f"{name}_max_abs"] = float(t.abs().max().item())
    if model.sharing == "full":
        out["shared_rms"] = out["encoder_embed_rms"]
        out["shared_max_abs"] = out["encoder_embed_max_abs"]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Three-way shared vocabulary encoder-decoder sentinel.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--arm",
        choices=["adamw_adamw", "cap_adamw", "adamw_rms", "cap_rms", "muon", "sign", "afmoun"],
        default="cap_rms",
    )
    p.add_argument("--sharing", choices=["full", "untied"], default="full")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--train-tokens", type=int, default=100_000_000)
    p.add_argument("--eval-tokens", type=int, default=1_000_000)
    p.add_argument("--vocab-size", type=int, default=0)
    p.add_argument("--dtype", default="")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--encoder-layers", type=int, default=4)
    p.add_argument("--decoder-layers", type=int, default=4)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--source-len", type=int, default=256)
    p.add_argument("--target-len", type=int, default=256)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--vector-lr", type=float, default=3e-4)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--matrix-weight-decay", type=float, default=0.1)
    p.add_argument("--hybrid-aux-weight-decay", type=float, default=0.01)
    p.add_argument("--sign-aux-weight-decay", type=float, default=0.0)
    p.add_argument("--afmoun-aux-weight-decay", type=float, default=0.0)
    p.add_argument("--tied-weight-decay", type=float, default=0.0)
    p.add_argument("--vector-weight-decay", type=float, default=0.0)
    p.add_argument("--rho-hidden", type=float, default=50.0)
    p.add_argument("--rho-output", type=float, default=3000.0)
    p.add_argument("--tied-cap", type=float, default=3.0)
    p.add_argument("--tied-scale", type=float, default=0.5)
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--ns-steps", type=int, default=5)
    p.add_argument("--bisection-steps", type=int, default=32)
    p.add_argument("--max-bracket-steps", type=int, default=128)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eval-every-steps", type=int, default=500)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--diag-every-steps", type=int, default=500)
    p.add_argument("--encoder-ablation-every-steps", type=int, default=0)
    p.add_argument("--checkpoint-every-steps", type=int, default=0)
    p.add_argument("--bos-token-id", type=int, default=0)
    p.add_argument("--bf16", action="store_true", help="Use BF16 autocast only; parameters/state remain FP32.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    dtype = args.dtype or metadata.get("dtype", "uint16")
    train_path = resolve_data_path(data_dir, metadata["train_path"])
    eval_path = resolve_data_path(data_dir, metadata["eval_path"])
    seq_len = int(args.source_len) + int(args.target_len)
    tokens_per_step = int(args.micro_batch_size) * int(args.gradient_accumulation_steps) * int(args.target_len)
    source_tokens_per_step = int(args.micro_batch_size) * int(args.gradient_accumulation_steps) * int(args.source_len)
    total_raw_tokens_per_step = tokens_per_step + source_tokens_per_step
    max_steps = max(1, int(math.ceil(int(args.train_tokens) / tokens_per_step)))
    train_examples = max_steps * int(args.micro_batch_size) * int(args.gradient_accumulation_steps)
    eval_examples = max(1, math.ceil(int(args.eval_tokens) / int(args.target_len)))
    vocab_size = int(args.vocab_size)
    if vocab_size <= 0:
        vocab_size = int(metadata.get("vocab_size") or metadata.get("tokenizer_vocab_size") or metadata.get("config_vocab_size") or 0)
    if vocab_size <= 0:
        sample = np.memmap(train_path, dtype=np.uint32 if dtype == "uint32" else np.uint16, mode="r")[: min(5_000_000, seq_len * train_examples)]
        vocab_size = int(np.max(sample)) + 1
    if not (0 <= int(args.bos_token_id) < vocab_size):
        raise ValueError(f"bos_token_id={args.bos_token_id} is outside vocab_size={vocab_size}")
    token_checks = {}
    token_checks |= validate_token_range(train_path, dtype=dtype, needed_tokens=train_examples * seq_len, vocab_size=vocab_size, label="train")
    token_checks |= validate_token_range(eval_path, dtype=dtype, needed_tokens=eval_examples * seq_len, vocab_size=vocab_size, label="eval")

    train_ds = Seq2SeqTokenDataset(train_path, seq_len=seq_len, num_examples=train_examples, dtype=dtype)
    eval_ds = Seq2SeqTokenDataset(eval_path, seq_len=seq_len, num_examples=eval_examples, dtype=dtype)
    gen = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size, shuffle=True, generator=gen, num_workers=0, pin_memory=torch.cuda.is_available())
    eval_loader = DataLoader(eval_ds, batch_size=args.micro_batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    diag_batch = next(iter(eval_loader))

    model = ThreeWaySharedVocabSeq2Seq(
        vocab_size=vocab_size,
        dim=args.dim,
        heads=args.heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        ff_mult=args.ff_mult,
        source_len=args.source_len,
        target_len=args.target_len,
        max_positions=max(args.source_len, args.target_len),
        dropout=0.0,
        sharing=args.sharing,
    )
    if args.sharing == "full":
        assert_three_way_tie(model)
    else:
        assert_fully_untied(model)
    model = model.float().to(device)
    model.train()
    if args.sharing == "full":
        assert_three_way_tie(model)
    else:
        assert_fully_untied(model)
    opt, parts = build_optimizer(model, args)
    if {str(p.dtype) for p in model.parameters()} != {"torch.float32"}:
        raise AssertionError("t5test expects FP32 parameters; BF16 is autocast-only")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    config = vars(args) | {
        "metadata": metadata,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "tokens_per_step": tokens_per_step,
        "target_tokens_per_step": tokens_per_step,
        "source_tokens_per_step": source_tokens_per_step,
        "total_raw_tokens_per_step": total_raw_tokens_per_step,
        "max_steps": max_steps,
        "target_train_tokens": int(args.train_tokens),
        "raw_source_plus_target_tokens": int(max_steps) * int(total_raw_tokens_per_step),
        "param_count": sum(p.numel() for p in model.parameters()),
        "sharing": args.sharing,
        "role_names": parts["names"],
        "weight_decay_protocol": {
            "matrix_weight_decay_all_arms": float(args.matrix_weight_decay),
            "hybrid_aux_weight_decay": float(args.hybrid_aux_weight_decay),
            "sign_tied_weight_decay": float(args.sign_aux_weight_decay),
            "sign_vector_weight_decay": float(args.sign_aux_weight_decay),
            "afmoun_tied_weight_decay": float(args.afmoun_aux_weight_decay),
            "afmoun_vector_weight_decay": float(args.afmoun_aux_weight_decay),
            "factorial_tied_weight_decay": float(args.tied_weight_decay),
            "factorial_vector_weight_decay": float(args.vector_weight_decay),
            "factorial_modes": getattr(opt, "afmoun_factorial_modes", {}),
            "tied_cap": float(args.tied_cap),
            "tied_scale": float(args.tied_scale),
            "tied_decay_clock": "vector_lr",
        },
        "token_checks": token_checks,
        "three_way_tie": {
            "enabled": args.sharing == "full",
            "same_python_id": args.sharing == "full",
            "data_ptr": int(model.encoder_embed.weight.data_ptr()),
            "parameter_id": id(model.encoder_embed.weight),
        },
        "objective": "encoder prefix to decoder continuation",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    start = time.time()
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(step: int, target_tokens_seen: int) -> None:
        payload = {
            "step": int(step),
            "target_tokens_seen": int(target_tokens_seen),
            "source_tokens_seen": int(step) * int(source_tokens_per_step),
            "total_raw_tokens_seen": int(step) * int(total_raw_tokens_per_step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "config": config,
        }
        atomic_torch_save(payload, checkpoint_dir / "latest_full.pt")
        if int(args.checkpoint_every_steps) > 0 and step > 0 and step % int(args.checkpoint_every_steps) == 0:
            atomic_torch_save(payload, checkpoint_dir / f"full_step{step:07d}_target_tokens{target_tokens_seen}.pt")

    ev_loss, ev_ppl, ev_tokens = evaluate(model, eval_loader, args, device=device)
    if not math.isfinite(ev_loss):
        raise RuntimeError(f"non-finite initialization eval loss: {ev_loss}")
    append_jsonl(
        metrics_path,
        {
            "phase": "eval",
            "step": 0,
            "tokens_seen": 0,
            "target_tokens_seen": 0,
            "source_tokens_seen": 0,
            "total_raw_tokens_seen": 0,
            "eval_loss": ev_loss,
            "eval_ppl": ev_ppl,
            "eval_tokens": ev_tokens,
            "seconds": time.time() - start,
        },
    )

    running_loss = 0.0
    running_count = 0
    tokens_seen = 0
    micro_step = 0
    global_step = 0
    train_iter = iter(train_loader)
    while global_step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = batch.to(device, non_blocking=True)
        source, decoder_input, labels = split_batch(batch, source_len=args.source_len, target_len=args.target_len, bos_token_id=args.bos_token_id)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bool(args.bf16 and device.type == "cuda")):
            out = model(source, decoder_input, labels)
            loss = out["loss"] / float(args.gradient_accumulation_steps)
        loss.backward()
        running_loss += float(loss.detach().cpu()) * float(args.gradient_accumulation_steps)
        running_count += 1
        micro_step += 1
        if micro_step % int(args.gradient_accumulation_steps) != 0:
            continue

        grad_norm = None
        if float(args.max_grad_norm) > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm)).detach().cpu())
        opt.step()
        opt.zero_grad(set_to_none=True)
        global_step += 1
        tokens_seen += tokens_per_step
        source_tokens_seen = global_step * source_tokens_per_step
        total_raw_tokens_seen = global_step * total_raw_tokens_per_step

        if global_step == 1:
            if not all(torch.isfinite(p.detach()).all().item() for p in model.parameters()):
                raise RuntimeError("non-finite parameter detected after step 1")
            param_dtypes = sorted({str(p.dtype) for p in model.parameters()})
            state_dtypes = optimizer_state_dtypes(opt)
        else:
            param_dtypes = None
            state_dtypes = None

        if global_step == 1 or global_step % int(args.log_every_steps) == 0:
            avg = running_loss / max(1, running_count)
            row = {
                "phase": "train",
                "step": global_step,
                "micro_step": micro_step,
                "tokens_seen": tokens_seen,
                "target_tokens_seen": tokens_seen,
                "source_tokens_seen": source_tokens_seen,
                "total_raw_tokens_seen": total_raw_tokens_seen,
                "loss": avg,
                "ppl": math.exp(min(20.0, avg)),
                "grad_norm_preclip": grad_norm,
                "seconds": time.time() - start,
            }
            if param_dtypes is not None:
                row["param_dtypes"] = param_dtypes
                row["optimizer_state_dtypes"] = state_dtypes
            append_jsonl(metrics_path, row)
            running_loss = 0.0
            running_count = 0

        if global_step == 1 or global_step % int(args.diag_every_steps) == 0:
            probe = role_gradient_probe(model, diag_batch, args, device=device)
            append_jsonl(
                metrics_path,
                {
                    "phase": "diagnostic",
                    "step": global_step,
                    "tokens_seen": tokens_seen,
                    "target_tokens_seen": tokens_seen,
                    "source_tokens_seen": source_tokens_seen,
                    "total_raw_tokens_seen": total_raw_tokens_seen,
                    **probe,
                    **vocab_stats(model),
                    "seconds": time.time() - start,
                },
            )

        if global_step % int(args.eval_every_steps) == 0 or global_step == max_steps:
            ev_loss, ev_ppl, ev_tokens = evaluate(model, eval_loader, args, device=device)
            row = {
                "phase": "eval",
                "step": global_step,
                "tokens_seen": tokens_seen,
                "target_tokens_seen": tokens_seen,
                "source_tokens_seen": source_tokens_seen,
                "total_raw_tokens_seen": total_raw_tokens_seen,
                "eval_loss": ev_loss,
                "eval_ppl": ev_ppl,
                "eval_tokens": ev_tokens,
                "seconds": time.time() - start,
            }
            if int(args.encoder_ablation_every_steps) > 0 and (
                global_step == 1
                or global_step % int(args.encoder_ablation_every_steps) == 0
                or global_step == max_steps
            ):
                zero_loss, zero_ppl, _ = evaluate(model, eval_loader, args, device=device, encoder_mode="zero")
                shuffle_loss, shuffle_ppl, _ = evaluate(model, eval_loader, args, device=device, encoder_mode="shuffle")
                row |= {
                    "zero_encoder_eval_loss": zero_loss,
                    "zero_encoder_eval_ppl": zero_ppl,
                    "shuffle_encoder_eval_loss": shuffle_loss,
                    "shuffle_encoder_eval_ppl": shuffle_ppl,
                    "zero_encoder_loss_delta": zero_loss - ev_loss,
                    "shuffle_encoder_loss_delta": shuffle_loss - ev_loss,
                }
            append_jsonl(metrics_path, row)
        if int(args.checkpoint_every_steps) > 0 and global_step % int(args.checkpoint_every_steps) == 0:
            save_checkpoint(global_step, tokens_seen)

    save_checkpoint(global_step, tokens_seen)


if __name__ == "__main__":
    main()
