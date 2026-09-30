from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from .muon import muon_matrix_update, unique_named_parameters


def trust_cap_scale(param_norm: torch.Tensor, update_norm: torch.Tensor, lr: float, target_ratio: float, eps: float) -> torch.Tensor:
    if target_ratio <= 0:
        return torch.ones_like(update_norm)
    proposed = update_norm * float(lr)
    return (float(target_ratio) * param_norm / proposed.clamp_min(eps)).clamp(max=1.0)


@torch.no_grad()
def vector_rms_update(param: torch.Tensor, grad: torch.Tensor, state: dict, *, beta: float, lr: float, eps: float, target_ratio: float = 0.0) -> torch.Tensor:
    buf = state.setdefault("momentum_buffer", torch.zeros_like(param))
    buf.mul_(beta).add_(grad)
    u = buf.detach().float()
    rms = u.square().mean().sqrt().clamp_min(eps)
    update = (u / rms).to(dtype=param.dtype)
    scale = trust_cap_scale(param.detach().float().norm().clamp_min(eps), update.detach().float().norm().clamp_min(eps), lr, target_ratio, eps)
    return update * scale.to(dtype=param.dtype)


@torch.no_grad()
def chunked_tied_clipped_row_update(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: dict,
    *,
    beta: float,
    lr: float,
    cap: float,
    eps: float = 1e-8,
    target_ratio: float = 0.0,
    chunk_rows: int = 2048,
) -> torch.Tensor:
    """Legacy V1 tied-table clipped-row prototype.

    This historical implementation is kept only for provenance. It is not the
    support-aware V2 oracle used in the paper experiments. In particular, it
    applies a post-clipping row renormalization and therefore should not be used
    to reproduce the paper's finite-cap tied-table update.
    """
    if grad.ndim != 2:
        return vector_rms_update(param, grad, state, beta=beta, lr=lr, eps=eps, target_ratio=target_ratio)

    buf = state.setdefault("momentum_buffer", torch.zeros_like(param))
    buf.mul_(beta).add_(grad)

    rows, d_raw = int(param.shape[0]), int(param.shape[1])
    d = max(1, d_raw)
    row_norm_target = math.sqrt(float(d))
    if math.isinf(float(cap)):
        c = row_norm_target
    else:
        c = max(float(cap), 1e-12)

    update = torch.empty_like(param)
    target_sq = float(d)
    for start in range(0, rows, int(chunk_rows)):
        end = min(rows, start + int(chunk_rows))
        x = buf[start:end].detach().float()
        if c <= 1.0 + 1e-12:
            raw = x.sign()
        elif c >= row_norm_target - 1e-12:
            rms = x.square().mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
            raw = x / rms
        else:
            abs_x = x.abs()
            row_norm = x.norm(dim=1, keepdim=True).clamp_min(eps)
            lo = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
            hi = row_norm_target / row_norm
            cap_sq = c * c
            for _ in range(8):
                row_sq = torch.minimum((hi * abs_x).square(), torch.full_like(abs_x, cap_sq)).sum(dim=1, keepdim=True)
                need = row_sq < target_sq
                if not bool(need.any().item()):
                    break
                hi = torch.where(need, hi * 2.0, hi)
            for _ in range(24):
                mid = (lo + hi) * 0.5
                row_sq = torch.minimum((mid * abs_x).square(), torch.full_like(abs_x, cap_sq)).sum(dim=1, keepdim=True)
                lo = torch.where(row_sq < target_sq, mid, lo)
                hi = torch.where(row_sq >= target_sq, mid, hi)
            raw = x.sign() * torch.minimum(hi * abs_x, torch.full_like(abs_x, c))
        raw_norm = raw.norm(dim=1, keepdim=True).clamp_min(eps)
        update_f = raw * (row_norm_target / raw_norm)
        update[start:end].copy_(update_f.to(dtype=param.dtype))

    update_norm = update.detach().float().norm().clamp_min(eps)
    scale = trust_cap_scale(param.detach().float().norm().clamp_min(eps), update_norm, lr, target_ratio, eps)
    return update * scale.to(dtype=param.dtype)


class AFMuon(Optimizer):
    """Legacy AF-Muon V1 optimizer prototype, not used in the paper runs."""

    def __init__(self, param_groups):
        super().__init__(param_groups, {})

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
            wd = float(group.get("weight_decay", 0.0))
            eps = float(group.get("eps", 1e-8))
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if role == "matrix":
                    buf = st.setdefault("momentum_buffer", torch.zeros_like(p))
                    update = muon_matrix_update(p.grad, buf, beta=beta, ns_steps=int(group.get("ns_steps", 5)))
                    p.mul_(1.0 - lr * wd)
                    p.add_(update.reshape_as(p), alpha=-lr)
                elif role == "tied":
                    d_model = int(p.shape[1])
                    tie_lr = lr * float(group.get("rho_output", 3000.0)) / float(group.get("rho_hidden", 50.0))
                    tie_lr *= float(group.get("tied_scale", 0.5)) / max(1, d_model)
                    update = chunked_tied_clipped_row_update(
                        p,
                        p.grad,
                        st,
                        beta=beta,
                        lr=tie_lr,
                        cap=float(group.get("tied_cap", 3.0)),
                        eps=eps,
                        target_ratio=float(group.get("target_ratio", 0.0)),
                        chunk_rows=int(group.get("chunk_rows", 2048)),
                    )
                    p.add_(update, alpha=-tie_lr)
                elif role == "vector":
                    update = vector_rms_update(p, p.grad, st, beta=beta, lr=lr, eps=eps, target_ratio=float(group.get("target_ratio", 0.0)))
                    p.add_(update, alpha=-lr)
                else:
                    raise ValueError(f"unknown role: {role}")
        return loss


def _tied_weight(model):
    if not (hasattr(model, "get_input_embeddings") and hasattr(model, "get_output_embeddings")):
        return None
    inp = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if inp is None or out is None or getattr(out, "weight", None) is not inp.weight:
        return None
    return inp.weight


def build_afmoun(
    model,
    *,
    muon_lr: float = 0.02,
    vector_lr: float = 3e-4,
    momentum: float = 0.95,
    matrix_weight_decay: float = 0.1,
    tied_cap: float = 3.0,
    tied_scale: float = 0.5,
    rho_hidden: float = 50.0,
    rho_output: float = 3000.0,
    chunk_rows: int = 2048,
    ns_steps: int = 5,
) -> AFMuon:
    tied = _tied_weight(model)
    if tied is None:
        raise ValueError("AF-Muon release code expects a tied input embedding / LM-head weight.")

    matrix, tied_params, vector = [], [], []
    for name, p in unique_named_parameters(model):
        if p is tied:
            tied_params.append(p)
        elif p.ndim >= 2:
            matrix.append(p)
        elif p.ndim <= 1:
            vector.append(p)
        else:
            raise ValueError(f"unassigned parameter role for {name}: shape={tuple(p.shape)}")

    return AFMuon([
        {"params": matrix, "role": "matrix", "lr": muon_lr, "momentum": momentum, "weight_decay": matrix_weight_decay, "ns_steps": ns_steps},
        {"params": tied_params, "role": "tied", "lr": muon_lr, "momentum": momentum, "tied_cap": tied_cap, "tied_scale": tied_scale, "rho_hidden": rho_hidden, "rho_output": rho_output, "chunk_rows": chunk_rows},
        {"params": vector, "role": "vector", "lr": vector_lr, "momentum": momentum, "weight_decay": 0.0},
    ])


def build_scion_sign(model, **kwargs) -> AFMuon:
    """SCION-style tied Sign endpoint used in the finalist comparison."""
    kwargs = dict(kwargs)
    kwargs["tied_cap"] = 1.0
    kwargs["tied_scale"] = kwargs.get("tied_scale", 1.0)
    return build_afmoun(model, **kwargs)
