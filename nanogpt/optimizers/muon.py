from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(grad: Tensor, steps: int = 5) -> Tensor:
    """Muon Newton-Schulz polar approximation used in the experiments."""
    assert grad.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad.bfloat16()
    transposed = grad.size(-2) > grad.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(int(steps)):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.mT
    return x


def muon_matrix_update(
    grad: Tensor,
    momentum: Tensor,
    *,
    beta: float = 0.95,
    ns_steps: int = 5,
) -> Tensor:
    """Muon hidden-matrix update: EMA momentum + Nesterov signal + NS map."""
    momentum.lerp_(grad, 1.0 - beta)
    signal = grad.lerp(momentum, beta)
    if signal.ndim == 4:
        signal = signal.reshape(signal.shape[0], -1)
    update = zeropower_via_newtonschulz5(signal, steps=ns_steps)
    update *= max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adamw_aux_update(
    grad: Tensor,
    state: dict,
    *,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-10,
) -> Tensor:
    """AdamW-style auxiliary direction matching the Hybrid Muon fallback."""
    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(grad)
        state["exp_avg_sq"] = torch.zeros_like(grad)
        state["step"] = 0
    state["step"] += 1
    state["exp_avg"].lerp_(grad, 1.0 - beta1)
    state["exp_avg_sq"].lerp_(grad.square(), 1.0 - beta2)
    m_hat = state["exp_avg"] / (1.0 - beta1 ** state["step"])
    v_hat = state["exp_avg_sq"] / (1.0 - beta2 ** state["step"])
    return m_hat / (v_hat.sqrt() + eps)


class HybridMuon(Optimizer):
    """Single-device Hybrid Muon: Muon hidden matrices + AdamW aux tensors."""

    def __init__(self, param_groups):
        defaults = {}
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            role = group["role"]
            lr = float(group["lr"])
            wd = float(group.get("weight_decay", 0.0))
            beta = float(group.get("momentum", 0.95))
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if role == "matrix":
                    buf = st.setdefault("momentum_buffer", torch.zeros_like(p))
                    update = muon_matrix_update(p.grad, buf, beta=beta, ns_steps=int(group.get("ns_steps", 5)))
                    p.mul_(1.0 - lr * wd)
                    p.add_(update.reshape_as(p), alpha=-lr)
                elif role == "aux":
                    update = adamw_aux_update(
                        p.grad,
                        st,
                        beta1=float(group.get("beta1", 0.9)),
                        beta2=float(group.get("beta2", 0.95)),
                        eps=float(group.get("eps", 1e-10)),
                    )
                    p.mul_(1.0 - lr * wd)
                    p.add_(update, alpha=-lr)
                else:
                    raise ValueError(f"unknown role: {role}")
        return loss


def unique_named_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    seen = set()
    out = []
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        out.append((name, p))
    return out


def build_hybrid_muon(
    model,
    *,
    muon_lr: float = 0.02,
    aux_lr: float = 3e-4,
    momentum: float = 0.95,
    matrix_weight_decay: float = 0.1,
    aux_weight_decay: float = 0.0,
    ns_steps: int = 5,
) -> HybridMuon:
    matrix, aux = [], []
    tied = None
    if hasattr(model, "get_input_embeddings") and hasattr(model, "get_output_embeddings"):
        inp = model.get_input_embeddings()
        out = model.get_output_embeddings()
        if inp is not None and out is not None and getattr(out, "weight", None) is inp.weight:
            tied = inp.weight
    for _, p in unique_named_parameters(model):
        if p is tied:
            aux.append(p)
        elif p.ndim >= 2:
            matrix.append(p)
        else:
            aux.append(p)
    groups = [
        {"params": matrix, "role": "matrix", "lr": muon_lr, "momentum": momentum, "weight_decay": matrix_weight_decay, "ns_steps": ns_steps},
        {"params": aux, "role": "aux", "lr": aux_lr, "weight_decay": aux_weight_decay, "beta1": 0.9, "beta2": 0.95, "eps": 1e-10},
    ]
    return HybridMuon(groups)
