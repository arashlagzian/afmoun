from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from .afmoun_v2 import (
    chunked_tied_clipped_row_update_v2,
    partition_afmoun_v2_parameters,
    vector_rms_update_v2,
)
from .muon import adamw_aux_update, muon_matrix_update


class NanoGPTFactorialMuon(Optimizer):
    """Muon matrices with independent tied-table and 1D auxiliary rules."""

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

            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]

                if role == "matrix":
                    buf = st.setdefault("momentum_buffer", torch.zeros_like(p))
                    update = muon_matrix_update(
                        p.grad,
                        buf,
                        beta=beta,
                        ns_steps=int(group.get("ns_steps", 5)),
                    )
                    p.mul_(1.0 - lr * wd)
                    p.add_(update.reshape_as(p), alpha=-lr)

                elif role == "tied_adamw":
                    update = adamw_aux_update(
                        p.grad,
                        st,
                        beta1=float(group.get("beta1", 0.9)),
                        beta2=float(group.get("beta2", 0.95)),
                        eps=float(group.get("eps", 1e-10)),
                    )
                    p.mul_(1.0 - lr * wd)
                    p.add_(update, alpha=-lr)

                elif role == "tied_afmoun":
                    if p.ndim != 2:
                        raise ValueError(f"tied_afmoun expects a 2D table, got {tuple(p.shape)}")
                    d_model = int(p.shape[1])
                    rho_hidden = float(group["rho_hidden"])
                    rho_output = float(group["rho_output"])
                    tied_scale = float(group.get("tied_scale", 0.5))
                    tied_lr = lr * rho_output / rho_hidden * tied_scale / float(d_model)
                    update = chunked_tied_clipped_row_update_v2(
                        p,
                        p.grad,
                        st,
                        beta=beta,
                        cap=float(group.get("tied_cap", 3.0)),
                        chunk_rows=int(group.get("chunk_rows", 2048)),
                        bisection_steps=int(group.get("bisection_steps", 32)),
                        max_bracket_steps=int(group.get("max_bracket_steps", 128)),
                    )
                    p.add_(update, alpha=-tied_lr)

                elif role == "vector_adamw":
                    update = adamw_aux_update(
                        p.grad,
                        st,
                        beta1=float(group.get("beta1", 0.9)),
                        beta2=float(group.get("beta2", 0.95)),
                        eps=float(group.get("eps", 1e-10)),
                    )
                    p.mul_(1.0 - lr * wd)
                    p.add_(update, alpha=-lr)

                elif role == "vector_rms":
                    if p.ndim > 1:
                        raise ValueError(f"vector_rms expects scalar/1D tensors, got {tuple(p.shape)}")
                    update = vector_rms_update_v2(
                        p,
                        p.grad,
                        st,
                        beta=beta,
                        eps=float(group.get("eps", 1e-8)),
                    )
                    p.add_(update, alpha=-lr)

                else:
                    raise ValueError(f"unknown factorial role: {role}")

        return loss


def build_nanogpt_factorial_muon(
    model,
    *,
    tied_rule: str,
    vector_rule: str,
    muon_lr: float = 0.02,
    vector_lr: float = 3e-4,
    momentum: float = 0.95,
    matrix_weight_decay: float = 0.1,
    tied_weight_decay: float = 0.0,
    vector_weight_decay: float = 0.0,
    tied_cap: float = 3.0,
    tied_scale: float = 0.5,
    rho_hidden: float = 50.0,
    rho_output: float = 3000.0,
    chunk_rows: int = 2048,
    ns_steps: int = 5,
    bisection_steps: int = 32,
    max_bracket_steps: int = 128,
) -> NanoGPTFactorialMuon:
    if tied_rule not in {"adamw", "afmoun"}:
        raise ValueError(f"unknown tied_rule: {tied_rule}")
    if vector_rule not in {"adamw", "rms"}:
        raise ValueError(f"unknown vector_rule: {vector_rule}")
    if float(tied_cap) < 1.0 or math.isnan(float(tied_cap)):
        raise ValueError("tied_cap must satisfy c >= 1")

    part = partition_afmoun_v2_parameters(model)
    tied_role = "tied_adamw" if tied_rule == "adamw" else "tied_afmoun"
    vector_role = "vector_adamw" if vector_rule == "adamw" else "vector_rms"

    groups = [
        {
            "params": part["matrix"],
            "role": "matrix",
            "lr": float(muon_lr),
            "momentum": float(momentum),
            "weight_decay": float(matrix_weight_decay),
            "ns_steps": int(ns_steps),
        },
        {
            "params": part["tied"],
            "role": tied_role,
            "lr": float(muon_lr if tied_rule == "afmoun" else vector_lr),
            "momentum": float(momentum),
            "weight_decay": float(tied_weight_decay),
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-10,
            "tied_cap": float(tied_cap),
            "tied_scale": float(tied_scale),
            "rho_hidden": float(rho_hidden),
            "rho_output": float(rho_output),
            "chunk_rows": int(chunk_rows),
            "bisection_steps": int(bisection_steps),
            "max_bracket_steps": int(max_bracket_steps),
        },
        {
            "params": part["vector"],
            "role": vector_role,
            "lr": float(vector_lr),
            "momentum": float(momentum),
            "weight_decay": float(vector_weight_decay),
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-8 if vector_rule == "rms" else 1e-10,
        },
    ]
    opt = NanoGPTFactorialMuon(groups)
    opt.factorial_role_names = part["names"]
    opt.factorial_rules = {"tied_rule": tied_rule, "vector_rule": vector_rule}
    return opt
