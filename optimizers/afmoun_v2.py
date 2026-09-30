from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from .muon import muon_matrix_update, unique_named_parameters


__all__ = [
    "AFMuonV2",
    "build_afmoun_v2",
    "build_scion_sign_v2",
    "chunked_tied_clipped_row_update_v2",
    "partition_afmoun_v2_parameters",
    "vector_rms_update_v2",
]


def _validate_momentum(beta: float) -> None:
    if not 0.0 <= float(beta) < 1.0:
        raise ValueError(f"momentum must lie in [0, 1), got {beta}")


@torch.no_grad()
def vector_rms_update_v2(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: dict,
    *,
    beta: float,
    eps: float,
) -> torch.Tensor:
    """Unnormalized first momentum followed by the stabilized RMS oracle."""
    _validate_momentum(beta)
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if param.shape != grad.shape:
        raise ValueError(
            f"parameter/gradient shape mismatch: {tuple(param.shape)} vs {tuple(grad.shape)}"
        )
    if param.numel() == 0:
        raise ValueError("AF-Muon V2 does not support empty vector parameters")

    buf = state.setdefault("momentum_buffer", torch.zeros_like(param))
    buf.mul_(beta).add_(grad)

    direction = buf.detach().float()
    rms = direction.square().mean().sqrt().clamp_min(float(eps))
    return (direction / rms).to(dtype=param.dtype)


def _row_rms_oracle_fp32(x: torch.Tensor) -> torch.Tensor:
    """Exact row-RMS direction for nonzero fp32 rows, with zero rows mapped to zero."""
    max_abs = x.abs().amax(dim=1, keepdim=True)
    nonzero = max_abs > 0
    safe_max = torch.where(nonzero, max_abs, torch.ones_like(max_abs))
    scaled = x / safe_max
    scaled_rms = scaled.square().mean(dim=1, keepdim=True).sqrt()
    safe_rms = torch.where(nonzero, scaled_rms, torch.ones_like(scaled_rms))
    return torch.where(nonzero, scaled / safe_rms, torch.zeros_like(scaled))


def _capped_row_sq(q: torch.Tensor, tau: torch.Tensor, cap: float) -> torch.Tensor:
    capped = torch.minimum(tau * q, torch.full_like(q, float(cap)))
    return capped.square().sum(dim=1, keepdim=True)


def _finite_cap_oracle_fp32(
    x: torch.Tensor,
    *,
    cap: float,
    bisection_steps: int,
    max_bracket_steps: int,
) -> torch.Tensor:
    """Support-aware finite-cap row oracle on an fp32 row chunk.

    For each row b, this approximates

        sign(b_j) min(cap, tau |b_j|)

    with squared row target min(d, cap**2 * |supp(b)|). Rows whose support
    cannot fill the row-l2 radius saturate every useful coordinate at the cap.
    Radius-limited rows use a dynamically bracketed scalar bisection. The
    returned bisection iterate is the feasible lower endpoint, so finite solver
    tolerance cannot violate either the row radius or the coordinate cap.
    """
    rows, d = int(x.shape[0]), int(x.shape[1])
    if rows == 0:
        return torch.empty_like(x)

    abs_x = x.abs()
    signs = x.sign()

    # The oracle is invariant to positive row scaling. Normalizing by the
    # maximum magnitude avoids overflow/underflow in the root solve.
    max_abs = abs_x.amax(dim=1, keepdim=True)
    nonzero_rows = max_abs > 0
    safe_max = torch.where(nonzero_rows, max_abs, torch.ones_like(max_abs))
    q = abs_x / safe_max
    support = q > 0
    support_size = support.sum(dim=1, keepdim=True)

    cap_sq = float(cap) * float(cap)
    support_capacity = support_size.to(dtype=x.dtype) * cap_sq
    support_limited = (support_capacity <= float(d)).squeeze(1)
    radius_limited = ~support_limited

    result = torch.zeros_like(x)

    # This includes zero rows. Since sign(0) = 0, they remain exactly zero.
    if bool(support_limited.any().item()):
        result[support_limited] = signs[support_limited] * float(cap)

    if not bool(radius_limited.any().item()):
        return result

    q_radius = q[radius_limited]
    sign_radius = signs[radius_limited]
    row_norm_target = math.sqrt(float(d))

    # The uncapped row-RMS scale is a lower bracket: clipping can only reduce
    # its squared norm from d. Expansion continues until every row is bracketed
    # or fails explicitly rather than silently returning an infeasible update.
    q_norm = q_radius.square().sum(dim=1, keepdim=True).sqrt()
    lo = torch.zeros_like(q_norm)
    hi = row_norm_target / q_norm
    target_sq = float(d)

    row_sq = _capped_row_sq(q_radius, hi, cap)
    need_expansion = row_sq < target_sq
    for _ in range(int(max_bracket_steps)):
        if not bool(need_expansion.any().item()):
            break
        lo = torch.where(need_expansion, hi, lo)
        candidate = hi * 2.0
        if not bool(torch.isfinite(candidate[need_expansion]).all().item()):
            raise RuntimeError(
                "clipped-row LMO bracket overflowed; the momentum row has "
                "unsupported floating-point dynamic range"
            )
        hi = torch.where(need_expansion, candidate, hi)
        row_sq = _capped_row_sq(q_radius, hi, cap)
        need_expansion = row_sq < target_sq

    if bool(need_expansion.any().item()):
        count = int(need_expansion.sum().item())
        raise RuntimeError(
            "failed to bracket the clipped-row LMO root for "
            f"{count} row(s) after {max_bracket_steps} expansions"
        )

    for _ in range(int(bisection_steps)):
        mid = (lo + hi) * 0.5
        row_sq = _capped_row_sq(q_radius, mid, cap)
        below_target = row_sq < target_sq
        lo = torch.where(below_target, mid, lo)
        hi = torch.where(below_target, hi, mid)

    # lo is feasible by construction. It approaches the exact root from below
    # and preserves the KKT form without a post-clipping renormalization.
    magnitude = torch.minimum(lo * q_radius, torch.full_like(q_radius, float(cap)))
    result[radius_limited] = sign_radius * magnitude
    return result


@torch.no_grad()
def chunked_tied_clipped_row_update_v2(
    param: torch.Tensor,
    grad: torch.Tensor,
    state: dict,
    *,
    beta: float,
    cap: float,
    chunk_rows: int = 2048,
    bisection_steps: int = 32,
    max_bracket_steps: int = 128,
) -> torch.Tensor:
    """Support-aware, complete-row chunked tied-table clipped LMO.

    The persistent state is one parameter-dtype first-moment buffer. Solver
    intermediates are chunked by complete rows in fp32; the returned update is a
    full parameter-dtype tensor.
    """
    _validate_momentum(beta)
    if param.ndim != 2 or grad.ndim != 2:
        raise ValueError(
            "the AF-Muon V2 tied-table oracle requires 2D parameter and gradient tensors"
        )
    if param.shape != grad.shape:
        raise ValueError(
            f"parameter/gradient shape mismatch: {tuple(param.shape)} vs {tuple(grad.shape)}"
        )
    if int(param.shape[1]) <= 0:
        raise ValueError("the tied table must have a positive model width")
    if int(chunk_rows) <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    if int(bisection_steps) <= 0:
        raise ValueError(f"bisection_steps must be positive, got {bisection_steps}")
    if int(max_bracket_steps) <= 0:
        raise ValueError(
            f"max_bracket_steps must be positive, got {max_bracket_steps}"
        )

    cap = float(cap)
    if math.isnan(cap) or cap < 1.0:
        raise ValueError(f"cap must satisfy c >= 1 or c = inf, got {cap}")

    buf = state.setdefault("momentum_buffer", torch.zeros_like(param))
    buf.mul_(beta).add_(grad)

    rows, d = int(param.shape[0]), int(param.shape[1])
    row_norm_target = math.sqrt(float(d))
    update = torch.empty_like(param)

    for start in range(0, rows, int(chunk_rows)):
        end = min(rows, start + int(chunk_rows))
        x = buf[start:end].detach().float()

        if cap == 1.0:
            direction = x.sign()
        elif math.isinf(cap) or cap >= row_norm_target:
            direction = _row_rms_oracle_fp32(x)
        else:
            direction = _finite_cap_oracle_fp32(
                x,
                cap=cap,
                bisection_steps=int(bisection_steps),
                max_bracket_steps=int(max_bracket_steps),
            )

        update[start:end].copy_(direction.to(dtype=param.dtype))

    return update


class AFMuonV2(Optimizer):
    """Support-aware AF-Muon V2 for tied-head RoPE decoder models."""

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
            weight_decay = float(group.get("weight_decay", 0.0))
            eps = float(group.get("eps", 1e-8))

            _validate_momentum(beta)
            if lr < 0.0:
                raise ValueError(f"learning rate must be nonnegative, got {lr}")
            if weight_decay < 0.0:
                raise ValueError(
                    f"weight decay must be nonnegative, got {weight_decay}"
                )

            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                if role == "matrix":
                    if param.ndim != 2:
                        raise ValueError(
                            "AF-Muon V2 hidden-matrix parameters must be 2D, "
                            f"got shape {tuple(param.shape)}"
                        )
                    momentum = state.setdefault(
                        "momentum_buffer", torch.zeros_like(param)
                    )
                    update = muon_matrix_update(
                        param.grad,
                        momentum,
                        beta=beta,
                        ns_steps=int(group.get("ns_steps", 5)),
                    )
                    param.mul_(1.0 - lr * weight_decay)
                    param.add_(update.reshape_as(param), alpha=-lr)

                elif role == "tied":
                    if param.ndim != 2:
                        raise ValueError(
                            "the tied token embedding / LM-head parameter must be 2D"
                        )
                    d_model = int(param.shape[1])
                    rho_hidden = float(group.get("rho_hidden", 50.0))
                    rho_output = float(group.get("rho_output", 3000.0))
                    tied_scale = float(group.get("tied_scale", 0.5))
                    if rho_hidden <= 0.0:
                        raise ValueError(
                            f"rho_hidden must be positive, got {rho_hidden}"
                        )
                    if rho_output < 0.0:
                        raise ValueError(
                            f"rho_output must be nonnegative, got {rho_output}"
                        )
                    if tied_scale < 0.0:
                        raise ValueError(
                            f"tied_scale must be nonnegative, got {tied_scale}"
                        )

                    tied_lr = (
                        lr
                        * rho_output
                        / rho_hidden
                        * tied_scale
                        / float(d_model)
                    )
                    update = chunked_tied_clipped_row_update_v2(
                        param,
                        param.grad,
                        state,
                        beta=beta,
                        cap=float(group.get("tied_cap", 3.0)),
                        chunk_rows=int(group.get("chunk_rows", 2048)),
                        bisection_steps=int(group.get("bisection_steps", 32)),
                        max_bracket_steps=int(
                            group.get("max_bracket_steps", 128)
                        ),
                    )
                    param.add_(update, alpha=-tied_lr)

                elif role == "vector":
                    if param.ndim > 1:
                        raise ValueError(
                            "AF-Muon V2 vector-class parameters must be scalar or 1D, "
                            f"got shape {tuple(param.shape)}"
                        )
                    update = vector_rms_update_v2(
                        param,
                        param.grad,
                        state,
                        beta=beta,
                        eps=eps,
                    )
                    param.add_(update, alpha=-lr)

                else:
                    raise ValueError(f"unknown AF-Muon V2 role: {role}")

        return loss


def _tied_weight(model):
    if not (
        hasattr(model, "get_input_embeddings")
        and hasattr(model, "get_output_embeddings")
    ):
        return None
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        return None
    if getattr(output_embeddings, "weight", None) is not input_embeddings.weight:
        return None
    return input_embeddings.weight


def partition_afmoun_v2_parameters(model):
    """Return the explicit tied-head RoPE parameter partition used by V2.

    Hidden matrices are weights owned by torch.nn.Linear modules. The shared
    input/output table is handled first. Scalar and 1D tensors use vector RMS.
    Any other learned tensor, including an additional embedding table, is
    rejected instead of being silently sent to Muon.
    """
    tied = _tied_weight(model)
    if tied is None:
        raise ValueError(
            "AF-Muon V2 expects the input embedding and LM head to share the "
            "same parameter object"
        )

    linear_weight_ids: set[int] = set()
    embedding_weight_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.weight is not None:
            linear_weight_ids.add(id(module.weight))
        if isinstance(module, torch.nn.Embedding) and module.weight is not None:
            embedding_weight_ids.add(id(module.weight))

    matrix: list[torch.nn.Parameter] = []
    tied_params: list[torch.nn.Parameter] = []
    vector: list[torch.nn.Parameter] = []
    names = {"matrix": [], "tied": [], "vector": []}

    for name, param in unique_named_parameters(model):
        if param is tied:
            tied_params.append(param)
            names["tied"].append(name)
        elif id(param) in linear_weight_ids:
            if param.ndim != 2:
                raise ValueError(
                    f"hidden Linear weight {name} must be 2D, got {tuple(param.shape)}"
                )
            matrix.append(param)
            names["matrix"].append(name)
        elif param.ndim <= 1:
            vector.append(param)
            names["vector"].append(name)
        else:
            kind = (
                "additional embedding table"
                if id(param) in embedding_weight_ids
                else "unrecognized learned tensor"
            )
            raise ValueError(
                f"AF-Muon V2 does not assign {kind} {name} with shape "
                f"{tuple(param.shape)}; provide an explicit role or use a "
                "supported tied-head RoPE architecture"
            )

    if len(tied_params) != 1:
        raise ValueError(
            "the tied input embedding / LM-head parameter must occur exactly "
            f"once after identity deduplication, found {len(tied_params)}"
        )

    return {
        "matrix": matrix,
        "tied": tied_params,
        "vector": vector,
        "names": names,
    }


def build_afmoun_v2(
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
    bisection_steps: int = 32,
    max_bracket_steps: int = 128,
    eps: float = 1e-8,
) -> AFMuonV2:
    """Build the support-aware AF-Muon V2 optimizer.

    Learning-rate schedules remain external, as in V1: update the matrix and
    tied group ``lr`` with eta_M,t and the vector group ``lr`` with eta_vec,t.
    """
    _validate_momentum(momentum)
    if float(muon_lr) < 0.0 or float(vector_lr) < 0.0:
        raise ValueError("learning rates must be nonnegative")
    if float(matrix_weight_decay) < 0.0:
        raise ValueError("matrix_weight_decay must be nonnegative")
    if float(rho_hidden) <= 0.0:
        raise ValueError("rho_hidden must be positive")
    if float(rho_output) < 0.0:
        raise ValueError("rho_output must be nonnegative")
    if float(tied_scale) < 0.0:
        raise ValueError("tied_scale must be nonnegative")
    if int(chunk_rows) <= 0:
        raise ValueError("chunk_rows must be positive")
    if int(ns_steps) <= 0:
        raise ValueError("ns_steps must be positive")
    if int(bisection_steps) <= 0:
        raise ValueError("bisection_steps must be positive")
    if int(max_bracket_steps) <= 0:
        raise ValueError("max_bracket_steps must be positive")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")
    tied_cap = float(tied_cap)
    if math.isnan(tied_cap) or tied_cap < 1.0:
        raise ValueError("tied_cap must satisfy c >= 1 or c = inf")

    partition = partition_afmoun_v2_parameters(model)
    optimizer = AFMuonV2(
        [
            {
                "params": partition["matrix"],
                "role": "matrix",
                "lr": float(muon_lr),
                "momentum": float(momentum),
                "weight_decay": float(matrix_weight_decay),
                "ns_steps": int(ns_steps),
            },
            {
                "params": partition["tied"],
                "role": "tied",
                "lr": float(muon_lr),
                "momentum": float(momentum),
                "tied_cap": tied_cap,
                "tied_scale": float(tied_scale),
                "rho_hidden": float(rho_hidden),
                "rho_output": float(rho_output),
                "chunk_rows": int(chunk_rows),
                "bisection_steps": int(bisection_steps),
                "max_bracket_steps": int(max_bracket_steps),
            },
            {
                "params": partition["vector"],
                "role": "vector",
                "lr": float(vector_lr),
                "momentum": float(momentum),
                "weight_decay": 0.0,
                "eps": float(eps),
            },
        ]
    )
    optimizer.afmoun_v2_role_names = partition["names"]
    return optimizer


def build_scion_sign_v2(model, **kwargs) -> AFMuonV2:
    """Build the support-aware Sign endpoint using the AF-Muon V2 framework."""
    kwargs = dict(kwargs)
    kwargs["tied_cap"] = 1.0
    kwargs["tied_scale"] = kwargs.get("tied_scale", 1.0)
    return build_afmoun_v2(model, **kwargs)
