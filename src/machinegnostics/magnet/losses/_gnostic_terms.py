"""Shared numerical helpers for stable gnostic losses."""

from __future__ import annotations

import torch

EPS = 1e-6
MAX_ABS_TWO_THETA = 30.0
MAX_MAGNITUDE = 1e6
MAX_GRADIENT = 1e6


def _clip_gradient(gradient: torch.Tensor) -> torch.Tensor:
    return torch.clamp(gradient, min=-MAX_GRADIENT, max=MAX_GRADIENT)


def compute_terms(diff: torch.Tensor, scale: float = 1.0) -> dict[str, torch.Tensor]:
    scale_value = max(abs(float(scale)), EPS)
    if diff.requires_grad:
        diff.register_hook(_clip_gradient)
    theta = diff / scale_value
    two_theta = torch.clamp(2.0 * theta, min=-MAX_ABS_TWO_THETA, max=MAX_ABS_TWO_THETA)
    fi = torch.clamp(1.0 / torch.cosh(two_theta), min=EPS, max=1.0)
    fj = torch.clamp(torch.cosh(two_theta), min=1.0, max=MAX_MAGNITUDE)
    hi = torch.tanh(two_theta)
    hj = torch.clamp(torch.sinh(two_theta), min=-MAX_MAGNITUDE, max=MAX_MAGNITUDE)
    p_i = torch.clamp((1.0 - hi) / 2.0, min=EPS, max=1.0 - EPS)
    return {
        "scale": torch.as_tensor(scale_value, device=diff.device, dtype=diff.dtype),
        "theta": theta,
        "two_theta": two_theta,
        "fi": fi,
        "fj": fj,
        "hi": hi,
        "hj": hj,
        "estimating_entropy": 1.0 - fi,
        "quantifying_entropy": torch.clamp(fj - 1.0, min=0.0, max=MAX_MAGNITUDE),
        "information": -(p_i * torch.log(p_i) + (1.0 - p_i) * torch.log(1.0 - p_i)),
    }
