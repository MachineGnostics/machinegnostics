"""Shared numerical helpers for stable gnostic losses."""

from __future__ import annotations

import numpy as np
import torch

EPS = 1e-6
MAX_ABS_TWO_THETA = 30.0
MAX_MAGNITUDE = 1e6
MAX_GRADIENT = 1e6
MAX_GNOSTIC_SCALE = 2.0
MIN_GNOSTIC_SCALE = 1e-2


def _clip_gradient(gradient: torch.Tensor) -> torch.Tensor:
    return torch.clamp(gradient, min=-MAX_GRADIENT, max=MAX_GRADIENT)


def compute_terms(diff: torch.Tensor, scale: float = 1.0) -> dict[str, torch.Tensor]:
    scale_value = max(abs(float(scale)), EPS)
    theta = diff / scale_value
    two_theta_raw = 2.0 * theta
    two_theta = torch.clamp(two_theta_raw, min=-MAX_ABS_TWO_THETA, max=MAX_ABS_TWO_THETA)
    clip_mask = (torch.abs(two_theta_raw) < MAX_ABS_TWO_THETA)

    fi_raw = 1.0 / torch.cosh(two_theta)
    fi = torch.clamp(fi_raw, min=EPS, max=1.0)

    fj_raw = torch.cosh(two_theta)
    fj = torch.clamp(fj_raw, min=1.0, max=MAX_MAGNITUDE)

    hi = torch.tanh(two_theta)

    hj_raw = torch.sinh(two_theta)
    hj = torch.clamp(hj_raw, min=-MAX_MAGNITUDE, max=MAX_MAGNITUDE)

    p_raw = (1.0 - hi) / 2.0
    p_i = torch.clamp(p_raw, min=EPS, max=1.0 - EPS)

    return {
        "scale": torch.as_tensor(scale_value, device=diff.device, dtype=diff.dtype),
        "theta": theta,
        "two_theta": two_theta,
        "clip_mask": clip_mask.to(dtype=diff.dtype),
        "fi": fi,
        "fi_active": (clip_mask & (fi_raw > EPS)).to(dtype=diff.dtype),
        "fj": fj,
        "fj_active": (clip_mask & (fj_raw < MAX_MAGNITUDE)).to(dtype=diff.dtype),
        "hi": hi,
        "hj": hj,
        "hj_active": (clip_mask & (torch.abs(hj_raw) < MAX_MAGNITUDE)).to(dtype=diff.dtype),
        "estimating_entropy": 1.0 - fi,
        "quantifying_entropy": torch.clamp(fj - 1.0, min=0.0, max=MAX_MAGNITUDE),
        "p_i": p_i,
        "p_active": (clip_mask & (p_raw > EPS) & (p_raw < 1.0 - EPS)).to(dtype=diff.dtype),
        "information": -(p_i * torch.log(p_i) + (1.0 - p_i) * torch.log(1.0 - p_i)),
    }


def resolve_gnostic_scale(diff: torch.Tensor, scale: float | str = "auto") -> float:
    """Resolve a user scale or derive one from the current batch."""
    if isinstance(scale, str):
        if scale != "auto":
            raise ValueError("scale must be a real number or the string 'auto'")
        from machinegnostics.magcal import ScaleParam

        preliminary = compute_terms(diff, scale=1.0)
        fi_mean = float(preliminary["fi"].mean().item())
        resolved = float(ScaleParam()._gscale_loc(fi_mean))
        if not np.isfinite(resolved):
            raise ValueError("auto scale resolution produced a non-finite value")
        return float(min(max(abs(resolved), MIN_GNOSTIC_SCALE), MAX_GNOSTIC_SCALE))

    resolved = float(scale)
    if not np.isfinite(resolved):
        raise ValueError("scale must be finite")
    if resolved <= 0.0 or resolved > MAX_GNOSTIC_SCALE:
        raise ValueError("scale must be in the range (0, 2]")
    return resolved


def compute_gnostic_terms(diff: torch.Tensor, scale: float | str = "auto") -> dict[str, torch.Tensor]:
    """Compute stable gnostic terms after resolving the effective scale."""
    return compute_terms(diff, scale=resolve_gnostic_scale(diff, scale))
