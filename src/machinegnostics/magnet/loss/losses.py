"""Loss functions for Magnet ANN workflows.

Author: Nirmal Parmar

Notes:
- Losses return a scalar `Tensor` compatible with `backward()`.
- `CrossEntropyLoss` expects logits and class-index targets.
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor


class Loss:
    """Base loss callable."""

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return self.forward(y_pred, y_true)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        raise NotImplementedError


class MSELoss(Loss):
    """Mean squared error loss."""

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred - y_true
        return (diff * diff).mean()


class MAELoss(Loss):
    """Mean absolute error loss."""

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred.data - y_true.data
        out_data = np.mean(np.abs(diff))
        out = Tensor(out_data, requires_grad=y_pred.requires_grad, device=y_pred.device, dtype=y_pred.data.dtype)
        out._prev = {y_pred}

        def _backward():
            if out.grad is None:
                return
            if y_pred.requires_grad:
                grad = np.sign(diff) / diff.size
                y_pred.grad += out.grad * grad

        out._backward_fn = _backward
        return out


class BCELoss(Loss):
    """Binary cross-entropy loss for probability inputs."""

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        probs = y_pred.clip(self.eps, 1.0 - self.eps)
        return -((y_true * probs.log()) + ((1.0 - y_true) * (1.0 - probs).log())).mean()


class CrossEntropyLoss(Loss):
    """Multi-class cross-entropy from logits and integer class targets."""

    def __init__(self, reduction: str = "mean"):
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be either 'mean' or 'sum'.")
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        logits_data = logits.data
        target_idx = targets.data.astype(np.int64).reshape(-1)

        shifted = logits_data - np.max(logits_data, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        n = logits_data.shape[0]
        losses = -np.log(probs[np.arange(n), target_idx] + 1e-12)
        out_value = losses.mean() if self.reduction == "mean" else losses.sum()

        out = Tensor(out_value, requires_grad=logits.requires_grad, device=logits.device, dtype=logits.data.dtype)
        out._prev = {logits}

        def _backward():
            if out.grad is None:
                return
            if logits.requires_grad:
                grad = probs.copy()
                grad[np.arange(n), target_idx] -= 1.0
                if self.reduction == "mean":
                    grad /= n
                logits.grad += out.grad * grad

        out._backward_fn = _backward
        return out
