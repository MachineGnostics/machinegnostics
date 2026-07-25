"""Early stopping callback."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from .base import Callback


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving."""

    def __init__(self, monitor: str = "val_loss", patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best = None
        self.best_weights = None
        self.wait = 0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.best = None
        self.best_weights = None
        self.wait = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.best is None or current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = deepcopy(self.model.get_weights())
            return

        self.wait += 1
        if self.wait >= self.patience:
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)
