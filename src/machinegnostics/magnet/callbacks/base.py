"""Callback base class."""

from __future__ import annotations

from typing import Any, Dict, Optional


class Callback:
    """Base callback with TF-style hooks."""

    def set_model(self, model: Any) -> None:
        self.model = model

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        return None

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        return None

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        return None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        return None
