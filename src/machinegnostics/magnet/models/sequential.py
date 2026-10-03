"""Sequential model container for MAGNET.

``Sequential`` is the user-facing ordered model container. It inherits
the training, evaluation, and parameter-management behavior from
:class:`machinegnostics.magnet.models.base.Model` while presenting the
familiar stack-of-layers workflow common in neural-network libraries.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..core.callbacks import Callback
from ..core.history import History
from ..layers.base import Layer
from ..losses import Loss
from .base import Model


class _GnosticParameterTracker(Callback):
    """Collect epoch-wise ``S`` and ``z0`` values from gnostic layers."""

    def __init__(self) -> None:
        super().__init__(verbose=False, name="_GnosticParameterTracker")
        self.S_history: dict[str, dict[str, list[float]]] = {"raw": {}, "effective": {}}
        self.z0_history: dict[str, list[float]] = {}
        self._tracked_layers: list[tuple[str, Any]] = []

    def set_model(self, model) -> None:
        super().set_model(model)
        self._tracked_layers = []
        self.S_history = {"raw": {}, "effective": {}}
        self.z0_history = {}
        for index, layer in enumerate(model.layers):
            if not hasattr(layer, "S") or not hasattr(layer, "z0"):
                continue
            layer_key = f"{index}:{layer.name}"
            self._tracked_layers.append((layer_key, layer))
            self.S_history["raw"][layer_key] = []
            self.S_history["effective"][layer_key] = []
            self.z0_history[layer_key] = []

    def on_epoch_end(self, epoch, logs=None):
        del epoch, logs
        for layer_key, layer in self._tracked_layers:
            self.S_history["raw"][layer_key].append(float(np.asarray(layer.S.data).reshape(-1)[0]))
            self.S_history["effective"][layer_key].append(float(np.asarray(layer.effective_S.data).reshape(-1)[0]))
            self.z0_history[layer_key].append(float(np.asarray(layer.z0.data).reshape(-1)[0]))
        return None


class Sequential(Model):
    """Apply layers in the order they are provided.

    ``Sequential`` is the simplest MAGNET model container. Each layer's
    output becomes the next layer's input, which makes the class ideal
    for dense feed-forward networks, compact gnostic pipelines, and
    introductory examples.
    """

    def __init__(self, layers: Iterable[Layer] | None = None, verbose: bool = False):
        """Create a sequential model from an optional layer list."""
        super().__init__(layers=layers, verbose=verbose)

    def compile(self, loss: Loss | str | None = None, optimizer: Any | str | None = None) -> None:
        """Attach a loss function and optimizer to the sequential model."""
        return super().compile(loss=loss, optimizer=optimizer)

    def fit(
        self,
        x: Any,
        y: Any,
        epochs: int = 10,
        batch_size: int = 32,
        validation_data: tuple[Any, Any] | None = None,
        shuffle: bool = True,
        callbacks: Iterable[Callback] | None = None,
    ) -> History:
        """Train the sequential model on a dataset.

        Parameters
        ----------
        x : array-like
            Training inputs with shape ``(n_samples, n_features, ...)``.
        y : array-like
            Training targets aligned with ``x``.
        epochs : int, optional
            Number of full passes over the training set.
        batch_size : int, optional
            Number of samples per optimization step.
        validation_data : tuple of array-like, optional
            Optional ``(x_val, y_val)`` pair evaluated after each epoch.
        shuffle : bool, optional
            Whether to shuffle the training data before each epoch.
        callbacks : iterable of Callback, optional
            Callback hooks invoked during training.

        Returns
        -------
        History
            Training history containing recorded loss values. When centered
            gnostic activations are present, the returned history also includes
            ``history["S_history"]`` with raw/effective scale trajectories and
            ``history["z0_history"]`` with learned center trajectories.
        """
        callback_list = list(callbacks or [])
        tracker = _GnosticParameterTracker()
        callback_list.append(tracker)

        history = super().fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            shuffle=shuffle,
            callbacks=callback_list,
        )
        if tracker.S_history["raw"]:
            history["S_history"] = tracker.S_history
            history["z0_history"] = tracker.z0_history
        return history

    def predict(self, x: Any, batch_size: int | None = None) -> np.ndarray:
        """Run inference and return NumPy predictions.

        Parameters
        ----------
        x : array-like
            Input samples to evaluate.
        batch_size : int or None, optional
            Batch size used during inference. ``None`` processes the full
            array at once.

        Returns
        -------
        numpy.ndarray
            Predicted outputs for each input sample.
        """
        return super().predict(x, batch_size=batch_size)
