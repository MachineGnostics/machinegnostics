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
            Training history containing recorded loss values.
        """
        return super().fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            shuffle=shuffle,
            callbacks=callbacks,
        )

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
