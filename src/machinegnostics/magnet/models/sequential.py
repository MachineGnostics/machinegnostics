"""Sequential model container and training helpers for Magnet.

Author: Nirmal Parmar

Notes:
- Designed to feel familiar to Keras users while staying lightweight.
- Supports layer stacking, training/eval mode switching, and mini-batch fit loops.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..data import batch_iterator
from ..layers import Layer
from ..tensor import Tensor


class Sequential(Layer):
    """A simple ordered container of layers.

    Parameters
    ----------
    layers:
        Optional iterable of layers to initialize the sequence.
    """

    def __init__(self, layers: Optional[Iterable[Layer]] = None):
        super().__init__()
        self.layers: List[Layer] = list(layers) if layers is not None else []

    def add(self, layer: Layer) -> None:
        """Append a layer to the model."""
        self.layers.append(layer)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x, batch_size: int = 128) -> np.ndarray:
        """Run forward-only inference and return NumPy predictions."""
        self.eval()
        outputs = []
        x_np = np.asarray(x)
        for xb in batch_iterator(x_np, batch_size=batch_size, shuffle=False):
            yb = self(Tensor(xb, requires_grad=False))
            outputs.append(yb.data)
        return np.concatenate(outputs, axis=0)

    def evaluate(self, x, y, loss_fn, batch_size: int = 128) -> Dict[str, float]:
        """Evaluate model with a provided loss function."""
        self.eval()
        losses = []
        x_np = np.asarray(x)
        y_np = np.asarray(y)
        for xb, yb in batch_iterator(x_np, y_np, batch_size=batch_size, shuffle=False):
            preds = self(Tensor(xb, requires_grad=False))
            loss = loss_fn(preds, Tensor(yb, requires_grad=False))
            losses.append(float(loss.data))
        return {"loss": float(np.mean(losses)) if losses else 0.0}

    def fit(
        self,
        x,
        y,
        loss_fn,
        optimizer,
        epochs: int = 10,
        batch_size: int = 32,
        shuffle: bool = True,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the model with mini-batch gradient descent.

        Parameters mirror common Keras conventions for easier adoption.
        """
        x_np = np.asarray(x)
        y_np = np.asarray(y)

        history = {"loss": []}
        if validation_data is not None:
            history["val_loss"] = []

        for epoch in range(epochs):
            self.train()
            batch_losses = []

            for xb, yb in batch_iterator(x_np, y_np, batch_size=batch_size, shuffle=shuffle):
                preds = self(Tensor(xb, requires_grad=False))
                loss = loss_fn(preds, Tensor(yb, requires_grad=False))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses.append(float(loss.data))

            epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            history["loss"].append(epoch_loss)

            if validation_data is not None:
                val_x, val_y = validation_data
                val_metrics = self.evaluate(val_x, val_y, loss_fn=loss_fn, batch_size=batch_size)
                history["val_loss"].append(val_metrics["loss"])

            if verbose:
                if validation_data is not None:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"loss: {history['loss'][-1]:.6f} - "
                        f"val_loss: {history['val_loss'][-1]:.6f}"
                    )
                else:
                    print(f"Epoch {epoch + 1}/{epochs} - loss: {history['loss'][-1]:.6f}")

        return history