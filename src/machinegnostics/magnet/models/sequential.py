"""
Magnet Sequential Model API
==========================

User Documentation
------------------
The `Sequential` class is the primary high-level API for building and training neural networks in Magnet.
It provides a familiar, Keras-like interface for stacking layers, running predictions, and fitting models with flexible loss and callback support.

**Key Features:**
- Stack layers in order for feedforward computation.
- `.fit()` method supports single or multiple loss functions, boolean verbosity, and callbacks (EarlyStopping, LRScheduler, etc.).
- `.predict()` and `.evaluate()` for inference and validation.
- Compatible with all Magnet layers, losses, and optimizers.

**Basic Usage:**
```python
from machinegnostics.magnet import Sequential, Dense, ReLU, MSELoss, Adam
model = Sequential([
    Dense(1, 32), ReLU(), Dense(32, 1)
])
opt = Adam(model.parameters(), lr=0.01)
loss = MSELoss()
model.fit(X_train, y_train, loss_function=loss, optimizer=opt, epochs=50, verbose=True)
```

**Advanced Usage:**
- Provide a list of loss functions for multi-task or auxiliary loss tracking.
- Pass a list of callbacks (e.g., EarlyStopping, LRScheduler, custom) to control training.

Developer Notes
---------------
- The Sequential class inherits from Layer and manages an ordered list of layers.
- The `.fit()` method supports both single and multiple loss functions, and calls callbacks at the end of each epoch.
- Callbacks must implement `on_epoch_end(history, epoch, model, ...)` if they wish to be triggered.
- Losses are always tracked as `loss_0`, `loss_1`, ... in the history dict for consistency.
- Designed for extensibility: new callbacks, loss types, or training hooks can be added with minimal changes.
- Verbosity is now a boolean for clarity and modern API style.

Author: Nirmal Parmar
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..data import batch_iterator
from ..layers import Layer
from ..tensor import Tensor
from .base_model import BaseModel


class Sequential(Layer, BaseModel):
    """
    Magnet Sequential Model
    ----------------------
    An ordered container for stacking neural network layers and managing training/inference.

    This class provides a high-level, Keras-like API for building, training, and evaluating neural networks in Magnet.

    Parameters
    ----------
    layers : iterable of Layer, optional
        List or iterable of Magnet Layer objects to initialize the model.

    Example
    -------
    ```python
    from machinegnostics.magnet import Sequential, Dense, ReLU, MSELoss, Adam

    model = Sequential([
        Dense(1, 32), ReLU(), Dense(32, 1)
    ])
    opt = Adam(model.parameters(), lr=0.01)
    loss = MSELoss()
    model.fit(X_train, y_train, loss_function=loss, optimizer=opt, epochs=50, verbose=True)
    ```
    """

    def __init__(self, layers: Optional[Iterable[Layer]] = None):
        super().__init__()
        self.layers: List[Layer] = list(layers) if layers is not None else []

    def add(self, layer: Layer) -> None:
        """
        Append a layer to the end of the model.

        Parameters
        ----------
        layer : Layer
            The Magnet Layer instance to add.
        """
        self.layers.append(layer)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all layers in the model.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor after passing through all layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x, batch_size: int = 128) -> np.ndarray:
        """
        Run forward-only inference and return NumPy predictions.

        Parameters
        ----------
        x : array-like
            Input data (NumPy array or compatible).
        batch_size : int, default=128
            Batch size for inference (for memory efficiency).

        Returns
        -------
        np.ndarray
            Model predictions as a NumPy array.
        """
        self.eval()
        outputs = []
        x_np = np.asarray(x)
        for xb in batch_iterator(x_np, batch_size=batch_size, shuffle=False):
            yb = self(Tensor(xb, requires_grad=False))
            outputs.append(yb.data)
        return np.concatenate(outputs, axis=0)

    def evaluate(self, x, y, loss_function, batch_size: int = 128) -> Dict[str, float]:
        """
        Evaluate the model on a dataset using a provided loss function.

        Parameters
        ----------
        x : array-like
            Input data.
        y : array-like
            Target data.
        loss_function : callable
            Loss function to compute.
        batch_size : int, default=128
            Batch size for evaluation.

        Returns
        -------
        dict
            Dictionary with average loss (key: 'loss').
        """
        self.eval()
        losses = []
        x_np = np.asarray(x)
        y_np = np.asarray(y)
        for xb, yb in batch_iterator(x_np, y_np, batch_size=batch_size, shuffle=False):
            preds = self(Tensor(xb, requires_grad=False))
            loss = loss_function(preds, Tensor(yb, requires_grad=False))
            losses.append(float(loss.data))
        return {"loss": float(np.mean(losses)) if losses else 0.0}

    def fit(
        self,
        x,
        y,
        loss_function,
        optimizer,
        epochs: int = 10,
        batch_size: int = 32,
        shuffle: bool = True,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
        callbacks: Optional[list] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model using mini-batch gradient descent.

        Parameters
        ----------
        x : array-like
            Training input data.
        y : array-like
            Training target data.
        loss_function : callable or list of callables
            One or more loss functions to compute per batch/epoch. If a list is provided, all losses are tracked.
        optimizer : Optimizer
            Magnet optimizer instance.
        epochs : int, default=10
            Number of training epochs.
        batch_size : int, default=32
            Mini-batch size.
        shuffle : bool, default=True
            Whether to shuffle the data each epoch.
        validation_data : tuple (x_val, y_val), optional
            Validation data for loss tracking.
        verbose : bool, default=True
            If True, prints progress each epoch.
        callbacks : list, optional
            List of callback objects (e.g., EarlyStopping, LRScheduler, custom). Each callback can implement `on_epoch_end(history, epoch, model, ...)`.

        Returns
        -------
        history : dict
            Dictionary of loss and val_loss histories for each loss function.

        Example
        -------
        >>> model.fit(X, y, loss_function=[MSELoss(), MAELoss()], optimizer=opt, epochs=10, callbacks=[EarlyStopping()])
        """
        x_np = np.asarray(x)
        y_np = np.asarray(y)

        # Normalize loss_function to a list
        if not isinstance(loss_function, (list, tuple)):
            loss_functions = [loss_function]
        else:
            loss_functions = list(loss_function)

        history = {f"loss_{i}": [] for i in range(len(loss_functions))}
        if validation_data is not None:
            for i in range(len(loss_functions)):
                history[f"val_loss_{i}"] = []

        if callbacks is None:
            callbacks = []

        for epoch in range(epochs):
            self.train()
            batch_losses = [[] for _ in loss_functions]

            for xb, yb in batch_iterator(x_np, y_np, batch_size=batch_size, shuffle=shuffle):
                preds = self(Tensor(xb, requires_grad=False))
                losses = [lf(preds, Tensor(yb, requires_grad=False)) for lf in loss_functions]

                optimizer.zero_grad()
                # Only the first loss is used for backward by default
                losses[0].backward()
                optimizer.step()

                for i, loss in enumerate(losses):
                    batch_losses[i].append(float(loss.data))

            for i, losses_i in enumerate(batch_losses):
                epoch_loss = float(np.mean(losses_i)) if losses_i else 0.0
                history[f"loss_{i}"].append(epoch_loss)

            if validation_data is not None:
                val_x, val_y = validation_data
                val_preds = self(Tensor(val_x, requires_grad=False))
                val_losses = [lf(val_preds, Tensor(val_y, requires_grad=False)) for lf in loss_functions]
                for i, vloss in enumerate(val_losses):
                    history[f"val_loss_{i}"].append(float(vloss.data))

            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs}"
                for i in range(len(loss_functions)):
                    msg += f" - loss_{i}: {history[f'loss_{i}'][-1]:.6f}"
                    if validation_data is not None:
                        msg += f" - val_loss_{i}: {history[f'val_loss_{i}'][-1]:.6f}"
                print(msg)

            # Callbacks (called at end of epoch)
            stop_training = False
            for cb in callbacks:
                if hasattr(cb, "on_epoch_end"):
                    cb.on_epoch_end(history=history, epoch=epoch, model=self)
                # If callback signals stop_training, break
                if hasattr(cb, "stop_training") and getattr(cb, "stop_training", False):
                    stop_training = True
            if stop_training:
                print(f"[Callback] Training stopped early at epoch {epoch+1}")
                break

        return history