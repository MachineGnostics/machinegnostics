"""Model containers and training loop."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from .callbacks import Callback
from .history import History
from .losses import LossLike, get_loss
from .optimizers import Optimizer, get_optimizer
from .tensor import Tensor
from .layers import Layer


class Model:
    """Base ANN model with a Keras-style compile/fit/predict API."""

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__.lower()
        self.loss_fn = None
        self.optimizer: Optimizer | None = None
        self._history = History()
        self.stop_training = False

    @property
    def params(self) -> List[Tensor]:
        return []

    def compile(self, optimizer: Union[str, Optimizer, None] = None, loss: LossLike = "mse") -> None:
        """Attach optimizer and loss function before training."""

        self.optimizer = get_optimizer(optimizer)
        self.loss_fn = get_loss(loss)

    def forward(self, inputs: Tensor, training: bool = True) -> Tensor:
        raise NotImplementedError

    def _tensorize(self, x, requires_grad: bool = False) -> Tensor:
        return Tensor(np.asarray(x, dtype=np.float64), requires_grad=requires_grad)

    def get_weights(self):
        return [param.data.copy() for param in self.params]

    def set_weights(self, weights) -> None:
        for param, weight in zip(self.params, weights):
            param.data = np.asarray(weight, dtype=np.float64)

    def _prepare_callbacks(self, callbacks: Optional[Sequence[Callback]]) -> List[Callback]:
        callback_list = list(callbacks or [])
        for callback in callback_list:
            callback.set_model(self)
        return callback_list

    def fit(
        self,
        x,
        y,
        epochs: int = 1,
        batch_size: int = 32,
        validation_split: float = 0.0,
        validation_data=None,
        callbacks: Optional[Sequence[Callback]] = None,
        shuffle: bool = True,
        verbose: int = 1,
    ) -> History:
        if self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("Call compile() before fit().")

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if validation_data is not None:
            x_val, y_val = validation_data
            x_train, y_train = x, y
        elif validation_split > 0.0:
            split_index = int(len(x) * (1.0 - validation_split))
            x_train, x_val = x[:split_index], x[split_index:]
            y_train, y_val = y[:split_index], y[split_index:]
        else:
            x_train, y_train = x, y
            x_val = y_val = None

        callback_list = self._prepare_callbacks(callbacks)
        self.stop_training = False
        for callback in callback_list:
            callback.on_train_begin({})

        history = History()

        for epoch in range(epochs):
            for callback in callback_list:
                callback.on_epoch_begin(epoch, {})

            if shuffle:
                indices = np.random.permutation(len(x_train))
                x_train = x_train[indices]
                y_train = y_train[indices]

            epoch_losses = []
            for start in range(0, len(x_train), batch_size):
                end = start + batch_size
                batch_x = self._tensorize(x_train[start:end])
                batch_y = self._tensorize(y_train[start:end])

                predictions = self.forward(batch_x, training=True)
                loss = self.loss_fn(batch_y, predictions)
                epoch_losses.append(float(loss.data))

                loss.backward()
                self.optimizer.step(self.params)
                self.optimizer.zero_grad(self.params)

            logs = {"loss": float(np.mean(epoch_losses))}

            if x_val is not None and y_val is not None:
                val_predictions = self.predict(x_val)
                val_loss = self.loss_fn(self._tensorize(y_val), self._tensorize(val_predictions))
                logs["val_loss"] = float(val_loss.data)

            history.append(logs)
            self._history.append(logs)

            for callback in callback_list:
                callback.on_epoch_end(epoch, logs)

            if verbose:
                metrics_text = ", ".join(f"{key}: {value:.6f}" for key, value in logs.items())
                print(f"Epoch {epoch + 1}/{epochs} - {metrics_text}")

            if self.stop_training:
                break

        for callback in callback_list:
            callback.on_train_end(self._history.as_dict())

        return history

    def predict(self, x):
        inputs = self._tensorize(x)
        outputs = self.forward(inputs, training=False)
        return outputs.data


class Sequential(Model):
    """A simple stack of layers for ANN workflows."""

    def __init__(self, layers: Optional[Sequence[Layer]] = None, name: str | None = None):
        super().__init__(name=name)
        self.layers: List[Layer] = list(layers or [])

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    @property
    def params(self) -> List[Tensor]:
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.params)
        return params

    def forward(self, inputs: Tensor, training: bool = True) -> Tensor:
        output = inputs
        for layer in self.layers:
            output = layer(output, training=training)
        return output

    def summary(self) -> None:
        """Print a small human-readable model summary."""

        print(f"Model: {self.name}")
        for index, layer in enumerate(self.layers, start=1):
            print(f"  {index}. {layer.__class__.__name__} name={layer.name} trainable={layer.trainable}")
