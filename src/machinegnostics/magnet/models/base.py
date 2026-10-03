"""Base model container for MAGNET.

Developer note
--------------
Author: Nirmal Parmar

``Model`` is the shared training container for MAGNET models. It owns the
layer stack, collects trainable parameters, and provides the common
compile/fit/evaluate/predict workflow used by ``Sequential`` and other future
model types.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from ..core.callbacks import Callback
from ..core.history import History
from ..core.tensor import Tensor
from ..layers.base import Layer
from ..losses import Loss, get_loss
from ..optimizers import get_optimizer
from ..utils.logging import get_logger


def _format_progress_bar(current: int, total: int, width: int = 20) -> str:
	"""Return an ASCII progress bar for training status output."""
	if total <= 0:
		return "[--------------------] 0%"
	current = min(max(current, 0), total)
	filled = int(round(width * current / total))
	filled = min(filled, width)
	bar = "#" * filled + "-" * (width - filled)
	percent = int(round(100 * current / total))
	return f"[{bar}] {percent:3d}%"


class Model:
	"""Base MAGNET model container.

	The model wires layers together, manages parameters, and runs training.
	Subclasses such as ``Sequential`` should inherit this behavior instead of
	re-implementing the core training flow unless they need custom orchestration.
	"""

	def __init__(self, layers: Iterable[Layer] | None = None, verbose: bool = False):
		"""Create a model from an optional list of layers.

		Parameters
		----------
		layers:
			Optional iterable of layers to seed the model with.
		verbose:
			If ``True``, enable debug-level logging for the model instance.
		"""
		self.layers: list[Layer] = list(layers or [])
		self.loss_fn: Loss | None = None
		self.optimizer: Any | None = None
		self._history = History()
		self.history = self._history
		self.stop_training = False
		self.verbose = verbose
		self.logger = get_logger(self.__class__.__name__, logging.INFO if verbose else logging.WARNING)
		if self.verbose:
			self.logger.info("%s initialized.", self.__class__.__name__)

	def __repr__(self) -> str:
		"""Return a concise debug representation for the model."""
		return f"<{self.__class__.__name__}: {len(self.layers)} layers>"

	@property
	def params(self) -> list[Tensor]:
		"""Return all trainable tensors exposed by the model."""
		parameters: list[Tensor] = []
		for layer in self.layers:
			if getattr(layer, "trainable", True):
				parameters.extend(list(layer.parameters()))
		if self.verbose:
			self.logger.debug("Collected %s trainable parameters.", len(parameters))
		return parameters

	def add(self, layer: Layer) -> None:
		"""Append a new layer to the model."""
		self.layers.append(layer)
		if self.verbose:
			self.logger.info("Added layer %s.", layer.__class__.__name__)

	def compile(self, loss: Loss | str | None = None, optimizer: Any | str | None = None) -> None:
		"""Attach a loss function and optimizer to the model.

		Parameters
		----------
		loss:
			Loss name, class, or instance resolved through ``get_loss``.
		optimizer:
			Optimizer name or instance resolved through ``get_optimizer``.
		"""
		self.loss_fn = get_loss(loss)
		self.optimizer = get_optimizer(optimizer)
		if self.verbose and self.loss_fn is not None and self.optimizer is not None:
			self.logger.info(
				"Compiled model with loss=%s and optimizer=%s.",
				self.loss_fn.__class__.__name__,
				self.optimizer.__class__.__name__,
			)

	def forward(self, x: Any, training: bool = True) -> Tensor:
		"""Run a forward pass through every layer in the model."""
		output = x if isinstance(x, Tensor) else Tensor(x)
		for layer in self.layers:
			output = layer(output, training=training)
		if self.verbose:
			self.logger.debug("Ran forward pass with output shape %s.", getattr(output, "shape", None))
		return output

	def __call__(self, x: Any, training: bool = True) -> Tensor:
		"""Alias for ``forward`` so models can be used like layers."""
		return self.forward(x, training=training)

	def predict(self, x: Any, batch_size: int | None = None) -> np.ndarray:
		"""Return model predictions as NumPy arrays."""
		array = np.asarray(x, dtype=np.float64)
		if self.verbose:
			self.logger.debug("Predict called with input shape %s and batch_size=%s.", array.shape, batch_size)
		if batch_size is None:
			return self.forward(array, training=False).data
		outputs = []
		for index in range(0, len(array), batch_size):
			outputs.append(self.forward(array[index : index + batch_size], training=False).data)
		return np.concatenate(outputs, axis=0)

	def evaluate(self, x: Any, y: Any, batch_size: int = 32) -> float:
		"""Evaluate the current model on a full dataset."""
		array_x = np.asarray(x, dtype=np.float64)
		array_y = np.asarray(y, dtype=np.float64)
		if self.verbose:
			self.logger.debug("Evaluate called with input shape %s and batch_size=%s.", array_x.shape, batch_size)
		total_loss = 0.0
		n_batches = 0
		for index in range(0, len(array_x), batch_size):
			xb, yb = array_x[index : index + batch_size], array_y[index : index + batch_size]
			y_pred = self.forward(xb, training=False)
			loss = self.loss_fn(y_pred, yb)
			total_loss += loss.data.item() if isinstance(loss, Tensor) else float(loss)
			n_batches += 1
		return total_loss / max(n_batches, 1)

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
		"""Train the model and return the recorded history."""
		array_x = np.asarray(x, dtype=np.float64)
		array_y = np.asarray(y, dtype=np.float64)
		callback_list = list(callbacks or [])
		if self.loss_fn is None or self.optimizer is None:
			raise RuntimeError("Model must be compiled before calling fit()")
		if self.verbose:
			self.logger.info(
				"Training for %s epochs on %s samples (batch_size=%s, validation=%s, shuffle=%s).",
				epochs,
				len(array_x),
				batch_size,
				validation_data is not None,
				shuffle,
			)
		self.stop_training = False
		self._history = History()
		self.history = self._history

		for callback in callback_list:
			if hasattr(callback, "set_model"):
				callback.set_model(self)
			if hasattr(callback, "on_train_begin"):
				callback.on_train_begin({})

		for epoch in range(epochs):
			epoch_start = perf_counter()
			for callback in callback_list:
				if hasattr(callback, "on_epoch_begin"):
					callback.on_epoch_begin(epoch, {})

			if shuffle:
				indices = np.random.permutation(len(array_x))
				array_x = array_x[indices]
				array_y = array_y[indices]

			epoch_loss = 0.0
			n_batches = 0
			for index in range(0, len(array_x), batch_size):
				xb, yb = array_x[index : index + batch_size], array_y[index : index + batch_size]
				y_pred = self.forward(xb, training=True)
				loss = self.loss_fn(y_pred, yb)
				if isinstance(loss, Tensor):
					loss.backward()
					self.optimizer.step(self.params)
					self.optimizer.zero_grad(self.params)
					batch_loss = loss.data.item()
				else:
					batch_loss = float(loss)
				epoch_loss += batch_loss
				n_batches += 1

			epoch_loss /= max(n_batches, 1)
			logs = {"loss": epoch_loss}
			self._history.record(logs)

			if validation_data is not None:
				val_x, val_y = validation_data
				val_loss = self.evaluate(val_x, val_y, batch_size=batch_size)
				self._history.record({"val_loss": val_loss})
				logs["val_loss"] = val_loss

			for layer in self.layers:
				if hasattr(layer, "sync_grads"):
					layer.sync_grads()

			for callback in callback_list:
				if hasattr(callback, "on_epoch_end"):
					callback.on_epoch_end(epoch, logs)

			stopped_now = self.stop_training
			if self.verbose:
				elapsed = perf_counter() - epoch_start
				progress = _format_progress_bar(epoch + 1, epochs)
				message = f"Epoch {epoch + 1}/{epochs} {progress} - {elapsed:.2f}s - loss: {epoch_loss:.4f}"
				if validation_data is not None:
					message += f" - val_loss: {logs['val_loss']:.4f}"
				if stopped_now:
					message += " - stopped early"
				self.logger.info(message)

			if self.stop_training:
				if self.verbose:
					stopper = next(
						(
							callback
							for callback in callback_list
							if hasattr(callback, "stopped_epoch") and getattr(callback, "stopped_epoch", None) is not None
						),
						None,
					)
					if stopper is not None:
						stop_message = f"Training stopped early at epoch {stopper.stopped_epoch}/{epochs}"
						if getattr(stopper, "best_epoch", None) is not None:
							stop_message += f"; best {stopper.monitor}={stopper.best:.4f} at epoch {stopper.best_epoch}"
						if getattr(stopper, "stopped_value", None) is not None:
							stop_message += f"; last {stopper.monitor}={stopper.stopped_value:.4f}"
						self.logger.info(stop_message)
					else:
						self.logger.info("Training stopped early at epoch %s/%s.", epoch + 1, epochs)
				break

		for callback in callback_list:
			if hasattr(callback, "on_train_end"):
				callback.on_train_end({"loss": self._history.get("loss", []), "val_loss": self._history.get("val_loss", [])})

		return self._history

	def get_weights(self) -> list[np.ndarray]:
		"""Return copies of the model parameters as NumPy arrays."""
		return [param.data.copy() for param in self.params]

	def set_weights(self, weights: Iterable[Any]) -> None:
		"""Load a list of NumPy arrays back into the model parameters."""
		for param, weight in zip(self.params, weights):
			param.data = np.asarray(weight, dtype=np.float64).copy()
		if self.verbose:
			self.logger.debug("Updated model weights from %s tensors.", len(self.params))

	def _infer_summary_input_shape(self) -> tuple[int, ...] | None:
		"""Infer a feature-shape for summary tracing."""
		if not self.layers:
			return None

		first_layer = self.layers[0]
		explicit_shape = getattr(first_layer, "input_shape", None)
		if explicit_shape is not None:
			explicit_shape = tuple(explicit_shape)
			if explicit_shape:
				if explicit_shape[0] is None:
					return tuple(int(dim) for dim in explicit_shape[1:])
				if len(explicit_shape) == 1:
					return (int(explicit_shape[0]),)
				return tuple(int(dim) for dim in explicit_shape[1:])

		params = getattr(first_layer, "params", {})
		for key in ("W", "weight", "kernel", "gamma", "beta"):
			param = params.get(key)
			if param is not None and hasattr(param, "shape") and param.shape:
				return (int(param.shape[0]),)

		for param in params.values():
			if hasattr(param, "shape") and param.shape:
				return (int(param.shape[0]),)

		for attr in ("in_features", "input_dim", "input_size", "n_in"):
			value = getattr(first_layer, attr, None)
			if value is not None:
				return (int(value),)

		return None

	@staticmethod
	def _format_summary_shape(shape: tuple[int, ...] | None, batch_size: int = 1) -> str:
		"""Format a traced tensor shape for the summary table."""
		if shape is None:
			return "?"
		if len(shape) == 0:
			return "()"
		return str(tuple(batch_size if dim is None else int(dim) for dim in shape))

	@staticmethod
	def _format_summary_feature_shape(shape: tuple[int, ...] | None) -> str:
		"""Format a summary shape without the batch axis."""
		if shape is None:
			return "?"
		if len(shape) == 0:
			return "()"
		feature_shape = tuple(int(dim) for dim in shape[1:]) if len(shape) > 1 else tuple(int(dim) for dim in shape)
		if len(feature_shape) == 0:
			return "()"
		if len(feature_shape) == 1:
			return str(feature_shape[0])
		return ", ".join(str(dim) for dim in feature_shape)

	def _infer_layer_summary_shape(self, layer: Layer, input_shape: tuple[int, ...] | None) -> tuple[int, ...] | None:
		"""Infer a layer's output shape without executing backend operations."""
		params = getattr(layer, "params", {})
		layer_name = layer.__class__.__name__

		if "W" in params and hasattr(params["W"], "shape") and len(params["W"].shape) >= 2:
			batch_dim = None if input_shape is None or len(input_shape) == 0 else input_shape[0]
			return (batch_dim, int(params["W"].shape[1]))

		if layer_name == "Flatten":
			if input_shape is None or len(input_shape) < 2:
				return None
			return (input_shape[0], int(np.prod(input_shape[1:])))

		if layer_name in {
			"ReLU",
			"Sigmoid",
			"Tanh",
			"Softmax",
			"Step",
			"LeakyReLU",
			"ELU",
			"Softplus",
			"Swish",
			"Square",
			"BatchNorm",
			"Fi",
			"Fj",
			"Hi",
			"Hj",
			"Ei",
		}:
			return input_shape

		if {"gamma", "beta"}.issubset(params.keys()):
			return input_shape

		return input_shape

	def summary(self, input_shape: tuple[int, ...] | None = None, batch_size: int = 1) -> None:
		"""Print a compact parameter summary for the model."""
		print(f"{'Layer':<20}{'Shape':<38}{'Param #':<10}")
		print("-" * 68)
		feature_shape = input_shape if input_shape is not None else self._infer_summary_input_shape()
		traced_input_shapes: list[tuple[int, ...] | None] = []
		traced_output_shapes: list[tuple[int, ...] | None] = []
		current_shape = (batch_size, *feature_shape) if feature_shape is not None else None
		for layer in self.layers:
			traced_input_shapes.append(current_shape)
			current_shape = self._infer_layer_summary_shape(layer, current_shape)
			traced_output_shapes.append(current_shape)

		total_params = 0
		for layer, input_shape_item, output_shape in zip(self.layers, traced_input_shapes, traced_output_shapes):
			n_params = sum(param.data.size for param in layer.parameters())
			total_params += n_params
			shape_text = (
				f"({self._format_summary_feature_shape(input_shape_item)}, {self._format_summary_feature_shape(output_shape)})"
			)
			print(f"{layer.name:<20}{shape_text:<38}{n_params:<10}")
		print("-" * 68)
		print(f"Total trainable params: {total_params}")
		if self.verbose:
			self.logger.info("Printed model summary with %s total parameters.", total_params)
