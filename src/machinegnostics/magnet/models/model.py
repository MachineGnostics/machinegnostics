"""Model and Sequential containers for MAGNET.

Developer note
-------------
Author: Nirmal Parmar

This module implements the training loop for MAGNET (Machine Gnostics Neural
Networks). The API is intentionally close to common deep-learning libraries so
users can build small examples quickly.

Examples
--------
>>> import numpy as np
>>> from machinegnostics.magnet import Sequential, Dense, Sigmoid, MSE, Adam
>>> model = Sequential([Dense(2, 1), Sigmoid()])
>>> model.compile(loss=MSE(), optimizer=Adam(lr=0.01))
>>> X = np.array([[0., 0.], [1., 1.]])
>>> y = np.array([[0.], [1.]])
>>> history = model.fit(X, y, epochs=2, batch_size=2, verbose=False)
>>> list(history.keys())
['loss']
"""

from __future__ import annotations

import logging
from time import perf_counter

import numpy as np

from ..core.history import History
from ..losses import get_loss, Loss
from ..optimizers import get_optimizer
from ..core.tensor import Tensor
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

	def __init__(self, layers=None, verbose: bool = False):
		"""Create a model from an optional list of layers.

		Parameters
		----------
		layers:
			Optional iterable of layers to seed the model with.
		verbose:
			If ``True``, enable debug-level logging for the model instance.
		"""
		self.layers = list(layers or [])
		self.loss_fn: Loss | None = None
		self.optimizer = None
		self._history = History()
		self.history = self._history
		self.stop_training = False
		self.verbose = verbose
		self.logger = get_logger(self.__class__.__name__, logging.INFO if verbose else logging.WARNING)
		if self.verbose:
			self.logger.info(f"{self.__class__.__name__} initialized.")

	@property
	def params(self):
		"""Return all trainable tensors exposed by the model.

		Returns
		-------
		list[Tensor]
			All parameter tensors from trainable layers.
		"""
		parameters = []
		for layer in self.layers:
			if getattr(layer, "trainable", True):
				parameters.extend(list(layer.parameters()))
		if self.verbose:
			self.logger.debug(f"Collected {len(parameters)} trainable parameters.")
		return parameters

	def add(self, layer):
		"""Append a new layer to the model.

		Parameters
		----------
		layer:
			Layer instance to append to the current stack.
		"""
		self.layers.append(layer)
		if self.verbose:
			self.logger.info(f"Added layer {layer.__class__.__name__}.")

	def compile(self, loss, optimizer):
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
		if self.verbose:
			self.logger.info(
				f"Compiled model with loss={self.loss_fn.__class__.__name__} "
				f"and optimizer={self.optimizer.__class__.__name__}."
			)

	def forward(self, x, training=True):
		"""Run a forward pass through every layer in the model.

		Parameters
		----------
		x:
			Input array or tensor.
		training:
			Whether the model should run in training mode.

		Returns
		-------
		Tensor
			Final output of the stacked layers.
		"""
		output = x if isinstance(x, Tensor) else Tensor(x)
		for layer in self.layers:
			output = layer(output, training=training)
		if self.verbose:
			self.logger.debug(f"Ran forward pass with output shape {getattr(output, 'shape', None)}.")
		return output

	def predict(self, x, batch_size=None):
		"""Return model predictions as NumPy arrays.

		Parameters
		----------
		x:
			Input data to predict on.
		batch_size:
			Optional batch size for chunked prediction.

		Returns
		-------
		numpy.ndarray
			Predicted outputs.
		"""
		array = np.asarray(x, dtype=np.float64)
		if self.verbose:
			self.logger.debug(f"Predict called with input shape {array.shape} and batch_size={batch_size}.")
		if batch_size is None:
			return self.forward(array, training=False).data
		outputs = []
		for index in range(0, len(array), batch_size):
			outputs.append(self.forward(array[index : index + batch_size], training=False).data)
		return np.concatenate(outputs, axis=0)

	def evaluate(self, x, y, batch_size=32):
		"""Evaluate the current model on a full dataset.

		Parameters
		----------
		x:
			Evaluation inputs.
		y:
			Ground-truth targets.
		batch_size:
			Mini-batch size used during evaluation.

		Returns
		-------
		float
			Average loss across the dataset.
		"""
		array_x = np.asarray(x, dtype=np.float64)
		array_y = np.asarray(y, dtype=np.float64)
		if self.verbose:
			self.logger.debug(f"Evaluate called with input shape {array_x.shape} and batch_size={batch_size}.")
		total_loss = 0.0
		n_batches = 0
		for index in range(0, len(array_x), batch_size):
			xb, yb = array_x[index : index + batch_size], array_y[index : index + batch_size]
			y_pred = self.forward(xb, training=False)
			loss = self.loss_fn(y_pred, yb)
			total_loss += loss.data.item() if isinstance(loss, Tensor) else float(loss)
			n_batches += 1
		return total_loss / max(n_batches, 1)

	def fit(self, x, y, epochs=10, batch_size=32, validation_data=None, shuffle=True, callbacks=None):
		"""Train the model and return the recorded history.

		Parameters
		----------
		x:
			Training inputs.
		y:
			Training targets.
		epochs:
			Number of training epochs.
		batch_size:
			Mini-batch size.
		validation_data:
			Optional ``(x_val, y_val)`` tuple for validation loss tracking.
		shuffle:
			Whether to shuffle the training data at the start of each epoch.
		callbacks:
			Optional callback objects invoked during training.

		Returns
		-------
		History
			Training history container populated with loss values.

		Examples
		--------
		>>> model = Sequential([Dense(2, 1), Sigmoid()], verbose=True)
		>>> history = model.fit(X, y, epochs=10, batch_size=4)
		>>> history["loss"][-1]
		"""
		array_x = np.asarray(x, dtype=np.float64)
		array_y = np.asarray(y, dtype=np.float64)
		callback_list = list(callbacks or [])
		if self.verbose:
			self.logger.info(
				f"Training for {epochs} epochs on {len(array_x)} samples "
				f"(batch_size={batch_size}, validation={validation_data is not None}, shuffle={shuffle})."
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
			self._history.setdefault("loss", []).append(epoch_loss)
			logs = {"loss": epoch_loss}

			if validation_data is not None:
				val_x, val_y = validation_data
				val_loss = self.evaluate(val_x, val_y, batch_size=batch_size)
				self._history.setdefault("val_loss", []).append(val_loss)
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
					stopper = next((callback for callback in callback_list if hasattr(callback, "stopped_epoch") and getattr(callback, "stopped_epoch", None) is not None), None)
					if stopper is not None:
						stop_message = f"Training stopped early at epoch {stopper.stopped_epoch}/{epochs}"
						if getattr(stopper, "best_epoch", None) is not None:
							stop_message += f"; best {stopper.monitor}={stopper.best:.4f} at epoch {stopper.best_epoch}"
						if getattr(stopper, "stopped_value", None) is not None:
							stop_message += f"; last {stopper.monitor}={stopper.stopped_value:.4f}"
						self.logger.info(stop_message)
					else:
						self.logger.info(f"Training stopped early at epoch {epoch + 1}/{epochs}.")
				break

		for callback in callback_list:
			if hasattr(callback, "on_train_end"):
				callback.on_train_end({"loss": self._history.get("loss", []), "val_loss": self._history.get("val_loss", [])})

		return self._history

	def get_weights(self):
		"""Return copies of the model parameters as NumPy arrays.

		Returns
		-------
		list[numpy.ndarray]
			A snapshot of the current parameter values.
		"""
		return [param.data.copy() for param in self.params]

	def set_weights(self, weights):
		"""Load a list of NumPy arrays back into the model parameters.

		Parameters
		----------
		weights:
			Iterable of NumPy arrays matching the model parameter shapes.
		"""
		for param, weight in zip(self.params, weights):
			param.data = np.asarray(weight, dtype=np.float64).copy()
		if self.verbose:
			self.logger.debug(f"Updated model weights from {len(self.params)} tensors.")

	def summary(self):
		"""Print a compact parameter summary for the model.

		The summary lists each layer with its parameter count and a total at the
		end. It is meant for quick inspection rather than full shape tracing.
		"""
		print(f"{'Layer':<20}{'Output Shape':<20}{'Param #':<10}")
		print("-" * 50)
		total_params = 0
		for layer in self.layers:
			n_params = sum(param.data.size for param in layer.parameters())
			total_params += n_params
			print(f"{layer.name:<20}{'?':<20}{n_params:<10}")
		print("-" * 50)
		print(f"Total trainable params: {total_params}")
		if self.verbose:
			self.logger.info(f"Printed model summary with {total_params} total parameters.")


class Sequential(Model):
	"""Sequential model container for MAGNET layers.

	This class is a thin semantic wrapper around ``Model`` for layer-by-layer
	network definitions. It does not add new training behavior; instead, it
	provides a clear place for future sequential-style model subclasses while
	keeping the inherited ``Model`` API intact.

	Inherited methods
	-----------------
	- ``add``
	- ``compile``
	- ``forward``
	- ``predict``
	- ``evaluate``
	- ``fit``
	- ``get_weights``
	- ``set_weights``
	- ``summary``

	Examples
	--------
	>>> from machinegnostics.magnet.models import Sequential
	>>> isinstance(Sequential(), Model)
	True
	"""
	pass
