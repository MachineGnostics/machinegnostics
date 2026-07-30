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

import numpy as np

from ..core.history import History
from ..losses import get_loss, Loss
from ..optimizers import get_optimizer
from ..core.tensor import Tensor


class Model:
	"""Base MAGNET model container.

	The model wires layers together, manages parameters, and runs training.
	"""

	def __init__(self, layers=None):
		"""Create a model from an optional list of layers."""
		self.layers = list(layers or [])
		self.loss_fn: Loss | None = None
		self.optimizer = None
		self._history = History()
		self.history = self._history
		self.stop_training = False

	@property
	def params(self):
		"""Return all trainable tensors exposed by the model."""
		parameters = []
		for layer in self.layers:
			if getattr(layer, "trainable", True):
				parameters.extend(list(layer.parameters()))
		return parameters

	def add(self, layer):
		"""Append a new layer to the model."""
		self.layers.append(layer)

	def compile(self, loss, optimizer):
		"""Attach a loss function and optimizer to the model."""
		self.loss_fn = get_loss(loss)
		self.optimizer = get_optimizer(optimizer)

	def forward(self, x, training=True):
		"""Run a forward pass through every layer in the model."""
		output = x if isinstance(x, Tensor) else Tensor(x)
		for layer in self.layers:
			output = layer(output, training=training)
		return output

	def predict(self, x, batch_size=None):
		"""Return model predictions as NumPy arrays."""
		array = np.asarray(x, dtype=np.float64)
		if batch_size is None:
			return self.forward(array, training=False).data
		outputs = []
		for index in range(0, len(array), batch_size):
			outputs.append(self.forward(array[index : index + batch_size], training=False).data)
		return np.concatenate(outputs, axis=0)

	def evaluate(self, x, y, batch_size=32):
		"""Evaluate the current model on a full dataset."""
		array_x = np.asarray(x, dtype=np.float64)
		array_y = np.asarray(y, dtype=np.float64)
		total_loss = 0.0
		n_batches = 0
		for index in range(0, len(array_x), batch_size):
			xb, yb = array_x[index : index + batch_size], array_y[index : index + batch_size]
			y_pred = self.forward(xb, training=False)
			loss = self.loss_fn(y_pred, yb)
			total_loss += loss.data.item() if isinstance(loss, Tensor) else float(loss)
			n_batches += 1
		return total_loss / max(n_batches, 1)

	def fit(self, x, y, epochs=10, batch_size=32, validation_data=None, shuffle=True, verbose=True, callbacks=None):
		"""Train the model and return the recorded history.

		Examples
		--------
		>>> history = model.fit(X, y, epochs=10, batch_size=4)
		>>> history["loss"][-1]
		"""
		array_x = np.asarray(x, dtype=np.float64)
		array_y = np.asarray(y, dtype=np.float64)
		callback_list = list(callbacks or [])
		self.stop_training = False
		self._history = History()
		self.history = self._history

		for callback in callback_list:
			if hasattr(callback, "set_model"):
				callback.set_model(self)
			if hasattr(callback, "on_train_begin"):
				callback.on_train_begin({})

		for epoch in range(epochs):
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

			if verbose:
				message = f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f}"
				if validation_data is not None:
					message += f" - val_loss: {logs['val_loss']:.4f}"
				print(message)

			if self.stop_training:
				break

		for callback in callback_list:
			if hasattr(callback, "on_train_end"):
				callback.on_train_end({"loss": self._history.get("loss", []), "val_loss": self._history.get("val_loss", [])})

		return self._history

	def get_weights(self):
		"""Return copies of the model parameters as NumPy arrays."""
		return [param.data.copy() for param in self.params]

	def set_weights(self, weights):
		"""Load a list of NumPy arrays back into the model parameters."""
		for param, weight in zip(self.params, weights):
			param.data = np.asarray(weight, dtype=np.float64).copy()

	def summary(self):
		"""Print a compact parameter summary for the model."""
		print(f"{'Layer':<20}{'Output Shape':<20}{'Param #':<10}")
		print("-" * 50)
		total_params = 0
		for layer in self.layers:
			n_params = sum(param.data.size for param in layer.parameters())
			total_params += n_params
			print(f"{layer.name:<20}{'?':<20}{n_params:<10}")
		print("-" * 50)
		print(f"Total trainable params: {total_params}")


class Sequential(Model):
	"""Sequential model container for MAGNET layers.

	This is the preferred entry point for small tutorial-style networks.
	"""
	pass
