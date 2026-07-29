"""Training callbacks for MAGNET (Machine Gnostics Neural Networks).

Developer note
-------------
Author: Nirmal Parmar

Examples
--------
>>> from machinegnostics.magnet.callbacks import EarlyStopping
>>> stopper = EarlyStopping(monitor="val_loss", patience=3)
"""

from __future__ import annotations

from copy import deepcopy


class Callback:
	"""Base callback interface used by the MAGNET training loop."""

	def set_model(self, model):
		"""Attach the current model to the callback."""
		self.model = model

	def on_train_begin(self, logs=None):
		return logs

	def on_train_end(self, logs=None):
		return logs

	def on_epoch_begin(self, epoch, logs=None):
		return logs

	def on_epoch_end(self, epoch, logs=None):
		return logs


class EarlyStopping(Callback):
	"""Stop training when a monitored metric stops improving.

	Parameters
	----------
	monitor:
		Metric name to watch, typically ``"val_loss"``.
	patience:
		Number of epochs without improvement before stopping.
	min_delta:
		Minimum improvement required to reset patience.
	restore_best_weights:
		Restore the best observed weights when stopping.
	"""
	def __init__(self, monitor="val_loss", patience=5, min_delta=0.0, restore_best_weights=True):
		"""Create a new early-stopping controller."""
		self.monitor = monitor
		self.patience = patience
		self.min_delta = min_delta
		self.restore_best_weights = restore_best_weights
		self.best = None
		self.best_weights = None
		self.wait = 0
		self.model = None

	def on_train_begin(self, logs=None):
		"""Reset the early-stopping state at the start of training."""
		self.best = None
		self.best_weights = None
		self.wait = 0

	def on_epoch_end(self, epoch, logs=None):
		"""Inspect the latest logs and decide whether training should stop."""
		logs = logs or {}
		current = logs.get(self.monitor)
		if current is None:
			return logs
		if self.best is None or current < self.best - self.min_delta:
			self.best = current
			self.wait = 0
			if self.restore_best_weights and self.model is not None:
				self.best_weights = deepcopy(self.model.get_weights())
			return logs
		self.wait += 1
		if self.wait >= self.patience and self.model is not None:
			self.model.stop_training = True
			if self.restore_best_weights and self.best_weights is not None:
				self.model.set_weights(self.best_weights)
		return logs
