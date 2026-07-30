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

import logging

from ..utils.logging import get_logger

logger = get_logger(__name__)


class Callback:
	"""Base callback interface used by the MAGNET training loop.

	Callbacks let you observe and influence training without changing the model
	implementation itself. A callback is attached to a model before training and
	then receives lifecycle events at the start and end of training, and at the
	start and end of every epoch.

	Typical uses
	------------
	- stop training early when validation loss stops improving;
	- save the best weights during training;
	- collect custom logs or metrics;
	- inspect the model state at specific training milestones.

	Parameters
	----------
	verbose:
		If ``True``, emit callback-level log messages through the shared logger.
	name:
		Optional display name used by the logger and debug output.

	Examples
	--------
	>>> from machinegnostics.magnet.core import Callback
	>>> callback = Callback(verbose=True)
	>>> callback.name
	'Callback'
	"""

	def __init__(self, verbose: bool = False, name: str | None = None):
		self.verbose = verbose
		self.name = name or self.__class__.__name__
		self.logger = get_logger(self.name, logging.INFO if verbose else logging.WARNING)
		if self.verbose:
			self.logger.info("Callback initialized.")

	def set_model(self, model):
		"""Attach the current model to the callback.

		The training loop calls this before the first epoch so callback methods
		can access ``self.model`` when they need to read weights, stop training,
		or restore state.
		"""
		self.model = model
		if self.verbose:
			self.logger.info("Attached model %s.", model.__class__.__name__)

	def on_train_begin(self, logs=None):
		"""Hook called once before training starts.

		Subclasses can override this method to reset internal state or allocate
		training-time resources.
		"""
		return logs

	def on_train_end(self, logs=None):
		"""Hook called once after training finishes.

		Subclasses can use this to release resources, summarize results, or flush
		collected logs.
		"""
		return logs

	def on_epoch_begin(self, epoch, logs=None):
		"""Hook called at the beginning of each epoch.

		Parameters
		----------
		epoch:
			Zero-based epoch index.
		logs:
			Mutable dictionary passed through the training loop.
		"""
		return logs

	def on_epoch_end(self, epoch, logs=None):
		"""Hook called at the end of each epoch.

		This is the most common hook for monitoring metrics, triggering early
		stopping, or persisting intermediate training state.
		"""
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
	def __init__(self, monitor="val_loss", patience=5, min_delta=0.0, restore_best_weights=True, verbose: bool = False):
		"""Initialize the early stopping callback.

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
		verbose:
			If ``True``, emit callback-level log messages through the shared logger.

		Examples
		--------
		>>> from machinegnostics.magnet import EarlyStopping
		>>> stopper = EarlyStopping(monitor="val_loss", patience=3, min_delta=0.001, restore_best_weights=True, verbose=True)
		"""
		super().__init__(verbose=verbose)
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
		if self.verbose:
			self.logger.info("EarlyStopping state reset.")

	def on_epoch_end(self, epoch, logs=None):
		"""Inspect the latest logs and decide whether training should stop."""
		logs = logs or {}
		current = logs.get(self.monitor)
		if current is None:
			if self.verbose:
				self.logger.info("Monitor %s missing at epoch %s.", self.monitor, epoch)
			return logs
		if self.best is None or current < self.best - self.min_delta:
			self.best = current
			self.wait = 0
			if self.restore_best_weights and self.model is not None:
				self.best_weights = deepcopy(self.model.get_weights())
			if self.verbose:
				self.logger.info("New best %s=%.6f at epoch %s.", self.monitor, current, epoch)
			return logs
		self.wait += 1
		if self.wait >= self.patience and self.model is not None:
			self.model.stop_training = True
			if self.restore_best_weights and self.best_weights is not None:
				self.model.set_weights(self.best_weights)
			if self.verbose:
				self.logger.info("Stopping training at epoch %s after %s stale epochs.", epoch, self.wait)
		return logs
