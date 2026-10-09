"""History container used by MAGNET training loops.

Developer note
-------------
Author: Nirmal Parmar

Examples
--------
>>> from machinegnostics.magnet.history import History
>>> history = History().record({"loss": 0.5})
>>> history.last("loss")
0.5
"""

from __future__ import annotations

import logging

from ..utils.logging import get_logger


class History(dict):
	"""Dictionary-like training history with convenience helpers."""

	def __init__(self, *args, verbose: bool = False, **kwargs):
		super().__init__(*args, **kwargs)
		self.verbose = verbose
		self.logger = get_logger(self.__class__.__name__, logging.INFO if verbose else logging.WARNING)
		if self.verbose:
			self.logger.info("History initialized.")

	def record(self, logs):
		"""Append a log dictionary to the stored history."""
		for key, value in logs.items():
			self.setdefault(key, []).append(value)
		if self.verbose:
			self.logger.info("Recorded history keys: %s.", ", ".join(logs.keys()))
		return self

	def last(self, key, default=None):
		"""Return the last recorded value for a metric key."""
		values = self.get(key)
		if not values:
			if self.verbose:
				self.logger.info("History key %s is empty; returning default.", key)
			return default
		if self.verbose:
			self.logger.info("Retrieved last history value for key %s.", key)
		return values[-1]
