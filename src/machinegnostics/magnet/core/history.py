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


class History(dict):
	"""Dictionary-like training history with convenience helpers."""

	def record(self, logs):
		"""Append a log dictionary to the stored history."""
		for key, value in logs.items():
			self.setdefault(key, []).append(value)
		return self

	def last(self, key, default=None):
		"""Return the last recorded value for a metric key."""
		values = self.get(key)
		if not values:
			return default
		return values[-1]
