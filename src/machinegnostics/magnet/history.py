"""History container used by magnet training loops."""

from __future__ import annotations


class History(dict):
	def record(self, logs):
		for key, value in logs.items():
			self.setdefault(key, []).append(value)
		return self

	def last(self, key, default=None):
		values = self.get(key)
		if not values:
			return default
		return values[-1]
