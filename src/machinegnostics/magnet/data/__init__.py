"""Data utilities for Magnet training loops.

Author: Nirmal Parmar
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np


def batch_iterator(x, y=None, batch_size: int = 32, shuffle: bool = True):
	"""Yield mini-batches from arrays.

	Parameters
	----------
	x:
		Input features.
	y:
		Optional labels/targets.
	batch_size:
		Number of rows per mini-batch.
	shuffle:
		Whether to shuffle sample indices before batching.
	"""
	x_np = np.asarray(x)
	y_np = None if y is None else np.asarray(y)

	n = x_np.shape[0]
	indices = np.arange(n)
	if shuffle:
		np.random.shuffle(indices)

	for start in range(0, n, batch_size):
		end = min(start + batch_size, n)
		idx = indices[start:end]
		if y_np is None:
			yield x_np[idx]
		else:
			yield x_np[idx], y_np[idx]


def train_val_split(x, y, val_ratio: float = 0.2, shuffle: bool = True, random_state: Optional[int] = None):
	"""Split arrays into train/validation subsets."""
	if val_ratio <= 0.0 or val_ratio >= 1.0:
		raise ValueError("val_ratio must be in (0, 1).")

	x_np = np.asarray(x)
	y_np = np.asarray(y)
	n = x_np.shape[0]

	rng = np.random.default_rng(random_state)
	indices = np.arange(n)
	if shuffle:
		rng.shuffle(indices)

	split = int(n * (1.0 - val_ratio))
	train_idx, val_idx = indices[:split], indices[split:]
	return (x_np[train_idx], y_np[train_idx]), (x_np[val_idx], y_np[val_idx])


__all__ = ["batch_iterator", "train_val_split"]
