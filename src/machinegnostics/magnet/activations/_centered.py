"""Centered gnostic activation base for MAGNET.

Developer note
--------------
Author: Nirmal Parmar, OSS Machine Gnostics

This module implements the shared center/scale machinery used by the
parameterized gnostic activations. It maps an input tensor into normalized
characteristic space using a learnable or fixed center ``z0`` and scale ``S``.
The subclasses provide the actual characteristic transform while this base
class handles initialization, bounded scale parameterization, and tensor
plumbing.
"""

from __future__ import annotations

import numpy as np

from ._shared import Tensor, _gnostic_activation_tensor, compute_characteristics, custom_tensor, Layer
from .activation import Activation


class _CenteredCharacteristicActivation(Activation):
	"""Shared centered characteristic activation with optional fixed scale.

	Subclasses implement :meth:`_transform` to map the normalized coordinate
	``theta = (x - z0) / S`` onto the desired gnostic characteristic.
	"""

	def __init__(self, S: float | str = "auto", z0_init: float | str = "median", S_init: float = 1.0, name=None, verbose: bool = False):
		"""Create a centered characteristic activation.

		Parameters
		----------
		S:
			Scale value or ``"auto"`` to learn a bounded scale parameter.
		z0_init:
			Initial center for the characteristic transform.
		S_init:
			Initial scale used when ``S="auto"``.
		name:
			Optional layer name.
		verbose:
			Enable debug logging.
		"""
		super().__init__(name, verbose=verbose)
		self.z0_init = z0_init
		self.S_init = float(S_init)
		self.S = S
		if self.S != "auto":
			self.S = float(self.S)
		if self.S != "auto" and not (0.01 <= self.S <= 2.0):
			raise ValueError("S must be between 0.01 and 2.0 when provided.")
		self._initialized = False

	def _initialize_params(self, x: Tensor) -> None:
		"""Allocate ``z0`` and ``S`` parameters from the first input batch."""
		if x.ndim == 0:
			feature_shape = ()
		elif x.ndim == 1:
			feature_shape = (x.shape[0],)
		else:
			feature_shape = tuple(x.shape[1:])

		if self.z0_init == "mean":
			z0_value = np.mean(x.data, axis=0, keepdims=x.ndim > 1)
		elif self.z0_init == "median":
			z0_value = np.median(x.data, axis=0, keepdims=x.ndim > 1)
		else:
			z0_value = np.full(feature_shape or (1,), float(self.z0_init), dtype=np.float64)

		self.params["z0"] = Tensor(z0_value, requires_grad=True)
		if self.S == "auto":
			s_init = float(np.clip(self.S_init, 1e-4, 1.9999))
			s_raw_value = np.full(np.shape(z0_value) or (1,), np.log(s_init / (2.0 - s_init)), dtype=np.float64)
			self.params["S_raw"] = Tensor(s_raw_value, requires_grad=True)
			self.params["S"] = Tensor(np.full(np.shape(z0_value) or (1,), s_init, dtype=np.float64), requires_grad=False)
			self.grads["S_raw"] = None
			self.grads["S"] = None
		else:
			self.params["S"] = Tensor(np.full(np.shape(z0_value) or (1,), self.S, dtype=np.float64), requires_grad=False)
		self.grads["z0"] = None
		self._initialized = True

	def _scale_tensor(self):
		"""Return the bounded scale tensor used by the characteristic map."""
		if self.S == "auto":
			s_raw = self.params["S_raw"]._tensor
			s = torch.clamp(2.0 * torch.sigmoid(s_raw), 1e-4, 1.9999)
			
			# ⚠️  CRITICAL: Do NOT detach S from computation graph!
			# This allows gradients to flow back to S_raw parameter.
			self.params["S"]._tensor = s.detach()  # Keep only the inspected value here
			return s
		
		s = self.params["S"]._tensor
		self.params["S"]._tensor = s.detach()  # Non-trainable S, no gradient needed
		return s

	def _transform(self, theta: torch.Tensor) -> torch.Tensor:
		"""Transform normalized coordinates into the target characteristic."""
		raise NotImplementedError

	def forward(self, x, training=True):
		"""Apply the centered characteristic transform to ``x``."""
		x = x if isinstance(x, Tensor) else Tensor(x)
		if not self._initialized:
			self._initialize_params(x)

		z0 = self.params["z0"]._tensor
		s = self._scale_tensor()
		theta = (x._tensor - z0) / s
		out = self._transform(theta)
		self.theta = Tensor.from_torch(theta)
		self.out = Tensor.from_torch(out)
		return Tensor.from_torch(out)
