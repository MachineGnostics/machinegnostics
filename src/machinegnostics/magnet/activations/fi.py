"""Dedicated public activation module for fi.

Developer note
--------------
This module keeps the public MAGNET activation API one class per file.
"""

from __future__ import annotations

import numpy as np
import torch

from ._shared import Tensor, compute_characteristics, custom_tensor, _gnostic_activation_tensor, Layer
from .activation import Activation

class Fi(Activation):
	"""
	Gnostic Fidelity (Fi) Activation Layer with Learnable Center and Scale.

	This layer implements the gnositc fidelity characteristic. The Fi layer learns a concept center (z0) and maintains a bounded scale parameter (S) to transform inputs into the gnostic
	fidelity space.

	Examples
	--------
	>>> import numpy as np
	>>> from machinegnostics.magnet import Dense, Fi, Sequential
	>>> model = Sequential([Dense(2, 1), Fi()])
	>>> model(np.array([[0.0, 1.0]])).shape
	(1, 1)
	"""
	
	def __init__(self, S: float | str = "auto", z0_init: float | str = "median", 
	             S_init: float = 1.0, name=None, verbose: bool = False):
		"""
		Create a trainable fidelity activation layer.

		Parameters
		----------
		S : float or "auto"
			If "auto" (default), S is learned via S_raw parameter
			If float in [0.01, 2.0], S is fixed and non-trainable
		z0_init : float, "mean", or "median"
			Initial concept center:
			- float: Use this value for all neurons
			- "mean": Initialize from batch mean (computed in first forward)
			- "median": Initialize from batch median (computed in first forward)
		S_init : float
			Initial value for S when S="auto". Transformed to S_raw via:
			S_raw = -log(2/S_init - 1)
			Helps control initialization stability.
		name : str, optional
			Layer name for tracking in model
		verbose : bool
			Enable debug output during training
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

	@staticmethod
	def _stable_sech(x: torch.Tensor) -> torch.Tensor:
		"""
		Numerically stable sech function: sech(x) = 1/cosh(x).

		For |x| <= 20: Use direct computation (numerically stable)
		For |x| > 20:  Use approximation: sech(x) ≈ 2*exp(-|x|)
		                (cosh(x) ≈ exp(|x|)/2 for large |x|)

		Parameters
		----------
		x : torch.Tensor
			Input values

		Returns
		-------
		torch.Tensor
			Stable sech values, all in (0, 1]
		"""
		abs_x = torch.abs(x)
		return torch.where(abs_x <= 20.0, 1.0 / torch.cosh(torch.clamp(x, -20.0, 20.0)), 
		                   2.0 * torch.exp(-abs_x))

	def _initialize_params(self, x: Tensor) -> None:
		"""
		Lazy initialization of learnable parameters from first batch.

		Called on first forward pass to establish parameter shape and values.
		Handles both 1D (batch size only) and multi-dimensional inputs.

		Parameters
		----------
		x : Tensor
			Input batch with shape (batch_size, ..., feature_dim) or similar
		"""
		if x.ndim == 0:
			feature_shape = ()
		elif x.ndim == 1:
			feature_shape = (x.shape[0],)
		else:
			feature_shape = tuple(x.shape[1:])

		# Initialize z0 (concept center)
		if self.z0_init == "mean":
			z0_value = np.mean(x.data, axis=0, keepdims=x.ndim > 1)
		elif self.z0_init == "median":
			z0_value = np.median(x.data, axis=0, keepdims=x.ndim > 1)
		else:
			z0_value = np.full(feature_shape or (1,), float(self.z0_init), dtype=np.float64)

		self.params["z0"] = Tensor(z0_value, requires_grad=True)
		
		# Initialize S and S_raw (scale parameter)
		if self.S == "auto":
			# S_raw is unconstrained; S is the computed bounded version
			s_init = float(np.clip(self.S_init, 1e-4, 1.9999))
			
			# Transform S_init to S_raw via inverse sigmoid:
			# If S = 2*sigmoid(S_raw), then
			# S_raw = log((S/2) / (1 - S/2)) = -log(2/S - 1)
			s_raw_value = np.full(np.shape(z0_value) or (1,), 
			                      -np.log(2.0 / s_init - 1.0), dtype=np.float64)
			
			self.params["S_raw"] = Tensor(s_raw_value, requires_grad=True)
			self.params["S"] = Tensor(np.full(np.shape(z0_value) or (1,), s_init, 
			                                 dtype=np.float64), requires_grad=False)
			self.grads["S_raw"] = None
			self.grads["S"] = None
		else:
			# S is fixed and non-trainable
			self.params["S"] = Tensor(np.full(np.shape(z0_value) or (1,), self.S, 
			                                 dtype=np.float64), requires_grad=False)
		
		self.grads["z0"] = None
		self._initialized = True

	def forward(self, x, training=True):
		"""
		Forward pass: compute sech(2*theta) with learned center and scale.

		Parameters
		----------
		x : Tensor or array-like
			Input batch

		training : bool
			If True, apply lazy initialization (if needed)

		Returns
		-------
		Tensor
			Activated output with shape matching input
		"""
		x = x if isinstance(x, Tensor) else Tensor(x)
		if not self._initialized:
			self._initialize_params(x)

		z0 = self.params["z0"]._tensor
		
		if self.S == "auto":
			# S_raw is the actual learned parameter (unconstrained)
			# S is derived from it and must remain in the computation graph
			s_raw = self.params["S_raw"]._tensor
			s = torch.clamp(2.0 * torch.sigmoid(s_raw), 1e-4, 1.9999)
			
			# ⚠️  CRITICAL FIX: Do NOT detach S from the graph!
			# Original (broken): self.params["S"]._tensor = s.detach().clone()
			# This breaks gradient flow to S_raw.
			# Store only a detached copy for inspection; the live tensor ``s``
			# still drives the forward computation and gradients to S_raw.
			self.params["S"]._tensor = s.detach()
		else:
			s = self.params["S"]._tensor

		# Compute normalized deviation and activation
		theta = (x._tensor - z0) / s
		out = self._stable_sech(2.0 * theta)
		
		# Cache for potential later inspection
		self.theta = Tensor.from_torch(theta)
		self.out = Tensor.from_torch(out)
		
		return Tensor.from_torch(out)
