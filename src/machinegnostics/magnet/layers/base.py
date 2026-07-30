"""Base layer class for MAGNET (Machine Gnostics Neural Networks).

Developer note
--------------
``Layer`` is the shared contract for every MAGNET building block. It does not
perform any computation by itself. Instead, subclasses store trainable tensors
in ``params``, implement ``forward`` for the actual math, and optionally
override ``backward`` when autograd is not enough.

The class also exposes an optional logger, following the same project pattern
used in ``magcal`` and model classes. Logging is intentionally quiet by default
and becomes useful when you want to trace mode changes or parameter handling
during debugging.

At runtime, a layer can be used in two ways:
- call it directly like a function, for example ``y = layer(x)``;
- place it inside a ``Model`` or ``Sequential`` container and train it with
	``fit``/``predict``.

The public helpers are intentionally small:
- ``parameters()`` exposes trainable tensors to optimizers;
- ``sync_grads()`` copies tensor gradients into ``grads`` for inspection;
- ``set_mode(training)`` records training vs inference state.

Author: Nirmal Parmar
"""

from __future__ import annotations

import numpy as np
import logging
from ..utils.logging import get_logger


class Layer:
	"""Common base class for all MAGNET layers.

	Subclasses define the actual computation by overriding ``forward`` and, when
	needed, ``backward``. The base class provides the small layer contract used
	throughout MAGNET: trainable parameters live in ``params``, gradients are
	cached in ``grads``, and layers can be called like functions.

	Minimal working example
	-----------------------
	The example below defines a tiny scaling layer and shows the public layer
	API in one place.

	>>> import numpy as np
	>>> from machinegnostics.magnet import Tensor, Layer
	>>>
	>>> class Scale(Layer):
	... 	def __init__(self):
	... 		super().__init__()
	... 		self.params["w"] = Tensor(np.array([2.0]), requires_grad=True)
	... 	def forward(self, x, training=True):
	... 		x = x if isinstance(x, Tensor) else Tensor(x)
	... 		return x * self.params["w"]
	... 	def backward(self, grad_output):
	... 		raise NotImplementedError
	...
	>>> layer = Scale()
	>>> layer(np.array([3.0])).data
	array([6.])
	>>> list(layer.parameters())[0].data
	array([2.])
	>>> layer.set_mode(False)
	>>> layer._training
	False
	"""

	def __init__(self, name=None, verbose: bool = False):
		"""Initialize the layer bookkeeping fields.

		Parameters
		----------
		name:
			Optional display name. If omitted, the class name is used.
		verbose:
			If ``True``, enable debug-level logging for the layer instance.
		"""
		self.name = name or self.__class__.__name__
		self.params = {}
		self.grads = {}
		self.trainable = True
		self._training = True
		self.logger = get_logger(self.name, logging.DEBUG if verbose else logging.WARNING)
		self.logger.debug("Layer initialized.")

	def forward(self, x, training=True):
		"""Transform the input tensor in the forward pass.

		Subclasses must override this method. The ``training`` flag lets the same
		layer behave differently during training and inference when needed.
		"""
		raise NotImplementedError

	def backward(self, grad_output):
		"""Backward pass hook for non-autograd layers.

		MAGNET's tensor autograd handles most layers automatically, so this hook
		is only needed for layers that implement a manual backward pass.
		"""
		raise NotImplementedError

	def __call__(self, x, training=True):
		"""Alias for ``forward`` so layers can be called like functions.

		Examples
		--------
		>>> import numpy as np
		>>> from machinegnostics.magnet import Tensor, Layer
		>>>
		>>> class Shift(Layer):
		... 	def forward(self, x, training=True):
		... 		x = x if isinstance(x, Tensor) else Tensor(x)
		... 		return x + 1
		... 	def backward(self, grad_output):
		... 		raise NotImplementedError
		...
		>>> Shift()(np.array([1.0, 2.0])).data
		array([2., 3.])
		"""
		return self.forward(x, training=training)

	def parameters(self):
		"""Yield all trainable tensors owned by the layer.

		Examples
		--------
		>>> import numpy as np
		>>> from machinegnostics.magnet import Tensor, Layer
		>>>
		>>> class WithWeight(Layer):
		... 	def __init__(self):
		... 		super().__init__()
		... 		self.params["w"] = Tensor(np.array([1.0]), requires_grad=True)
		... 	def forward(self, x, training=True):
		... 		raise NotImplementedError
		... 	def backward(self, grad_output):
		... 		raise NotImplementedError
		...
		>>> [param.data.tolist() for param in WithWeight().parameters()]
		[[1.0]]
		"""
		self.logger.debug("Collecting layer parameters.")
		for param in self.params.values():
			yield param

	def sync_grads(self):
		"""Cache tensor gradients into the layer-level ``grads`` mapping.

		This is useful after autograd has populated each tensor's ``grad`` field
		and you want a plain NumPy copy on the layer for inspection or logging.

		Examples
		--------
		>>> import numpy as np
		>>> from machinegnostics.magnet import Tensor, Layer
		>>>
		>>> class WithWeight(Layer):
		... 	def __init__(self):
		... 		super().__init__()
		... 		self.params["w"] = Tensor(np.array([1.0]), requires_grad=True)
		... 	def forward(self, x, training=True):
		... 		raise NotImplementedError
		... 	def backward(self, grad_output):
		... 		raise NotImplementedError
		...
		>>> layer = WithWeight()
		>>> layer.params["w"].grad = np.array([0.25])
		>>> layer.sync_grads()
		>>> layer.grads["w"]
		array([0.25])
		"""
		self.logger.debug("Synchronizing parameter gradients.")
		for key, param in self.params.items():
			self.grads[key] = None if param.grad is None else np.asarray(param.grad, dtype=np.float64).copy()

	def get_params_and_grads(self):
		"""Yield parameter and gradient pairs for the optimizer.

		The iterator returns ``(parameter, gradient)`` tuples, where gradients are
		NumPy arrays when available and ``None`` otherwise.
		"""
		self.logger.debug("Collecting parameters and gradients.")
		for key, param in self.params.items():
			yield param, None if param.grad is None else np.asarray(param.grad, dtype=np.float64)

	def set_mode(self, training: bool):
		"""Record whether the layer is in training or inference mode.

		Parameters
		----------
		training:
			``True`` for training mode, ``False`` for inference mode.
		"""
		self._training = training
		self.logger.debug("Layer mode set to %s.", "training" if training else "inference")

	def __repr__(self):
		"""Return a concise debug representation.

		Examples
		--------
		>>> from machinegnostics.magnet.layers.base import Layer
		>>> repr(Layer())
		'<Layer: 0 params>'
		"""
		n_params = sum(param.data.size for param in self.params.values())
		return f"<{self.name}: {n_params} params>"
