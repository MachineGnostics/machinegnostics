"""Dense (fully connected) layer for MAGNET.

``Dense`` implements the affine transform ``y = x @ W + b`` and stores
its parameters as MAGNET tensors backed by torch. This keeps the public
API familiar while allowing gradients to flow automatically through the
hidden autograd backend.

Author: Nirmal Parmar
Machine Gnostics (MAGNET)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..core.tensor import Tensor
from ..initializers import XavierUniform, Zeros, get_initializer
from .base import Layer


class Dense(Layer):
    """Apply a trainable affine transformation to the last input axis.

    ``Dense`` is MAGNET's standard fully connected layer. It learns a
    weight matrix ``W`` and, optionally, a bias vector ``b`` so the
    layer output is ``x @ W + b``. The class supports both the current
    initializer API and older compatibility aliases, which makes it a
    stable building block across the MAGNET refactor.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    weight_init : callable or None, optional
        Initializer for the weight matrix.
    bias_init : callable or None, optional
        Initializer for the bias vector.
    name : str or None, optional
        Name of the layer.
    verbose : bool, optional
        If ``True``, enable verbose logging.
    backward_fn : callable or None, optional
        Custom backward function for the layer.
    n_in : int or None, optional
        Alias for ``in_features``.
    n_out : int or None, optional
        Alias for ``out_features``.
    use_bias : bool, optional
        Whether to include a bias term.
    kernel_initializer : callable or None, optional
        Alias for ``weight_init``.
    bias_initializer : callable or None, optional
        Alias for ``bias_init``.

    Attributes
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    n_in : int
        Alias for ``in_features``.
    n_out : int
        Alias for ``out_features``.
    use_bias : bool
        Whether the layer includes a bias term.
    trainable : bool
        Whether the layer's parameters are trainable.
    params : dict
        Dictionary storing the layer's parameters as MAGNET tensors.
    grads : dict
        Dictionary storing the gradients of the layer's parameters.
    W : Tensor
        Weight matrix.
    b : Tensor or None
        Bias vector if ``use_bias`` is ``True``, else ``None``.
    backward_fn : callable or None
        Custom backward function for the layer.

    Notes
    -----
    The ``Dense`` layer performs a standard affine transformation on the last input axis.
    It is a fundamental building block for constructing fully connected neural networks
    within the MAGNET framework. The layer supports both explicit initializers and
    backward function customization for advanced use cases.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet import Dense
    >>> Dense(in_features=3, out_features=2)(np.array([[1.0, 2.0, 3.0]])).data.shape
    (1, 2)

    """

    def __init__(
        self,
        in_features=None,
        out_features=None,
        weight_init=None,
        bias_init=None,
        name=None,
        verbose: bool = False,
        backward_fn: Callable[['Dense', Any], Any] | None = None,
        *,
        n_in=None,
        n_out=None,
        use_bias: bool = True,
        kernel_initializer=None,
        bias_initializer=None,
    ):
        """Initialize a dense layer and its trainable parameters.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        weight_init : callable or None, optional
            Initializer for the weight matrix.
        bias_init : callable or None, optional
            Initializer for the bias vector.
        name : str or None, optional
            Name of the layer.
        verbose : bool, optional
            If ``True``, enable verbose logging.
        backward_fn : callable or None, optional
            Custom backward function for the layer.
        n_in : int or None, optional
            Alias for ``in_features``.
        n_out : int or None, optional
            Alias for ``out_features``.
        use_bias : bool, optional
            Whether the layer includes a bias term.
        kernel_initializer : callable or None, optional
            Alias for ``weight_init``.
        bias_initializer : callable or None, optional
            Alias for ``bias_init``.

        Notes
        -----
        The ``Dense`` layer performs a standard affine transformation on the last input axis.
        It is a fundamental building block for constructing fully connected neural networks
        within the MAGNET framework. The layer supports both explicit initializers and
        backward function customization for advanced use cases.

        Examples
        --------
        >>> import numpy as np
        >>> from machinegnostics.magnet import Dense
        >>> Dense(in_features=3, out_features=2)(np.array([[1.0, 2.0, 3.0]])).data.shape
        (1, 2)
        """
        super().__init__(name, verbose=verbose)
        in_features = in_features if in_features is not None else n_in
        out_features = out_features if out_features is not None else n_out
        if in_features is None or out_features is None:
            raise ValueError('Dense requires in_features/out_features or n_in/n_out')

        weight_init = self._resolve_initializer(
            kernel_initializer if kernel_initializer is not None else weight_init
        )
        bias_init = self._resolve_initializer(
            bias_initializer if bias_initializer is not None else bias_init,
            default=Zeros(),
        )

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n_in = self.in_features
        self.n_out = self.out_features
        self.use_bias = bool(use_bias)
        self.trainable = True

        self.params['W'] = self._ensure_parameter_tensor(
            weight_init((self.in_features, self.out_features)),
            f'{self.name}_W',
        )
        self.W = self.params['W']
        self.grads['W'] = None

        if self.use_bias:
            self.params['b'] = self._ensure_parameter_tensor(
                bias_init((self.out_features,)),
                f'{self.name}_b',
            )
            self.b = self.params['b']
            self.grads['b'] = None
        else:
            self.b = None

        self.backward_fn = backward_fn
        self.logger.debug(
            'Dense initialized with in_features=%s, out_features=%s, use_bias=%s.',
            self.in_features,
            self.out_features,
            self.use_bias,
        )

    @staticmethod
    def _resolve_initializer(initializer, default=None):
        """Resolve initializer aliases used across the MAGNET refactor."""
        if initializer is None:
            return default if default is not None else XavierUniform(seed=42)
        if isinstance(initializer, str):
            alias_map = {
                'glorot_uniform': 'xavieruniform',
                'glorot-uniform': 'xavieruniform',
                'random_normal': 'randomnormal',
                'he_normal': 'henormal',
            }
            return get_initializer(alias_map.get(initializer, initializer))
        return initializer

    @staticmethod
    def _ensure_parameter_tensor(value, name: str) -> Tensor:
        """Wrap raw initializer output in a trainable MAGNET tensor."""
        tensor = value if isinstance(value, Tensor) else Tensor(value)
        tensor.name = name
        tensor.requires_grad = True
        return tensor

    def forward(self, x, training=True):
        """Apply the affine transformation ``x @ W + b``.

        Parameters
        ----------
        x : Tensor or array-like
            Input batch whose last dimension must equal
            ``self.in_features``.
        training : bool, optional
            Compatibility flag for the layer API.

        Returns
        -------
        Tensor
            Output tensor with shape ``(..., out_features)``.
        """
        x = x if isinstance(x, Tensor) else Tensor(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f'Dense expected input with {self.in_features} features, got shape {x.shape}'
            )
        self.input = x
        self.logger.debug('Running dense forward pass with input shape %s.', x.shape)
        output = x @ self.W
        if self.use_bias and self.b is not None:
            output = output + self.b
        return output

    def backward(self, grad_output):
        """Compute manual affine gradients for compatibility workflows."""
        if self.backward_fn is not None:
            self.logger.debug('Dense.backward delegated to custom backward_fn.')
            return self.backward_fn(self, grad_output)

        if not hasattr(self, 'input') or self.input is None:
            raise RuntimeError('Dense.backward requires a prior forward pass')

        grad_output_tensor = grad_output if isinstance(grad_output, Tensor) else Tensor(grad_output)
        input_data = self.input.data
        grad_output_data = grad_output_tensor.data
        weight_data = self.W.data

        grad_w = input_data.T @ grad_output_data
        grad_input = grad_output_data @ weight_data.T

        self.grads['W'] = grad_w
        self.W.grad = grad_w
        if self.use_bias and self.b is not None:
            grad_b = np.asarray(grad_output_data.sum(axis=0), dtype=np.float64)
            self.grads['b'] = grad_b
            self.b.grad = grad_b
        self.logger.debug(
            'Dense.backward computed gradients with input shape %s and grad_output shape %s.',
            self.input.shape,
            grad_output_tensor.shape,
        )
        return Tensor(grad_input)
