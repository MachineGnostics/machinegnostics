"""Activation exports for MAGNET.

This module keeps the standard activation layers in one shallow namespace
and re-exports the dedicated gnostic activations implemented in their own
files. The gnostic classes are imported rather than defined here so the
public API stays flat without duplicating implementation logic.

Examples
--------
>>> import numpy as np
>>> from machinegnostics.magnet.activations import ReLU, Fi
>>> ReLU()(np.array([-1.0, 0.0, 1.0])).shape
(3,)
>>> Fi(learnable_S=False, learnable_z0=False)(np.array([0.0])).shape
(1,)
"""

from __future__ import annotations

import numpy as np

from ..core._gnostic import custom_tensor
from ..core.tensor import Tensor
from ..layers.base import Layer
from .base import Activation
from .ei import Ei, ei
from .fi import Fi, fi
from .fj import Fj, fj
from .hi import Hi, hi
from .hj import Hj, hj


def _activation_tensor(x: Tensor, value, prime) -> Tensor:
    """Wrap a NumPy activation result in a differentiable MAGNET tensor.

    Parameters
    ----------
    x : Tensor
        Source tensor that provides the upstream autograd connection.
    value : array-like
        Activated values to expose through the MAGNET tensor facade.
    prime : array-like
        Elementwise derivative used by MAGNET's custom tensor bridge.

    Returns
    -------
    Tensor
        Tensor that preserves the forward value and local derivative.
    """
    return custom_tensor(value, x, np.asarray(prime, dtype=np.float64))


class ReLU(Activation):
    """Apply the rectified linear unit activation.

    ReLU returns zero for negative inputs and preserves positive values.
    It is the default hidden-layer activation in many feed-forward
    networks because it is simple, inexpensive, and avoids the heavy
    saturation behavior of bounded nonlinearities.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet.activations import ReLU
    >>> ReLU()(np.array([-2.0, 0.0, 3.0])).data.tolist()
    [0.0, 0.0, 3.0]
    """

    def forward(self, x, training: bool = True) -> Tensor:
        """Return ``max(0, x)`` elementwise.

        Parameters
        ----------
        x : Tensor or array-like
            Input values of any shape.
        training : bool, optional
            Unused compatibility flag maintained for the layer API.

        Returns
        -------
        Tensor
            Activated tensor with the same shape as ``x``.

        Examples
        --------
        >>> import numpy as np
        >>> from machinegnostics.magnet.activations import ReLU
        >>> ReLU()(np.array([-1.0, 2.0])).shape
        (2,)
        """
        x = x if isinstance(x, Tensor) else Tensor(x)
        return x.relu()


class Sigmoid(Activation):
    """Apply the logistic sigmoid activation.

    Sigmoid maps each value into the open interval ``(0, 1)`` and is
    commonly used for binary output layers, probabilities, and gates.
    The function is smooth and differentiable everywhere, which makes it
    convenient for optimization when bounded outputs are required.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet.activations import Sigmoid
    >>> Sigmoid()(np.array([0.0])).data.round(3).tolist()
    [0.5]
    """

    def forward(self, x, training: bool = True) -> Tensor:
        """Return the elementwise sigmoid of the input.

        Parameters
        ----------
        x : Tensor or array-like
            Input values of any shape.
        training : bool, optional
            Unused compatibility flag maintained for the layer API.

        Returns
        -------
        Tensor
            Tensor with values between 0 and 1.
        """
        x = x if isinstance(x, Tensor) else Tensor(x)
        return x.sigmoid()


class Tanh(Activation):
    """Apply the hyperbolic tangent activation.

    Tanh squashes inputs into ``(-1, 1)`` with zero-centered output,
    which can be useful when subsequent layers benefit from positive and
    negative activations of similar scale.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet.activations import Tanh
    >>> Tanh()(np.array([0.0])).data.tolist()
    [0.0]
    """

    def forward(self, x, training: bool = True) -> Tensor:
        """Return the elementwise hyperbolic tangent of the input.

        Parameters
        ----------
        x : Tensor or array-like
            Input values of any shape.
        training : bool, optional
            Unused compatibility flag maintained for the layer API.

        Returns
        -------
        Tensor
            Tensor with the same shape as ``x`` and values in ``(-1, 1)``.
        """
        x = x if isinstance(x, Tensor) else Tensor(x)
        return x.tanh()


class Softmax(Activation):
    """Convert logits into normalized probabilities.

    Softmax is typically used in multiclass classification heads. The
    implementation subtracts the maximum logit along the last axis before
    exponentiation to improve numerical stability.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet.activations import Softmax
    >>> Softmax()(np.array([[1.0, 2.0, 3.0]])).shape
    (1, 3)
    """

    def forward(self, x, training: bool = True) -> Tensor:
        """Normalize the last axis into probabilities.

        Parameters
        ----------
        x : Tensor or array-like
            Logits with any leading batch dimensions.
        training : bool, optional
            Unused compatibility flag maintained for the layer API.

        Returns
        -------
        Tensor
            Probability tensor with the same shape as ``x``.

        Notes
        -----
        The output sums to 1 across the final axis.
        """
        x = x if isinstance(x, Tensor) else Tensor(x)
        shifted = x - Tensor(np.max(x.data, axis=-1, keepdims=True))
        exp = shifted.exp()
        return exp / exp.sum(axis=-1, keepdims=True)


class Step(Activation):
    """Apply a hard threshold activation.

    Step maps values above the threshold to 1 and all others to 0. It is
    mainly useful for explicit thresholding or post-processing because it
    is not smooth and therefore provides limited gradient information.

    Attributes
    ----------
    threshold : float
        Decision boundary used by the activation.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet.activations import Step
    >>> Step(threshold=0.5)(np.array([0.25, 0.75])).data.tolist()
    [0.0, 1.0]
    """

    def __init__(self, threshold: float = 0.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.threshold = float(threshold)

    def forward(self, x, training: bool = True) -> Tensor:
        """Apply the configured threshold elementwise.

        Parameters
        ----------
        x : Tensor or array-like
            Input values of any shape.
        training : bool, optional
            Unused compatibility flag maintained for the layer API.

        Returns
        -------
        Tensor
            Tensor of zeros and ones with the same shape as ``x``.
        """
        x = x if isinstance(x, Tensor) else Tensor(x)
        data = (x.data > self.threshold).astype(np.float64)
        prime = np.zeros_like(x.data, dtype=np.float64)
        return _activation_tensor(x, data, prime)


class LeakyReLU(Activation):
    """Apply the leaky rectified linear unit activation.

    LeakyReLU behaves like ReLU for positive inputs but keeps a small
    slope for negative inputs so gradients can continue to flow through
    units that would otherwise become inactive.

    Attributes
    ----------
    alpha : float
        Multiplicative slope applied to negative inputs.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet.activations import LeakyReLU
    >>> LeakyReLU(alpha=0.1)(np.array([-2.0, 3.0])).data.tolist()
    [-0.2, 3.0]
    """

    def __init__(self, alpha: float = 0.01, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.alpha = float(alpha)

    def forward(self, x, training: bool = True) -> Tensor:
        """Return ``x`` for positive inputs and ``alpha * x`` otherwise.

        Parameters
        ----------
        x : Tensor or array-like
            Input values of any shape.
        training : bool, optional
            Unused compatibility flag maintained for the layer API.

        Returns
        -------
        Tensor
            Activated tensor with the same shape as ``x``.
        """
        x = x if isinstance(x, Tensor) else Tensor(x)
        data = np.where(x.data > 0.0, x.data, self.alpha * x.data)
        prime = np.where(x.data > 0.0, 1.0, self.alpha)
        return _activation_tensor(x, data, prime)


class ELU(Activation):
    """Apply the exponential linear unit activation.

    ELU matches the identity for positive inputs and transitions smoothly
    into an exponential curve for negative inputs. This can reduce the
    dead-unit behavior seen in plain ReLU while keeping positive-region
    gradients intact.

    Attributes
    ----------
    alpha : float
        Saturation scale for negative inputs.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet.activations import ELU
    >>> ELU(alpha=1.0)(np.array([-1.0, 1.0])).shape
    (2,)
    """

    def __init__(self, alpha: float = 1.0, name: str | None = None, verbose: bool = False):
        super().__init__(name=name, verbose=verbose)
        self.alpha = float(alpha)

    def forward(self, x, training: bool = True) -> Tensor:
        """Apply the ELU transform elementwise.

        Parameters
        ----------
        x : Tensor or array-like
            Input values of any shape.
        training : bool, optional
            Unused compatibility flag maintained for the layer API.

        Returns
        -------
        Tensor
            Activated tensor with the same shape as ``x``.
        """
        x = x if isinstance(x, Tensor) else Tensor(x)
        positive = x.data > 0.0
        data = np.where(positive, x.data, self.alpha * np.expm1(x.data))
        prime = np.where(positive, 1.0, data + self.alpha)
        return _activation_tensor(x, data, prime)


class Softplus(Activation):
    """Apply the softplus activation.

    Softplus is a smooth approximation to ReLU. It is useful when the
    model benefits from strictly positive, differentiable outputs without
    introducing a hard corner at zero.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet.activations import Softplus
    >>> Softplus()(np.array([0.0])).data.round(3).tolist()
    [0.693]
    """

    def forward(self, x, training: bool = True) -> Tensor:
        """Return a smooth positive transform of the input.

        Parameters
        ----------
        x : Tensor or array-like
            Input values of any shape.
        training : bool, optional
            Unused compatibility flag maintained for the layer API.

        Returns
        -------
        Tensor
            Activated tensor with the same shape as ``x``.
        """
        x = x if isinstance(x, Tensor) else Tensor(x)
        data = np.log1p(np.exp(-np.abs(x.data))) + np.maximum(x.data, 0.0)
        prime = 1.0 / (1.0 + np.exp(-np.clip(x.data, -500.0, 500.0)))
        return _activation_tensor(x, data, prime)


class Swish(Activation):
    """Apply the swish activation ``x * sigmoid(x)``.

    Swish is a smooth self-gated nonlinearity that often performs well
    as a drop-in replacement for ReLU in dense networks. It preserves
    small negative responses while remaining fully differentiable.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet.activations import Swish
    >>> Swish()(np.array([0.0, 1.0])).shape
    (2,)
    """

    def forward(self, x, training: bool = True) -> Tensor:
        """Return the swish transform of the input.

        Parameters
        ----------
        x : Tensor or array-like
            Input values of any shape.
        training : bool, optional
            Unused compatibility flag maintained for the layer API.

        Returns
        -------
        Tensor
            Activated tensor with the same shape as ``x``.
        """
        x = x if isinstance(x, Tensor) else Tensor(x)
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x.data, -500.0, 500.0)))
        data = x.data * sigmoid
        prime = sigmoid + x.data * sigmoid * (1.0 - sigmoid)
        return _activation_tensor(x, data, prime)


class Square(Activation):
    """Square the input elementwise.

    Square is a simple polynomial activation that emphasizes magnitude
    regardless of sign. It can be useful in feature engineering,
    energy-like objectives, and diagnostic experiments where a symmetric
    response is desirable.

    Examples
    --------
    >>> import numpy as np
    >>> from machinegnostics.magnet.activations import Square
    >>> Square()(np.array([-2.0, 3.0])).data.tolist()
    [4.0, 9.0]
    """

    def forward(self, x, training: bool = True) -> Tensor:
        """Return the elementwise square of the input.

        Parameters
        ----------
        x : Tensor or array-like
            Input values of any shape.
        training : bool, optional
            Unused compatibility flag maintained for the layer API.

        Returns
        -------
        Tensor
            Tensor containing ``x ** 2`` with the same shape as ``x``.
        """
        x = x if isinstance(x, Tensor) else Tensor(x)
        data = np.square(x.data)
        prime = 2.0 * x.data
        return _activation_tensor(x, data, prime)


def get_activation(activation, verbose: bool = False):
    """Resolve an activation specification into a layer instance.

    Parameters
    ----------
    activation : None, str, Layer, or callable
        Activation specification. String values are resolved against the
        built-in registry of standard and gnostic activations.
    verbose : bool, optional
        When a string is resolved to a class, forward the verbosity flag
        to the created instance.

    Returns
    -------
    Activation or Layer or None
        Resolved activation instance, or ``None`` when ``activation`` is
        ``None``.

    Raises
    ------
    ValueError
        If the string name is unknown.
    TypeError
        If the specification cannot be interpreted.

    Examples
    --------
    >>> get_activation('relu').__class__.__name__
    'ReLU'
    >>> get_activation('fi').__class__.__name__
    'Fi'
    """
    if activation is None:
        return None
    if isinstance(activation, Layer):
        return activation
    if isinstance(activation, str):
        registry = {
            'relu': ReLU,
            'step': Step,
            'threshold': Step,
            'heaviside': Step,
            'leakyrelu': LeakyReLU,
            'elu': ELU,
            'sigmoid': Sigmoid,
            'softplus': Softplus,
            'tanh': Tanh,
            'swish': Swish,
            'softmax': Softmax,
            'square': Square,
            'fi': Fi,
            'fj': Fj,
            'hi': Hi,
            'hj': Hj,
            'ei': Ei,
        }
        key = activation.replace('_', '').replace('-', '').lower()
        try:
            return registry[key](verbose=verbose)
        except KeyError as exc:
            raise ValueError(f'Unknown activation: {activation}') from exc
    if callable(activation):
        return activation
    raise TypeError(f'Unsupported activation specification: {type(activation)!r}')


__all__ = [
    'Activation',
    'ReLU',
    'Sigmoid',
    'Tanh',
    'Softmax',
    'Step',
    'LeakyReLU',
    'ELU',
    'Softplus',
    'Swish',
    'Square',
    'Fi',
    'Fj',
    'Hi',
    'Hj',
    'Ei',
    'fi',
    'fj',
    'hi',
    'hj',
    'ei',
    'get_activation',
]
