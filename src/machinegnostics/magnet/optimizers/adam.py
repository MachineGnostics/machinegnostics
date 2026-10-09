"""Adam optimizer for MAGNET."""

from __future__ import annotations

from typing import Iterable

import torch

from ..core.tensor import Tensor
from .base import Optimizer


class Adam(Optimizer):
    """Adam optimizer with automatic scaling for MAGNET special parameters.
    
    Parameters
    ----------
    learning_rate : float
        Step size used for updates.
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the second moment estimates.
    epsilon : float
        Small constant that prevents division by zero.
    lr : float, optional
        Alias for ``learning_rate``.
    gradient_scale_factor : float, optional
        Factor to scale the gradients for MAGNET special parameters.
    verbose : bool, optional
        Enable debug logging for the optimizer instance.

    Attributes
    ----------
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the second moment estimates.
    epsilon : float
        Small constant that prevents division by zero.
    _m : dict
        Dictionary storing the first moment estimates for each parameter.
    _v : dict
        Dictionary storing the second moment estimates for each parameter.
    _t : int
        Time step counter.

    Notes
    -----
    The Adam optimizer combines the advantages of both AdaGrad and RMSProp and is well-suited for problems with large datasets and high-dimensional parameter spaces.
    It maintains per-parameter learning rates that are adapted based on the first and second moments of the gradients.

    Examples
    --------
    >>> from machinegnostics.magnet import Adam
    >>> optimizer = Adam(learning_rate=0.001)
    >>> optimizer.step(params)  # where `params` is an iterable of Tensor objects
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        lr: float | None = None,
        gradient_scale_factor: float = 1.0,
        verbose: bool = False,
    ):
        """Create an Adam optimizer.

        Parameters
        ----------
        learning_rate:
            Step size used for updates.
        beta1:
            Exponential decay rate for the first moment estimates.
        beta2:
            Exponential decay rate for the second moment estimates.
        epsilon:
            Small constant that prevents division by zero.
        lr:
            Alias for ``learning_rate``.
        gradient_scale_factor:
            Factor to scale the gradients for MAGNET special parameters.
        verbose:
            Enable debug logging for the optimizer instance.

        Notes
        -----
        The Adam optimizer combines the advantages of both AdaGrad and RMSProp and is well-suited for problems with large datasets and high-dimensional parameter spaces.
        It maintains per-parameter learning rates that are adapted based on the first and second moments of the gradients.

        Examples
        --------
        >>> from machinegnostics.magnet import Adam
        >>> optimizer = Adam(learning_rate=0.001)
        >>> optimizer.step(params)  # where `params` is an iterable of Tensor objects
        """
        super().__init__(
            learning_rate=learning_rate,
            lr=lr,
            gradient_scale_factor=gradient_scale_factor,
            verbose=verbose,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m = {}
        self._v = {}
        self._t = 0

    def step(self, params: Iterable[Tensor]) -> None:
        """Update each parameter tensor using the Adam rule.

        Parameters
        ----------
        params : iterable of Tensor
            Parameter tensors to be updated.

        Returns
        -------
        None
        """
        self._t += 1
        for param in params:
            if param._tensor.grad is None:
                continue
            key = id(param)
            m = self._m.get(key)
            v = self._v.get(key)
            if m is None:
                m = torch.zeros_like(param._tensor)
            if v is None:
                v = torch.zeros_like(param._tensor)

            grad = param._tensor.grad
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)

            m_hat = m / (1.0 - self.beta1 ** self._t)
            v_hat = v / (1.0 - self.beta2 ** self._t)
            lr = self.get_adaptive_lr(self.get_param_name(param), grad)

            with torch.no_grad():
                param._tensor.add_(-lr * m_hat / (torch.sqrt(v_hat) + self.epsilon))

            self._m[key] = m.detach()
            self._v[key] = v.detach()
