"""Stochastic gradient descent optimizer for MAGNET."""

from __future__ import annotations

from typing import Iterable

import torch

from ..core.tensor import Tensor
from .base import Optimizer


class SGD(Optimizer):
    """SGD with automatic gradient-based scaling for ``S`` and ``z0``.

    Parameters
    ----------
    learning_rate : float
        Step size used for updates.
    momentum : float
        Momentum factor for the updates.
    lr : float, optional
        Alias for ``learning_rate``.
    gradient_scale_factor : float, optional
        Factor to scale the gradients for MAGNET special parameters.
    verbose : bool, optional
        Enable debug logging for the optimizer instance.

    Attributes
    ----------
    momentum : float
        Momentum factor for the updates.
    _velocity : dict
        Dictionary storing the velocity for each parameter.
    
    Notes
    -----
    The SGD optimizer updates parameters using the gradient of the loss function with respect to the parameters.
    It can optionally use momentum to accelerate convergence and smooth updates.

    Examples
    --------
    >>> from machinegnostics.magnet import SGD
    >>> optimizer = SGD(learning_rate=0.01, momentum=0.9)
    >>> optimizer.step(params)  # where `params` is an iterable of Tensor objects
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        lr: float | None = None,
        gradient_scale_factor: float = 1.0,
        verbose: bool = False,
    ):
        """Create an SGD optimizer.

        Parameters
        ----------
        learning_rate:
            Step size used for updates.
        momentum:
            Momentum factor for the updates.
        lr:
            Alias for ``learning_rate``.
        gradient_scale_factor:
            Factor to scale the gradients for MAGNET special parameters.
        verbose:
            Enable debug logging for the optimizer instance.

        Notes
        -----
        The SGD optimizer updates parameters using the gradient of the loss function with respect to the parameters.
        It can optionally use momentum to accelerate convergence and smooth updates.

        Examples
        --------
        >>> from machinegnostics.magnet import SGD
        >>> optimizer = SGD(learning_rate=0.01, momentum=0.9)
        >>> optimizer.step(params)  # where `params` is an iterable of Tensor objects
        """
        super().__init__(
            learning_rate=learning_rate,
            lr=lr,
            gradient_scale_factor=gradient_scale_factor,
            verbose=verbose,
        )
        self.momentum = momentum
        self._velocity = {}

    def step(self, params: Iterable[Tensor]) -> None:
        """Update each parameter tensor using SGD with optional momentum.

        Parameters
        ----------
        params : iterable of Tensor
            Parameter tensors to be updated.

        Returns
        -------
        None
        """
        for param in params:
            if param._tensor.grad is None:
                continue
            key = id(param)
            velocity = self._velocity.get(key)
            if velocity is None:
                velocity = torch.zeros_like(param._tensor)
            grad = param._tensor.grad
            lr = self.get_adaptive_lr(self.get_param_name(param), grad)
            with torch.no_grad():
                velocity = self.momentum * velocity - lr * grad
                param._tensor.add_(velocity)
            self._velocity[key] = velocity.detach()
