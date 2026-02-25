"""Base module abstractions for Magnet layers.

Author: Nirmal Parmar

Notes:
- `Layer` is the common base abstraction for all trainable/non-trainable blocks.
- `Parameter` is a trainable tensor with gradients enabled by default.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from ..tensor import Tensor


class Parameter(Tensor):
    """Trainable tensor container used by optimizers."""

    def __init__(self, data, name=None, requires_grad=True, device="cpu", dtype=None):
        super().__init__(
            data=data,
            name=name,
            requires_grad=requires_grad,
            device=device,
            dtype=dtype if dtype is not None else None,
        )


class Layer:
    """Base class for all model components in Magnet."""

    def __init__(self):
        self.training = True

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError("Layer subclasses must implement forward().")

    def parameters(self) -> List[Parameter]:
        """Recursively collect all trainable parameters."""
        params: List[Parameter] = []
        seen = set()

        def collect(value):
            if isinstance(value, Parameter):
                if id(value) not in seen:
                    seen.add(id(value))
                    params.append(value)
            elif isinstance(value, Layer):
                for p in value.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        params.append(p)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        for attr in self.__dict__.values():
            collect(attr)

        return params

    def get_params(self) -> List[Parameter]:
        """Backward-compatible alias for older code paths."""
        return self.parameters()

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()

    def train(self) -> "Layer":
        self.training = True
        for child in self.children():
            child.train()
        return self

    def eval(self) -> "Layer":
        self.training = False
        for child in self.children():
            child.eval()
        return self

    def children(self) -> List["Layer"]:
        children: List[Layer] = []
        for value in self.__dict__.values():
            if isinstance(value, Layer):
                children.append(value)
            elif isinstance(value, (list, tuple)):
                children.extend([item for item in value if isinstance(item, Layer)])
            elif isinstance(value, dict):
                children.extend([item for item in value.values() if isinstance(item, Layer)])
        return children