"""Weight initializers for MAGNET."""

from .base import Initializer
from .glorot_normal import GlorotNormal
from .glorot_uniform import GlorotUniform
from .he_normal import HeNormal
from .he_uniform import HeUniform
from .normal import Normal
from .ones import Ones
from .uniform import Uniform
from .zeros import Zeros

# Backward-compatible aliases kept in the flat MAGNET namespace.
RandomNormal = Normal
XavierUniform = GlorotUniform
XavierNormal = GlorotNormal


def get_initializer(name_or_callable=None):
    """Return an initializer instance from a name or callable specification.

    Parameters
    ----------
    name_or_callable : str or callable or Initializer or None, optional
        Initializer name, custom callable, initializer instance, or ``None``.
        ``None`` returns the default :class:`GlorotUniform` initializer.

    Returns
    -------
    callable
        A callable initializer.

    Raises
    ------
    ValueError
        If the initializer name is unknown.
    TypeError
        If the specification type is unsupported.
    """
    if name_or_callable is None:
        return GlorotUniform()

    if isinstance(name_or_callable, Initializer):
        return name_or_callable

    if callable(name_or_callable) and not isinstance(name_or_callable, str):
        return name_or_callable

    if isinstance(name_or_callable, str):
        key = name_or_callable.replace("-", "_").lower()
        registry = {
            "glorot_uniform": GlorotUniform(),
            "xavier_uniform": GlorotUniform(),
            "xavieruniform": GlorotUniform(),
            "glorot_normal": GlorotNormal(),
            "xavier_normal": GlorotNormal(),
            "xaviernormal": GlorotNormal(),
            "he_uniform": HeUniform(),
            "heuniform": HeUniform(),
            "he_normal": HeNormal(),
            "henormal": HeNormal(),
            "normal": Normal(),
            "random_normal": Normal(),
            "randomnormal": Normal(),
            "uniform": Uniform(),
            "zeros": Zeros(),
            "ones": Ones(),
        }
        if key not in registry:
            raise ValueError(f"Unknown initializer: {name_or_callable}")
        return registry[key]

    raise TypeError(f"Unsupported initializer specification: {type(name_or_callable)!r}")


__all__ = [
    "Initializer",
    "GlorotUniform",
    "GlorotNormal",
    "HeUniform",
    "HeNormal",
    "Normal",
    "Uniform",
    "Zeros",
    "Ones",
    "RandomNormal",
    "XavierUniform",
    "XavierNormal",
    "get_initializer",
]
