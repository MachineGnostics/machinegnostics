from fontTools.unicodedata import name
import numpy as np
from .layer import Layer

class Dropout(Layer):
    """Inverted dropout. Active only during training; a no-op during inference."""
    def __init__(self, rate=0.5, seed=None, name=None):
        super().__init__(name)
        assert 0.0 <= rate < 1.0
        self.rate = rate
        self.rng = np.random.default_rng(seed)

    def forward(self, x, training=True):
        if training and self.rate > 0.0:
            self.mask = (self.rng.random(x.shape) > self.rate) / (1.0 - self.rate)
            return x * self.mask
        self.mask = None
        return x

    def backward(self, grad_output):
        if self.mask is not None:
            return grad_output * self.mask
        return grad_output