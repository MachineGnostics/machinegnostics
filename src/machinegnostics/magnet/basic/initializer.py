import numpy as np

class Initializer:
    def __call__(self, shape):
        raise NotImplementedError

class Zeros(Initializer):
    """Good for biases. Never use for weight matrices (breaks symmetry)."""
    def __call__(self, shape):
        return np.zeros(shape, dtype=np.float64)

class Ones(Initializer):
    def __call__(self, shape):
        return np.ones(shape, dtype=np.float64)

class RandomNormal(Initializer):
    def __init__(self, mean=0.0, stddev=0.01, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape):
        return self.rng.normal(self.mean, self.stddev, size=shape)
class XavierUniform(Initializer):
    """
    Glorot/Xavier uniform initialization.
    Good default for tanh / sigmoid activations.
    Draws from U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out)).
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape):
        fan_in, fan_out = _compute_fans(shape)
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self.rng.uniform(-limit, limit, size=shape)
class HeNormal(Initializer):
    """
    He initialization. Good default for ReLU / LeakyReLU activations.
    Draws from N(0, sqrt(2 / fan_in)).
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape):
        fan_in, _ = _compute_fans(shape)
        stddev = np.sqrt(2.0 / fan_in)
        return self.rng.normal(0.0, stddev, size=shape)
    
def _compute_fans(shape):
    """
    Works for Dense weight shapes (fan_in, fan_out) and
    Conv2D kernel shapes (out_ch, in_ch, kh, kw).
    """
    if len(shape) == 2:
        fan_in, fan_out = shape
    elif len(shape) == 4:
        out_ch, in_ch, kh, kw = shape
        receptive_field = kh * kw
        fan_in = in_ch * receptive_field
        fan_out = out_ch * receptive_field
    else:
        fan_in = fan_out = np.prod(shape)
    return fan_in, fan_out