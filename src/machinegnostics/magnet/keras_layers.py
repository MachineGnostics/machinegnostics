import numpy as np

# Local activation implementations to avoid circular/package import issues
class _Linear:
    def forward(self, x):
        return x
    def derivative(self, x):
        return np.ones_like(x)

class _ReLU:
    def forward(self, x):
        return np.maximum(0, x)
    def derivative(self, x):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))

class _Sigmoid:
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    def derivative(self, x):
        f = self.forward(x)
        return f * (1.0 - f)

class _Tanh:
    def forward(self, x):
        return np.tanh(x)
    def derivative(self, x):
        return 1.0 - np.square(np.tanh(x))

class _SoftPlus:
    def forward(self, x):
        return np.log1p(np.exp(x))
    def derivative(self, x):
        return 1.0 / (1.0 + np.exp(-x))


class Dense:
    """
    Keras-like Dense layer for numpy-based networks.

    Parameters
    - units: int
    - activation: str | None ('linear','relu','sigmoid','tanh','softplus','quadratic')
    - use_bias: bool
    - name: str | None
    - use_gnostic: bool (alias: gnostic_weights)
    - geometry: str | None ('E'|'Q') for layer-level override
    - gdf: str | None ('global'|'local'|'egdf'|'eldf'|'qgdf'|'qldf')
    - gdf_influence: float in [0,1] to mix GDF modifier into weights
    """

    def __init__(self,
                 units: int,
                 activation: str | None = 'linear',
                 use_bias: bool = True,
                 name: str | None = None,
                 use_gnostic: bool = True,
                 gnostic_weights: bool | None = None,
                 geometry: str | None = None,
                 gdf: str | None = None,
                 gdf_influence: float = 0.0):
        self.units = int(units)
        self.activation_name = activation or 'linear'
        self.use_bias = bool(use_bias)
        self.name = name or f"dense_{id(self)%10000}"
        # gnostic controls
        self.use_gnostic = bool(gnostic_weights if gnostic_weights is not None else use_gnostic)
        self.geometry = geometry
        self.gdf = gdf
        self.gdf_influence = float(gdf_influence)

        # params set in build
        self.W = None
        self.b = None
        self.input_dim = None
        self.output_shape = None

        # caches for backward
        self._x = None
        self._z = None

        # activation dispatch
        self._act = self._get_activation(self.activation_name)

    def _get_activation(self, name: str):
        if name is None or name == 'linear':
            return _Linear()
        if name == 'relu':
            return _ReLU()
        if name == 'sigmoid':
            return _Sigmoid()
        if name == 'tanh':
            return _Tanh()
        if name == 'softplus':
            return _SoftPlus()
        if name == 'quadratic':
            # inline quadratic to avoid circular import
            class Quadratic:
                def __call__(self, x):
                    return x * x
                def forward(self, x):
                    return x * x
                def derivative(self, x):
                    return 2.0 * x
            return Quadratic()
        raise ValueError(f"Unsupported activation: {name}")

    def build(self, input_shape):
        # input_shape like (n_samples, n_features)
        if isinstance(input_shape, (tuple, list)):
            self.input_dim = int(input_shape[-1])
        elif hasattr(input_shape, 'shape'):
            self.input_dim = int(input_shape.shape[-1])
        else:
            raise ValueError("Invalid input_shape for Dense.build")
        # Xavier-like init
        limit = np.sqrt(6.0 / (self.input_dim + self.units))
        self.W = np.random.uniform(-limit, limit, size=(self.input_dim, self.units)).astype(np.float64)
        if self.use_bias:
            self.b = np.zeros((1, self.units), dtype=np.float64)
        self.output_shape = (None, self.units)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.W is None:
            self.build(x.shape)
        self._x = np.asarray(x)
        z = self._x.dot(self.W)
        if self.use_bias:
            z = z + self.b
        self._z = z
        return self._act.forward(z)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # grad_out is dL/dy where y = activation(z)
        dz = grad_out * self._act.derivative(self._z)
        # keep last layer error signal for gnostic backward scaling
        self._last_error = dz
        # grads
        dW = self._x.T.dot(dz)
        if self.use_bias:
            db = dz.sum(axis=0, keepdims=True)
        else:
            db = None
        # propagate to previous layer: dL/dx = dz * W^T
        grad_in = dz.dot(self.W.T)
        # stash grads for optimizer access
        self._grads = {'w': dW}
        if db is not None:
            self._grads['b'] = db
        return grad_in

    def get_params(self):
        params = {'w': self.W}
        if self.use_bias:
            params['b'] = self.b
        return params

    def get_grads(self):
        return getattr(self, '_grads', {})
