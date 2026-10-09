import numpy as np
from .layer import Layer
from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

class Sigmoid(Layer):
    def forward(self, x, training=True):
        self.output = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1.0 - self.output)
    
class Tanh(Layer):
    def forward(self, x, training=True):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1.0 - self.output ** 2)
    
class ReLU(Layer):
    def forward(self, x, training=True):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)
    
class LeakyReLU(Layer):
    def __init__(self, alpha=0.01, name=None):
        super().__init__(name)
        self.alpha = alpha

    def forward(self, x, training=True):
        self.input = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad_output):
        dx = np.ones_like(self.input)
        dx[self.input <= 0] = self.alpha
        return grad_output * dx

class Softmax(Layer):
    """
    Numerically stable softmax over the last axis.
    NOTE: When paired with categorical cross-entropy loss, prefer computing
    the combined softmax+CE gradient in the loss function itself (see
    losses.py) instead of chaining through this backward(), since the
    combined gradient (y_pred - y_true) is far simpler and more stable.
    Use this standalone Softmax layer for inference or when not paired
    directly with cross-entropy.
    """
    def forward(self, x, training=True):
            shifted = x- np.max(x, axis=-1, keepdims=True)
            exp = np.exp(shifted)
            self.output= exp / np.sum(exp, axis=-1, keepdims=True)
            return self.output

    def backward(self, grad_output):
        # Full Jacobian-vector product, per-sample.
        batch_size, n = self.output.shape
        dx = np.empty_like(grad_output)
        for i in range(batch_size):
            y = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(y) - y @ y.T
            dx[i] = jacobian @ grad_output[i]
        return dx

class Fidelity(Layer):
    """
    Fidelity activation function.
    """
    def __init__(self, S: float|str = 1, name=None):
        super().__init__(name)
        self.S = S

    def _get_fidelity(self, x):
        z_x = np.exp(x - np.median(x))  # avoid division by zero
        gnostic_charc = GnosticsCharacteristics(R=z_x)
        # S = auto
        if isinstance(self.S, str) and self.S == "auto":
            scale_param = ScaleParam()
            q, q1 = gnostic_charc._get_q_q1(S=1)
            fidelity = gnostic_charc._fi(q, q1)
            self.S_local = scale_param._gscale_loc(np.mean(fidelity))
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            hi = gnostic_charc._hi(q, q1)
            fidelity = gnostic_charc._fi(q, q1)
        else:
            self.S_local = self.S
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fidelity = gnostic_charc._fi(q, q1)
            hi = gnostic_charc._hi(q, q1)
        return fidelity, hi

    def forward(self, x, training=True):
        self.input = x
        self.fidelity, self.hi = self._get_fidelity(x)  # Using x as both y_pred and y_true for activation
        return self.fidelity

    def backward(self, grad_output):
        # prime is (-2/s)(fi*hi)
        prime = (-2 * self.fidelity * self.hi + np.finfo(float).eps)/self.S # avoid division by zero
        return grad_output * prime

class Infidelity(Layer):
    """
    Infidelity activation function.
    """
    def __init__(self, S: float|str = 1, name=None):
        super().__init__(name)
        self.S = S

    def _get_infidelity(self, x):
        z_x = np.exp(x - np.median(x))  # avoid division by zero
        gnostic_charc = GnosticsCharacteristics(R=z_x)
        # S = auto
        if isinstance(self.S, str) and self.S == "auto":
            scale_param = ScaleParam()
            q, q1 = gnostic_charc._get_q_q1(S=1)
            fidelity = gnostic_charc._fi(q, q1)
            self.S_local = scale_param._gscale_loc(np.mean(fidelity))
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            infidelity = gnostic_charc._fj(q, q1)
            hi = gnostic_charc._hi(q, q1)
        else:
            self.S_local = self.S
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            infidelity = gnostic_charc._fj(q, q1)
            hi = gnostic_charc._hi(q, q1)
        return infidelity, hi

    def forward(self, x, training=True):
        self.input = x
        self.infidelity, self.hi = self._get_infidelity(x)  # Using x as both y_pred and y_true for activation
        return self.infidelity

    def backward(self, grad_output):
        # prime is (2/s)(fj*hi)
        prime = (2 * self.hi + np.finfo(float).eps)/(self.S * self.infidelity)# avoid division by zero
        return grad_output * prime

class Irrelevance(Layer):
    """
    Irrelevance activation function. (hi gnostic characteristic)
    """
    def __init__(self, S: float|str = 1, name=None):
        super().__init__(name)
        self.S = S

    def _get_irrelevance(self, x):
        z_x = np.exp(x - np.median(x))  # avoid division by zero
        gnostic_charc = GnosticsCharacteristics(R=z_x)
        # S = auto
        if isinstance(self.S, str) and self.S == "auto":
            scale_param = ScaleParam()
            q, q1 = gnostic_charc._get_q_q1(S=1)
            fidelity = gnostic_charc._fi(q, q1)
            self.S_local = scale_param._gscale_loc(np.mean(fidelity))
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            irrelevance = gnostic_charc._hi(q, q1)
            fi = gnostic_charc._fi(q, q1)
        else:
            self.S_local = self.S
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            irrelevance = gnostic_charc._hi(q, q1)
            fi = gnostic_charc._fi(q, q1)
        return irrelevance, fi

    def forward(self, x, training=True):
        self.input = x
        self.irrelevance, self.fi = self._get_irrelevance(x)  # Using x as both y_pred and y_true for activation
        return self.irrelevance

    def backward(self, grad_output):
        # prime is (2/s)(hi*hi)
        prime = (2 / self.S) * self.fi**2  # avoid division by zero
        return grad_output * prime

class Relevance(Layer):
    """
    Relevance activation function. (hj gnostic characteristic)
    """
    def __init__(self, S: float|str = 1, name=None):
        super().__init__(name)
        self.S = S

    def _get_relevance(self, x):
        z_x = np.exp(x - np.median(x))  # avoid division by zero
        gnostic_charc = GnosticsCharacteristics(R=z_x)
        # S = auto
        if isinstance(self.S, str) and self.S == "auto":
            scale_param = ScaleParam()
            q, q1 = gnostic_charc._get_q_q1(S=1)
            fidelity = gnostic_charc._fi(q, q1)
            self.S_local = scale_param._gscale_loc(np.mean(fidelity))
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            relevance = gnostic_charc._hj(q, q1)
            fi = gnostic_charc._fi(q, q1)
        else:
            self.S_local = self.S
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            relevance = gnostic_charc._hj(q, q1)
            fi = gnostic_charc._fi(q, q1)
        return relevance, fi

    def forward(self, x, training=True):
        self.input = x
        self.relevance, self.fi = self._get_relevance(x)  # Using x as both y_pred and y_true for activation
        return self.relevance

    def backward(self, grad_output):
        # prime is (-2/s)(fj*hi)
        prime = (2 / self.S) * self.fi  # avoid division by zero
        return grad_output * prime