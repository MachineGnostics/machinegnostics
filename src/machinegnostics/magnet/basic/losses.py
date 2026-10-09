import numpy as np
from machinegnostics.magcal import GnosticsCharacteristics, ScaleParam

class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

class MSE(Loss):
    """Mean Squared Error, for regression."""
    def forward(self, y_pred, y_true):
        self.y_pred, self.y_true= y_pred, y_true
        return np.mean((y_pred- y_true) ** 2)

    def backward(self):
        n = self.y_pred.shape[0]
        return 2.0 * (self.y_pred- self.y_true) / n
    
class BinaryCrossEntropy(Loss):
    """
    Binary cross-entropy. Expects y_pred to already be sigmoid probabilities
    in (0, 1) — i.e. use this after a Sigmoid layer.
    """
    def forward(self, y_pred, y_true, eps=1e-12):
        self.y_pred= np.clip(y_pred, eps, 1- eps)
        self.y_true= y_true
        return -np.mean(y_true * np.log(self.y_pred) + (1- y_true) * np.log(1- self.y_pred))

    def backward(self):
        n = self.y_pred.shape[0]
        return (self.y_pred- self.y_true) / (self.y_pred * (1- self.y_pred) * n)


class _BaseGnosticCharc:
    def __init__(self, S: float|str = 1):
        self.S = S

    def _get_fidelity(self, y_pred, y_true):
        y_diff = y_pred - y_true
        z_y_diff = np.exp(y_diff)  # avoid division by zero
        gnostic_charc = GnosticsCharacteristics(R=z_y_diff)
        # S = auto
        if isinstance(self.S, str) and self.S == "auto":
            scale_param = ScaleParam()
            q, q1 = gnostic_charc._get_q_q1(S=1)
            fidelity = gnostic_charc._fi(q, q1)
            self.S_local = scale_param._gscale_loc(np.mean(fidelity))
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fidelity = gnostic_charc._fi(q, q1)
        else:
            self.S_local = self.S
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fidelity = gnostic_charc._fi(q, q1)
        return fidelity

    def _get_gw(self, y_pred, y_true):
        fi = self._get_fidelity(y_pred, y_true)
        fi2 = fi ** 2
        gw = fi2 / np.sum(fi2 + np.finfo(float).eps)  # avoid division by zero
        return gw

    def _get_rentropy(self, y_pred, y_true):
        y_diff = y_pred - y_true
        z_y_diff = np.exp(y_diff)  # avoid division by zero
        gnostic_charc = GnosticsCharacteristics(R=z_y_diff)
        # S = auto
        if isinstance(self.S, str) and self.S == "auto":
            scale_param = ScaleParam()
            q, q1 = gnostic_charc._get_q_q1(S=1)
            fidelity = gnostic_charc._fi(q, q1)
            self.S_local = scale_param._gscale_loc(np.mean(fidelity))
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fi = gnostic_charc._fi(q, q1)
            fj = gnostic_charc._fj(q, q1)
            hi = gnostic_charc._hi(q, q1)
            rentropy = gnostic_charc._rentropy(fi, fj)
        else:
            self.S_local = self.S
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fi = gnostic_charc._fi(q, q1)
            fj = gnostic_charc._fj(q, q1)
            hi = gnostic_charc._hi(q, q1)
            rentropy = gnostic_charc._rentropy(fi, fj)
        return rentropy, fi, hi

    def _get_information(self, y_pred, y_true):
        y_diff = y_pred - y_true
        z_y_diff = np.exp(y_diff)  # avoid division by zero
        gnostic_charc = GnosticsCharacteristics(R=z_y_diff)
        # S = auto
        if isinstance(self.S, str) and self.S == "auto":
            scale_param = ScaleParam()
            q, q1 = gnostic_charc._get_q_q1(S=1)
            fidelity = gnostic_charc._fi(q, q1)
            self.S_local = scale_param._gscale_loc(np.mean(fidelity))
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fi = gnostic_charc._fi(q, q1)
            hi = gnostic_charc._hi(q, q1)
            p = gnostic_charc._idistfun(hi)
            information = gnostic_charc._info_i(p)
        else:
            self.S_local = self.S
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fi = gnostic_charc._fi(q, q1)
            hi = gnostic_charc._hi(q, q1)
            p = gnostic_charc._idistfun(hi)
            information = gnostic_charc._info_i(p)
        return information, fi, p

    # def _get_fihi(self, y_pred, y_true):
    #     y_diff = y_pred - y_true
    #     z_y_diff = np.exp(y_diff)
    #     gnostic_charc = GnosticsCharacteristics(R=z_y_diff)
    #     q, q1 = gnostic_charc._get_q_q1(S=self.S)
    #     fi = gnostic_charc._fj(q, q1)
    #     hi = gnostic_charc._hi(q, q1)
    #     return fi, hi

    def _get_fihi(self, y_pred, y_true):
        y_diff = y_pred - y_true
        z_y_diff = np.exp(y_diff)
        gnostic_charc = GnosticsCharacteristics(R=z_y_diff)
        # S = auto
        if isinstance(self.S, str) and self.S == "auto":
            scale_param = ScaleParam()
            q, q1 = gnostic_charc._get_q_q1(S=1)
            fi = gnostic_charc._fi(q, q1)
            scale =scale_param._gscale_loc(np.mean(fi))
            self.S_local = scale
            q, q1 = gnostic_charc._get_q_q1(S=scale)
            fi = gnostic_charc._fi(q, q1)
            hi = gnostic_charc._hi(q, q1)
        else:
            self.S_local = self.S
            q, q1 = gnostic_charc._get_q_q1(S=self.S_local)
            fi = gnostic_charc._fi(q, q1)
            hi = gnostic_charc._hi(q, q1)
        return fi, hi, self.S_local

class GnosticFidelity(Loss, _BaseGnosticCharc):
    """Gnostic Fidelity Loss."""

    def __init__(self, S: float|str = 1):
        super().__init__(S=S)
        self.S = S

    def forward(self, y_pred, y_true):
        self.y_pred, self.y_true = y_pred, y_true
        fi, _, _ = self._get_fihi(y_pred, y_true)
        return np.mean(fi)
    
    def backward(self):
        n = self.y_pred.shape[0]
        if isinstance(self.S, str) and self.S == "auto":
            fi, hi, self.S_local = self._get_fihi(self.y_pred, self.y_true)
        else:
            fi, hi, _ = self._get_fihi(self.y_pred, self.y_true)
        grad = -(2 / self.S_local) * fi * hi
        return grad

class GnosticInfidelity(Loss, _BaseGnosticCharc):
    """Gnostic Infidelity Loss."""

    def __init__(self, S: float|str = 1):
        super().__init__(S=S)
        self.S = S

    def forward(self, y_pred, y_true):
        self.y_pred, self.y_true = y_pred, y_true
        fi, _, _ = self._get_fihi(y_pred, y_true)
        infidelity = 1 / (fi + 1e-21)  # avoid division by zero
        return np.mean(infidelity)
    
    def backward(self):
        n = self.y_pred.shape[0]
        if isinstance(self.S, str) and self.S == "auto":
            fi, hi, self.S_local = self._get_fihi(self.y_pred, self.y_true)
        else:  
            fi, hi, _ = self._get_fihi(self.y_pred, self.y_true)
        grad = (2 / self.S_local) * (hi / (fi + np.finfo(float).eps))  # avoid division by zero
        # cap the gradient to avoid extreme values
        grad = np.clip(grad, -1e12, 1e12)
        return grad

class GnosticInformation(Loss, _BaseGnosticCharc):
    """Gnostic Information Loss."""

    def __init__(self, S: float|str = 1):
        super().__init__(S=S)
        self.S = S

    def forward(self, y_pred, y_true):
        self.y_pred, self.y_true = y_pred, y_true
        information = self._get_information(y_pred, y_true)
        return np.mean(information)
    
    def backward(self):
        n = self.y_pred.shape[0]
        if isinstance(self.S, str) and self.S == "auto":
            i, fi, p = self._get_information(self.y_pred, self.y_true)
        else:
            i, fi, p = self._get_information(self.y_pred, self.y_true)
        grad =  (1 / self.S_local) * fi**2 * (-np.log(p) + np.log(1-p))
        return grad

class GnosticResidualEntropy(Loss, _BaseGnosticCharc):
    """Gnostic Residual Entropy Loss."""

    def __init__(self, S: float|str = 1):
        super().__init__(S=S)
        self.S = S

    def forward(self, y_pred, y_true):
        self.y_pred, self.y_true = y_pred, y_true
        rentropy = self._get_rentropy(y_pred, y_true)
        return np.mean(rentropy)
    
    def backward(self):
        n = self.y_pred.shape[0]
        if isinstance(self.S, str) and self.S == "auto":
            rentropy, fi, hi = self._get_rentropy(self.y_pred, self.y_true)
        else:
            rentropy, fi, hi = self._get_rentropy(self.y_pred, self.y_true)
        grad = (2 / self.S_local) * (-fi*hi + (hi/fi))
        return grad

class GnosticMSE(Loss, _BaseGnosticCharc):
    """Gnostic Mean Squared Error Loss."""

    def __init__(self, S: float|str = 1):
        super().__init__(S=S)
        self.S = S

    def forward(self, y_pred, y_true):
        self.y_pred, self.y_true= y_pred, y_true
        # gnostic weighted MSE
        self.gw = self._get_gw(y_pred, y_true)
        return np.mean(self.gw * (y_pred- y_true) ** 2)

    def backward(self):
        n = self.y_pred.shape[0]
        return 2.0 * self.gw * (self.y_pred- self.y_true) / n

class GnosticBinaryCrossEntropy(Loss, _BaseGnosticCharc):
    """Gnostic Binary Cross-Entropy Loss."""

    def __init__(self, S: float|str = 1):
        super().__init__(S=S)
        self.S = S

    def forward(self, y_pred, y_true, eps=1e-12):
        self.y_pred= np.clip(y_pred, eps, 1- eps)
        self.y_true= y_true
        # gnostic weighted BCE
        self.gw = self._get_gw(y_pred, y_true)
        return -np.mean(self.gw * (y_true * np.log(self.y_pred) + (1- y_true) * np.log(1- self.y_pred)))

    def backward(self):
        n = self.y_pred.shape[0]
        return self.gw * (self.y_pred- self.y_true) / (self.y_pred * (1- self.y_pred) * n)