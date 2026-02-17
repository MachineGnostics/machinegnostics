"""
Neural network models: ANN, CNN, RNN with gnostic-aware training.

Terminology
-----------
- geometry: selects gnostic geometry used by the training pipeline.
    - 'E' → Estimating (Euclidian)
    - 'Q' → Quantification (Minkowskian)
- gdf: local vs global Gnostic Distribution Function variant applied to weights.
- scale: S parameter (auto or numeric) passed into gnostic criterion.

This follows the regression-style logging and mg_loss integration,
and uses E/Q neuron types to mirror the selected geometry semantics.
"""
import numpy as np
import logging
from typing import List, Tuple
from machinegnostics.magcal.util.logging import get_logger
from .losses import GnosticLoss
from .gnostics_engine import GnosticEngine

class BaseModel:
    def __init__(self,
                 layers: List,
                 optimizer,
                 mg_loss: str = 'hi',
                 data_form: str = 'a',
                 geometry: str = 'E', # 'E' Estimating (Euclidian) and 'Q' is Quantification (Minkowskian)
                 gdf: str = None, # global or local GDF modifier
                 scale: 'str | int | float' = 'auto',
                 early_stopping: bool = True,
                 tolerance: float = 1e-6,
                 verbose: bool = False):
        self.layers = layers
        self.optimizer = optimizer
        self.mg_loss = mg_loss
        self.data_form = data_form
        self.geometry = geometry
        self.gdf = gdf
        self.scale = scale
        self.early_stopping = early_stopping
        self.tolerance = tolerance
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.gloss = GnosticLoss(mg_loss=mg_loss, data_form=data_form, verbose=verbose)
        self.gengine = GnosticEngine(geometry=geometry, gdf=gdf, verbose=verbose, S=self.scale)
        self.history = []
        # Expose latest per-sample gnostic weights to users after fit
        self.gnostic_weights_ = None

    def build(self, input_shape):
        shape = input_shape
        for layer in self.layers:
            layer.build(shape)
            shape = layer.output_shape

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        g = grad
        for layer in reversed(self.layers):
            g = layer.backward(g)
        # collect grads
        grads = {}
        params = {}
        for layer in self.layers:
            for k, v in layer.get_grads().items():
                grads[f"{layer.name}:{k}"] = v
            for k, v in layer.get_params().items():
                params[f"{layer.name}:{k}"] = v
        return params, grads

    def step(self, params, grads):
        self.optimizer.step(params, grads)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32):
        self.build(X.shape)
        n = X.shape[0]
        for epoch in range(1, epochs + 1):
            # forward full-batch to compute gnostic loss & weights
            y_pred = self.forward(X)
            H_loss, rentropy = self.gloss.compute(y_true=y, y_pred=y_pred, scale=self.scale)
            residuals = (y_pred - y).reshape(n, -1).mean(axis=1)
            # without use GDF DistFuncEngine to compute sample weights based on residuals
            weights = self.gengine.compute_sample_weights(residuals)
            # Enforce 1D per-sample weights
            weights = np.asarray(weights).reshape(-1)
            if weights.size != n:
                weights = np.ones(n) / n
            if self.gdf is not None:
                # optional geometry/GDF modifier
                try:
                    weights = self.gengine.apply_gdf_modifier(residuals, weights)
                    weights = np.asarray(weights).reshape(-1)
                    if weights.size != n:
                        weights = np.ones(n) / n
                except Exception:
                    pass
            # store last computed per-sample weights for user access
            self.gnostic_weights_ = weights
            # mini-batch SGD with sample weights
            idx = np.arange(n)
            np.random.shuffle(idx)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                bi = idx[start:end]
                xb = X[bi]
                yb = y[bi]
                wb_raw = weights[bi]
                wb_arr = np.asarray(wb_raw)
                # Reduce any extra dimensions to per-sample weights
                if wb_arr.ndim > 1:
                    # average across all non-sample axes
                    wb_arr = wb_arr.mean(axis=tuple(range(1, wb_arr.ndim)))
                # Safety: ensure length matches batch
                if wb_arr.shape[0] != len(bi):
                    wb_arr = np.ones(len(bi)) / len(bi)
                wb = wb_arr.reshape(-1, 1)
                ypb = self.forward(xb)
                # simple squared error gradient scaled by sample weights
                # grad wrt predictions
                grad_pred = 2.0 * (ypb - yb) * wb
                # backprop
                params, grads = self.backward(grad_pred)
                # apply optimizer
                self.step(params, grads)
            # recompute metrics after epoch
            y_pred = self.forward(X)
            H_loss_new, rentropy_new = self.gloss.compute(y_true=y, y_pred=y_pred, scale=self.scale)
            self.history.append({
                'epoch': epoch,
                'H_loss': H_loss_new,
                'rentropy': rentropy_new
            })
            if self.verbose:
                self.logger.info(f"Epoch {epoch}: H_loss={H_loss_new:.6f}, rentropy={rentropy_new:.6f}")
            # early stopping based on both entropy and H_loss
            if self.early_stopping and epoch > 1:
                prev = self.history[-2]
                if (abs(prev['H_loss'] - H_loss_new) < self.tolerance) or (abs(prev['rentropy'] - rentropy_new) < self.tolerance):
                    if self.verbose:
                        self.logger.info(f"Early stop at epoch {epoch} (converged by H_loss/rentropy)")
                    break

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

class ANN(BaseModel):
    pass

class CNN(BaseModel):
    pass

class RNN(BaseModel):
    pass
