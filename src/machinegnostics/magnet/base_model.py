"""
Magnet Models — Architecture and Training

Developer Notes (Author: Nirmal Parmar, Machine Gnostics)
- Architecture breakdown aligns with ML library design:
    [BaseModel] → [BaseMethods mixin] → [BaseCalculations] → [HistoryTracker] → [Public classes]
- Public methods are concise; internal helpers start with `_internal_...`.
- TensorFlow-like API: `Sequential` (alias to ANN), `Dense`, `Conv2D`, `SimpleRNN`.
- Gnostic weights are computed from residuals and multiplied with regular gradients.
    Per-layer gnostic settings (geometry, gdf, use_gnostic) are supported and
    combined by averaging per-layer weights, normalized.

General Notes
- geometry: 'E' (Estimating/Euclidian) or 'Q' (Quantification/Minkowskian)
- gdf: 'local' or 'global' variant of GDF modifier
- scale: S parameter (auto or numeric) passed to the gnostic criterion
"""
import numpy as np
import logging
from typing import List, Tuple
from machinegnostics.magcal.util.logging import get_logger
from .losses import GnosticLoss
from .gnostics_engine import GnosticEngine
from machinegnostics.magcal.util.narwhals_df import narwhalify


class HistoryTracker:
    """Track per-epoch training history in a list-like structure.

    Provides append/access helpers while staying compatible with
    existing code expecting list semantics (supports `__getitem__`).
    """
    def __init__(self):
        self.records: List[dict] = []

    def append(self, epoch: int, H_loss: float, rentropy: float):
        self.records.append({'epoch': epoch, 'H_loss': H_loss, 'rentropy': rentropy})

    def __getitem__(self, idx):
        return self.records[idx]

    def to_list(self):
        return list(self.records)

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
        self.history = HistoryTracker()
        # Expose latest per-sample gnostic weights to users after fit
        self.gnostic_weights_ = None

    def build(self, input_shape):
        shape = input_shape
        for layer in self.layers:
            layer.build(shape)
            shape = layer.output_shape

    # Internal/public method split
    def _internal_forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _internal_backward(self, grad):
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

    def _internal_compute_sample_weights(self, residuals: np.ndarray) -> np.ndarray:
        """Compute per-sample weights, combining per-layer gnostic configs if present.

        Strategy:
        - For each layer with `use_gnostic=True`, create a temporary engine
          using layer-level `geometry`/`gdf` (falling back to model defaults)
          and compute weights from residuals.
        - If no layer opts in, use model-level engine.
        - Average the resulting weight vectors elementwise and normalize.
        """
        n = residuals.shape[0]
        weights_list = []
        for lyr in self.layers:
            if getattr(lyr, 'use_gnostic', True):
                geom = getattr(lyr, 'geometry', None) or self.geometry
                gdf = getattr(lyr, 'gdf', None) or self.gdf
                eng = GnosticEngine(geometry=geom, gdf=gdf, verbose=self.verbose, S=self.scale)
                w = np.asarray(eng.compute_sample_weights(residuals)).reshape(-1)
                if w.size == n:
                    weights_list.append(w)
        if not weights_list:
            w = np.asarray(self.gengine.compute_sample_weights(residuals)).reshape(-1)
            if w.size != n:
                w = np.ones(n) / n
            return w
        W = np.stack(weights_list, axis=1)
        w_avg = W.mean(axis=1)
        s = w_avg.sum()
        if s <= 0 or not np.isfinite(s):
            return np.ones(n) / n
        return w_avg / s

    @narwhalify
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32):
        self.build(X.shape)
        n = X.shape[0]
        for epoch in range(1, epochs + 1):
            # forward full-batch to compute gnostic loss & weights
            y_pred = self._internal_forward(X)
            H_loss, rentropy = self.gloss.compute(y_true=y, y_pred=y_pred, scale=self.scale)
            residuals = (y_pred - y).reshape(n, -1).mean(axis=1)
            # combine per-layer gnostic weights (or fallback to model-level)
            weights = self._internal_compute_sample_weights(residuals)
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
                ypb = self._internal_forward(xb)
                # simple squared error gradient scaled by sample weights
                # grad wrt predictions
                grad_pred = 2.0 * (ypb - yb) * wb
                # backprop
                params, grads = self._internal_backward(grad_pred)
                # apply optimizer
                self.step(params, grads)
            # recompute metrics after epoch
            y_pred = self._internal_forward(X)
            H_loss_new, rentropy_new = self.gloss.compute(y_true=y, y_pred=y_pred, scale=self.scale)
            self.history.append(epoch=epoch, H_loss=H_loss_new, rentropy=rentropy_new)
            if self.verbose:
                self.logger.info(f"Epoch {epoch}: H_loss={H_loss_new:.6f}, rentropy={rentropy_new:.6f}")
            # early stopping based on both entropy and H_loss
            if self.early_stopping and epoch > 1:
                prev = self.history[-2]
                if (abs(prev['H_loss'] - H_loss_new) < self.tolerance) or (abs(prev['rentropy'] - rentropy_new) < self.tolerance):
                    if self.verbose:
                        self.logger.info(f"Early stop at epoch {epoch} (converged by H_loss/rentropy)")
                    break

    @narwhalify
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._internal_forward(X)

class ANN(BaseModel):
    """Alias model matching TensorFlow-like `Sequential` semantics."""
    pass

class Sequential(ANN):
    """TensorFlow-like API alias for ANN."""
    pass

class CNN(BaseModel):
    pass

class RNN(BaseModel):
    pass
