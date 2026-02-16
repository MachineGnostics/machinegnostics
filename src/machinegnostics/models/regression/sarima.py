'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2026  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
'''

import numpy as np
import pandas as pd
import itertools
from typing import Tuple, Union, Optional
from machinegnostics.models.base_io_models import DataProcessLayerBase
from machinegnostics.models.regression.base_regressor_history import HistoryRegressorBase
from machinegnostics.magcal import disable_parent_docstring
from machinegnostics.metrics import robr2
import logging
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.magcal.util.narwhals_df import narwhalify

class SARIMA(HistoryRegressorBase, DataProcessLayerBase):
    """
    Gnostic SARIMA (Seasonal ARIMA) with robust iterative reweighting.

    Overview
    --------
    This implementation extends ARIMA to include seasonal components `(P, D, Q, s)` and
    integrates the Machine Gnostics reweighting loop for noise/outlier robustness.
    The model operates on a doubly-differenced series `y''` (seasonal differencing `D`
    followed by regular differencing `d`) and builds a linear additive approximation
    with AR/SAR and MA/SMA terms, optionally with a linear trend.

    Model (on doubly-differenced series)
    ------------------------------------
        y''_t = c
                 + Σ(φ_i · y''_{t-i})                # non-seasonal AR(p)
                 + Σ(Φ_j · y''_{t-j·s})              # seasonal AR(P) with period s
                 + Σ(θ_k · ε_{t-k})                  # non-seasonal MA(q)
                 + Σ(Θ_l · ε_{t-l·s})                # seasonal MA(Q) with period s
                 + ε_t

    where:
    - `y''_t` is the doubly-differenced series (first seasonal `D`, then regular `d`).
    - `s` is the seasonality period (e.g., 12 for monthly data).
    - `φ, Φ` are AR and seasonal AR parameters.
    - `θ, Θ` are MA and seasonal MA parameters.
    - `ε_t` is the residual error.

    Robust Reweighting
    ------------------
    The underlying linear regression is fitted using the Machine Gnostics IO and
    reweighting layers. Weights are learned using gnostic characteristics based on
    irrelevance metrics. The choice of `mg_loss` controls the objective:
    - `hi`: minimize irrelevance of observed data (estimation-focus).
    - `hj`: minimize irrelevance of ideal/reference (quantification-focus).

    Key Features
    ------------
    - Seasonal and non-seasonal orders: `(p, d, q) × (P, D, Q, s)`.
    - Optional trend: `'c'` (constant), `'ct'` (constant + linear time), `'n'` (none).
    - Robust reweighting with `scale='auto'` or a numeric value and `data_form` `'a'`/`'m'`.
    - Convergence control via `max_iter`, `learning_rate`, `tolerance`, `early_stopping`.
    - Training history collection for diagnostics (`_history`).

    Parameters
    ----------
    order : tuple[int, int, int], default=(1, 0, 0)
        Non-seasonal order `(p, d, q)`.
    seasonal_order : tuple[int, int, int, int], default=(0, 0, 0, 0)
        Seasonal order `(P, D, Q, s)`; `s` is the periodicity.
    optimize : bool, default=False
        Placeholder for automatic order selection within `max_order_search`. Currently
        not performing exhaustive model selection.
    max_order_search : tuple[int, int, int], default=(2, 1, 2)
        Limits for automatic non-seasonal order search when `optimize=True`.
    trend : {'c', 'ct', 'n'}, default='c'
        Trend option: constant only, constant+time, or none.
    scale : {'auto', int, float}, default='auto'
        Scaling for gnostic calculations; `'auto'` estimates a suitable scale.
    max_iter : int, default=100
        Maximum number of reweighting iterations.
    learning_rate : float, default=0.1
        Step size used by the reweighting optimizer.
    tolerance : float, default=1e-3
        Convergence tolerance for the reweighting loop.
    mg_loss : {'hi', 'hj'}, default='hi'
        Loss selection for gnostic weighting (see Robust Reweighting).
    early_stopping : bool, default=True
        Stop iterations when progress falls below `tolerance`.
    verbose : bool, default=False
        Enable internal logging; uses `DEBUG` when `True`, `WARNING` otherwise.
    data_form : {'a', 'm'}, default='a'
        Data form for IO conversions: `'a'` additive or `'m'` multiplicative.
    gnostic_characteristics : bool, default=True
        Whether to compute gnostic characteristics in the IO layer.
    history : bool, default=True
        Collect training history in `_history` for diagnostics.

    Attributes
    ----------
    p, d, q : int
        Non-seasonal orders.
    P, D, Q, s : int
        Seasonal orders and seasonality period.
    training_data_raw_ : np.ndarray | None
        Original training series `y`.
    training_data_diff_ : np.ndarray | None
        Series after seasonal and regular differencing.
    training_residuals_ : np.ndarray | None
        Initial residuals estimate used for MA/SMA features.
    _history : list[dict] | None
        Training history entries: `iteration`, `h_loss`, `coefficients`, `rentropy`, `weights`.
    logger : logging.Logger
        Component logger (level controlled by `verbose`).

    Methods Summary
    ---------------
    fit(y, X=None) -> SARIMA
        Fit the model on `y`; builds features on the doubly-differenced series.
    predict(steps=1, future=True) -> np.ndarray
        Forecast `steps` ahead and invert differencing back to the original domain.
    score(y, X=None) -> float
        Robust R² computed on the differenced training series; returns `-inf`
        for non-training data.
    summary() -> None
        Print a human-readable model summary.

    Raises
    ------
    ValueError
        If the series is too short for requested differencing/lags; if residuals
        are required (MA terms present) but unavailable; or when feature creation
        cannot proceed due to insufficient data.

    Notes
    -----
    - Residuals are initialized via a long AR(m) heuristic to support MA/SMA terms.
      Future residuals during forecasting are assumed to be zero.
    - Trend `'ct'` uses a simple time index aligned to the training target length.
    - Logging leverages `machinegnostics.magcal.util.logging.get_logger` and honors
      `verbose` for level selection.
    - History collection is enabled when `history=True`; the first entry is a
      placeholder capturing initial state.

    Example
    -------
    >>> import numpy as np
    >>> from machinegnostics.models import SARIMA
    >>> y = np.sin(np.linspace(0, 4*np.pi, 240)) + 0.1*np.random.randn(240)
    >>> model = SARIMA(order=(2,1,1), seasonal_order=(1,1,1,12), trend='c', verbose=True)
    >>> model.fit(y)
    >>> forecast = model.predict(steps=12)
    >>> r2 = model.score(y)
    >>> model.summary()
    """
    @disable_parent_docstring
    def __init__(self,
                 order: Tuple[int, int, int] = (1, 0, 0),
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 optimize: bool = False,
                 max_order_search: Tuple[int, int, int] = (2, 1, 2),
                 trend: str = 'c',
                 scale: Union[str, int, float] = 'auto',
                 max_iter: int = 100,
                 learning_rate: float = 0.1,
                 tolerance: float = 1e-3,
                 mg_loss: str = 'hi',
                 early_stopping: bool = True,
                 verbose: bool = False,
                 data_form: str = 'a',
                 gnostic_characteristics: bool = True,
                 history: bool = True):
        
        self.order = order
        self.p, self.d, self.q = order
        
        self.seasonal_order = seasonal_order
        self.P, self.D, self.Q, self.s = seasonal_order
        
        self.optimize = optimize
        self.max_order_search = max_order_search
        self.trend = trend
        
        # Initialize base
        super().__init__(
            degree=1, 
            max_iter=max_iter,
            learning_rate=learning_rate,
            tolerance=tolerance,
            mg_loss=mg_loss,
            early_stopping=early_stopping,
            verbose=verbose,
            scale=scale,
            data_form=data_form,
            gnostic_characteristics=gnostic_characteristics,
            history=history
        )
        
        if self.history and isinstance(self._history, bool):
            self._history = []
            self._history.append({
                'iteration': 0,
                'h_loss': None,
                'coefficients': None,
                'rentropy': None,
                'weights': None,
            })

        self.training_data_raw_ = None
        self.training_data_diff_ = None # After applying BOTH differences
        self.training_residuals_ = None
        
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)

    def _difference(self, y: np.ndarray, d: int, s: int = 1) -> np.ndarray:
        """Apply d differencing operations with lag s."""
        if d == 0:
            return y
        res = y
        for _ in range(d):
            # Diff with lag s
            # y_t - y_{t-s}
            # This reduces length by s
            if len(res) <= s:
                raise ValueError(f"Series too short for differencing with lag {s}")
            res = res[s:] - res[:-s]
        return res

    def _inverse_difference(self, 
                          original_data: np.ndarray, 
                          forecast_diff: np.ndarray, 
                          d: int,
                          s: int = 1) -> np.ndarray:
        """
        Revert differencing (lag s).
        """
        if d == 0:
            return forecast_diff
            
        current_forecast = forecast_diff
        
        # Reconstruction needs history.
        # Structure: We need the last 's' values for EACH level of differencing.
        # But simpler: Reconstruct one level at a time.
        
        # We need to compute the history at the specific diff level to get the anchors.
        
        # Example: D=1, s=12. y_t = y_{t-12} + y'_t
        # To predict next step, need y_{T+1-12}.
        
        # We handle D layers iteratively.
        working_history = original_data.copy()
        
        # We need to store snapshots of history before each diff to serve as anchors
        history_snapshots = []
        
        # Forward pass to generate intermediate histories
        temp = working_history
        for i in range(d):
            history_snapshots.append(temp)
            # diff
            if len(temp) > s:
                temp = temp[s:] - temp[:-s]
            else:
                 # Should not happen if fit worked
                 pass
                 
        # Reverse pass to integrate
        # current_forecast is at level D
        # we want to add it to level D-1
        
        for i in reversed(range(d)):
            prev_level_history = history_snapshots[i]
            # y_{t} = y_{t-s} + y'_{t}
            # We need the last 's' values of prev_level_history to start bootstrapping
            
            anchors = list(prev_level_history[-s:])
            
            # Perform integration
            level_restored = []
            for k in range(len(current_forecast)):
                # y_{t} = y'_{t} + y_{t-s}
                val = current_forecast[k] + anchors[k] # anchor[k] is y_{t-s} relative to current step
                level_restored.append(val)
                anchors.append(val) # Add new val to usage for s steps later
                
            current_forecast = np.array(level_restored)
            
        return current_forecast

    def _create_features(self, y: np.ndarray, residuals: Optional[np.ndarray] = None):
        """
        Create (AR + SAR + MA + SMA) feature matrix.
        """
        n_samples = len(y)
        
        # Lags needed
        # Regular AR: p
        # Seasonal AR: P * s
        # Regular MA: q
        # Seasonal MA: Q * s
        
        max_ar_lag = max(self.p, self.P * self.s) if (self.p > 0 or self.P > 0) else 0
        max_ma_lag = max(self.q, self.Q * self.s) if (self.q > 0 or self.Q > 0) else 0
        
        max_lag = max(max_ar_lag, max_ma_lag)
        
        if self.q > 0 or self.Q > 0:
            if residuals is None:
                raise ValueError("Residuals required for MA terms")
            max_lag = max(max_lag, max_ma_lag)

        if n_samples <= max_lag:
             raise ValueError(f"Not enough data ({n_samples}) for max lag {max_lag} (Seasonality s={self.s}).")

        X_features = []
        y_target = []
        
        for i in range(max_lag, n_samples):
            features = []
            
            # --- Regular AR (1..p) ---
            if self.p > 0:
                features.extend(y[i-self.p : i][::-1])
                
            # --- Seasonal AR (s, 2s, .. Ps) ---
            if self.P > 0:
                for k in range(1, self.P + 1):
                    features.append(y[i - k * self.s])
                    
            # --- Regular MA (1..q) ---
            if self.q > 0:
                features.extend(residuals[i-self.q : i][::-1])
                
            # --- Seasonal MA (s, 2s, .. Qs) ---
            if self.Q > 0:
                for k in range(1, self.Q + 1):
                    features.append(residuals[i - k * self.s])
            
            X_features.append(features)
            y_target.append(y[i])
            
        return np.array(X_features), np.array(y_target)

    def _estimate_initial_residuals(self, y_diff: np.ndarray) -> np.ndarray:
        """
        Estimate residuals using a high-order AR model suitable for seasonality.
        We approximate using a long regular AR, ensuring it covers seasonal lags.
        """
        # Heuristic: Cover at least one or two seasons plus regular dynamics
        m = max(self.p, self.q) + self.s * max(self.P, self.Q) + self.s
        
        n = len(y_diff)
        if len(y_diff) < 2 * m + 10:
             # If short data, reduce m conservatively
             m = max(1, len(y_diff) // 3)
             
        if n <= m:
            return np.zeros(n)
            
        # OLS AR(m)
        X_long = []
        y_long = []
        for i in range(m, n):
            X_long.append(y_diff[i-m:i][::-1])
            y_long.append(y_diff[i])
            
        X_long = np.array(X_long)
        y_long = np.array(y_long)
        
        X_long_bias = np.c_[np.ones(X_long.shape[0]), X_long]
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_long_bias, y_long, rcond=None)
            y_pred_valid = X_long_bias @ coeffs
            residuals_valid = y_long - y_pred_valid
            return np.concatenate([np.zeros(m), residuals_valid])
        except:
            return np.zeros(n)

    @narwhalify
    def fit(self, y: np.ndarray, X=None):
        """Fit Gnostic SARIMA."""
        self.logger.info("Starting fit process for Gnostic SARIMA.")
        
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        y = np.array(y).flatten()
        
        self.training_data_raw_ = y
        
        # 1. Seasonal Differencing
        y_sdiff = self._difference(y, self.D, self.s)
        
        # 2. Regular Differencing
        y_final_diff = self._difference(y_sdiff, self.d, 1)
        self.training_data_diff_ = y_final_diff
        
        # 3. Residuals
        residuals = None
        if self.q > 0 or self.Q > 0:
            residuals = self._estimate_initial_residuals(y_final_diff)
            self.training_residuals_ = residuals
            
        # 4. Features
        try:
            X_train, y_train = self._create_features(y_final_diff, residuals)
        except ValueError as e:
            self.logger.error(f"Fit failed during feature creation: {e}")
            raise
            
        # 5. Trend
        if self.trend == 'ct':
            # Add time column
            # Be careful aligning time with X_train
            # y_train corresponds to indices in y_final_diff
            # We need absolute time if trend is global? 
            # Usually simple index 0..N for the regression works
            t_idx = np.arange(len(y_train)).reshape(-1, 1)
            X_train = np.hstack([X_train, t_idx])
            
        # 6. Fit
        Xc, yc = super()._fit_io(X_train, y_train)
        super()._fit(Xc, yc)
        
        return self

    @narwhalify
    def predict(self, steps: int = 1, future: bool = True) -> np.ndarray:
        """Forecast."""
        if self.training_data_diff_ is None:
            raise ValueError("Model not fitted.")
            
        # Histories
        history_diff = list(self.training_data_diff_)
        history_res = list(self.training_residuals_) if self.training_residuals_ is not None else []
        
        forecast_diff = []
        current_t = len(self.training_data_diff_) # For 'ct' trend relative logic
        
        # Recursion
        for _ in range(steps):
            features = []
            
            # Regular AR
            if self.p > 0:
                features.extend(history_diff[-self.p:][::-1])
                
            # Seasonal AR
            if self.P > 0:
                for k in range(1, self.P + 1):
                    idx = len(history_diff) - k * self.s
                    if idx < 0:
                        # Should not happen if training data was sufficient
                        features.append(0) 
                    else:
                        features.append(history_diff[idx])
                        
            # Regular MA
            if self.q > 0:
                # Resid is 0 for future
                # Check if we have history
                needed = self.q
                avail = len(history_res)
                # We need residuals at -1, -2 ... -q
                # Append 0s for previous forecast steps
                # Or we can just maintain history_res with 0s appended as we go
                chunk = history_res[-self.q:][::-1]
                # If chunk is smaller than self.q (unlikely if padded), fill 0s
                if len(chunk) < self.q:
                     chunk = [0.0]*(self.q - len(chunk)) + chunk
                features.extend(chunk)
                
            # Seasonal MA
            if self.Q > 0:
                for k in range(1, self.Q + 1):
                    idx = len(history_res) - k * self.s
                    if idx < 0:
                        features.append(0.0)
                    else:
                        features.append(history_res[idx])
                        
            feat_vector = np.array(features).reshape(1, -1)
            
            if self.trend == 'ct':
                feat_vector = np.hstack([feat_vector, [[current_t]]])
            
            # Predict
            feat_vec_trans = super()._predict_io(feat_vector)
            pred_val = super()._predict(feat_vec_trans)[0]
            
            forecast_diff.append(pred_val)
            history_diff.append(pred_val)
            
            if self.q > 0 or self.Q > 0:
                history_res.append(0.0) # Assume 0 residual for prediction
                
            current_t += 1
            
        # Inverse Transforms
        # 1. Inverse Regular Diff d
        forecast_regular = self._inverse_difference(
            self._difference(self.training_data_raw_, self.D, self.s), # History for regular diff layer is the Seasonally Diffed series
            np.array(forecast_diff), 
            self.d, 
            1
        )
        
        # 2. Inverse Seasonal Diff D
        forecast_final = self._inverse_difference(
            self.training_data_raw_, # History for seasonal layer is raw
            forecast_regular,
            self.D,
            self.s
        )
        
        return forecast_final

    @narwhalify
    def score(self, y: np.ndarray, X=None) -> float:
        """
        Score using Robust R2 on the fully differenced series (in-sample only).
        """
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        y = np.array(y).flatten()
        
        # If y is same as training data...
        if np.array_equal(y, self.training_data_raw_):
            # Calculate in-sample robust R2 of y_diff (final)
            
            # Recreate the final diff series
            # 1. Seasonal Differencing
            y_sdiff = self._difference(y, self.D, self.s)
            # 2. Regular Differencing
            y_final_diff = self._difference(y_sdiff, self.d, 1)
            
            # Recalculate features
            residuals = self.training_residuals_
            X_train, y_train = self._create_features(y_final_diff, residuals)
            
            if self.trend == 'ct':
                 t_idx = np.arange(len(y_train)).reshape(-1, 1)
                 X_train = np.hstack([X_train, t_idx])
            
            # Predict
            X_trans = super()._predict_io(X_train)
            y_pred_diff = super()._predict(X_trans)
            
            return robr2(y_train, y_pred_diff)
            
        else:
            self.logger.warning("Scoring on new data not fully supported for recursive SARIMA models yet. Returning -inf.")
            return -np.inf

    def summary(self):
        """Show model summary."""
        print(f"Gnostic SARIMA(p={self.p}, d={self.d}, q={self.q})x(P={self.P}, D={self.D}, Q={self.Q}, s={self.s})")
        print(f"Trend: '{self.trend}'")
        print(f"Iterations: {len(self._history) if self._history else 'N/A'}")
        if self.weights is not None:
             print("Final Weights Stats:")
             print(f"  Min: {np.min(self.weights):.4f}")
             print(f"  Max: {np.max(self.weights):.4f}")
             print(f"  Mean: {np.mean(self.weights):.4f}")

    def __repr__(self):
        return f"SARIMA(order={self.order}, seasonal_order={self.seasonal_order}, trend='{self.trend}')"
