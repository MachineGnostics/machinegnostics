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

class ARIMA(HistoryRegressorBase, DataProcessLayerBase):
    """
    Gnostic ARIMA (AutoRegressive Integrated Moving Average) with Robust Iterative Reweighting.

    This model implements an ARIMA(p, d, q) process for time series forecasting,
    empowered by Mathematical Gnostics weights.

    Model:
        y'_t = c + Σ(φ_i * y'_{t-i}) + Σ(θ_j * ε_{t-j}) + ε_t

    Where:

    y'_t : d-th differenced series
    φ    : Autoregressive parameters
    θ    : Moving Average parameters
    ε_t  : Residual error term

    Key Features
    ------------
    - **Robust Forecasting**: Resilient to outliers via Gnostic Weights.
    - **Integrated (d)**: Supports differencing for non-stationary data.
    - **Moving Average (q)**: Supports moving average terms using Hannan-Rissanen estimation.
    - **Trend Support**: Supports constant ('c') and constant+linear ('ct') trends.

    Parameters
    ----------
    order : tuple, default=(1, 0, 0)
        The (p, d, q) order of the model for the number of AR parameters,
        differences, and MA parameters.
    trend : str, {'c', 'ct', 'n'}, default='c'
        Trend to include in the model:
        - 'c': Constant (bias/intercept).
        - 'ct': Constant and linear time trend.
        - 'n': No trend.
    scale : {'auto', int, float}, default='auto'
    optimize : bool, default=False
        If True, the model automatically selects the best (p, d, q) order 
        from the range (1..max_p, 0..max_d, 0..max_q) minimizing RMSE on a validation set.
    max_order_search : tuple, default=(5, 1, 5)
        The maximum (p, d, q) values to search when optimize=True.
        Scaling method or value for gnostic calculations.
    max_iter : int, default=100
        Maximum number of gnostic reweighting iterations.
    tolerance : float, default=1e-3
        Convergence tolerance for weights.
    mg_loss : str, default='hi'
        Gnostic loss function ('hi' or 'hj').
    early_stopping : bool, default=True
        Whether to stop training early upon convergence.
    verbose : bool, default=False
        If True, prints training progress.
    history : bool, default=True
        Whether to tracking training history (weights, loss).

    Examples
    --------
    >>> from machinegnostics.models import ARIMA
    >>> # Initialize ARIMA(1, 1, 1)
    >>> model = ARIMA(order=(1, 1, 1), trend='c')
    >>> model.fit(y_train)
    >>> forecast optimize: bool = False,
                 max_order_search: Tuple[int, int, int] = (5, 1, 5),
                 = model.predict(steps=10)
    """
    @disable_parent_docstring
    def __init__(self,
                 order: Tuple[int, int, int] = (1, 0, 0),
                 optimize: bool = False,
                 max_order_search: Tuple[int, int, int] = (5, 1, 5),
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
        self.optimize = optimize
        self.max_order_search = max_order_search
        self.trend = trend
        
        # Initialize base
        # Degree 1 because AR/MA formulation is linear in parameters
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

        self.training_data_raw_ = None  # Original series in original scale
        self.training_data_diff_ = None # Differenced series
        self.training_residuals_ = None # Residuals for MA
        
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)

    def _difference(self, y: np.ndarray, d: int) -> np.ndarray:
        """Apply d differencing operations."""
        if d == 0:
            return y
        res = np.diff(y, n=d)
        return res

    def _inverse_difference(self, 
                          original_data: np.ndarray, 
                          forecast_diff: np.ndarray, 
                          d: int) -> np.ndarray:
        """
        Revert differencing to get actual forecast.
        
        Parameters
        ----------
        original_data : Pre-differencing historical data (needed for last values)
        forecast_diff : Forecasted values in the differenced domain
        d : order of differencing
        """
        if d == 0:
            return forecast_diff
            
        # We need the last d values of the original series (or intermediate series)
        # to reconstruct. 
        # For d=1: y_t = y_{t-1} + y'_t
        # For d>1: Recursive reconstruction
        
        current_forecast = forecast_diff
        
        # We process differencing layers one by one from d down to 1
        # To do this correctly, we conceptually need the history at each level of differencing.
        # Alternatively, we can use np.cumsum and add the appropriate initial value.
        
        # Easier: Reconstruct one level at a time.
        # Level d -> d-1 -> ... -> 0
        
        # We need to re-compute lower order differences of history to get the "last value" at that level.
        # Or simply:
        # y_restored = np.r_[last_value, forecast_diff].cumsum()[1:]
        
        # We need to handle this iteratively for d > 1
        
        # Create a working copy of history that we can difference
        history = original_data.copy()
        
        # We need the last values at each level of differencing.
        # Let's store them.
        last_values = []
        for i in range(d):
            last_values.append(history[-1])
            history = np.diff(history)
            
        # Now reconstruct
        # last_values[0] is y[-1] (level 0)
        # last_values[1] is y'[-1] (level 1)
        # ...
        # last_values[d-1] is y^{(d-1)}[-1]
        
        # We are given forecast at level d. We add to level d-1.
        current_level_forecast = current_forecast
        
        for i in reversed(range(d)):
            last_val = last_values[i]
            # Cumulative sum starts from the last value of the previous level
            # prediction[0] = last_val + current_level_forecast[0]
            # prediction[1] = prediction[0] + current_level_forecast[1]
            
            # np.cumsum of forecast
            csc = np.cumsum(current_level_forecast)
            current_level_forecast = csc + last_val
            
        return current_level_forecast

    def _create_features(self, y: np.ndarray, residuals: Optional[np.ndarray] = None):
        """
        Create (AR + MA) feature matrix.
        
        y: Time series (already differenced if d>0)
        residuals: Estimated residuals (required if q > 0)
        """
        n_samples = len(y)
        
        # Determine start index based on max required lag
        max_lag = self.p
        if self.q > 0:
            if residuals is None:
                raise ValueError("Residuals required for q > 0")
            if len(residuals) != n_samples:
                raise ValueError("Residuals length mismatch")
            max_lag = max(max_lag, self.q)
            
        if n_samples <= max_lag:
            raise ValueError(f"Not enough data samples ({n_samples}) for max lag {max_lag}.")
            
        X_features = []
        y_target = []
        
        for i in range(max_lag, n_samples):
            features = []
            
            # AR Lags: y_{t-1}, ..., y_{t-p}
            if self.p > 0:
                ar_window = y[i-self.p:i][::-1]
                features.extend(ar_window)
                
            # MA Lags: e_{t-1}, ..., e_{t-q}
            if self.q > 0:
                ma_window = residuals[i-self.q:i][::-1]
                features.extend(ma_window)
                
            X_features.append(features)
            y_target.append(y[i])
            
        return np.array(X_features), np.array(y_target)

    def _estimate_initial_residuals(self, y_diff: np.ndarray) -> np.ndarray:
        """
        Estimate residuals using a high-order AR model (Hannan-Rissanen Step 1).
        """
        # Heuristic for long AR order
        m = max(self.p, self.q) + 4
        if len(y_diff) < 2 * m:
             # Fallback if short data
             m = max(1, len(y_diff) // 4)
        
        # We use a temporary standard OLS (via numpy) or simple robust fit for speed/stability
        # Create lags for AR(m)
        n = len(y_diff)
        if n <= m:
            return np.zeros(n) # Cannot estimate
            
        X_long = []
        y_long = []
        for i in range(m, n):
            X_long.append(y_diff[i-m:i][::-1])
            y_long.append(y_diff[i])
            
        X_long = np.array(X_long)
        y_long = np.array(y_long)
        
        # Add intercept column
        X_long_bias = np.c_[np.ones(X_long.shape[0]), X_long]
        
        # Simple LSTSQ
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_long_bias, y_long, rcond=None)
            
            # Predict on full series (padding start with 0 residuals)
            # Cannot predict first m points accurately without history.
            # We assume residuals are 0 for the first m points.
            
            # Predict valid range
            y_pred_valid = X_long_bias @ coeffs
            residuals_valid = y_long - y_pred_valid
            
            # Pad beginning
            residuals = np.concatenate([np.zeros(m), residuals_valid])
            return residuals
            
        except np.linalg.LinAlgError:
            self.logger.warning("Failed to estimate initial residuals. Assuming zeros.")
            return np.zeros(n)

    def _optimize_order(self, y: np.ndarray) -> Tuple[int, int, int]:
        """
        Find optimal (p, d, q) order by minimizing RMSE on validation split.
        """
        import itertools # Local import to ensure availability during autoreload
        self.logger.info(f"Optimizing (p, d, q) with max_order_search={self.max_order_search}")
        
        # Split data (80/20 split)
        n = len(y)
        if n < 20: 
             # Too small to split effectively, return current order or minimal defaults
             self.logger.warning("Data too small for optimization. Using default order.")
             return self.order

        split_point = int(n * 0.8)
        y_train = y[:split_point]
        y_valid = y[split_point:]
        
        max_p, max_d, max_q = self.max_order_search
        
        best_rmse = float('inf')
        best_order = self.order
        
        p_values = range(0, max_p + 1)
        d_values = range(0, max_d + 1)
        q_values = range(0, max_q + 1)
        
        combinations = list(itertools.product(p_values, d_values, q_values))
        
        self.logger.info(f"Grid search space size: {len(combinations)}")
        
        # print(f"DEBUG: Starting optimization. Space size: {len(combinations)}")

        for i, (p, d, q) in enumerate(combinations):
            if p == 0 and q == 0 and self.trend == 'n':
                # Pure noise model with no constant? Likely useless.
                continue

            # self.logger.info(f"Testing order {(p,d,q)}")
            try:
                # Create a temporary model instance
                # We intentionally disable optimization and history for speed/recursion avoidance
                model = self.__class__(
                    order=(p, d, q),
                    optimize=False,
                    max_order_search=self.max_order_search,
                    trend=self.trend,
                    max_iter=10, # Reduced iterations for speed during search
                    learning_rate=self.learning_rate,
                    tolerance=self.tolerance * 10, # Looser tolerance for search
                    mg_loss=self.mg_loss,
                    early_stopping=self.early_stopping,
                    verbose=False,
                    scale=self.scale,
                    data_form=self.data_form,
                    gnostic_characteristics=self.gnostic_characteristics,
                    history=False
                )
                
                model.fit(y_train)
                
                # Setup validation data in model for prediction
                # In standard ARIMA, we predict steps ahead from end of training.
                pred = model.predict(steps=len(y_valid), future=True)
                
                rmse = np.sqrt(np.mean((y_valid - pred)**2))
                
                # Valid RMSE check
                if np.isnan(rmse) or np.isinf(rmse):
                    continue

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_order = (p, d, q)
                    self.logger.info(f"New best: {best_order} with RMSE {best_rmse:.4f}")
                    
            except Exception as e:
                # Helpful for debugging why specific orders fail
                self.logger.warning(f"Order {(p,d,q)} failed: {e}") 
                continue
        
        self.logger.info(f"Optimization finished. Best order: {best_order}, RMSE: {best_rmse:.4f}")
        return best_order

    @narwhalify
    def fit(self, y: np.ndarray, X=None):
        """
        Fit the ARIMA model.
        
        Parameters
        ----------
        y : array-like or series
            Target time series. Accepts NumPy arrays, Pandas Series/DataFrame column.
        X : Ignored
        """
        # print(f"DEBUG: Entering fit. Optimize={self.optimize}")
        self.logger.info("Starting fit process for Gnostic ARIMA.")
        
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        y = np.array(y).flatten()
        
        # Optimization logic
        if self.optimize:
            self.logger.info("Running hyperparameter optimization...")
            best_order = self._optimize_order(y)
            self.order = best_order
            self.p, self.d, self.q = best_order
            self.logger.info(f"Updated model order to {self.order}")
            
        self.training_data_raw_ = y
        
        # 1. Difference
        y_diff = self._difference(y, self.d)
        self.training_data_diff_ = y_diff
        
        # 2. MA Residual Estimation
        residuals = None
        if self.q > 0:
            residuals = self._estimate_initial_residuals(y_diff)
            self.training_residuals_ = residuals
            
        # 3. Create Features
        try:
            X_train, y_train = self._create_features(y_diff, residuals)
        except ValueError as e:
            self.logger.error(f"Data processing failed: {e}")
            raise
            
        # 4. Handle Trend
        if self.trend == 'ct':
            # Time index
            # Start index relative to start of y_diff? 
            # y_diff starts at index d of original.
            # X_train starts at max_lag of y_diff.
            max_lag_eff = max(self.p, self.q if self.q > 0 else 0)
            start_t = self.d + max_lag_eff
            t_idx = np.arange(start_t, start_t + len(y_train)).reshape(-1, 1)
            X_train = np.hstack([X_train, t_idx])
            
        elif self.trend == 'n':
            if self.verbose:
                self.logger.warning("Trend 'n' requested but Base Regressor enforces intercept.")

        # 5. Process IO & Fit
        Xc, yc = super()._fit_io(X_train, y_train)
        super()._fit(Xc, yc)
        
        return self

    @narwhalify
    def predict(self, steps: int = 1, future: bool = True) -> np.ndarray:
        """
        Forecast future values.
        """
        if self.training_data_raw_ is None:
             raise ValueError("Model is not fitted.")
             
        self.logger.info(f"Forecasting {steps} steps ahead.")
        
        # Working histories
        history_diff = list(self.training_data_diff_)
        history_res = list(self.training_residuals_) if self.q > 0 else []
        
        forecast_diff = []
        
        # Time index for 'ct'
        # Current length of original series
        current_t = len(self.training_data_raw_)
        
        for _ in range(steps):
            # Form feature vector
            features = []
            
            # AR part
            if self.p > 0:
                if len(history_diff) < self.p:
                     raise ValueError("Not enough history for AR prediction.")
                features.extend(history_diff[-self.p:][::-1])
                
            # MA part
            if self.q > 0:
                # We need residuals. For future steps, expected residual is 0.
                # However, if q extends into known history, we use it.
                # We have been appending 0 to history_res for forecasted steps?
                # Actually, in forecasting loop:
                # True value unknown -> Residual unknown -> usually set to 0.
                if len(history_res) < self.q:
                     # Should not happen if fit worked and q <= length
                     # Pad with 0?
                     needed = self.q - len(history_res)
                     features.extend([0.0]*needed + history_res[::-1])
                else:
                    features.extend(history_res[-self.q:][::-1])
            
            feat_vector = np.array(features).reshape(1, -1)
            
            # Trend
            if self.trend == 'ct':
                feat_vector = np.hstack([feat_vector, [[current_t]]])
            
            # Predict diff value
            # IO Predict
            feat_vector_transformed = super()._predict_io(feat_vector)
            pred_val_diff = super()._predict(feat_vector_transformed)[0]
            
            forecast_diff.append(pred_val_diff)
            
            # Update histories
            history_diff.append(pred_val_diff)
            if self.q > 0:
                # Expected residual for forecast is 0
                history_res.append(0.0)
                
            current_t += 1
            
        # Inverse Difference
        forecast_diff_arr = np.array(forecast_diff)
        final_forecast = self._inverse_difference(self.training_data_raw_, forecast_diff_arr, self.d)
        
        return final_forecast

    @narwhalify
    def score(self, y: np.ndarray, X=None) -> float:
        """
        Score using Robust R2.
        Note: Checks prediction on differenced series or original?
        Usually R2 on original series is what matters for user.
        """
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        y = np.array(y).flatten()
        
        # We can't really "score" easily in batch because ARIMA is recursive (especially MA and I).
        # We could perform one-step ahead predictions for the whole series?
        # Or simple: Use the fitted model to predict in-sample?
        
        # For simplicity, let's score on the internal regression task (prediction of y_diff)
        # OR score on re-constructed in-sample fit.
        
        # Let's Score on the regression task (y_diff) as that represents the model fit quality properly
        # w.r.t the optimization objective.
        
        # However, users compare y to y_pred.
        
        # Let's try to generate in-sample predictions on original scale?
        # That requires valid start conditions.
        
        # Fallback: Score on y_diff prediction (the stationarized series).
        
        # 1. Difference y
        y_diff = self._difference(y, self.d) # This is ground truth for diff
        
        # 2. Need residuals for these points?
        # Use simple estimation again? Or use training residuals if available?
        # If y is new data, we must re-estimate residuals?
        # For scoring separate test set, ARIMA is tricky.
        
        # If y is same as training data...
        if np.array_equal(y, self.training_data_raw_):
            # Calculate in-sample robust R2 of y_diff
            
            # Recalculate features
            residuals = self.training_residuals_
            X_train, y_train = self._create_features(y_diff, residuals)
            
            if self.trend == 'ct':
                 # Reconstruct t_idx as in fit
                 max_lag_eff = max(self.p, self.q if self.q > 0 else 0)
                 start_t = self.d + max_lag_eff
                 t_idx = np.arange(start_t, start_t + len(y_train)).reshape(-1, 1)
                 X_train = np.hstack([X_train, t_idx])
            
            # Predict
            X_trans = super()._predict_io(X_train)
            y_pred_diff = super()._predict(X_trans)
            
            return robr2(y_train, y_pred_diff)
            
        else:
            self.logger.warning("Scoring on new data not fully supported for ARIMA with MA due to recursive state. Returning -inf.")
            return -np.inf

    def summary(self):
        """Show model summary."""
        print(f"Gnostic ARIMA(p={self.p}, d={self.d}, q={self.q}, trend='{self.trend}')")
        print(f"Iterations: {len(self._history) if self._history else 'N/A'}")
        if self.weights is not None:
             print("Final Weights Stats:")
             print(f"  Min: {np.min(self.weights):.4f}")
             print(f"  Max: {np.max(self.weights):.4f}")
             print(f"  Mean: {np.mean(self.weights):.4f}")

    def __repr__(self):
        return f"ARIMA(order=({self.p}, {self.d}, {self.q}), trend='{self.trend}')"

