'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2026  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
'''

import numpy as np
import pandas as pd
from machinegnostics.models.base_io_models import DataProcessLayerBase
from machinegnostics.models.regression.base_regressor_history import HistoryRegressorBase
from machinegnostics.magcal import disable_parent_docstring
from machinegnostics.metrics import robr2
import logging
from machinegnostics.magcal.util.logging import get_logger

class AutoRegressor(HistoryRegressorBase, DataProcessLayerBase):
    """
    Gnostic AutoRegressor (AR) with Robust Iterative Reweighting.

    This model implements an Autoregressive (AR) process for time series forecasting,
    empowered by Mathematical Gnostics weights. Unlike standard AR models (OLS), this
    model iteratively reweights observations based on their residual gnostic probability,
    making the forecast robust to temporary anomalies and outliers in the training series.

    Model:
        y_t = c + w_1*y_{t-1} + ... + w_p*y_{t-p} + (trend) + error

    Key Features
    ------------
    - **Robust Forecasting**: Resilient to outliers in history via Gnostic Weights.
    - **Trend Support**: Supports constant ('c') and constant+linear ('ct') trends.
    - **Iterative Refinement**: Estimates coefficients using Gnostic Weighted Least Squares.
    - **Recursive Forecasting**: Supports multi-step ahead prediction.

    Parameters
    ----------
    lags : int, default=1
        Number of past observations to use (p).
    trend : str, {'c', 'ct'}, default='c'
        Trend to include in the model:
        - 'c': Constant (bias/intercept).
        - 'ct': Constant and linear time trend.
        - 'n': No trend (Note: underlying gnostics base might add bias naturally, careful usage).
    scale : {'auto', int, float}, default='auto'
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
    >>> import numpy as np
    >>> from machinegnostics.models import AutoRegressor
    >>> # Generate synthetic AR data
    >>> np.random.seed(42)
    >>> t = np.arange(100)
    >>> y = np.sin(t * 0.1) + np.random.normal(0, 0.1, 100)
    >>> # Initialize and train model
    >>> ar_model = AutoRegressor(lags=5, trend='c')
    >>> ar_model.fit(y)
    >>> # Forecast next 10 steps
    >>> forecast = ar_model.predict(steps=10)
    """
    @disable_parent_docstring
    def __init__(self,
                 lags: int = 1,
                 trend: str = 'c',
                 scale: 'str | int | float' = 'auto',
                 max_iter: int = 100,
                 learning_rate: float = 0.1,
                 tolerance: float = 1e-3,
                 mg_loss: str = 'hi',
                 early_stopping: bool = True,
                 verbose: bool = False,
                 data_form: str = 'a',
                 gnostic_characteristics: bool = True,
                 history: bool = True):
        
        self.lags = lags
        self.trend = trend
        
        # Initialize base with linear degree (AR is linear in parameters)
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

        # Workaround: Base class HistoryRegressorBase overwrites self._history with bool
        # Restore it to a list if needed to prevent AttributeError in _fit
        if self.history and isinstance(self._history, bool):
            self._history = []
            self._history.append({
                'iteration': 0,
                'h_loss': None,
                'coefficients': None,
                'rentropy': None,
                'weights': None,
            })

        self.training_data_ = None
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)

    def _create_lags(self, y: np.ndarray, lags: int):
        """Create lag features from 1D array."""
        n_samples = len(y)
        if n_samples <= lags:
            raise ValueError(f"Not enough data samples ({n_samples}) for {lags} lags.")
            
        X_lags = []
        y_target = []
        
        # X: [y_{t-1}, y_{t-2}, ..., y_{t-p}]
        # We start predicting at index `lags`
        for i in range(lags, n_samples):
            # lags window: y[i-lags : i]
            # to be consistent with AR notation y_{t-1}, y_{t-2}... reverse the window
            window = y[i-lags:i][::-1]
            X_lags.append(window)
            y_target.append(y[i])
            
        return np.array(X_lags), np.array(y_target)

    def fit(self, y: np.ndarray, X=None):
        """
        Fit the Autoregressor to the time series y.
        
        Parameters
        ----------
        y : array-like
            Target time series.
        X : Ignored
            Included for compatibility, but ignored.
        """
        self.logger.info("Starting fit process for GnosticAutoRegressor.")
        
        # Values check
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        
        y = np.array(y).flatten()
        self.training_data_ = y  # Store for forecasting
        
        # create lags
        X_train, y_train = self._create_lags(y, self.lags)
        
        # Handle Trend
        # Base Regressor handles 'bias' (constant) automatically via _generate_polynomial_features(degree=1)
        # which adds a column of 1s.
        # But if trend='ct', we need to add a time index column.
        
        if self.trend == 'ct':
            # Time index corresponding to targets. Target starts at index `lags`.
            # We normalize time to start at 1 for stability? Or just use raw index.
            # Using raw index relative to series start.
            t_idx = np.arange(self.lags, self.lags + len(y_train)).reshape(-1, 1)
            X_train = np.hstack([X_train, t_idx])
        elif self.trend == 'n':
            # Theoretically we should remove intercept handling from base, but Base enforces it.
            # Warning user or accepting constant is standard limitation for now unless modified base.
            if self.verbose:
                self.logger.warning("Trend 'n' (no constant) requested but Base Regressor currently enforces intercept. Bias will be included.")
        
        # Process IO (Scaling, Checks)
        Xc, yc = super()._fit_io(X_train, y_train)
        
        # Fit using Gnostic Robust Regression logic
        super()._fit(Xc, yc)
        
        return self

    def predict(self, steps: int = 1, future: bool = True) -> np.ndarray:
        """
        Forecast future values.

        Parameters
        ----------
        steps : int, default=1
            Number of steps to forecast into the future.
        future : bool, default=True
            Forecasting mode. (Currently only future=True supported).
        
        Returns
        -------
        forecast : np.ndarray
            Predicted values for the next `steps`.
        """
        if self.training_data_ is None:
             raise ValueError("Model is not fitted.")
             
        self.logger.info(f"Forecasting {steps} steps ahead.")
        
        history = list(self.training_data_)
        forecast = []
        
        # For 'ct' trend, we need to know the next time index
        next_t = len(history) 
        
        for _ in range(steps):
            # Extract lag features
            # Get last `lags` elements
            if len(history) < self.lags:
                 raise ValueError("History insufficient for lags.")
                 
            # Lags: [y_{t-1}, y_{t-2} ... ]
            current_lags = np.array(history[-self.lags:][::-1])
            
            # Construct feature vector
            feat_vector = current_lags.reshape(1, -1)
            
            if self.trend == 'ct':
                 feat_vector = np.hstack([feat_vector, [[next_t]]])
            
            # Predict
            # Note: _predict handles scaling/poly transform if any, but we did IO manually in fit?
            # DataProcessLayerBase.predict calls _predict_io then _predict.
            # We need to ensure feat_vector goes through same scaling pipeline if it exists.
            
            # Using public predict interface from base would require hacking shape?
            # We call the internal io and internal predict steps
            
            # 1. IO Checks & Transforms (Scaling)
            # BaseIO fits scalers on X_train.
            feat_vector_transformed = super()._predict_io(feat_vector)
            
            # 2. Predict (Poly expansion + Dot product)
            # Base implementation: X_poly @ coeff
            pred = super()._predict(feat_vector_transformed)[0]
            
            forecast.append(pred)
            history.append(pred)
            next_t += 1
            
        return np.array(forecast)

    def score(self, y: np.ndarray, X=None) -> float:
        """
        Score the model on a time series y using Robust R2.
        
        Parameters
        ----------
        y : array-like
            Time series to evaluate.
        X : Ignored
        
        Returns
        -------
        score : float
            Robust R2 score.
        """
        self.logger.info("Calculating score.")
        
        # Values check
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        y = np.array(y).flatten()
        
        # create lags
        try:
            X_lags, y_target = self._create_lags(y, self.lags)
        except ValueError:
            self.logger.warning("Not enough data to score.")
            return -np.inf
            
        # Handle Trend (Same as fit)
        if self.trend == 'ct':
            # Assuming relative time index from 0 for scoring segment
            # Note: This implies the trend re-starts for the scored segment.
            # Ideally, one should pass absolute time, but standard fit(y) API limits this.
            # Consistent with 'fit' behavior on new data.
            t_idx = np.arange(self.lags, self.lags + len(y_target)).reshape(-1, 1)
            X_lags = np.hstack([X_lags, t_idx])
            
        # Predict
        # IO Predict (Scaling)
        X_trans = super()._predict_io(X_lags)
        # Model Predict
        y_pred = super()._predict(X_trans)
        
        return robr2(y_target, y_pred)

    def summary(self):
        """Show model summary."""
        print(f"Gnostic AutoRegressor (lags={self.lags}, trend='{self.trend}')")
        print(f"Iterations: {len(self._history) if self._history else 'N/A'}")
        if self.weights is not None:
             print("Final Weights Stats:")
             print(f"  Min: {np.min(self.weights):.4f}")
             print(f"  Max: {np.max(self.weights):.4f}")
             print(f"  Mean: {np.mean(self.weights):.4f}")

    def __repr__(self):
        return (f"GnosticAutoRegressor(lags={self.lags}, "
                f"trend='{self.trend}', "
                f"iterations={self.max_iter})")
