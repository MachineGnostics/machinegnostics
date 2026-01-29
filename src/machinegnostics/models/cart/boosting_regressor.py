'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2026  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.
'''

import numpy as np
from machinegnostics.models.base_io_models import DataProcessLayerBase
from machinegnostics.models.cart.base_boosting_history import HistoryBoostingBase
from machinegnostics.metrics import robr2
from machinegnostics.magcal import disable_parent_docstring
import logging
from machinegnostics.magcal.util.logging import get_logger

class GnosticBoostingRegressor(HistoryBoostingBase, DataProcessLayerBase):
    """
    Boosting Regressor with Robust Gnostic Learning.

    The Gnostic Boosting Regressor extends the Gradient Boosting approach by integrating
    Mathematical Gnostics principles. It employs an iterative reweighting scheme (`gnostic_weights`)
    that assesses the quality of each data sample based on the residuals of the previous iteration's
    model. This allows the model to autonomously down-weight outliers and noise.
    
    This implementation wraps the XGBoost library.

    Key Features
    ------------
    - **Robustness to Outliers**: Automatically identifies and down-weights anomalous samples.
    - **Iterative Refinement**: Optimizes sample weights over multiple Gnostic iterations.
    - **Boosted Performance**: Leverages the power of Boosting as the underlying estimator.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of gradient boosted trees. Equivalent to number of boosting rounds.
    max_depth : int, default=6
        Maximum tree depth for base learners.
    learning_rate : float, default=0.3
        Boosting learning rate (eta).
    gnostic_weights : bool, default=True
        If True, enables the iterative gnostic reweighting process.
    max_iter : int, default=10
        The maximum number of iterations for the gnostic weight update loop.
    tolerance : float, default=1e-4
        Convergence tolerance for gnostic weights.
    data_form : str, default='a'
        Data form for gnostic conversions ('a' for additive, 'm' for multiplicative).
    verbose : bool, default=False
        Controls the verbosity.
    random_state : int, default=None
        Random number seed.
    history : bool, default=True
        Whether to record training history (loss, rentropy, weights).
    scale : str, default='auto'
        Scale parameter for Gnostic calculations.
    early_stopping : bool, default=True
        Whether to stop gnostic iterations early if convergence is detected.
    **kwargs
        Additional keyword arguments passed to `xgboost.XGBRegressor`.
        
    Example
    -------
    >>> from machinegnostics.models.cart.boosting_regressor import GnosticBoostingRegressor
    >>> import numpy as np
    >>> X = np.random.rand(100, 5)
    >>> y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 100)
    >>> model = GnosticBoostingRegressor(n_estimators=50, gnostic_weights=True)
    >>> model.fit(X, y)
    >>> preds = model.predict(X)
    """
    @disable_parent_docstring
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.3,
                 gnostic_weights: bool = True,
                 max_iter: int = 10,
                 tolerance: float = 1e-4,
                 data_form: str = 'a',
                 verbose: bool = False,
                 random_state: int = None,
                 history: bool = True,
                 scale: str = 'auto',
                 early_stopping: bool = True,
                 **kwargs):
        
        self.gnostic_weights = gnostic_weights
        _max_iter = max_iter if gnostic_weights else 0
        
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=_max_iter,
            tolerance=tolerance,
            data_form=data_form,
            verbose=verbose,
            random_state=random_state,
            history=history,
            scale=scale,
            early_stopping=early_stopping,
            estimator_type='regressor',
            **kwargs
        )
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        self.logger.info("Starting fit process for GnosticBoostingRegressor.")
        # Data process layer IO
        Xc, yc = super()._fit_io(X, y)
        super()._fit(Xc, yc)
        return self

    def predict(self, model_input: np.ndarray) -> np.ndarray:
        """Predict regression target."""
        self.logger.info("Making predictions with GnosticBoostingRegressor.")
        model_input_c = super()._predict_io(model_input)
        return super()._predict_boosting_impl(model_input_c, self.model)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the robust coefficient of determination R^2 of the prediction."""
        y_pred = self.predict(X)
        return robr2(y, y_pred, w=self.weights)

    def __repr__(self):
        return (f"GnosticBoostingRegressor(n_estimators={self.n_estimators}, "
                f"learning_rate={self.learning_rate}, "
                f"gnostic_weights={self.gnostic_weights}, "
                f"max_depth={self.max_depth}, "
                f"max_iter={self.max_iter})")
