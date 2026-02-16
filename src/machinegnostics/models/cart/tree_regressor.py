'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2026  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
'''

import numpy as np
from machinegnostics.models.base_io_models import DataProcessLayerBase
from machinegnostics.models.cart.base_cart_history import HistoryCartBase
from machinegnostics.metrics import robr2
from machinegnostics.magcal import disable_parent_docstring
import logging
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.magcal.util.narwhals_df import narwhalify

class GnosticDecisionTreeRegressor(HistoryCartBase, DataProcessLayerBase):
    """
    Gnostic Decision Tree Regressor.
    
    Implements a single Decision Tree with iterative gnostic reweighting to handle outliers and data quality.
    Instead of bootstrapping/bagging like a forest, this fits one tree using gnostic weights directly.
    
    Key Features
    ------------
    - **Robust Single Tree**: Fits a decision tree that is robust to outliers by down-weighting them.
    - **Iterative Refinement**: Updates sample weights based on residuals over multiple iterations.
    - **Interpretable**: Maintains the interpretability of a single decision tree.
    
    Parameters
    ----------
    max_depth : int, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum samples to split a node.
    gnostic_weights : bool, default=True
        Whether to use iterative gnostic weights.
    max_iter : int, default=10
        Maximum gnostic iterations.
    tolerance : float, default=1e-4
        Convergence tolerance.
    data_form : str, default='a'
        Data form: 'a' (additive) or 'm' (multiplicative).
    verbose : bool, default=False
        Verbosity.
    random_state : int, default=None
        Random seed.
    history : bool, default=True
        Whether to record training history.
    
    Example
    -------
    >>> import numpy as np
    >>> from machinegnostics.models import GnosticDecisionTreeRegressor
    >>>
    >>> # Generate synthetic data with outliers
    >>> X = np.random.rand(100, 1) * 10
    >>> y = 2 * X.ravel() + 1 + np.random.normal(0, 0.5, 100)
    >>> y[::10] += 20  # Add strong outliers
    >>>
    >>> # Initialize and fit the robust tree model
    >>> model = GnosticDecisionTreeRegressor(
    ...     max_depth=5,
    ...     gnostic_weights=True,
    ...     max_iter=10
    ... )
    >>> model.fit(X, y)
    >>>
    >>> # Make predictions
    >>> preds = model.predict(X[:5])
    >>> print("Predictions:", preds)
    """
    @disable_parent_docstring
    def __init__(self,
                 max_depth: int = None,
                 min_samples_split: int = 2,
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
        
        # If gnostic_weights is False, we set max_iter=0 for the base class so loop runs once (standard Tree)
        self.gnostic_weights = gnostic_weights
        _max_iter = max_iter if gnostic_weights else 0
        
        super().__init__(
            n_estimators=1, # Ignored for tree type, but passed to satisfy base init
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_iter=_max_iter,
            tolerance=tolerance,
            data_form=data_form,
            verbose=verbose,
            random_state=random_state,
            history=history,
            scale=scale,
            early_stopping=early_stopping,
            estimator_type='tree'
        )
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)

    @narwhalify
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model.

        Parameters
        ----------
        X : array-like or dataframe of shape (n_samples, n_features)
            Input features. Accepts NumPy arrays, Pandas DataFrame.
        y : array-like or series of shape (n_samples,)
            Target values. Accepts NumPy arrays, Pandas Series/DataFrame column.
        """
        self.logger.info("Starting fit process for GnosticDecisionTreeRegressor.")
        # Data process layer IO
        Xc, yc = super()._fit_io(X, y)
        # Call base fit (which does logic)
        super()._fit(Xc, yc)
        return self

    @narwhalify
    def predict(self, model_input: np.ndarray) -> np.ndarray:
        """Predict outcomes.

        Parameters
        ----------
        model_input : array-like or dataframe of shape (n_samples, n_features)
            Input features for prediction.

        Returns
        -------
        array-like
            Predicted values. Returns native type (NumPy array or Pandas Series) based on input.
        """
        self.logger.info("Making predictions with GnosticDecisionTreeRegressor.")
        model_input_c = super()._predict_io(model_input)
        return super()._predict(model_input_c)

    @narwhalify
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score the model.

        Parameters
        ----------
        X : array-like or dataframe of shape (n_samples, n_features)
            Input features for scoring.
        y : array-like or series of shape (n_samples,)
            True target values.
        """
        self.logger.info("Calculating score.")
        X_checked, y_checked = super()._score_io(X, y)
        y_pred = self.predict(X_checked)
        return robr2(y_checked, y_pred, w=self.weights)

    def __repr__(self):
        return (f"GnosticDecisionTreeRegressor(gnostic_weights={self.gnostic_weights}, "
                f"max_depth={self.max_depth}, "
                f"max_iter={self.max_iter})")
