'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2026  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.
'''

import numpy as np
from machinegnostics.models.base_io_models import DataProcessLayerBase
from machinegnostics.models.cart.base_boosting_classifier_history import HistoryBoostingClassifierBase
from machinegnostics.metrics import accuracy_score
from machinegnostics.magcal import disable_parent_docstring
import logging
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.magcal.util.narwhals_df import narwhalify

class GnosticBoostingClassifier(HistoryBoostingClassifierBase, DataProcessLayerBase):
    """
    Boosting Classifier with Robust Gnostic Learning.

    The Gnostic Boosting Classifier extends the Gradient Boosting approach by integrating
    Mathematical Gnostics principles. It employs an iterative reweighting scheme (`gnostic_weights`)
    that assesses the quality of each data sample based on the probability residuals of the previous iteration's
    model. This allows the model to autonomously down-weight outliers and noise (mislabeled samples).
    
    This implementation wraps the XGBoost library.

    Key Features
    ------------
    - **Robustness to Label Noise**: Automatically identifies and down-weights samples with high residual norms.
    - **Iterative Refinement**: Optimizes sample weights over multiple Gnostic iterations.
    - **Boosted Performance**: Leverages the power of boosting as the underlying estimator.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of gradient boosted trees.
    max_depth : int, default=6
        Maximum tree depth for base learners.
    learning_rate : float, default=0.3
        Boosting learning rate.
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
        Additional keyword arguments passed to `xgboost.XGBClassifier`.
        
    Example
    -------
    >>> from machinegnostics.models.cart.boosting_classifier import GnosticBoostingClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_classes=2, random_state=42)
    >>> model = GnosticBoostingClassifier(n_estimators=50, gnostic_weights=True)
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
            estimator_type='classifier',
            **kwargs
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
            Target class labels. Accepts NumPy arrays, Pandas Series/DataFrame column.
        """
        self.logger.info("Starting fit process for GnosticBoostingClassifier.")
        # Data process layer IO
        Xc, yc = super()._fit_io(X, y)
        super()._fit(Xc, yc)
        return self

    @narwhalify
    def predict(self, model_input: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        model_input : array-like or dataframe of shape (n_samples, n_features)
            Input features for prediction.

        Returns
        -------
        array-like
            Predicted class labels. Returns native type (NumPy array or Pandas Series) based on input.
        """
        self.logger.info("Making predictions with GnosticBoostingClassifier.")
        model_input_c = super()._predict_io(model_input)
        return super()._predict_boosting_impl(model_input_c, self.model)

    @narwhalify
    def predict_proba(self, model_input: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        model_input : array-like or dataframe of shape (n_samples, n_features)
            Input features for probability prediction.

        Returns
        -------
        array-like
            Predicted probabilities. Returns native type (NumPy array or Pandas Series/DataFrame) based on input.
        """
        self.logger.info("Making probability predictions with GnosticBoostingClassifier.")
        model_input_c = super()._predict_io(model_input)
        return super()._predict_proba_boosting_impl(model_input_c, self.model)

    @narwhalify
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score (Accuracy).

        Parameters
        ----------
        X : array-like or dataframe of shape (n_samples, n_features)
            Input features for evaluation.
        y : array-like or series of shape (n_samples,)
            True class labels.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def __repr__(self):
        return (f"GnosticBoostingClassifier(n_estimators={self.n_estimators}, "
                f"learning_rate={self.learning_rate}, "
                f"gnostic_weights={self.gnostic_weights}, "
                f"max_depth={self.max_depth}, "
                f"max_iter={self.max_iter})")
