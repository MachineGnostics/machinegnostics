'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2026  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
'''

import numpy as np
from machinegnostics.models.base_io_models import DataProcessLayerBase
from machinegnostics.models.cart.base_cart_classifier_history import HistoryCartClassifierBase
from machinegnostics.metrics import accuracy_score
from machinegnostics.magcal import disable_parent_docstring
import logging
from machinegnostics.magcal.util.logging import get_logger

class GnosticRandomForestClassifier(HistoryCartClassifierBase, DataProcessLayerBase):
    """
    Gnostic Random Forest Classifier.
    
    Implements a Random Forest Classifier with iterative gnostic reweighting to handle outliers and data quality.
    
    Key Features
    ------------
    - **Robust Ensemble Classification**: Combines Random Forest with Gnostic reweighting.
    - **Iterative Refinement**: Updates sample weights based on ensemble vote probabilities over iterations.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=None
        Maximum depth of the trees.
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
    >>> from machinegnostics.models import GnosticRandomForestClassifier
    >>>
    >>> model = GnosticRandomForestClassifier(n_estimators=50, max_depth=3, gnostic_weights=True)
    >>> model.fit(X, y)
    >>> preds = model.predict(X)
    """
    @disable_parent_docstring
    def __init__(self,
                 n_estimators: int = 100,
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
        
        self.gnostic_weights = gnostic_weights
        _max_iter = max_iter if gnostic_weights else 0
        
        super().__init__(
            n_estimators=n_estimators,
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
            estimator_type='forest'
        )
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        self.logger.info("Starting fit process for GnosticRandomForestClassifier.")
        # Data process layer IO
        Xc, yc = super()._fit_io(X, y)
        super()._fit(Xc, yc)
        return self

    def predict(self, model_input: np.ndarray) -> np.ndarray:
        """Predict outcomes."""
        self.logger.info("Making predictions with GnosticRandomForestClassifier.")
        model_input_c = super()._predict_io(model_input)
        return super()._predict(model_input_c)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score (Accuracy)."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def __repr__(self):
        return (f"GnosticRandomForestClassifier(n_estimators={self.n_estimators}, "
                f"gnostic_weights={self.gnostic_weights}, "
                f"max_depth={self.max_depth}, "
                f"max_iter={self.max_iter})")
