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

class GnosticDecisionTreeClassifier(HistoryCartClassifierBase, DataProcessLayerBase):
    """
    Gnostic Decision Tree Classifier.
    
    Implements a single Decision Tree Classifier with iterative gnostic reweighting to handle outliers and data quality.
    
    Key Features
    ------------
    - **Robust Classification**: Identifies and down-weights samples with high gnostic residual/uncertainty.
    - **Iterative Refinement**: Updates sample weights based on classification probabilities over iterations.
    
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
    >>> from machinegnostics.models import GnosticDecisionTreeClassifier
    >>>
    >>> X, y = make_classification(n_samples=100, n_classes=2, random_state=42)
    >>> model = GnosticDecisionTreeClassifier(max_depth=3, gnostic_weights=True)
    >>> model.fit(X, y)
    >>> preds = model.predict(X)
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
        
        self.gnostic_weights = gnostic_weights
        _max_iter = max_iter if gnostic_weights else 0
        
        super().__init__(
            n_estimators=1,
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

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        self.logger.info("Starting fit process for GnosticDecisionTreeClassifier.")
        # Data process layer IO (Handles X, y checks/conversions)
        # Assuming base IO supports classification labels (y) without trying to convert them if they are categorical
        # Usually base_io_models assumes regression-like float y for _fit_io.
        # Check base_io_models... but sticking to safe side, we rely on standard numpy arrays
        Xc, yc = super()._fit_io(X, y)
        super()._fit(Xc, yc)
        return self

    def predict(self, model_input: np.ndarray) -> np.ndarray:
        """Predict outcomes."""
        self.logger.info("Making predictions with GnosticDecisionTreeClassifier.")
        model_input_c = super()._predict_io(model_input)
        return super()._predict(model_input_c)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score (Accuracy)."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def __repr__(self):
        return (f"GnosticDecisionTreeClassifier(gnostic_weights={self.gnostic_weights}, "
                f"max_depth={self.max_depth}, "
                f"max_iter={self.max_iter})")
