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
    Random Forest Classifier with Robust Gnostic Learning.

    The Gnostic Forest Classifier extends the standard random forest approach by integrating
    Mathematical Gnostics principles. It employs an iterative reweighting scheme (`gnostic_weights`)
    that assesses the quality of each data sample based on the consistency of the previous iteration's
    model predictions. This allows the forest to autonomously down-weight outliers and noise (mislabeled samples), resulting in a more robust predictive model.

    Key Features
    ------------
    - **Robustness to Label Noise**: Automatically identifies and down-weights samples with low classification confidence or high residual norms.
    - **Iterative Refinement**: Optimizes sample weights over multiple iterations until convergence or maximum iterations are reached.
    - **Gnostic History**: (Optional) Tracks the evolution of loss, entropy, and weights across iterations for diagnostic analysis.
    - **Standard Forest Capabilities**: Supports standard hyperparameters like `n_estimators`, `max_depth`, and `min_samples_split`.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : int, default=None
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure
        or until all leaves contain less than min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    gnostic_weights : bool, default=True
        If True, enables the iterative gnostic reweighting process. If False, the model behaves
        like a standard Random Forest Classifier with uniform weights (single iteration).
    max_iter : int, default=10
        The maximum number of iterations for the gnostic weight update loop.
        Only effective if `gnostic_weights=True`.
    tolerance : float, default=1e-4
        The convergence tolerance. If the change in Gnostic Loss or Rentropy between iterations
        drops below this threshold, the training stops early.
    data_form : str, default='a'
        The form of the data for Gnostic calculations:
        - 'a': Additive (standard real-valued data).
        - 'm': Multiplicative.
    verbose : bool, default=False
        If True, prints progress logs and convergence information during training.
    random_state : int, default=None
        Controls the randomness of the bootstrapping of the samples used when building trees,
        ensuring reproducible results.
    history : bool, default=True
        If True, records the training history (loss, entropy, weights) which is accessible
        via the `_history` attribute.
    scale : str or float, default='auto'
        The scale parameter 'S' for Gnostic calculations.
        - 'auto': Automatically estimated from the data.
        - float: A fixed scale value.
    early_stopping : bool, default=True
        If True, allows the iterative process to stop before `max_iter` if the convergence criteria are met.

    Attributes
    ----------
    weights : np.ndarray
        The final calibrated sample weights assigned to the training data. Lower weights indicate
        potential outliers or low-fidelity data points.
    trees : list
        The list of underlying decision trees (estimators) that make up the forest.
    classes_ : np.ndarray
        The classes labels (single output problem).
    _history : list of dict
        A record of training metrics (iteration, h_loss, rentropy, weights) for each step, available if `history=True`.

    Methods
    -------
    fit(X, y)
        Fit the Gnostic Forest model to the training data.
    predict(X)
        Predict class labels for input samples X.
    score(X, y)
        Return the mean accuracy on the given test data and labels.

    Example
    -------
    >>> from machinegnostics.models import GnosticRandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> # Generate synthetic data
    >>> X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    >>>
    >>> # Initialize and fit the robust model
    >>> model = GnosticRandomForestClassifier(
    ...     n_estimators=50,
    ...     gnostic_weights=True,
    ...     max_iter=5,
    ...     verbose=True
    ... )
    >>> model.fit(X, y)
    >>>
    >>> # Make predictions
    >>> preds = model.predict(X[:5])
    >>> print("Predictions:", preds)
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
