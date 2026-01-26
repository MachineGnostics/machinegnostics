'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar

Description:
This module implements a robust k-means clustering model using mathematical gnostics principles.
'''

import numpy as np
from machinegnostics.models.clustering.base_io_clustering import DataProcessClusteringBase
from machinegnostics.models.clustering.base_clustering_history import HistoryClusteringBase
import logging
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.magcal import disable_parent_docstring

class KMeansClustering(HistoryClusteringBase, DataProcessClusteringBase):
    """
    Robust K-Means Clustering using Mathematical Gnostics principles.

    This clustering model fits k-means to data using robust, gnostic loss functions
    and adaptive sample weights. It is designed to be resilient to outliers and non-Gaussian noise,
    making it suitable for scientific and engineering applications where data quality may vary.

    Key Features
    ------------
    - Robust to outliers: Uses gnostic loss functions and adaptive sample weights.
    - Iterative optimization: Supports early stopping and convergence tolerance.
    - Tracks detailed history: Optionally records loss, weights, entropy, and gnostic characteristics at each iteration.
    - Gnostic weight multiplication: Weights are multiplied by gnostic weights (gw) for robust clustering.
    - Compatible with numpy arrays for input/output.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of clusters to form.
    scale : {'auto', int, float}, default='auto'
        Scaling method or value for gnostic calculations.
    max_iter : int, default=100
        Maximum number of optimization iterations.
    tolerance : float, default=1e-1
        Tolerance for convergence.
    mg_loss : str, default='hi'
        Loss function to use ('hi' or 'hj').
    early_stopping : bool, default=True
        Whether to stop early if convergence is detected.
    verbose : bool, default=False
        If True, prints progress and diagnostics during fitting.
    data_form : str, default='a'
        Internal data representation format ('a' for additive, 'm' for multiplicative).
    gnostic_characteristics : bool, default=False
        If True, computes and records gnostic characteristics.
    history : bool, default=True
        If True, records the optimization history for analysis.
    init : str, default='random'
        Method for initialization ('random' or 'kmeans++').

    Attributes
    ----------
    centroids : np.ndarray
        Fitted cluster centroids of shape (n_clusters, n_features).
    labels : np.ndarray
        Cluster labels for each sample.
    weights : np.ndarray
        Final sample weights after robust fitting.
    params : list of dict
        List of parameter snapshots (loss, weights, centroids, gnostic properties) at each iteration.
    _history : list
        Internal optimization history (if enabled).
    All configuration parameters as set at initialization.

    Methods
    -------
    fit(X, y=None)
        Fit the k-means clustering model to input features X.
    predict(X)
        Predict cluster labels for new input features X.
    score(X, y=None)
        Compute the negative inertia score for input features X.

    Example
    -------
    >>> from machinegnostics.models import KMeansClustering
    >>> model = KMeansClustering(n_clusters=3, max_iter=100, verbose=True)
    >>> model.fit(X_train)
    >>> labels = model.predict(X_test)
    >>> score = model.score(X_test)

    Notes
    -----
    - This model is part of the Machine Gnostics library, which implements advanced machine learning techniques
      based on mathematical gnostics principles.
    - The key innovation is the multiplication of sample weights with gnostic weights (gw) for robust clustering.
    - For more information, visit: https://machinegnostics.info/
    """
    @disable_parent_docstring
    def __init__(
        self, 
        n_clusters: int = 3,
        scale: str | int | float = 'auto',
        max_iter: int = 100,
        tolerance: float = 1e-1,
        mg_loss: str = 'hi',
        early_stopping: bool = True,
        verbose: bool = False,
        data_form: str = 'a',
        gnostic_characteristics: bool = False,
        history: bool = True,
        init: str = 'random'
    ):
        """
        Initialize a KMeansClustering instance with robust, gnostic clustering settings.

        Parameters
        ----------
        n_clusters : int, default=3
            The number of clusters to form.
        scale : {'auto', int, float}, default='auto'
            Scaling method or value for gnostic calculations.
        max_iter : int, default=100
            Maximum number of optimization iterations.
        tolerance : float, default=1e-2
            Tolerance for convergence.
        mg_loss : str, default='hi'
            Loss function to use ('hi' or 'hj').
        early_stopping : bool, default=True
            Whether to stop early if convergence is detected.
        verbose : bool, default=False
            If True, prints progress and diagnostics during fitting.
        data_form : str, default='a'
            Internal data representation format ('a' for additive, 'm' for multiplicative).
        gnostic_characteristics : bool, default=False
            If True, computes and records gnostic properties (fi, hi, etc.).
        history : bool, default=True
            If True, records the optimization history for analysis.
        init : str, default='random'
            Method for initialization ('random' or 'kmeans++').

        Notes
        -----
        All configuration parameters are stored as attributes for later reference.
        """
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tolerance=tolerance,
            mg_loss=mg_loss,
            early_stopping=early_stopping,
            verbose=verbose,
            scale=scale,
            data_form=data_form,
            gnostic_characteristics=gnostic_characteristics,
            history=history,
            init=init
        )
        
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.mg_loss = mg_loss
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.scale = scale
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self._record_history = history
        self.init = init
        self.params = []
        
        # history option
        if history:
            self._history = []
        else:
            self._history = None
        
        # logger
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.info(f"{self.__class__.__name__} initialized:")

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit the robust k-means clustering model to the provided data.

        This method performs robust k-means clustering using the specified gnostic loss function,
        iteratively optimizing the cluster centroids and sample weights to minimize the influence of outliers.
        If history tracking is enabled, it records loss, weights, centroids, and gnostic properties at each iteration.

        The key innovation is the multiplication of sample weights with gnostic weights (gw):
        new_weights = weights * gw

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features).
        y : np.ndarray, optional
            Not used, present for API consistency.

        Returns
        -------
        self : KMeansClustering
            Returns the fitted model instance for chaining or further use.

        Notes
        -----
        - After fitting, the model's centroids, labels, and sample weights are available in the 
          `centroids`, `labels`, and `weights` attributes.
        - If `history=True`, the optimization history is available in the `params` and `_history` attributes.

        Example
        -------
        >>> model = KMeansClustering(n_clusters=3, max_iter=100, verbose=True)
        >>> model.fit(X_train)
        >>> print(model.centroids)
        >>> print(model.labels)
        >>> print(model.weights)
        """
        self.logger.info("Starting fit process for KMeansClustering.")
        # fit for data processing
        Xc, _ = super()._fit_io(X, y)
        # fit for robust clustering
        super()._fit(Xc, y)
        return self
    
    def predict(self, model_input: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels using the fitted k-means clustering model.

        Parameters
        ----------
        model_input : np.ndarray
            Input features for prediction, shape (n_samples, n_features).

        Returns
        -------
        labels : np.ndarray
            Predicted cluster labels, shape (n_samples,).

        Example
        -------
        >>> model = KMeansClustering(n_clusters=3, max_iter=100, verbose=True)
        >>> model.fit(X_train)
        >>> labels = model.predict(X_test)
        """
        self.logger.info("Making predictions with KMeansClustering.")
        model_input_c = super()._predict_io(model_input)
        return super()._predict(model_input_c)
    
    def score(self, X: np.ndarray, y: np.ndarray = None) -> float:
        """
        Compute the negative inertia score for the k-means clustering model.

        The inertia is the sum of squared distances of samples to their closest cluster center.
        A negative value is returned for sklearn compatibility (higher is better).

        Parameters
        ----------
        X : np.ndarray
            Input features for scoring, shape (n_samples, n_features).
        y : np.ndarray, optional
            Not used, present for API consistency.

        Returns
        -------
        score : float
            Negative inertia score of the model on the provided data.
        
        Example
        -------
        >>> model = KMeansClustering(n_clusters=3, max_iter=100, verbose=True)
        >>> model.fit(X_train)
        >>> score = model.score(X_test)
        >>> print(f"Negative inertia score: {score}")
        """
        self.logger.info("Calculating inertia score with KMeansClustering.")
        # check
        X_checked, _ = super()._score_io(X, y)
        # Call the score method
        score = super()._score(X_checked, y)
        return score
    
    def __repr__(self):
        """String representation of the KMeansClustering instance."""
        return (f"KMeansClustering(n_clusters={self.n_clusters}, max_iter={self.max_iter}, "
                f"tol={self.tolerance}, mg_loss='{self.mg_loss}', early_stopping={self.early_stopping}, "
                f"verbose={self.verbose}, scale={self.scale}, data_form='{self.data_form}', "
                f"gnostic_characteristics={self.gnostic_characteristics}, history={self._record_history}, "
                f"init='{self.init}')")
