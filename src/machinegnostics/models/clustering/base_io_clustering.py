'''
Module for DataProcessClusteringBase class to handle input/output processing for clustering models.

Handles data type checking, validation, and conversion to ensure that input data is in the correct format 
for clustering model training and prediction. Unlike supervised learning, clustering doesn't require target labels (y).

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
import pandas as pd
import logging
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.models.base_io_models import DataProcessLayerBase

class DataProcessClusteringBase(DataProcessLayerBase):
    """
    A class to handle input/output processing for clustering models.

    This class extends DataProcessLayerBase with clustering-specific behavior where target labels (y) 
    are optional and not required for fitting. This is suitable for unsupervised learning algorithms 
    like k-means clustering.

    Inherits
    --------
    DataProcessLayerBase
        Provides basic data processing functionality.
    """
    def __init__(self, verbose: bool = False, **kwargs):
        """
        Initialize the DataProcessClusteringBase with optional parameters.

        Parameters
        ----------
        verbose : bool, default=False
            If True, enables detailed logging.
        **kwargs : dict
            Additional parameters for configuration.
        """ 
        super().__init__(verbose=verbose, **kwargs)
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.info(f"{self.__class__.__name__} initialized for clustering (unsupervised learning).")

    def _fit_io(self, X, y=None):
        """
        Process and validate input data for fitting a clustering model.

        For clustering models, y is optional and typically None since clustering is unsupervised.

        Parameters
        ----------
        X : array-like, pandas DataFrame, or pyspark DataFrame
            Input features of shape (n_samples, n_features).
        y : array-like, optional
            Not used for clustering, but kept for API consistency. Default is None.

        Returns
        -------
        X_checked : np.ndarray
            Validated and processed input features.
        y : None
            Always returns None for clustering (kept for API consistency).
        """
        self.logger.info("Starting fit input/output processing for clustering.")
        X_checked = self._check_X(X)
        # For clustering, we don't need y, so return None
        return X_checked, None

    def _predict_io(self, X) -> np.ndarray:
        """
        Process and validate input data for prediction.

        Parameters
        ----------
        X : array-like, pandas DataFrame, or pyspark DataFrame
            Input features for prediction of shape (n_samples, n_features).

        Returns
        -------
        X_checked : np.ndarray
            Validated and processed input features.
        """
        self.logger.info("Starting predict input/output processing for clustering.")
        X_checked = self._check_X_predict(X)
        return X_checked
    
    def _score_io(self, X, y=None) -> tuple:
        """
        Process and validate input data for scoring.

        For clustering, y is optional since scoring typically uses inertia or silhouette score.

        Parameters
        ----------
        X : array-like, pandas DataFrame, or pyspark DataFrame
            Test samples of shape (n_samples, n_features).
        y : array-like, optional
            Not used for clustering scoring, but kept for API consistency.

        Returns
        -------
        X_checked : np.ndarray
            Validated and processed input features.
        y : None
            Always returns None for clustering.
        """
        self.logger.info("Starting score input/output processing for clustering.")
        X_checked = self._check_X(X)
        # For clustering, we don't use y for scoring
        return X_checked, None
