'''
ClusteringMethodsBase - Base class for Machine Gnostics Clustering Methods

This class serves as the foundational base for all Machine Gnostics clustering methods,
providing common attributes and functionalities that can be extended by specific
gnostics algorithms and models.

Currently supports:
- k-means clustering

Copyright (C) Machine Gnostics
Author: Nirmal Parmar
'''

import logging
from machinegnostics.magcal.util.logging import get_logger
import numpy as np
from machinegnostics.magcal import (GnosticsCharacteristics, 
                                    DataConversion,
                                    ScaleParam,
                                    EGDF, QGDF, ELDF, QLDF)
from machinegnostics.models.base_model import ModelBase
from machinegnostics.magcal.util.min_max_float import np_max_float, np_min_float, np_eps_float
from typing import Union

class ClusteringMethodsBase(ModelBase):
    """
    Base class for Machine Gnostics Clustering Methods.

    Provide support to design fit, predict, and score methods for gnostics clustering models.
    """
    def __init__(self,
                 n_clusters: int = 3,
                 max_iter: int = 100,
                 tolerance: float = 1e-8,
                 mg_loss: str = 'hi',
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: Union[str, int, float] = 'auto',
                 history: bool = True,
                 data_form: str = 'a',
                 gnostic_characteristics: bool = True,
                 init: str = 'random'):
        super().__init__(verbose=verbose)
        """
        Initialize the ClusteringMethodsBase class.
        
        Parameters
        ----------
        n_clusters : int, default=3
            Number of clusters to form.
        max_iter : int, default=100
            Maximum number of iterations.
        tolerance : float, default=1e-8
            Convergence tolerance.
        mg_loss : str, default='hi'
            Gnostic loss function ('hi' or 'hj').
        early_stopping : bool, default=True
            Whether to use early stopping.
        verbose : bool, default=False
            Verbosity mode.
        scale : Union[str, int, float], default='auto'
            Scale parameter for gnostic calculations.
        history : bool, default=True
            Whether to track history.
        data_form : str, default='a'
            Data form ('a' for additive, 'm' for multiplicative).
        gnostic_characteristics : bool, default=True
            Whether to compute gnostic characteristics.
        init : str, default='random'
            Initialization method for centroids ('random' or 'kmeans++').
        """

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.centroids = None
        self.weights = None
        self.labels = None
        self.early_stopping = early_stopping
        self.mg_loss = mg_loss
        self.verbose = verbose
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self.scale = scale
        self.history = history
        self.init = init

        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.info(f"{self.__class__.__name__} initialized:")

        if self.history:
            self._history = []
            # default history content
            self._history.append({
                'iteration': 0,
                'h_loss': None,
                'centroids': None,
                'labels': None,
                'rentropy': None,
                'fi': None,
                'fj': None,
                'hi': None,
                'hj': None,
                'pi': None,
                'pj': None,
                'ei': None,
                'ej': None,
                'infoi': None,
                'infoj': None,
                'weights': None,
                'scale': None,
            })
        else:
            self._history = None

    def _input_checks(self):
        """
        Perform input validation for model parameters.
        """
        self.logger.info("Performing input checks for arguments.")
        if not isinstance(self.n_clusters, int) or self.n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer.")
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(self.tolerance, (float, int)) or self.tolerance <= 0:
            raise ValueError("tolerance must be a positive float or int.")
        if self.mg_loss not in ['hi', 'hj']:
            raise ValueError("mg_loss must be either 'hi' or 'hj'.")
        if not isinstance(self.scale, (str, int, float)):
            raise ValueError("scale must be a string, int, or float.")
        if isinstance(self.scale, (int, float)) and (self.scale < 0 or self.scale > 2):
            raise ValueError("scale must be between 0 and 2 if it is a number.")
        if self.data_form not in ['a', 'm']:
            raise ValueError("data_form must be either 'a' (additive) or 'm' (multiplicative).")
        if self.init not in ['random', 'kmeans++']:
            raise ValueError("init must be either 'random' or 'kmeans++'.")

    def _weight_init(self, d: np.ndarray, like: str = 'one') -> np.ndarray:
        """
        Initialize weights based on the input data.

        Parameters
        ----------
        d : np.ndarray
            Input data.
        like : str, optional
            Type of initialization ('one', 'zero', 'random'). Default is 'one'.

        Returns
        -------
        np.ndarray
            Initialized weights.
        """
        self.logger.info(f"Initializing weights with method: {like}")
        if like == 'one':
            return np.ones(len(d))
        elif like == 'zero':
            return np.zeros(len(d))
        elif like == 'random':
            return np.random.rand(len(d))
        else:
            self.logger.error("Invalid weight initialization method.")
            raise ValueError("like must be 'one', 'zero', or 'random'.")

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize cluster centroids.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Initial centroids of shape (n_clusters, n_features).
        """
        self.logger.info(f"Initializing centroids using method: {self.init}")
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            # Randomly select n_clusters samples as initial centroids
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            return X[indices].copy()
        
        elif self.init == 'kmeans++':
            # k-means++ initialization
            centroids = np.zeros((self.n_clusters, n_features))
            
            # Choose first centroid randomly
            centroids[0] = X[np.random.choice(n_samples)]
            
            # Choose remaining centroids
            for i in range(1, self.n_clusters):
                # Compute distances to nearest centroid
                distances = np.min([np.sum((X - c)**2, axis=1) for c in centroids[:i]], axis=0)
                
                # Choose next centroid with probability proportional to distance squared
                probabilities = distances / distances.sum()
                cumulative_probs = probabilities.cumsum()
                r = np.random.rand()
                
                for j, p in enumerate(cumulative_probs):
                    if r < p:
                        centroids[i] = X[j]
                        break
            
            return centroids

    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        """
        Compute weighted distances between samples and centroids.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        centroids : np.ndarray
            Centroids of shape (n_clusters, n_features).
        weights : np.ndarray, optional
            Sample weights of shape (n_samples,).

        Returns
        -------
        np.ndarray
            Distances of shape (n_samples, n_clusters).
        """
        self.logger.info("Computing distances between samples and centroids.")
        n_samples = X.shape[0]
        n_clusters = centroids.shape[0]
        distances = np.zeros((n_samples, n_clusters))
        
        for i in range(n_clusters):
            diff = X - centroids[i]
            if weights is not None:
                # Apply weights to the distance calculation
                distances[:, i] = np.sqrt(np.sum(weights[:, np.newaxis] * diff**2, axis=1))
            else:
                distances[:, i] = np.sqrt(np.sum(diff**2, axis=1))
        
        return distances

    def _assign_clusters(self, distances: np.ndarray) -> np.ndarray:
        """
        Assign samples to nearest centroids.

        Parameters
        ----------
        distances : np.ndarray
            Distances of shape (n_samples, n_clusters).

        Returns
        -------
        np.ndarray
            Cluster labels of shape (n_samples,).
        """
        self.logger.info("Assigning samples to clusters.")
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        """
        Update centroids based on cluster assignments.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        labels : np.ndarray
            Cluster labels of shape (n_samples,).
        weights : np.ndarray, optional
            Sample weights of shape (n_samples,).

        Returns
        -------
        np.ndarray
            Updated centroids of shape (n_clusters, n_features).
        """
        self.logger.info("Updating centroids.")
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))
        
        for i in range(self.n_clusters):
            mask = labels == i
            if np.sum(mask) > 0:
                if weights is not None:
                    # Weighted centroid
                    cluster_weights = weights[mask]
                    weighted_sum = np.sum(cluster_weights[:, np.newaxis] * X[mask], axis=0)
                    new_centroids[i] = weighted_sum / np.sum(cluster_weights)
                else:
                    # Unweighted centroid
                    new_centroids[i] = np.mean(X[mask], axis=0)
            else:
                # If cluster is empty, keep the old centroid or reinitialize
                if self.centroids is not None:
                    new_centroids[i] = self.centroids[i]
                else:
                    new_centroids[i] = X[np.random.choice(X.shape[0])]
        
        return new_centroids

    def _data_conversion(self, z: np.ndarray) -> np.ndarray:
        """
        Convert data using specified data form.

        Parameters
        ----------
        z : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Converted data.
        """
        self.logger.info(f"Converting data using form: {self.data_form}")
        dc = DataConversion()
        if self.data_form == 'a':
            return dc._convert_az(z)
        elif self.data_form == 'm':
            return dc._convert_mz(z)
        else:
            raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative.")

    def _gnostic_criterion(self, z: np.ndarray, z0: np.ndarray, s) -> tuple:
        """
        Compute the gnostic criterion.

        Parameters
        ----------
        z : np.ndarray
            Input data.
        z0 : np.ndarray
            Reference data.
        s : int or np.ndarray
            Scale parameter for the gnostic criterion.

        Returns
        -------
        tuple
            Tuple containing the gnostic criterion values.
            (loss, rentropy, hi, hj, fi, fj, pi, pj, ei, ej, infoi, infoj)
        
        NOTE:
            normalized loss and rentropy are returned.
        """
        self.logger.info("Computing gnostic criterion.")
        q, q1 = self._compute_q(z, z0, s)

        # Default values for optional outputs
        pi = pj = ei = ej = infoi = infoj = None

        if self.mg_loss == 'hi':
            self.logger.info("Computing gnostic criterion for 'hi' loss.")
            hi = self.gc._hi(q, q1)
            fi = self.gc._fi(q, q1)
            fj = self.gc._fj(q, q1)
            re = self.gc._rentropy(fi, fj)
            if self.gnostic_characteristics:
                hj = self.gc._hj(q, q1)
                pi = self.gc._idistfun(hi)
                pj = self.gc._jdistfun(hj)
                infoi = self.gc._info_i(pi)
                infoj = self.gc._info_j(pj)
                ei = self.gc._ientropy(fi)
                ej = self.gc._jentropy(fj)
            else:
                hj = pi = pj = ei = ej = infoi = infoj = None

            # normalize hi and re
            re_norm = (re - np.min(re)) / (np.max(re) - np.min(re)) if np.max(re) != np.min(re) else re
            H = np.sum(hi ** 2)
            return H, np.mean(re_norm), hi, hj, fi, fj, pi, pj, ei, ej, infoi, infoj
        
        elif self.mg_loss == 'hj':
            self.logger.info("Computing gnostic criterion for 'hj' loss.")
            hj = self.gc._hj(q, q1)
            fi = self.gc._fi(q, q1)
            fj = self.gc._fj(q, q1)
            re = self.gc._rentropy(fi, fj)
            if self.gnostic_characteristics:
                hi = self.gc._hi(q, q1)
                pi = self.gc._idistfun(hi)
                pj = self.gc._jdistfun(hj)
                infoi = self.gc._info_i(pi)
                infoj = self.gc._info_j(pj)
                ei = self.gc._ientropy(fi)
                ej = self.gc._jentropy(fj)
            else:
                hi = pi = pj = ei = ej = infoi = infoj = None
            
            # normalize hj and re
            re_norm = (re - np.min(re)) / (np.max(re) - np.min(re)) if np.max(re) != np.min(re) else re
            H = np.sum(hj ** 2)
            return H, np.mean(re_norm), hi, hj, fi, fj, pi, pj, ei, ej, infoi, infoj

    def _compute_q(self, z, z0, s: int = 1):
        """
        Compute q and q1 for gnostic criterion.

        Parameters
        ----------
        z : np.ndarray
            Input data.
        z0 : np.ndarray
            Reference data.
        s : int, default=1
            Scale parameter.

        Returns
        -------
        tuple
            (q, q1) gnostic parameters.
        """
        self.logger.info("Computing q and q1 for gnostic criterion.")
        eps = np_eps_float()  # Small value to avoid division by zero
        z0_safe = np.where(np.abs(z0) < eps, eps, z0)
        zz = z / z0_safe
        self.gc = GnosticsCharacteristics(zz, verbose=self.verbose)
        q, q1 = self.gc._get_q_q1(S=s)
        return q, q1

    def _normalize_weights(self, weights):
        """
        Normalize weights to ensure they sum to 1.

        Parameters
        ----------
        weights : np.ndarray
            Weights to be normalized.

        Returns
        -------
        np.ndarray
            Normalized weights.
        """
        self.logger.info("Normalizing weights.")
        total_weight = np.sum(weights)
        if total_weight == 0:
            return np.ones_like(weights) / len(weights)
        return weights / total_weight

    def _fit(self, X, y=None):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like, optional
            Target values (not used in clustering).
        """
        # Placeholder for fitting logic
        pass

    def _predict(self, X):
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : array-like
            Input features for prediction.

        Returns
        -------
        labels : array-like
            Predicted cluster labels.
        """
        # Placeholder for prediction logic
        pass

    def _score(self, X, y=None):
        """
        Compute the score of the model.

        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like, optional
            True labels (not typically used in clustering).

        Returns
        -------
        score : float
            Score of the model (e.g., inertia).
        """
        # Placeholder for scoring logic
        pass
