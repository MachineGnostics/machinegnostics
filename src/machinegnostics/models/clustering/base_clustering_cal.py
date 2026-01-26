'''
ClusteringCalBase class for clustering models internal calculations.

Primary to develop methods like fit, predict, score in clustering models.

Clustering param base class that can be used for robust clustering models.
- k-means clustering

Machine Gnostics
Author: Nirmal Parmar
'''

import numpy as np
import logging
from machinegnostics.magcal.util.logging import get_logger
from machinegnostics.models.clustering.base_clustering_methods import ClusteringMethodsBase
from machinegnostics.magcal.util.min_max_float import np_max_float, np_min_float
from machinegnostics.magcal import (ScaleParam, 
                                    GnosticsWeights)


class ClusteringCalBase(ClusteringMethodsBase):
    """
    Base class for clustering model internal calculations.

    This class serves as a foundation for implementing methods such as fit, predict, and score
    in various clustering models. It extends the ClusteringMethodsBase class to provide core
    functionalities required for clustering tasks.

    Inherits
    --------
    ClusteringMethodsBase
        Provides basic methods for clustering models.
    """
    def __init__(self,
                 n_clusters: int = 3,
                 max_iter: int = 100,
                 tolerance: float = 1e-9,
                 mg_loss: str = 'hi',
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: 'str | int | float' = 'auto',
                 data_form: str = 'a',
                 gnostic_characteristics: bool = True,
                 history: bool = True,
                 init: str = 'random'):
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
        self.history = history
        self.init = init

        # Auto-enable history when early_stopping is True
        if self.early_stopping and not self.history:
            self.logger.warning(
                "early_stopping=True requires history=True. Automatically enabling history."
            )
            self.history = True
            self.logger.info("History has been enabled.")

        # history option
        if history:
            self._history = []
            # default history content
            self._history.append({
                'iteration': 0,
                'h_loss': None,
                'centroids': None,
                'labels': None,
                'rentropy': None,
                'weights': None,
            })
        else:
            self._history = None
        
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.info(f"{self.__class__.__name__} initialized:")
    
    def _fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit the clustering model to the data.
        
        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features).
        y : np.ndarray, optional
            Not used, present for API consistency.
        """
        self.logger.info("Starting fit process for ClusteringCalBase.")
        
        # Initialize weights
        self.weights = self._weight_init(d=X, like='one')
        
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        # Initialize labels
        self.labels = None
        
        for self._iter in range(self.max_iter):
            self._iter += 1
            self._prev_centroids = self.centroids.copy()
            
            try:
                # Compute distances with current weights
                distances = self._compute_distances(X, self.centroids, self.weights)
                
                # Assign clusters
                self.labels = self._assign_clusters(distances)
                
                # Update centroids with weighted samples
                self.centroids = self._update_centroids(X, self.labels, self.weights)
                
                # Compute residuals for gnostic weight update
                # Use distance from assigned centroid as residual
                residuals = np.zeros(X.shape[0])
                for i in range(len(X)):
                    residuals[i] = np.linalg.norm(X[i] - self.centroids[self.labels[i]])
                
                # Compute inertia (sum of squared distances to nearest centroid)
                inertia = np.sum(residuals ** 2)
                
                # Data conversion for gnostic calculations
                z_residuals = self._data_conversion(residuals)
                
                # Create a reference based on average distance
                z0_residuals = np.full_like(residuals, np.mean(residuals) + 1e-8)
                z0_residuals = self._data_conversion(z0_residuals)
                
                # Gnostic weights - KEY FEATURE: multiply weights with gw
                gwc = GnosticsWeights()
                gw = gwc._get_gnostic_weights(z_residuals)
                new_weights = self.weights * gw  # Multiply existing weights with gnostic weights

                # Compute scale and loss
                if self.scale == 'auto':
                    s = gwc.s
                else:
                    s = self.scale

                # Compute gnostic criterion
                self.loss, self.re, self.hi, self.hj, self.fi, self.fj, \
                self.pi, self.pj, self.ei, self.ej, self.infoi, self.infoj = self._gnostic_criterion(
                    z=z_residuals, z0=z0_residuals, s=s
                )

                # Normalize weights
                self.weights = new_weights / np.sum(new_weights)
                                                
                # Print loss
                if self.verbose:
                    self.logger.info(f'Iteration: {self._iter} - Machine Gnostic loss - {self.mg_loss}: {np.round(self.loss, 4)}, rentropy: {np.round(self.re, 4)}, inertia: {np.round(inertia, 4)}')

                # Capture history and append to history
                if self._history is not None:
                    self._history.append({
                        'iteration': self._iter,
                        'h_loss': self.loss,
                        'centroids': self.centroids.copy(),
                        'labels': self.labels.copy(),
                        'rentropy': self.re,
                        'weights': self.weights.copy(),
                        'inertia': inertia
                    })

                # Check convergence with early stopping
                centroid_shift = np.linalg.norm(self.centroids - self._prev_centroids)
                
                if self.early_stopping and self._history is not None:
                    if len(self._history) > 1:
                        prev_loss = self._history[-2]['h_loss']
                        prev_re = self._history[-2]['rentropy']
                        if (prev_loss is not None) and (prev_re is not None):
                            if (np.abs(self.loss - prev_loss) < self.tolerance) or (np.abs(self.re - prev_re) < self.tolerance):
                                if self.verbose:
                                    self.logger.info(f"Convergence reached at iteration {self._iter} with loss/rentropy change below tolerance.")
                                break
                
                # Also check centroid convergence
                if centroid_shift < self.tolerance:
                    if self.verbose:
                        self.logger.info(f"Convergence reached at iteration {self._iter} with centroid shift below tolerance.")
                    break
            
            except (ZeroDivisionError, np.linalg.LinAlgError) as e:
                if self.verbose:
                    self.logger.warning(f"Warning: {str(e)}. Using previous centroids.")
                self.centroids = self._prev_centroids
                break

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Internal prediction method for base class.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features to predict cluster labels for.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted cluster labels.
        """ 
        self.logger.info("Starting prediction for ClusteringCalBase.")
        
        if self.centroids is None:
            self.logger.error("Model has not been fitted yet.")
            raise ValueError("Model has not been fitted yet.")
        
        # Compute distances to centroids (without weights for prediction)
        distances = self._compute_distances(X, self.centroids, weights=None)
        
        # Assign to nearest centroid
        labels = self._assign_clusters(distances)
        
        return labels

    def _score(self, X: np.ndarray, y: np.ndarray = None) -> float:
        """
        Compute the inertia (sum of squared distances to nearest centroid).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like, optional
            Not used, present for API consistency.
            
        Returns
        -------
        float
            Negative inertia (higher is better for sklearn compatibility).
        """
        self.logger.info("Computing inertia score for ClusteringCalBase.")
        
        if self.centroids is None:
            self.logger.error("Model has not been fitted yet.")
            raise ValueError("Model has not been fitted yet.")
        
        # Predict labels
        labels = self._predict(X)
        
        # Compute inertia
        inertia = 0.0
        for i in range(len(X)):
            inertia += np.sum((X[i] - self.centroids[labels[i]]) ** 2)
        
        # Return negative inertia (higher is better)
        return -inertia
