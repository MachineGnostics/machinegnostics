'''
History class for the Robust Clustering model.

Machine Gnostics
Author: Nirmal Parmar
'''

import numpy as np
from machinegnostics.models.clustering.base_clustering_cal import ClusteringCalBase
import logging
from machinegnostics.magcal.util.logging import get_logger

class HistoryClusteringBase(ClusteringCalBase):
    """
    History class for the Robust Clustering model.
    
    This class extends ClusteringCalBase to maintain a history
    of model parameters and gnostic loss values during training iterations.
    
    Parameters needed to record history:
        - h_loss: Gnostic loss value at each iteration
        - iteration: The iteration number
        - weights: Sample weights at each iteration
        - centroids: Cluster centroids at each iteration
        - labels: Cluster labels at each iteration
        - n_clusters: Number of clusters
        - rentropy: Entropy of the model at each iteration
        - inertia: Sum of squared distances at each iteration
        - fi, hi, fj, hj, infoi, infoj, pi, pj, ei, ej: Additional gnostic information if calculated
    """
    
    def __init__(self,
                 n_clusters: int = 3,
                 max_iter: int = 100,
                 tolerance: float = 1e-3,
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
        self._history = history
        self.init = init
        
        self.params = [
            {
                'iteration': 0,
                'loss': None,
                'weights': None,
                'centroids': None,
                'labels': None,
                'n_clusters': self.n_clusters,
                'rentropy': None,
                'inertia': None,
                'fi': None,
                'hi': None,
                'fj': None,
                'hj': None,
                'infoi': None,
                'infoj': None,
                'pi': None,
                'pj': None,
                'ei': None,
                'ej': None
            }
        ]
        
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.info(f"{self.__class__.__name__} initialized:")
        self.logger.info("HistoryClusteringBase initialized.")
    
    def _fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit the model to the data and record history.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray, optional
            Not used, present for API consistency.
        """
        self.logger.info("Starting fit process for HistoryClusteringBase.")
        # Call the parent fit method to perform fitting
        super()._fit(X, y)

        # Record the final state in history as a dict
        params_dict = {}

        if self.gnostic_characteristics:
            # Compute final inertia
            inertia = 0.0
            if self.labels is not None and self.centroids is not None:
                for i in range(len(X)):
                    inertia += np.sum((X[i] - self.centroids[self.labels[i]]) ** 2)
            
            params_dict['iteration'] = self._iter
            params_dict['loss'] = self.loss
            params_dict['weights'] = self.weights.copy() if self.weights is not None else None
            params_dict['centroids'] = self.centroids.copy() if self.centroids is not None else None
            params_dict['labels'] = self.labels.copy() if self.labels is not None else None
            params_dict['n_clusters'] = self.n_clusters
            params_dict['rentropy'] = self.re
            params_dict['inertia'] = inertia
            params_dict['fi'] = self.fi
            params_dict['hi'] = self.hi
            params_dict['fj'] = self.fj
            params_dict['hj'] = self.hj
            params_dict['infoi'] = self.infoi
            params_dict['infoj'] = self.infoj
            params_dict['pi'] = self.pi
            params_dict['pj'] = self.pj
            params_dict['ei'] = self.ei
            params_dict['ej'] = self.ej
        else:
            # Compute final inertia
            inertia = 0.0
            if self.labels is not None and self.centroids is not None:
                for i in range(len(X)):
                    inertia += np.sum((X[i] - self.centroids[self.labels[i]]) ** 2)
            
            params_dict['iteration'] = self._iter if hasattr(self, '_iter') else 0
            params_dict['loss'] = None
            params_dict['weights'] = self.weights.copy() if self.weights is not None else None
            params_dict['centroids'] = self.centroids.copy() if self.centroids is not None else None
            params_dict['labels'] = self.labels.copy() if self.labels is not None else None
            params_dict['n_clusters'] = self.n_clusters
            params_dict['inertia'] = inertia

        self.params.append(params_dict)
