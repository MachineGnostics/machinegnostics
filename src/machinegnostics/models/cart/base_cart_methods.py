'''
CartMethodsBase - Base class for Machine Gnostics CART Methods

This class serves as the foundational base for Gnostic Random Forest and other CART methods.

Copyright (C) Machine Gnostics
'''

import logging
from machinegnostics.magcal.util.logging import get_logger
import numpy as np
from machinegnostics.magcal import (GnosticsCharacteristics, 
                                    DataConversion,
                                    GnosticsWeights)
from machinegnostics.models.base_model import ModelBase
from machinegnostics.magcal.util.min_max_float import np_max_float, np_min_float, np_eps_float
from sklearn.tree import DecisionTreeRegressor
from typing import Union, Optional

class CartMethodsBase(ModelBase):
    """
    Base class for Machine Gnostics CART Methods.
    """
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 max_iter: int = 10,
                 tolerance: float = 1e-4,
                 mg_loss: str = 'hi',
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: Union[str, int, float] = 'auto',
                 history: bool = True,
                 data_form: str = 'a',
                 gnostic_characteristics: bool = False,
                 random_state: Optional[int] = None,
                 estimator_type: str = 'forest'):
        super().__init__(verbose=verbose)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.mg_loss = mg_loss
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.scale = scale
        self.history = history
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self.random_state = random_state
        self.estimator_type = estimator_type
        
        self.trees = []
        self.tree = None
        self.weights = None
        
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)

        if self.history:
            self._history = []
        else:
            self._history = None

    def _input_checks(self):
        """Perform input validation."""
        self.logger.info("Performing input checks for arguments.")
        if not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("max_iter must be a non-negative integer.")
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

    def _weight_init(self, n_samples: int) -> np.ndarray:
        """Initialize weights to uniform."""
        return np.ones(n_samples)

    def _fit_forest_impl(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> list:
        """
        Implementation of Random Forest training using weighted bootstrap.
        
        Parameters
        ----------
        X : np.ndarray
            Features.
        y : np.ndarray
            Targets.
        sample_weight : np.ndarray
            Sample weights.
            
        Returns
        -------
        list
            List of trained DecisionTreeRegressor objects.
        """
        n_samples = X.shape[0]
        trees = []
        rng = np.random.RandomState(self.random_state)
        
        # Normalize weights for probability
        p = sample_weight / np.sum(sample_weight)
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling with weights
            indices = rng.choice(n_samples, size=n_samples, replace=True, p=p)
            X_sample = X[indices]
            y_sample = y[indices]
            # We don't need to pass sample_weight to tree fit if we resampled based on weights
            # But passing uniform weights avoids confusion, or we can just pass None.
            
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split,
                random_state=rng.randint(0, 2**32 - 1)
            )
            tree.fit(X_sample, y_sample)
            trees.append(tree)
            
        return trees

    def _fit_single_tree_impl(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> DecisionTreeRegressor:
        """
        Implementation of Single Tree training using sample weights.
        """
        rng = np.random.RandomState(self.random_state)
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=rng.randint(0, 2**32 - 1)
        )
        tree.fit(X, y, sample_weight=sample_weight)
        return tree

    def _predict_forest_impl(self, X: np.ndarray, trees: list) -> np.ndarray:
        """
        Predict using the forest.
        """
        if not trees:
             raise ValueError("Forest is empty.")
             
        predictions = np.zeros((X.shape[0], len(trees)))
        for i, tree in enumerate(trees):
            predictions[:, i] = tree.predict(X)
            
        # Standard averaging
        return np.mean(predictions, axis=1)

    def _predict_single_tree_impl(self, X: np.ndarray, tree: DecisionTreeRegressor) -> np.ndarray:
        """
        Predict using a single tree.
        """
        if tree is None:
             raise ValueError("Tree is not fitted.")
        return tree.predict(X)

    def _data_conversion(self, z: np.ndarray) -> np.ndarray:
        """Convert data using form."""
        dc = DataConversion()
        if self.data_form == 'a':
            return dc._convert_az(z)
        elif self.data_form == 'm':
            return dc._convert_mz(z)
        else:
            raise ValueError("data_form must be 'a' or 'm'.")

    def _compute_q(self, z, z0, s: int = 1):
        """Compute q and q1."""
        eps = np_eps_float()
        z0_safe = np.where(np.abs(z0) < eps, eps, z0)
        zz = z / z0_safe
        self.gc = GnosticsCharacteristics(zz, verbose=self.verbose)
        q, q1 = self.gc._get_q_q1(S=s)
        return q, q1

    def _gnostic_criterion(self, z: np.ndarray, z0: np.ndarray, s) -> tuple:
        """Compute gnostic criterion."""
        q, q1 = self._compute_q(z, z0, s)
        
        pi = pj = ei = ej = infoi = infoj = None

        if self.mg_loss == 'hi':
            hi = self.gc._hi(q, q1)
            fi = self.gc._fi(q, q1)
            fj = self.gc._fj(q, q1)
            re = self.gc._rentropy(fi, fj)
            if self.gnostic_characteristics:
                hj = self.gc._hj(q, q1)
                # Additional metrics if needed
            else:
                hj = None
            
            re_norm = (re - np.min(re)) / (np.max(re) - np.min(re)) if np.max(re) != np.min(re) else re
            H = np.sum(hi ** 2)
            return H, np.mean(re_norm), hi, hj, fi, fj, pi, pj, ei, ej, infoi, infoj
        elif self.mg_loss == 'hj':
            # Implement if needed, for now similar to 'hi' or just define basic
            hj = self.gc._hj(q, q1)
            fi = self.gc._fi(q, q1)
            fj = self.gc._fj(q, q1)
            re = self.gc._rentropy(fi, fj)
            hi = None
            re_norm = (re - np.min(re)) / (np.max(re) - np.min(re)) if np.max(re) != np.min(re) else re
            H = np.sum(hj ** 2)
            return H, np.mean(re_norm), hi, hj, fi, fj, pi, pj, ei, ej, infoi, infoj
        return 0, 0, None, None, None, None, None, None, None, None, None, None
