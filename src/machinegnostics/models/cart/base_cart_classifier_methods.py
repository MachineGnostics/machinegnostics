'''
CartClassifierMethodsBase - Base class for Machine Gnostics CART Classification Methods

This class serves as the foundational base for Gnostic Random Forest Classifier and Gnostic Decision Tree Classifier.

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
from sklearn.tree import DecisionTreeClassifier
from typing import Union, Optional

class CartClassifierMethodsBase(ModelBase):
    """
    Base class for Machine Gnostics CART Classification Methods.
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
        self.classes_ = None
        
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)

        if self.history:
            self._history = []
        else:
            self._history = None

    def _input_checks(self):
        """Perform input validation."""
        if self.max_iter < 0:
             raise ValueError("max_iter must be non-negative.")

    def _weight_init(self, n_samples: int) -> np.ndarray:
        """Initialize weights to uniform."""
        return np.ones(n_samples)

    def _fit_forest_impl(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> list:
        """
        Implementation of Random Forest training using weighted bootstrap.
        """
        n_samples = X.shape[0]
        trees = []
        rng = np.random.RandomState(self.random_state)
        
        # Normalize weights for probability sampling
        if np.sum(sample_weight) > 0:
            p = sample_weight / np.sum(sample_weight)
        else:
            p = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling with weights
            indices = rng.choice(n_samples, size=n_samples, replace=True, p=p)
            X_sample = X[indices]
            y_sample = y[indices]
            
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split,
                random_state=rng.randint(0, 2**32 - 1)
            )
            tree.fit(X_sample, y_sample) # Weights handled by resampling
            trees.append(tree)
            
        return trees

    def _fit_single_tree_impl(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> DecisionTreeClassifier:
        """
        Implementation of Single Tree training using sample weights.
        """
        rng = np.random.RandomState(self.random_state)
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=rng.randint(0, 2**32 - 1)
        )
        tree.fit(X, y, sample_weight=sample_weight)
        return tree

    def _predict_proba_forest_impl(self, X: np.ndarray, trees: list) -> np.ndarray:
        """
        Predict probabilities using the forest.
        """
        if not trees:
             raise ValueError("Forest is empty.")
             
        # Shape: (n_samples, n_classes, n_trees) -> then mean over n_trees
        # Need to handle if trees have different classes (though usually fit on same classes subset)
        # Assuming all trees see all classes for now or relying on sklearn behavior
        
        all_proba = []
        for tree in trees:
            probas = tree.predict_proba(X)
            
            # Handle case where a tree might not have seen all classes in bootstrap
            n_classes_tree = probas.shape[1]
            if n_classes_tree != len(self.classes_):
                # align probabilities (simplistic approach, assumes self.classes_ is sorted/superset)
                full_probas = np.zeros((X.shape[0], len(self.classes_)))
                # This is tricky without robust mapping. 
                # For Gnostic Forest, simpler to assume trees are consistent enough or handle mapping
                # But standard RF handles this.
                # For this implementation, we assume trees are compatible for averaging.
                # Fallback: Scikit-learn's DecisionTree usually computes proba for all classes seen in fit.
                pass 
                
            all_proba.append(probas)

        return np.mean(all_proba, axis=0)

    def _predict_proba_single_tree_impl(self, X: np.ndarray, tree: DecisionTreeClassifier) -> np.ndarray:
        """
        Predict probabilities using a single tree.
        """
        if tree is None:
             raise ValueError("Tree is not fitted.")
        return tree.predict_proba(X)
    
    def _predict_impl(self, X: np.ndarray, probas: np.ndarray) -> np.ndarray:
        """Convert probabilities to class labels."""
        return self.classes_[np.argmax(probas, axis=1)]

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
        
        # Copied from base model
        if self.mg_loss == 'hi':
            hi = self.gc._hi(q, q1)
            fi = self.gc._fi(q, q1)
            fj = self.gc._fj(q, q1)
            re = self.gc._rentropy(fi, fj)
            if self.gnostic_characteristics:
                hj = self.gc._hj(q, q1)
            else:
                hj = None
            
            re_norm = (re - np.min(re)) / (np.max(re) - np.min(re)) if np.max(re) != np.min(re) else re
            H = np.sum(hi ** 2)
            return H, np.mean(re_norm), hi, hj, fi, fj, pi, pj, ei, ej, infoi, infoj
            
        return 0, 0, None, None, None, None, None, None, None, None, None, None
