'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: ManGo Team, Nirmal Parmar
Date: 2025-10-01
Description: Machine Gnostics Robust Regression Machine Learning Model
This module implements a machine learning model for robust regression using the ManGo library.
This model is designed to handle various types of data and is particularly useful for applications in machine gnostics.
'''

import numpy as np
from src.magcal.base import RegressionBase
from src.magcal import GnosticsCharacteristics, DataConversion, GnosticCriterion

class RobustRegressor(RegressionBase):
    """
    Robust Regression Model for Machine Gnostics (Machine Learning). This model is based on non-statistical (Mathematical Gnostics) methods
    and is designed to be robust against outliers and noise in the data.


    """
    def __init__(self,
                 degree: int = 2,
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 mg_loss: str = 'hi',
                 early_stopping: bool = True,
                 verbose: bool = False,
                 history: bool = False):
        '''
        Robust Regressor - Machine Gnostics
        
        Initialize the regression model.
        
        Parameters:
        -----------
        degree : int
            Degree of polynomial features
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        early_stopping : int
            Number of iterations for early stopping check
        verbose: bool
            To print verbose
        history:
            To save history of the Gnostics Characteristics
        '''
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None
        self.weights = None
        self.early_stopping = early_stopping
        self.mg_loss = mg_loss
        self.verbose = verbose
        self.history = history

        if history:
            self._history = []

    def _generate_polynomial_features(self, X):
        """
        Generate polynomial features up to specified degree.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, 1)
            Input features
            
        Returns:
        --------
        array-like of shape (n_samples, degree + 1)
            Polynomial features including bias term
        """
        n_samples = len(X)
        X_poly = np.ones((n_samples, self.degree + 1))
        for d in range(1, self.degree + 1):
            X_poly[:, d] = X.ravel() ** d
        return X_poly
    
    def _compute_q(self, z, z0, s:int = 1):
        """
        For interval use only
        Compute q and q1."""
        self.gc = GnosticsCharacteristics(z/z0)
        q, q1 = self.gc._get_q_q1(S=s)     
        return q, q1
    
    def _compute_hi(self, q, q1):
        """Compute estimation relevance (hi)."""
        hi = self.gc._hi(q,q1)
        return hi
    
    def _gnostic_criterion(self, z, z0, s):
        """Compute the gnostic criterion."""
        q, q1 = self._compute_q(z, z0, s)
        if self.mg_loss == 'hi':
            hi = self.gc._hi(q, q1)
            return np.sum(hi ** 2)
        elif self.mg_loss == 'hj':
            hj = self.gc._hj(q, q1)
            return hj

    