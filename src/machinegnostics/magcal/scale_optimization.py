'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

Author: Nirmal Parmar
'''

import numpy as np
from scipy.optimize import minimize
from typing import Union
from machinegnostics.magcal.characteristics import GnosticsCharacteristics

class ScaleOptimization(GnosticsCharacteristics):
    """
    A class to perform scale optimization on a given matrix. This class is for internal use.

    Parameters
    ----------
    R : np.ndarray
        The input matrix for the scale optimization.
    eps : float, optional
        A small value to avoid division by zero (default is np.finfo(float).max).
    c : str [in {'i', 'j'}]
        The type of scale optimization to perform. 'i' for estimation and 'j' for quantification.
        c2 ∈{1,−1}and hc be the irrelevance, either hj (quantifying, c2 = 1) or hi (estimating, c2=−1)
    eps : float, optional
        A small value to avoid division by zero (default is np.finfo(float).eps).

    Attributes
    ----------
    q : np.ndarray
        The input matrix.
    q1 : np.ndarray
        The inverse of the input matrix, with protection against division by zero.
    """
    
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 c: str, 
                 eps: float = np.finfo(float).eps):
        """
        Initializes the ScaleOptimization class.

        Parameters
        ----------
        X : np.ndarray
            The input matrix for the scale optimization.
        y : np.ndarray
            The target values for the optimization.
        c : str [in {'i', 'j'}]
            The type of scale optimization to perform. 'i' for estimation and 'j' for quantification.
        eps : float, optional
            A small value to avoid division by zero (default is np.finfo(float).max).
        """
    
    def _F(self, C, x)-> float:
        """
        Computes the predicted values based on the coefficients C and input features x_obs.

        Parameters
        ----------
        C : np.ndarray
            Coefficients for the regression model.
        x_obs : np.ndarray
            Observed input features.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        return np.dot(x, C)
    
    def _recompute_q(self, Z, z0, S)-> np.ndarray:
        """
        Computes the q values for optimization.

        Parameters
        ----------
        Z_obs : np.ndarray
            Observed target values.
        z0 : np.ndarray
            Predicted target values.
        S : float
            Scale parameter.

        Returns
        -------
        np.ndarray
            Computed q values.
        """
        q = np.abs(Z / z0) / (S+self.eps) # to avoid division by zero
        q1 = 1 / q
        q1 = np.where(q1 != 0, q1, np.finfo(float).max)
        return q, q1
    
    def _recompute_h(self, q, q1, c:str)-> np.ndarray:
        """
        Computes the h values for optimization.

        Parameters
        ----------
        q : np.ndarray
            Computed q values.
        c : str [in {'i', 'j'}]
            The type of scale optimization to perform. 'i' for estimation and 'j' for quantification.

        Returns
        -------
        np.ndarray
            Computed h values.
        """
        if c == 'i':
            return self._hi(q, q1)
        elif c == 'j':
            return self._fj(q, q1)
        else:
            raise ValueError("Invalid value for c. Must be 'i' or 'j'.")
        
    def _criterion(self, C, S, x, Z):
        """
        Computes the criterion function for optimization.

        Parameters
        ----------
        C : np.ndarray
            Coefficients for the regression model.
        S : float
            Scale parameter.
        x : np.ndarray
            Input features.
        Z : np.ndarray
            Target values.

        Returns
        -------
        float
            Computed criterion value.
        """
        z0 = self._F(C, x)
        q, q1 = self._recompute_q(Z, z0, S)
        h = self._recompute_h(q, q1, self.c) 
        # Compute the criterion value (e.g., sum of squares)
        D_hi = h ** 2 # simple quadratic loss
        return np.sum(D_hi)
    
    def _optimize(self, x, Z):
        """
        Optimizes the scale parameter S and coefficients C.

        Parameters
        ----------
        x : np.ndarray
            Input features.
        Z : np.ndarray
            Target values.

        Returns
        -------
        tuple
            Optimized coefficients C and scale parameter S.
        """
        # Initial guess for C and S
        C0 = np.ones(x.shape[1])
        S0 = 1.0

        # Define the objective function to minimize
        def objective(params):
            C = params[:-1]
            S = np.abs(params[-1])
            return self._criterion(C, S, x, Z)

        # Initial guess for parameters
        initial_params = np.concatenate((C0, [S0]))

        # Perform optimization
        result = minimize(objective, initial_params, method='BFGS',)

        # Extract optimized coefficients and scale parameter
        optimized_params = result.x
        C_opt = optimized_params[:-1]
        S_opt = optimized_params[-1]

        return C_opt, S_opt
