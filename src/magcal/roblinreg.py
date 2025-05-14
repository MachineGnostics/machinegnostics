'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.
'''


import numpy as np
from scipy.optimize import minimize
from src.magcal.criterion import GnosticCriterion
from src.magcal.characteristics import GnosticsCharacteristics
from src.magcal.data_conversion import DataConversion
from src.magcal.base import RegressionBase
from abc import ABCMeta, abstractmethod
from scipy.optimize import approx_fprime

class MachineGnosticsRegression(RegressionBase):
    """
    A class to perform Machine Gnostics Regression based on the iterative formula
    described in Equation 19.2 of the provided reference.
    """

    def __init__(self, 
                 X:np.ndarray, 
                 y:np.ndarray, 
                 criterion:str="E1",
                 data_form:str='a',
                 data_varS:bool=True,
                 max_iter=100, 
                 eps=1e-6, 
                 deegree=1):
        """
        Initialize the regression model.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            The input feature matrix.
        - y: np.ndarray, shape (n_samples,)
            The target vector.
        - criterion: str
            The gnostic criterion to use (e.g., "Q1", "E1", "Q2", etc.).
        - data_form: str
            The form of the data (e.g., additive or multiplicative)
        - data_varS: bool
            Whether to use variable S in the data conversion.
            True: homoschedasticity
            False: heteroschedasticity
        - max_iter: int
            Maximum number of iterations for the regression.
        - eps: float
            Tolerance for convergence.
        - deegree: int
            The degree of the polynomial to fit.
        - case: str
            The case for the regression (e.g., "E1", "Q1", etc.).
        """
        self.X = X
        self.y = y
        self.criterion = criterion
        self.max_iter = max_iter
        self.tol = eps
        self.weights = np.ones(X.shape[0])  # Initialize weights to 1 NOTE; can be improved
        self.coefficients = None
        self.deegree = deegree
        self.data_form = data_form
        self.data_varS = data_varS
        self.history = []

        # data form check
        if self.data_form not in ['a', 'm']:
            raise ValueError("Invalid data form. Choose from ['a', 'm'], where 'a' is additive and 'm' is multiplicative.")
        # data varS check
        if self.data_varS not in [True, False]:
            raise ValueError("Invalid data varS. Choose from [True, False]. True for homoschedasticity and False for heteroschedasticity.")
        # criterion validation
        if self.criterion not in ['E1','E2', 'E3', 'Q1', 'Q2', 'Q3']:
            raise ValueError("Invalid criterion. Choose from ['E1', 'E2', 'E3', 'Q1', 'Q2', 'Q3']")
        # degree validation
        if self.deegree < 1:
            raise ValueError("Degree must be greater than or equal to 1.")
        
        def _get_z0(self, X, weights):
            '''
            Get the initial value of Z0
            '''
            return np.dot(X.T, weights)
        
        def _get_gc(self, Z, z0):
            '''
            Get the initial value of Gnostic Characteristics
            '''
            gc = GnosticsCharacteristics(Z/z0)
            fi = gc._fi()
            fj = gc._fj()
            hi = gc._hi()
            hj = gc._hj()
            Ii = gc._info_i()
            Ij = gc._info_j()
            return fi, fj, hi, hj, Ii, Ij

        def _compute_gi(self):
            """
            Compute the relative gradient (g_i) as described in Equation 19.4.

            Parameters:
            - F_prime: np.ndarray, shape (n_samples, n_features)
                The matrix of partial derivatives of the function F with respect to the coefficients.
            - z0: np.ndarray, shape (n_samples,)
                The predicted values (z0) from the current iteration.

            Returns:
            - g_i: np.ndarray, shape (n_samples, n_features)
                The relative gradient for each sample.
            """
            F_prime = self._compute_Fprime()  # Compute the partial derivatives

            # Ensure z0 does not contain zeros to avoid division by zero
            z0 = np.where(z0 == 0, 1e-8, z0)  # Replace zeros with a small value
            g_i = F_prime / z0[:, np.newaxis]  # Compute the relative gradient
            return g_i
            
            pass

        def _compute_Gi(self):
            '''
            Calculated Gi from equation 19.3
            '''
            pass

        def _compute_Ei(self):
            '''
            Calculated Ei from equation 19.5
            '''
            pass

        def _compute_Fprime(self, c, X):
            '''
            Calculated Fprime from equation 19.1

            Parameters:
            - c: np.ndarray, shape (n_features,)
                The coefficients of the regression model.
            - X: np.ndarray, shape (n_samples, n_features)
            '''
            # Define the prediction function
            def F(c, X):
                return np.dot(X, c)
            # Compute numerical partial derivatives
            epsilon = 1e-8  # Small step for numerical differentiation
            F_prime = np.array([approx_fprime(c, lambda c: F(c, X[i, :]), epsilon) for i in range(X.shape[0])])
            return F_prime
                        

        def _compute_coefficients(self):
            '''
            Calculated from equation 19.2
            '''
            pass

        def _optiimize_S(self):
            '''
            Optimized S from equation 19.6
            '''
            pass

        def _history(self):
            """
            Compute the history of the regression process.

            Returns:
            - history: list
                A list containing the history of the regression process.
            """
            return self.history

        def fit(self, X, y):
            '''
            Train the regression model using the input features and target values.
            Parameters:
            - X: np.ndarray, shape (n_samples, n_features)
                The input feature matrix.
            - y: np.ndarray, shape (n_samples,)
                The target vector.
            '''
            # data check
            if X.shape[0] != y.shape[0]:
                raise ValueError("Number of samples in X and y must be the same.")
            if X.ndim < 1:
                raise ValueError("X must be a 2D array.")
            if y.ndim != 1:
                raise ValueError("y must be a 1D array.")
            if self.deegree > X.shape[1]:
                raise ValueError("Degree must be less than or equal to the number of features in X.")
            # data homoscedasticity check
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                raise ValueError("Input data contains NaN values.")
            if np.any(np.isinf(X)) or np.any(np.isinf(y)):
                raise ValueError("Input data contains infinite values.")

            # convert data to az or mz and low values check
            
            # calculate q = z/z0, fi and S

            # re calculate q, gnostic characteristics (hi, hj, fi, fj, Ii, and Ij), with optimized S

            # get gnostic criterion

            # get filtering weights

            # get error function

            # gi calculation

            # Gi calculation

            # Ei calculation

            # calculate the coefficients using the iterative formula

            # Update the coefficients

            # Store the coefficients

            # Store the history of the regression process

            # Store the gnostic characteristics
            pass

        def predict(self, X):
            """
            Predict the target values for the given input features.

            Parameters:
            - X: np.ndarray, shape (n_samples, n_features)
                The input feature matrix.

            Returns:
            - y_pred: np.ndarray, shape (n_samples,)
                The predicted target values.
            """
            if self.coefficients is None:
                raise ValueError("Model is not fitted yet.")
            return np.dot(X, self.coefficients)