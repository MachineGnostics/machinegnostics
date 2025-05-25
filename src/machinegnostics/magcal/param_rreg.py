'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
Date: 2025-10-01
Description: Machine Gnostics Robust Regression Machine Learning Model
This module implements a machine learning model for robust regression using the Machine Gnostics library.
This model is designed to handle various types of data and is particularly useful for applications in machine gnostics.
'''

import numpy as np
from itertools import combinations_with_replacement
from machinegnostics.magcal import RegressorBase, GnosticsCharacteristics, DataConversion, ScaleParam, GnosticsWeights

class RegressorParamBase(RegressorBase):
    '''
    Parameter base class to perform calculation and record parameters

    Only for internal use.

    Methods:
    -------
    _generate_polynomial_features(self, X)

    _compute_q(z, z0, s)

    _compute_hi(q, q1)

    _gnostic_criterion(z, z0, s)

    _weighted_least_squares(X_poly, y, weights)

    _fit(X, y)

    _predict(X)

    _process_input()

    _process_output()
    '''
    def __init__(self,
                 degree: int = 1,
                 max_iter: int = 100,
                 tol: float = 1e-8,
                 mg_loss: str = 'hi',
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: [str, int, float] = 'auto',
                 history: bool = True,
                 data_form: str = 'a'):
        super().__init__()
    
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
        mg_loss: str
            Select the gnostic loss
        early_stopping : int
            Number of iterations for early stopping check
        verbose: bool
            To print verbose
        '''
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None
        self.weights = None
        self.early_stopping = early_stopping
        self.mg_loss = mg_loss
        self.verbose = verbose
        self.data_form = data_form
         # --- Scale input handling ---
        if isinstance(scale, str):
            if scale != 'auto':
                raise ValueError("scale must be 'auto' or a float between 0 and 2.")
            self.scale_value = 'auto'
        elif isinstance(scale, (int, float)):
            if not (0 <= scale <= 2):
                raise ValueError("scale must be 'auto' or a float between 0 and 2.")
            self.scale_value = float(scale)
        else:
            raise ValueError("scale must be 'auto' or a float between 0 and 2.")
        # data form check additive or multiplicative
        if self.data_form not in ['a', 'm']:
            raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative.")
        # history option
        if history:
            self._history = []
            # default history content
            self._history.append({
                'iteration': 0,
                'h_loss': None,
                'coefficients': None,
                'rentropy': None,
                'weights': None,
            })
        else:
            self._history = None

    def _process_input(self, X, y=None):
        '''
        Processing input

        Parameters:
            X: array-like, DataFrame, or Spark DataFrame
            y: array-like, DataFrame, Spark DataFrame, or None
        
        Returns:
            X: np.ndarray, shape (n_samples, n_features)
            y: np.ndarray or None, shape (n_samples,)
        '''
        import numpy as np

        # Identify input types
        is_pandas = False
        is_spark = False

        try:
            import pandas as pd
            if isinstance(X, (pd.DataFrame, pd.Series)):
                is_pandas = True
        except ImportError:
            pass

        try:
            from pyspark.sql import DataFrame as SparkDF
            if isinstance(X, SparkDF):
                is_spark = True
        except ImportError:
            pass

        # Convert X to NumPy array
        if is_pandas:
            X_np = X.to_numpy()
        elif is_spark:
            raise NotImplementedError("Processing Spark DataFrames requires distributed-safe logic. Consider collecting to Pandas first.")
        else:
            X_np = np.asarray(X)

        # Reshape X if needed
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        elif X_np.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")

        # Process y if provided
        if y is not None:
            if is_pandas:
                y_np = y.to_numpy().flatten()
            elif is_spark:
                raise NotImplementedError("Processing Spark DataFrames requires distributed-safe logic. Consider collecting to Pandas first.")
            else:
                y_np = np.asarray(y).flatten()

            # Validate y shape
            if y_np.ndim != 1:
                raise ValueError("y must be a 1D array of shape (n_samples,).")

            # Check samples consistency
            if X_np.shape[0] != y_np.shape[0]:
                raise ValueError(f"Number of samples in X and y must match. Got {X_np.shape[0]} and {y_np.shape[0]}.")

            return X_np, y_np

        else:
            # y is None, just return processed X and None
            return X_np
    
    def _process_output(self):
        '''
        Preparing output and saving params of the trained model

        For internal use only
        '''
        # capturing history
        if self._history is not None:
            self._history.append({
                'iteration': self._iter,
                'h_loss': None,  # Placeholder for loss, to be computed later
                'coefficients': self.coefficients.copy(),
                'rentropy': None,  # Placeholder for rentropy, to be computed later
                'weights': self.weights.copy(),
            })

    def _generate_polynomial_features(self, X):
        """
        Generate polynomial features for multivariate input up to specified degree.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input features

        Returns:
        --------
        X_poly : ndarray of shape (n_samples, n_output_features)
            Polynomial features including interaction terms
        """
        n_samples, n_features = X.shape
        combinations = []
        for degree in range(self.degree + 1):
            combinations += list(combinations_with_replacement(range(n_features), degree))

        X_poly = np.ones((n_samples, len(combinations)))
        for i, comb in enumerate(combinations):
            X_poly[:, i] = np.prod(X[:, comb], axis=1)
        
        return X_poly
    
    def _compute_q(self, z, z0, s:int = 1):
        """
        For interval use only
        Compute q and q1."""
        eps = np.finfo(float).eps
        z0_safe = np.where(np.abs(z0) < eps, eps, z0)
        zz = z / z0_safe
        self.gc = GnosticsCharacteristics(zz)
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
            fi = self.gc._fi(q, q1)
            fj = self.gc._fj(q, q1)
            re = self.gc._rentropy(fi, fj)
            H = np.sum(hi ** 2)
            return H, re.mean()
        elif self.mg_loss == 'hj':
            hj = self.gc._hj(q, q1)
            fi = self.gc._fi(q, q1)
            fj = self.gc._fj(q, q1)
            re = self.gc._rentropy(fi, fj)
            H = np.sum(hj ** 2)
            return H, re.mean()
    
    def _data_conversion(self, z):
        dc = DataConversion()
        if self.data_form == 'a':
            return dc._convert_az(z)
        elif self.data_form == 'm':
            return dc._convert_mz(z)
        else:
            raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative.")

    def _weighted_least_squares(self, X_poly, y, weights):
        """
        Solve weighted least squares using normal equations.
        
        Parameters:
        -----------
        X_poly : array-like
            Polynomial features matrix
        y : array-like
            Target values
        weights : array-like
            Sample weights
            
        Returns:
        --------
        array-like
            Estimated coefficients
        """
        eps = np.finfo(float).eps
        # Add small regularization term
        weights = np.clip(weights, eps, None)
        W = np.diag(weights)
        XtW = X_poly.T @ W
        XtWX = XtW @ X_poly + eps * np.eye(X_poly.shape[1])
        XtWy = XtW @ y
        
        try:
            return np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse for ill-conditioned matrices
            return np.linalg.pinv(XtWX) @ XtWy
    
    def _fit(self, X, y):
        '''
        Fit the Robust Regressor model using gnostic weights and polynomial features.

        This method trains the regression model on the given dataset `(X, y)` by iteratively applying
        weighted least squares. The weights are updated at each iteration using a mathematical
        gnostics-based approach that makes the model robust to noise and outliers.

        The process involves:
        - Expanding the input features into a polynomial basis.
        - Solving the weighted least squares problem to estimate model coefficients.
        - Computing residuals and transforming them into a gnostic space (via z-values).
        - Computing gnostic weights to adjust the influence of each data point.
        - Updating and normalizing the weights.
        - Evaluating convergence based on changes in loss and coefficient values.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Input feature values. Can be a 1D array or a single-column 2D array.
        
        y : array-like of shape (n_samples,)
            Target values corresponding to input samples.

        Attributes Updated
        ------------------
        self.coefficients : ndarray of shape (degree + 1,)
            Estimated polynomial regression coefficients after fitting.

        self.weights : ndarray of shape (n_samples,)
            Final sample weights after convergence.

        self.loss_history : list of float
            Gnostic criterion values computed at each iteration (if `history=True`).

        Notes
        -----
        - The method stops iterating when either:
            (1) The change in the recent loss values is below the specified tolerance `tol`, or
            (2) The change in coefficient values is below `tol`.
        - The convergence check uses the last `early_stopping` iterations.
        - The gnostic weighting and loss computations depend on the choice of `mg_loss`
        (e.g., `'hi'` or `'hj'`), which influences the robustness behavior.
        - Internal methods such as `_convert_az`, `_compute_gnostic_weights`, `criterion`, and `_gscale_loc`
        must be defined elsewhere in the class or imported.

        ''' 
        # processing input
        X, y = self._process_input(X, y)

        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X)
        
        # Initialize weights
        self.weights = np.ones(len(y))
        
        # Initialize coefficients to zeros
        self.coefficients = np.zeros(X_poly.shape[1])
        
        for self._iter in range(self.max_iter):
            self._iter += 1
            prev_coef = self.coefficients.copy()
            
            try:
                # Weighted least squares
                self.coefficients = self._weighted_least_squares(X_poly, y, self.weights)
                
                # Update weights using gnostic approach
                y0 = X_poly @ self.coefficients
                residuals = y - y0
                # Ensure residuals are not too close to zero
                # eps = np.finfo(float).eps
                # residuals = np.where(np.abs(residuals) < eps, eps, residuals)
                
                # dc = DataConversion()
                # z = dc._convert_az(residuals)
                z = self._data_conversion(residuals)
                gw = GnosticsWeights()
                gw = gw._get_gnostic_weights(z)
                new_weights = self.weights * gw

                # Compute scale and loss
                if self.scale_value == 'auto':
                    scale = ScaleParam()
                    s = scale._gscale_loc(np.mean(2 / (z + 1/z)))
                else:
                    s = self.scale_value

                loss, re = self._gnostic_criterion(z, y0, s)
                # Ensure weights are positive and normalized
                # new_weights = np.clip(new_weights, eps, None)
                # self.weights = self.weights * (s*loss**-1)
                self.weights = new_weights / np.mean(new_weights)
                                                
                # print loss
                if self.verbose:
                    print(f'Iteration: {self._iter} - Machine Gnostic loss - {self.mg_loss} : {np.round(loss, 4)}, rentropy: {np.round(re, 4)}')
                
                # processing output
                # capture history and append to history
                if self._history is not None:
                    self._history.append({
                        'iteration': self._iter+1,
                        'h_loss': loss,
                        'coefficients': self.coefficients.copy(),
                        'rentropy': re,
                        'weights': self.weights.copy(),
                    })

                # Check convergence with early stopping and rentropy
                # if entropy value is increasing, stop
                if self.early_stopping and self._history is not None:
                    if len(self._history) > 1:
                        prev_loss = self._history[-2]['h_loss']
                        prev_re = self._history[-2]['rentropy']
                        if (prev_loss is not None) and (prev_re is not None):
                            if (np.abs(loss - prev_loss) < self.tol) or (np.abs(re - prev_re) < self.tol):
                                if self.verbose:
                                    print(f"Convergence reached at iteration {self._iter} with loss/rentropy change below tolerance.")
                                break

                        
            except (ZeroDivisionError, np.linalg.LinAlgError) as e:
                if self.verbose:
                    print(f"Warning: {str(e)}. Using previous coefficients.")
                self.coefficients = prev_coef
                break
                
    def _predict(self, X):
        """
        Internal prediction method for base class.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features to predict for.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Process input and generate features
        X_processed = self._process_input(X)
        X_poly = self._generate_polynomial_features(X_processed)
        
        # Validate dimensions
        n_features_model = X_poly.shape[1]
        if n_features_model != len(self.coefficients):
            raise ValueError(
                f"Feature dimension mismatch. Model expects {len(self.coefficients)} "
                f"features but got {n_features_model} after polynomial expansion."
            )
        
        return X_poly @ self.coefficients