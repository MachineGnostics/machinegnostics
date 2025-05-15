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
from src.magcal import GnosticsCharacteristics, DataConversion, GnosticCriterion, ScaleParam, GnosticsWeights

class RobustRegressor(RegressionBase):
    """
    ## RobustRegressor: A Polynomial Regression Model Based on Machine Gnostics

    This class implements a robust regression model grounded in the principles of 
    Mathematical Gnostics — a non-statistical, deterministic framework for learning 
    from data. Unlike traditional statistical models that rely on probabilistic 
    assumptions, this approach uses algebraic and geometric structures to model 
    data while maintaining resilience to outliers, noise, and corrupted samples.

    The model fits a polynomial regression function to the input data, adjusting
    the influence of each data point through a gnostically-derived weighting scheme.
    It iteratively optimizes the regression coefficients using a custom criterion
    that minimizes a gnostic loss (e.g., `hi` or `hj`).

    Key Features
    ------------
    - Robust to outliers and heavy-tailed distributions
    - Polynomial feature expansion (up to configurable degree)
    - Gnostic-based iterative loss minimization
    - Custom weighting and scaling strategy
    - Early stopping and convergence control
    - Modular design for extensibility

    Parameters
    ----------
    degree : int, default=2
        Degree of the polynomial used to expand input features. A value of `2` fits
        a quadratic model; higher values increase model flexibility.

    max_iter : int, default=1000
        Maximum number of iterations for the training process.

    tol : float, default=1e-3
        Convergence threshold. Iteration stops if the change in loss or coefficients
        is below this tolerance for `early_stopping` consecutive iterations.

    mg_loss : str, default='hi'\
        Type of gnostic loss to use. Options:
            - `'hi'` : Estimation relevance loss
            - `'hj'` : Joint relevance loss
        Determines how residuals are transformed and weighted during training.

    early_stopping : int or bool, default=True
        Number of iterations over which to check for convergence. If set to `True`, 
        uses a default internal threshold (e.g., 10). If an integer, uses that value
        directly.

    verbose : bool, default=False
        If `True`, prints debug and progress messages during training.

    history : bool, default=False
        If `True`, stores the history of gnostic loss values across training iterations
        in `self._history`.

    Attributes
    ----------
    coefficients : ndarray of shape (degree + 1,)
        Final learned polynomial coefficients after training.

    weights : ndarray of shape (n_samples,)
        Final weights assigned to each sample based on gnostic transformations.

    _history : list of float
        List of gnostic loss values recorded at each iteration (if `history=True`).

    Methods
    -------
    fit(X, y)
        Fit the model to training data using polynomial expansion and gnostic loss minimization.

    predict(X)
        Predict output values for new input samples using the trained model.


    Example
    -------
    >>> from mango import RobustRegressor
    >>> model = RobustRegressor(degree=3, mg_loss='hi', verbose=True)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> print(model.coefficients)
    >>> print(model.weights)

    Resource:
    --------
    More information: https://machinegnostics.info/ 

    Github: https://github.com/MachineGnostics/ManGo
    """
    def __init__(self,
                 degree: int = 2,
                 max_iter: int = 100,
                 tol: float = 1e-8,
                 mg_loss: str = 'hi',
                 early_stopping: bool = True,
                 verbose: bool = False):
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
            return np.sum(hi ** 2)
        elif self.mg_loss == 'hj':
            hj = self.gc._hj(q, q1)
            return np.sum(hj**2)

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
    
    def fit(self, X, y):
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
        X = np.asarray(X)
        y = np.asarray(y)
        eps = np.finfo(float).min

        # Validate dimensions of X
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2 or X.shape[1] != 1:
            raise ValueError("X must be 1D or 2D with shape (n_samples,) or (n_samples, 1).")

        # Validate y
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of shape (n_samples,).")

        # Check for consistent sample sizes
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X and y must match. Got {X.shape[0]} and {y.shape[0]}.")

        
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X)
        
        # Initialize weights
        self.weights = np.ones(len(y))
        
        # Initialize coefficients to zeros
        self.coefficients = np.zeros(self.degree + 1)
        
        for _ in range(self.max_iter):
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
                
                dc = DataConversion()
                z = dc._convert_az(residuals)
                gw = GnosticsWeights()
                gw = gw._get_gnostic_weights(z)
                new_weights = self.weights * gw
                # Compute scale and loss
                scale = ScaleParam()
                s = scale._gscale_loc(np.mean(2 / (z + 1/z)))
                loss = self._gnostic_criterion(z, y0, s)
                self._history.append(loss)
                # Ensure weights are positive and normalized
                # new_weights = np.clip(new_weights, eps, None)
                # self.weights = self.weights * (s*loss**-1)
                self.weights = new_weights / np.mean(new_weights)
                                                
                # print loss
                if self.verbose:
                    print(f'Machine Gnostic loss - {self.mg_loss} : {np.round(loss, 4)}')
                
                # Check convergence
                if len(self._history) > self.early_stopping:
                    recent_losses = self._history[-self.early_stopping:]
                    if np.all(np.abs(np.diff(recent_losses)) < self.tol):
                        break
                    if np.all(np.abs(prev_coef - self.coefficients) < self.tol):
                        break
                        
            except (ZeroDivisionError, np.linalg.LinAlgError) as e:
                if self.verbose:
                    print(f"Warning: {str(e)}. Using previous coefficients.")
                self.coefficients = prev_coef
                break
                
    def predict(self, X):
        """
        Predict target values using the trained Robust Regressor model.

        This method applies the learned polynomial regression model to new input data
        and returns predicted values. It assumes that the `fit` method has already been
        called to estimate the model coefficients.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Input feature values for which predictions are to be made. Can be a 1D array
            or a single-column 2D array.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values corresponding to the input samples.

        Raises
        ------
        ValueError
            If the model has not been fitted or if the input shape is incompatible.

        Notes
        -----
        - This method expands the input features into the same polynomial basis as used during training.
        - Ensure `fit` has been called before using `predict`, otherwise `self.coefficients` will be `None`.
        - Input `X` will be converted to a NumPy array if it isn't already.
        """
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet.")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        X_poly = self._generate_polynomial_features(X)
        return X_poly @ self.coefficients