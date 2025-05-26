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

import os
import joblib
import pandas as pd
import numpy as np
from machinegnostics.magcal import _RobustRegressor

class RobustRegressor(_RobustRegressor):
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
    - Early stopping and convergence control - h loss and residual entropy
    - Modular design for extensibility
    - mlflow integration for model tracking and deployment
    - Save and load model using joblib

    Parameters
    ----------
    degree : int, default=1
        Degree of the polynomial for feature expansion. Must be >= 1.

    max_iter : int, default=100
        Maximum number of training iterations.

    tol : float, default=1e-8
        Convergence threshold for loss or coefficient changes.

    mg_loss : str, default='hi'
        Type of gnostic loss to use. Options:
            - 'hi': Estimation relevance loss
            - 'hj': Joint relevance loss

    early_stopping : bool or int, default=True
        If True, enables early stopping with a default window. If int, specifies the window size.

    verbose : bool, default=False
        If True, prints progress and debug information during training.

    scale : {'auto', float}, default='auto'
        Scaling strategy for the gnostic loss. 'auto' selects automatically.

    history : bool, default=True
        If True, records the training history (loss, coefficients, weights, etc.) at each iteration.

    data_form : str, default='a'
        Indicates the form of the input data:
            - 'a': Additive (default)
            - 'm': Multiplicative

    Attributes
    ----------
    coefficients : ndarray of shape (n_coeffs,)
        Final learned polynomial regression coefficients.

    weights : ndarray of shape (n_samples,)
        Final sample weights after convergence.

    _history : list of dict
        Training history. Each entry contains:
            - 'iteration': int, iteration number
            - 'h_loss': float, gnostic loss value
            - 'coefficients': list, regression coefficients at this iteration
            - 'rentropy': float, rentropy value
            - 'weights': list, sample weights at this iteration

    Methods
    -------
    fit(X, y)
        Fit the model to training data using polynomial expansion and gnostic loss minimization.

    predict(X)
        Predict output values for new input samples using the trained model.

    save_model(path)
        Save the trained model to disk using joblib.

    load_model(path)
        Load a previously saved model from disk.

    Example
    -------
    >>> from machinegnostics.models import RobustRegressor
    >>> model = RobustRegressor(degree=2, mg_loss='hi', verbose=True)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> print(model.coefficients)
    >>> print(model.weights)
    >>> # Save and load
    >>> model.save_model('./my_model')
    >>> loaded = RobustRegressor.load_model('./my_model')
    >>> y_pred2 = loaded.predict(X_test)

    Notes
    -----
    - The model is robust to outliers and is suitable for datasets with non-Gaussian noise.
    - Training history can be accessed via `model._history` for analysis and plotting.
    - For more information, visit: https://machinegnostics.info/

    Github: https://github.com/MachineGnostics/machinegnostics
    """
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
        super().__init__(degree, 
                         max_iter, 
                         tol, 
                         mg_loss, 
                         early_stopping, 
                         verbose,
                         scale,
                         history,
                         data_form)
        self.coefficients = None
        self.weights = None
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
        # only polynomial regression is supported
        if self.degree == 1:
            raise ValueError("Degree must be greater than 0 for polynomial regression.")
        if self.degree < 1:
            raise ValueError("Degree must be a non-negative integer.")

    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        '''
        Fit the Robust Regressor model using gnostic weights and polynomial features.

        This method trains the regression model on the given dataset `(X, y)` by iteratively applying
        weighted least squares. The weights are updated at each iteration using a mathematical
        gnostics-based approach that makes the model robust to noise and outliers.

        The process involves:
        - Expanding the input features into a polynomial basis.
        - Solving the gnostic influenced weighted least squares problem to estimate model coefficients.
        - Computing residuals and transforming them into a gnostic space.
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
        '''
        super()._fit(X, y)
        self.coefficients = self.coefficients
        self.weights = self.weights
    
                
    def predict(self, model_input)->np.ndarray:
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
        X = model_input
        return super()._predict(X)
    
    def save_model(self, path):
        """
        Save the model to the specified path using joblib.
        """
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, "model.pkl"))

    @classmethod
    def load_model(cls, path):
        """
        Load the model from the specified path using joblib.
        """
        return joblib.load(os.path.join(path, "model.pkl"))
        