'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
Date: 2025-10-01
Description: Machine Gnostics Robust Regression Machine Learning Model
This module implements a machine learning model for robust regression using the ManGo library.
This model is designed to handle various types of data and is particularly useful for applications in machine gnostics.
'''

import os
import joblib
import mlflow
import numpy as np
from machinegnostics.magcal import RegressorParamBase

class _RobustRegressor(RegressorParamBase, mlflow.pyfunc.PythonModel):
    """
    ## RobustRegressor: A Polynomial Regression Model Based on Machine Gnostics

    Key Features
    ------------
    - Robust to outliers and heavy-tailed distributions
    - Polynomial feature expansion (up to configurable degree)
    - Gnostic-based iterative loss minimization
    - Custom weighting and scaling strategy
    - Early stopping and convergence control
    - Modular design for extensibility
    - mlflow integration for model tracking and deployment
    - Save and load model using joblib

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
    
    params : bool, default=False,
        If 'True', store weights, coefficients, and gnostic loss in params

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

    """
    def __init__(self, 
                 degree = 2, 
                 max_iter = 100, 
                 tol = 1e-3, 
                 mg_loss = 'hi', 
                 early_stopping = True, 
                 verbose = False,
                 scale = 'auto',
                 history = True,
                 data_form = 'a'):
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
        # # Input validation and reshaping
        # X = np.asarray(X)
        # if X.ndim == 1:
        #     X = X.reshape(-1, 1)
            
        # # Verify feature dimensions match training data
        # n_features_trained = (len(self.coefficients) - 1) // (self.degree + 1)
        # n_features_input = X.shape[1]
        
        # if n_features_trained != n_features_input:
        #     raise ValueError(
        #         f"Model was trained with {n_features_trained} feature(s) but "
        #         f"received {n_features_input} feature(s) for prediction."
        #     )
        
        # Call base class prediction method
        # Handle pandas DataFrame
        if hasattr(model_input, "values"):
            X = model_input.values
        # Handle pyspark DataFrame (convert to pandas, then to numpy)
        elif "pyspark.sql.dataframe.DataFrame" in str(type(model_input)):
            X = model_input.toPandas().values
        # Assume it's already a numpy array
        else:
            X = np.asarray(model_input)
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
        