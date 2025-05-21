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
import mlflow
import numpy as np
from machinegnostics.magcal import RegressorParamBase

class _LinearRegressor(RegressorParamBase, mlflow.pyfunc.PythonModel):
    """
    ## LinearRegressor: A Simple Linear Regression Model Based on Machine Gnostics

    Key Features
    ------------
    - Fits a linear regression model (polynomial degree = 1)
    - Uses least squares estimation for coefficient calculation
    - Compatible with numpy arrays, pandas DataFrames, and pyspark DataFrames
    - mlflow integration for model tracking and deployment
    - Save and load model using joblib

    Parameters
    ----------
    max_iter : int, default=1
        Number of iterations (kept for interface compatibility; not used).

    tol : float, default=1e-8
        Convergence threshold (not used in linear regression).

    verbose : bool, default=False
        If `True`, prints debug and progress messages during training.

    Attributes
    ----------
    coefficients : ndarray of shape (n_features + 1,)
        Learned linear regression coefficients (including intercept).

    Methods
    -------
    fit(X, y)
        Fit the model to training data using least squares estimation.

    predict(X)
        Predict output values for new input samples using the trained model.

    save_model(path)
        Save the trained model to disk.

    load_model(path)
        Load a trained model from disk.

    """
    def __init__(self, 
                 max_iter:int = 100, 
                 tol = 1e-8, 
                 mg_loss = 'hi', 
                 early_stopping = True, 
                 verbose = False):
        super().__init__(max_iter=max_iter, 
                         tol=tol, 
                         mg_loss=mg_loss, 
                         early_stopping=early_stopping, 
                         verbose=verbose)
        self.coefficients = None
        self.weights = None
        self.degree = 1 # linear regression

    def fit(self, X, y):
        """
        Fit the Robust Linear Regressor model using gnostic weights and polynomial features.

        This method trains the regression model on the given dataset `(X, y)` by iteratively applying
        weighted least squares. The weights are calculated using a mathematical
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
        """
        super()._fit(X, y)
        self.coefficients = self.coefficients
        self.weights = self.weights
    

    def predict(self, model_input):
        """
        Predict target values using the trained Linear Regressor model.

        Parameters
        ----------
        model_input : array-like of shape (n_samples, n_features)
            Input feature values for prediction.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
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