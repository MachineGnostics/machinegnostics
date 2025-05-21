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
from machinegnostics.magcal import _LinearRegressor

class LinearRegressor(_LinearRegressor):
    """
    ## LinearRegressor: A Simple Linear Regression Model Based on Machine Gnostics

    This class implements a standard linear regression model using the Machine Gnostics framework.
    Unlike the classic linear regression version, this model fits a linear function (degree=1) to the input data
    on the mathematical gnostics methods. 

    Unlike traditional statistical models that rely on probabilistic 
    assumptions, this approach uses algebraic and geometric structures to model 
    data while maintaining resilience to outliers, noise, and corrupted samples.

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

    Example
    -------
    >>> from machinegnostics.models import LinearRegressor
    >>> model = LinearRegressor()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> print(model.coefficients)

    Resource:
    ---------
    More information: https://machinegnostics.info/
    Github: https://github.com/MachineGnostics/machinegnostics
    """
    def __init__(self,  
                 tol:int = 1e-8, 
                 mg_loss:str = 'hi', 
                 early_stopping:bool = True, 
                 verbose:bool = False):
        super().__init__(
                         tol=tol, 
                         mg_loss=mg_loss, 
                         early_stopping=early_stopping, 
                         verbose=verbose
                         )
        '''
        Robust Linear Regressor - Machine Gnostics
        
        Initialize the linear regression model.
        
        Parameters:
        -----------
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
        self.coefficients = None
        self.weights = None

    def fit(self, X:np.ndarray, y:np.ndarray):
        """
        Fit the Linear Regressor model using least squares.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature values. Can be a numpy array, pandas DataFrame, or pyspark DataFrame.

        y : array-like of shape (n_samples,)
            Target values corresponding to input samples.

        Returns
        -------
        None
        """
        super().fit(X, y)
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
        return super().predict(X)

    def save_model(self, path:str):
        """
        Save the model to the specified path using joblib.

        Parameters
        ----------
        path : str
            Directory path where the model will be saved.
        """
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, "model.pkl"))

    @classmethod
    def load_model(cls, path:str):
        """
        Load the model from the specified path using joblib.

        Parameters
        ----------
        path : str
            Directory path where the model is saved.
        Returns
        -------
        LinearRegressor
            An instance of the LinearRegressor class with the loaded model.
        """
        return joblib.load(os.path.join(path, "model.pkl"))