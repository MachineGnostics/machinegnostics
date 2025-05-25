'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
Date: 2025-10-01
Description: Machine Gnostics Logistic Regression Model (with MLflow support)
This module provides a robust, flexible logistic regression model based on the Machine Gnostics framework,
suitable for binary classification tasks, with built-in support for polynomial features, robust weighting,
early stopping, and MLflow model tracking and deployment.
'''

import os
import joblib
import mlflow
import numpy as np
from machinegnostics.magcal.mg_log_reg_mf import _LogisticRegressor

class LogisticRegressor(_LogisticRegressor):
    """
    LogisticRegressor: Robust Logistic Regression with Machine Gnostics and MLflow Integration

    This class implements a robust logistic regression model for binary classification, leveraging
    the Machine Gnostics framework. It is designed to be resilient to outliers and heavy-tailed
    distributions, and supports polynomial feature expansion, custom weighting, and early stopping.
    The model is fully compatible with MLflow for experiment tracking and deployment.

    Key Features
    ------------
    - Robust to outliers and non-Gaussian noise via gnostic weighting
    - Supports polynomial feature expansion (configurable degree)
    - Flexible probability output: gnostic or sigmoid
    - Customizable scaling of data (auto or manual)
    - Early stopping based on residual entropy or log loss
    - Full training history tracking (loss, entropy, coefficients, weights)
    - MLflow integration for model tracking, reproducibility, and deployment
    - Save and load model using joblib

    Parameters
    ----------
    degree : int, default=1
        Degree of the polynomial used to expand input features. A value of `1` fits
        a linear model; higher values increase model flexibility.

    max_iter : int, default=100
        Maximum number of iterations for the training process.

    tol : float, default=1e-8
        Convergence threshold. Iteration stops if the change in loss or coefficients
        is below this tolerance.

    scale : {'auto', float}, default='auto'
        Scaling mode for the gnostic transformation. Use 'auto' for automatic scaling,
        or provide a float value between 0 and 2 for manual scaling.

    early_stopping : bool, default=True
        If True, enables early stopping based on convergence criteria.

    history : bool, default=True
        If True, stores the history of loss, entropy, coefficients, and weights across
        training iterations in `self._history`.

    proba : {'gnostic', 'sigmoid'}, default='gnostic'
        Probability output mode. Use 'gnostic' for gnostic-based probabilities, or
        'sigmoid' for standard logistic regression probabilities.

    verbose : bool, default=False
        If True, prints debug and progress messages during training.

    Attributes
    ----------
    coefficients : ndarray of shape (n_features_poly,)
        Final learned polynomial coefficients after training.

    weights : ndarray of shape (n_samples,)
        Final weights assigned to each sample based on gnostic transformations.

    _history : list of dict
        List of dictionaries recording loss, entropy, coefficients, and weights at each iteration.

    Methods
    -------
    fit(X, y)
        Fit the model to training data using polynomial expansion and robust loss minimization.

    predict(X)
        Predict class labels (0 or 1) for new input samples using the trained model.

    predict_proba(X)
        Predict probabilities for new input samples using the trained model.

    save_model(path)
        Save the trained model to disk using joblib.

    load_model(path)
        Load a trained model from disk using joblib.

    Example
    -------
    >>> from machinegnostics.models import LogisticRegressor
    >>> model = LogisticRegressor(degree=2, proba='gnostic', verbose=True)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> y_proba = model.predict_proba(X_test)
    >>> print(model.coefficients)
    >>> print(model.weights)
    >>> model.save_model("my_logreg_model")
    >>> loaded = LogisticRegressor.load_model("my_logreg_model")
    >>> y_pred2 = loaded.predict(X_test)

    Notes
    -----
    - The model supports numpy arrays, pandas DataFrames, and pyspark DataFrames as input.
    - For best results, ensure input features are appropriately scaled and encoded.
    - For more information, visit: https://machinegnostics.info/
    """

    # All logic is inherited from _LogisticRegressor (mg_log_reg_mf.py)
    
    def __init__(self,
                 degree: int = 1,
                 max_iter: int = 100,
                 tol: float = 1e-8,
                 verbose: bool = False,
                 scale: [str, float, int] = 'auto', # if auto then automatically select scale based on the data else user can give float value between 0 to 2
                 early_stopping: bool = True,
                 history: bool = True,
                 proba: str = 'gnostic',
                 data_form:str = 'a'):
        """
        Initialize the LogisticRegressor with specified parameters.
        """
        super().__init__(degree=degree,
                         max_iter=max_iter,
                         tol=tol,
                         verbose=verbose,
                         scale=scale,
                         early_stopping=early_stopping,
                         history=history,
                         proba=proba,
                         data_form=data_form)
        
    def fit(self, X, y):
        """
        Fit the logistic regression model to training data.

        This method trains the LogisticRegressor using the provided input features `X` and binary target labels `y`.
        The model supports robust fitting using gnostic weighting, polynomial feature expansion, and early stopping.
        During training, the model iteratively updates its coefficients to minimize the chosen loss function
        (gnostic or sigmoid), and records the training history if enabled.

        Parameters
        ----------
        X : array-like, pandas.DataFrame, or numpy.ndarray of shape (n_samples, n_features)
            Training input samples. Can be a numpy array, pandas DataFrame, or similar structure.
        y : array-like or numpy.ndarray of shape (n_samples,)
            Target binary labels (0 or 1) for each sample.

        Returns
        -------
        self : LogisticRegressor
            Returns the fitted model instance (self), allowing for method chaining.

        Notes
        -----
        - The model will automatically expand features to the specified polynomial degree.
        - If `history=True`, training loss, entropy, coefficients, and weights are recorded at each iteration.
        - Early stopping is triggered based on the chosen convergence criteria.
        - Input data should be properly preprocessed (e.g., scaled, encoded) for best results.

        Example
        -------
        >>> model = LogisticRegressor(degree=2, proba='gnostic')
        >>> model.fit(X_train, y_train)
        """
        return super()._fit(X, y)

    def predict(self, model_input, context=None, params=None)-> np.ndarray:
        """
        Predict class labels for input samples.

        This method predicts binary class labels (0 or 1) for the provided input features `X` using the trained model.
        The prediction is based on the learned coefficients and the chosen probability output mode (gnostic or sigmoid).
        By default, a threshold of 0.5 is used to assign class labels.

        Parameters
        ----------
        model_input : array-like, pandas.DataFrame, or numpy.ndarray of shape (n_samples, n_features)
            Input samples for which to predict class labels.

        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,)
            Predicted binary class labels (0 or 1) for each input sample.

        Notes
        -----
        - The input features will be expanded to the polynomial degree used during training.
        - The model must be fitted before calling this method.

        Example
        -------
        >>> y_pred = model.predict(X_test)
        """
        return super()._predict(model_input)

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities for input samples.

        This method computes the predicted probabilities for the positive class (label 1)
        for each input sample in `X`, using the trained logistic regression model.
        The probability calculation is based on the selected mode (`proba` parameter):
        - If `proba='gnostic'`, uses the gnostic probability transformation.
        - If `proba='sigmoid'`, uses the standard logistic sigmoid function.

        Parameters
        ----------
        X : array-like, pandas.DataFrame, or numpy.ndarray of shape (n_samples, n_features)
            Input samples for which to predict probabilities. Can be a numpy array,
            pandas DataFrame, or similar structure.

        Returns
        -------
        proba : numpy.ndarray of shape (n_samples,)
            Predicted probabilities for the positive class (label 1) for each input sample.

        Notes
        -----
        - The input features will be expanded to the polynomial degree used during training.
        - The model must be fitted before calling this method.
        - The returned probabilities can be thresholded (e.g., at 0.5) to obtain class labels.

        Example
        -------
        >>> y_proba = model.predict_proba(X_test)
        >>> y_pred = (y_proba >= 0.5).astype(int)
        """
        return super()._predict_proba(X)

    def save_model(self, path):
        """
        Save the trained LogisticRegressor model to disk using joblib.

        This method serializes the entire model instance, including all learned parameters,
        training history, and configuration, to the specified directory. The saved model
        can later be loaded for inference or further training.

        Parameters
        ----------
        path : str
            Directory path to save the model. The model will be saved as 'model.pkl' in this directory.

        Returns
        -------
        None

        Example
        -------
        >>> model.save_model("my_logreg_model")
        """
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, "model.pkl"))

    @classmethod
    def load_model(cls, path):
        """
        Load a trained LogisticRegressor model from disk using joblib.

        This method restores a previously saved model from the specified directory,
        including all learned parameters, training history, and configuration.

        Parameters
        ----------
        path : str
            Directory path from which to load the model. The method expects a file named 'model.pkl' in this directory.

        Returns
        -------
        model : LogisticRegressor
            Loaded model instance, ready for inference or further training.

        Example
        -------
        >>> loaded = LogisticRegressor.load_model("my_logreg_model")
        >>> y_pred = loaded.predict(X_test)
        """
        return joblib.load(os.path.join(path, "model.pkl"))