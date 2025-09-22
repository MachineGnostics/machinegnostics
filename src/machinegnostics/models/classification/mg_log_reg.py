'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar

Description:
This module implements a logistic regression model using mathematical gnostics principles.
'''

import numpy as np
import pandas as pd
from machinegnostics.models.classification.layer_io_process_log_reg import DataProcessLogisticRegressor
from machinegnostics.metrics import f1_score
from machinegnostics.magcal import disable_parent_docstring
from typing import Union

class LogisticRegressor(DataProcessLogisticRegressor):
    """
    LogisticRegressor implements a logistic regression model based on Mathematical Gnostics principles.

    This class prepared with Machine Gnostic framework, feature-rich logistic regression
    implementation. It supports polynomial feature expansion, custom loss functions, early stopping, 
    gnostic-based probability estimation, and detailed training history tracking.

    Key Features:
        - Polynomial feature expansion up to a user-specified degree.
        - Choice of probability estimation method: 'gnostic' (default) or standard 'sigmoid'.
        - Calculation of gnostic characteristics for advanced model diagnostics.
        - Early stopping based on convergence of loss or entropy.
        - Verbose logging for monitoring training progress.
        - Optional scaling and data processing modes.
        - Maintains a history of model parameters and losses for analysis.

    Parameters
    ----------
    degree : int, default=1
        Degree of polynomial features to use for input expansion.
    max_iter : int, default=100
        Maximum number of iterations for the optimization algorithm.
    tol : float, default=1e-3
        Tolerance for convergence. Training stops if the change in loss or entropy is below this value.
    mg_loss : str, default='hi'
        Type of gnostic loss to use (e.g., 'hi', 'hj', etc.).
    early_stopping : bool, default=True
        Whether to stop training early if convergence is detected.
    verbose : bool, default=False
        If True, prints detailed logs during training.
    scale : str | int | float, default='auto'
        Scaling method for input features. Can be a string identifier or a numeric value.
    data_form : str, default='a'
        Data processing form: 'a' for additive, 'm' for multiplicative.
    gnostic_characteristics : bool, default=True
        If True, calculates and stores gnostic characteristics during training.
    history : bool, default=True
        If True, maintains a history of model parameters and losses.
    proba : str, default='gnostic'
        Probability estimation method: 'gnostic' for gnostic-based, 'sigmoid' for standard logistic regression.

    Attributes
    ----------
    coefficients : np.ndarray
        Fitted model coefficients after training.
    weights : np.ndarray
        Sample weights used during training.
    _history : list
        List of dictionaries containing training history (loss, coefficients, entropy, etc.).
    params : list
        List of model parameters (for compatibility and inspection).

    Methods
    -------
    fit(X, y)
        Fit the logistic regression model to the data.
    predict(model_input)
        Predict class labels for new data.
    predict_proba(model_input)
        Predict class probabilities for new data.
    score(X, y)
        Compute the F1 score of the model on given data.

    Examples
    --------
    >>> from machinegnostics.models.classification.mg_log_reg import LogisticRegressor
    >>> model = LogisticRegressor(degree=2, max_iter=200, verbose=True)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> print("F1 Score:", model.score(X_test, y_test))

    Notes
    -----
    - The model supports both binary and multiclass classification tasks.
    - More information on gnostic characteristics can be found in the Machine Gnostics documentation.
    - For more information, visit: https://machinegnostics.info/
    """
    
    @disable_parent_docstring
    def __init__(self,
                 degree: int = 1,
                 max_iter: int = 100,
                 tol: float = 1e-3,
                 mg_loss: str = 'hi',
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: 'str | int | float' = 'auto',
                 data_form: str = 'a',
                 gnostic_characteristics:bool=True,
                 history: bool = True,
                 proba:str = 'gnostic'):
        """
        Initialize the LogisticRegressor with specified parameters.

        Parameters:
            - degree: Degree of polynomial features.
            - max_iter: Maximum number of iterations for convergence.
            - tol: Tolerance for stopping criteria.
            - early_stopping: Whether to stop training early if convergence is reached.
            - verbose: Whether to print detailed logs during training.
            - scale: Scaling method for input features.
            - data_form: Form of data processing ('a' for additive, 'm' for multiplicative).
            - gnostic_characteristics: Whether to calculate gnostic characteristics.
            - history: Whether to maintain a history of model parameters and losses.
            - proba: Probability estimation method ('gnostic' or 'sigmoid').
        
        """
        super().__init__(
            degree=degree,
            max_iter=max_iter,
            tol=tol,
            mg_loss=mg_loss,
            early_stopping=early_stopping,
            verbose=verbose,
            scale=scale,
            data_form=data_form,
            gnostic_characteristics=gnostic_characteristics,
            proba=proba
        )
        
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.mg_loss = mg_loss
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.scale = scale
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self.history = history
        self.proba = proba
        self.params = []
        self._history = []

        # logger
        self.logger.info("LogisticRegressor initialized.")
    
    def fit(self, X, y):
        """
        Fit the LogisticRegressor model to the training data.

        This method trains the logistic regression model using the provided input features and target labels.
        It supports polynomial feature expansion, gnostic or sigmoid probability estimation, and early stopping
        based on convergence criteria. Training history, including loss and coefficients, is stored if enabled.

        Parameters
        ----------
        X : array-like or DataFrame
            Input features for training. Can be a NumPy array, pandas DataFrame, or compatible type.
        y : array-like
            Target labels for training. Should be a 1D array or Series of binary class labels (0 or 1).

        Returns
        -------
        self : LogisticRegressor
            Returns the fitted model instance for chaining.
        
        Raises
        ------
        ValueError
            If input shapes are incompatible or training fails due to numerical issues.

        Examples
        --------
        >>> model = LogisticRegressor(degree=2, max_iter=200)
        >>> model.fit(X_train, y_train)
        """
        self.logger.info("Starting fit process for LogisticRegressor.")
        super()._fit(X, y)
        
        self.coefficients = self.coefficients
        self.weights = self.weights
        return self
    
    def predict(self, model_input) -> np.ndarray:
        """
        Predict class labels for new input data.

        This method predicts binary class labels (0 or 1) for the provided input data using the trained model.
        It supports input as NumPy arrays, pandas DataFrames, or PySpark DataFrames (if supported by the parent class).
        The prediction threshold is typically 0.5 unless otherwise specified in the parent class.

        Parameters
        ----------
        model_input : array-like or DataFrame
            Input data for prediction. Can be a NumPy array, pandas DataFrame, or compatible type.

        Returns
        -------
        np.ndarray
            Array of predicted class labels (0 or 1).

        Examples
        --------
        >>> y_pred = model.predict(X_test)
        """
        self.logger.info("Making predictions with LogisticRegressor.")
        return super()._predict(model_input)
    
    def predict_proba(self, model_input) -> np.ndarray:
        """
        Predict class probabilities for new input data.

        This method returns the predicted probabilities for each input sample belonging to the positive class (label 1).
        It supports input as NumPy arrays, pandas DataFrames, or PySpark DataFrames (if supported by the parent class).
        The probability estimation method is determined by the `proba` parameter set during initialization
        ('gnostic' for gnostic-based probabilities or 'sigmoid' for standard logistic regression probabilities).

        Parameters
        ----------
        model_input : array-like or DataFrame
            Input data for probability prediction. Can be a NumPy array, pandas DataFrame, or compatible type.

        Returns
        -------
        np.ndarray
            Array of predicted probabilities for the positive class (values between 0 and 1).

        Examples
        --------
        >>> y_proba = model.predict_proba(X_test)
        >>> print(y_proba[:5])
        """
        self.logger.info("Calculating predicted probabilities with LogisticRegressor.")
        return super()._predict_proba(model_input)

    def score(self, X, y) -> float:
        """
        Compute the F1 score of the model on the provided test data.

        This method evaluates the performance of the trained model by computing the F1 score,
        which is the harmonic mean of precision and recall, on the given input features and true labels.

        Parameters
        ----------
        X : array-like or DataFrame
            Input features for evaluation.
        y : array-like
            True binary labels for evaluation.

        Returns
        -------
        float
            F1 score of the model predictions on the provided data.

        Examples
        --------
        >>> score = model.score(X_test, y_test)
        >>> print("F1 Score:", score)
        """
        self.logger.info("Calculating F1 score for LogisticRegressor.")
        y_pred = self.predict(X)
        return f1_score(y, y_pred)