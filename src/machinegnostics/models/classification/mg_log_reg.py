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
    Logistic Regression model using mathematical gnostics principles.
    
    This class extends DataProcessLogisticRegressor to implement a logistic regression model
    with additional functionalities for gnostic characteristics and history tracking.
    
    Parameters:
        - degree: Degree of polynomial features used in the model.
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
    
    def fit(self, X, y):
        """
        Fit the logistic regression model using the parent class logic.
        
        Parameters:
            - X: Input features.
            - y: Target labels.
        
        Returns:
            self: Fitted model instance.
        """
        super()._fit(X, y)
        
        self.coefficients = self.coefficients
        self.weights = self.weights
        return self
    
    def predict(self, model_input) -> np.ndarray:
        """
        Predict class labels for input data.
        
        Accepts numpy arrays, pandas DataFrames, or pyspark DataFrames.
        
        Parameters:
            - model_input: Input data for prediction.
        
        Returns:
            np.ndarray: Predicted class labels.
        """
        return super()._predict(model_input)
    
    def predict_proba(self, model_input) -> np.ndarray:
        """
        Predict probabilities for input data.
        
        Accepts numpy arrays, pandas DataFrames, or pyspark DataFrames.
        
        Parameters:
            - model_input: Input data for probability prediction.
        
        Returns:
            np.ndarray: Predicted probabilities.
        """
        return super()._predict_proba(model_input)

    def score(self, X, y) -> float:
        """
        Calculate the F1 score of the model on the given data.
        
        Parameters:
            - X: Input features.
            - y: True labels.
        
        Returns:
            float: F1 score of the model.
        """
        y_pred = self.predict(X)
        return f1_score(y, y_pred)