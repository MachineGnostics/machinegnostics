'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
Date: 2025-10-01
Description: Machine Gnostics logic for robust regression model and wrapping it with mlflow
'''

import os
import joblib
import mlflow
import numpy as np
from machinegnostics.models.classification.layer_history_log_reg import HistoryRobustRegressor

class InterfaceLogisticRegressor(HistoryRobustRegressor, mlflow.pyfunc.PythonModel):
    """
    _LogisticRegressor: MLflow-wrapped Gnostic Logistic Regression

    Developer Notes:
    ----------------
    - Inherits from _LogisticRegressorParamBase for core logic and mlflow.pyfunc.PythonModel for MLflow integration.
    - Supports saving/loading via joblib for reproducibility and deployment.
    - Handles numpy arrays, pandas DataFrames, and pyspark DataFrames for prediction.
    - Use fit(X, y) for training and predict(X) or predict_proba(X) for inference.
    - Use save_model(path) and load_model(path) for model persistence.
    """
    def __init__(self,
                 degree: int = 1,
                 max_iter: int = 100,
                 tol: float = 1e-3,
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: 'str | int | float' = 'auto',
                 data_form: str = 'a',
                 gnostic_characteristics:bool=True,
                 history: bool = True,
                 proba:str = 'gnostic'):
        super().__init__(
            degree=degree,
            max_iter=max_iter,
            tol=tol,
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
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.scale = scale
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self.history = history
        self.proba = proba

        # logger
        self.logger.info("InterfaceLogisticRegressor initialized.")


    def _fit(self, X, y):
        """
        Fit the logistic regression model using the parent class logic.
        """
        self.logger.info("Starting fit process for InterfaceLogisticRegressor.")
        super()._fit(X, y)

        self.coefficients = self.coefficients
        self.weights = self.weights
        return self

    def _predict(self, model_input) -> np.ndarray:
        """
        Predict class labels for input data.
        Accepts numpy arrays, pandas DataFrames, or pyspark DataFrames.
        """
        self.logger.info("Making predictions with InterfaceLogisticRegressor.")
        return super()._predict(model_input)

    def _predict_proba(self, model_input) -> np.ndarray:
        """
        Predict probabilities for input data.
        Accepts numpy arrays, pandas DataFrames, or pyspark DataFrames.
        """
        self.logger.info("Calculating predicted probabilities with InterfaceLogisticRegressor.")
        return super()._predict_proba(model_input)

    def save_model(self, path):
        """
        Save the trained model to disk using joblib.
        """
        self.logger.info(f"Saving model to {path}.")
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, "model.pkl"))

    @classmethod
    def load_model(cls, path):
        """
        Load a trained model from disk using joblib.
        """
        return joblib.load(os.path.join(path, "model.pkl"))