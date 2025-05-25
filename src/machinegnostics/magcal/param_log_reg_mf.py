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
from machinegnostics.magcal.param_log_reg import _LogisticRegressorParamBase

class _LogisticRegressor(_LogisticRegressorParamBase, mlflow.pyfunc.PythonModel):
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

    def fit(self, X, y):
        """
        Fit the logistic regression model using the parent class logic.
        """
        super().fit(X, y)

        self.coefficients = self.coefficients
        self.weights = self.weights
        return self

    def predict(self, model_input):
        """
        Predict class labels for input data.
        Accepts numpy arrays, pandas DataFrames, or pyspark DataFrames.
        """
        if hasattr(model_input, "values"):
            X = model_input.values
        elif "pyspark.sql.dataframe.DataFrame" in str(type(model_input)):
            X = model_input.toPandas().values
        else:
            X = np.asarray(model_input)
        return super().predict(X)

    def predict_proba(self, model_input):
        """
        Predict probabilities for input data.
        Accepts numpy arrays, pandas DataFrames, or pyspark DataFrames.
        """
        if hasattr(model_input, "values"):
            X = model_input.values
        elif "pyspark.sql.dataframe.DataFrame" in str(type(model_input)):
            X = model_input.toPandas().values
        else:
            X = np.asarray(model_input)
        return super().predict_proba(X)

    def save_model(self, path):
        """
        Save the trained model to disk using joblib.
        """
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, "model.pkl"))

    @classmethod
    def load_model(cls, path):
        """
        Load a trained model from disk using joblib.
        """
        return joblib.load(os.path.join(path, "model.pkl"))