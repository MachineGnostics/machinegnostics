import numpy as np
from machinegnostics.magcal import HistoryRobustRegressor
import mlflow
import os
import joblib

class MlflowInterfaceRobustRegressor(HistoryRobustRegressor, mlflow.pyfunc.PythonModel):
    """
    Interface for the Robust Regressor model with MLflow integration.
    
    This class extends HistoryRobustRegressor to provide an interface for
    logging and tracking model parameters and performance metrics using MLflow.
    
    Parameters needed for MLflow tracking:
        - experiment_name: Name of the MLflow experiment
        - run_name: Name of the MLflow run
    """
    
    def __init__(self,
                 degree: int = 1,
                 max_iter: int = 100,
                 tol: float = 1e-8,
                 mg_loss: str = 'hi',
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: [str, int, float] = 'auto',
                 data_form: str = 'a',
                 gnostic_characteristics: bool = True):
        super().__init__(
            degree=degree,
            max_iter=max_iter,
            tol=tol,
            mg_loss=mg_loss,
            early_stopping=early_stopping,
            verbose=verbose,
            scale=scale,
            data_form=data_form,
            gnostic_characteristics=gnostic_characteristics
        )
        self.coefficients = None
        self.weights = None

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to the data and log parameters to MLflow.
        
        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        """
        # Call the fit method from HistoryRobustRegressor
        super()._fit(X, y)
        self.coefficients = None
        self.weights = None
        return self

    def _predict(self, model_input) -> np.ndarray:
        """
        Predict class labels for input data and log predictions to MLflow.
        
        Accepts numpy arrays, pandas DataFrames, or pyspark DataFrames.
        
        Parameters
        ----------
        model_input : np.ndarray, pd.DataFrame, pyspark.sql.DataFrame
            Input data for prediction.
        
        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        predictions = super()._predict(model_input)
        return predictions
    
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