import numpy as np
from machinegnostics.models.regression.layer_histroy_rob_reg import HistoryRobustRegressor
import mlflow
import os
import joblib

class InterfaceRobustRegressor(HistoryRobustRegressor, mlflow.pyfunc.PythonModel):
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
                 scale: str | int | float = 'auto',
                 data_form: str = 'a',
                 gnostic_characteristics: bool = True, 
                 history: bool = True):
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
            history=history
        )
        self.coefficients = None
        self.weights = None
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.mg_loss = mg_loss
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.scale = scale
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self._history = history
        self.params = []

        # logger
        self.logger.info("InterfaceRobustRegressor initialized.")

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
        self.logger.info("Starting fit process for InterfaceRobustRegressor. Logging to MLflow available.")
        super()._fit(X, y)
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
        self.logger.info("Making predictions with InterfaceRobustRegressor.")
        predictions = super()._predict(model_input)
        return predictions
    
    def save_model(self, path:str):
        """
        Save the trained model to disk using joblib.

        Parameters
        ----------
        path : str
            Directory path where the model will be saved.
        If the directory does not exist, it will be created.
        If the model is already saved, it will be overwritten.
        This method saves the model in a directory with a file named "model.pkl".
        """
        self.logger.info(f"Saving model to {path}.")
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, "model.pkl"))

    @classmethod
    def load_model(cls, path:str):
        """
        Load a trained model from disk using joblib.

        Parameters
        ----------
        path : str
            Directory path where the model is saved.
        This method loads the model from a file named "model.pkl" in the specified directory.
        Returns
        -------
        MlflowInterfaceRobustRegressor
            An instance of the model loaded from the specified path.
        """
        return joblib.load(os.path.join(path, "model.pkl"))
        
    def save_model(self, path):
        """
        Save the trained model to disk using joblib.
        """
        self.logger.info(f"Saving model to {path}.")
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, "model.pkl"))