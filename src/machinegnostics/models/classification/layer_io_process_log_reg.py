import numpy as np
from machinegnostics.magcal.layer_io_process_base import DataProcessLayerBase
from machinegnostics.models.classification.layer_mlflow_log_reg import InterfaceLogisticRegressor
from machinegnostics.magcal import disable_parent_docstring

@disable_parent_docstring
class DataProcessLogisticRegressor(DataProcessLayerBase, InterfaceLogisticRegressor):
    """
    Data processing layer for the Robust Regressor model.
    Handles data preprocessing specific to the Robust Regressor model.
    """
    @disable_parent_docstring
    def __init__(self,
                 degree: int = 1,
                 max_iter: int = 100,
                 tol: float = 1e-3,
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: str | int | float = 'auto',
                 data_form: str = 'a',
                 gnostic_characteristics: bool = True,
                 history: bool = True,
                 proba: str = 'gnostic',
                 **kwargs):
        super().__init__(
            degree=degree,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            verbose=verbose,
            scale=scale,
            data_form=data_form,
            gnostic_characteristics=gnostic_characteristics,
            history=history,
            proba=proba,
            **kwargs
        )

        # logger
        self.logger.info("DataProcessLogisticRegressor initialized.")

        # --- argument checks ---
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("Degree must be a positive integer.")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(tol, (float, int)) or tol <= 0:
            raise ValueError("tol must be a positive float or int.")
        if not isinstance(scale, (str, int, float)):
            raise ValueError("scale must be a string, int, or float.")
        if isinstance(scale, (int, float)) and (scale < 0 or scale > 2):
            raise ValueError("scale must be between 0 and 2 if it is a number.")
        if data_form not in ['a', 'm']:
            raise ValueError("data_form must be either 'a' (additive) or 'm' (multiplicative).")
        if proba not in ['gnostic', 'sigmoid']:
            raise ValueError("proba must be either 'gnostic' or 'sigmoid'.")
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.scale = scale
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self.history = history
        self.params = []

    @disable_parent_docstring
    def _fit(self, X, y):
        """
        Fit the model to the data and preprocess it.
        """
        self.logger.info("Starting fit process for DataProcessLogisticRegressor.")
        X, y = self._fit_io(X, y)
        # Call the fit method from the next class in the MRO
        return super()._fit(X, y)

    @disable_parent_docstring
    def _predict(self, X) -> np.ndarray:
        """
        Predict using the model after preprocessing the input data.
        """
        self.logger.info("Making predictions with DataProcessLogisticRegressor.")
        X = self._predict_io(X)
        y_pred = super()._predict(X)
        # y_pred = self._convert_output(y_pred, self.data_form)
        return y_pred
    
    @disable_parent_docstring
    def _predict_proba(self, X) -> np.ndarray:
        """
        Predict probabilities using the model after preprocessing the input data.
        """
        self.logger.info("Calculating predicted probabilities with DataProcessLogisticRegressor.")
        X = self._predict_io(X)
        y_proba = super()._predict_proba(X)
        # y_proba = self._convert_output(y_proba, self.data_form)
        return y_proba