import numpy as np
from machinegnostics.magcal import DataProcessLayerBase, MlflowInterfaceRobustRegressor

class DataProcessRobustRegressor(DataProcessLayerBase, MlflowInterfaceRobustRegressor):
    """
    Data processing layer for the Robust Regressor model.
    Handles data preprocessing specific to the Robust Regressor model.
    """

    def __init__(self,
                 degree: int = 1,
                 max_iter: int = 100,
                 tol: float = 1e-3,
                 mg_loss: str = 'hi',
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: str | int | float = 'auto',
                 data_form: str = 'a',
                 gnostic_characteristics: bool = True,
                 history: bool = True,
                 **kwargs):
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
            history=history,
            **kwargs
        )

        # --- argument checks ---
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("Degree must be a positive integer.")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(tol, (float, int)) or tol <= 0:
            raise ValueError("tol must be a positive float or int.")
        if mg_loss not in ['hi', 'hj']:
            raise ValueError("mg_loss must be either 'hi' or 'hj'.")
        if not isinstance(scale, (str, int, float)):
            raise ValueError("scale must be a string, int, or float.")
        if isinstance(scale, (int, float)) and (scale < 0 or scale > 2):
            raise ValueError("scale must be between 0 and 2 if it is a number.")
        if data_form not in ['a', 'm']:
            raise ValueError("data_form must be either 'a' (additive) or 'm' (multiplicative).")
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.mg_loss = mg_loss
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.scale = scale
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self._record_history = history

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to the data and preprocess it.
        """
        X, y = self._fit_io(X, y)
        # Call the fit method from the next class in the MRO
        return super()._fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model after preprocessing the input data.
        """
        X = self._predict_io(X)
        return super()._predict(X)