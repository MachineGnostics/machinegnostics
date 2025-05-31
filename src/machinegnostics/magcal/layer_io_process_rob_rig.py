import numpy as np
from machinegnostics.magcal import DataProcessLayerBase, MlflowInterfaceRobustRegressor
from machinegnostics.magcal import disable_parent_docstring

class DataProcessRobustRegressor(DataProcessLayerBase, MlflowInterfaceRobustRegressor):
    """
    Data processing layer for the Robust Regressor model.
    
    This class extends DataProcessLayerBase to handle data preprocessing
    specific to the Robust Regressor model, including polynomial feature generation
    and scaling of input data.
    
    Parameters needed for data processing:
        - degree: Degree of polynomial features to generate
        - scale: Scaling method for input data (e.g., 'auto', 'minmax', etc.)
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
    
    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to the data and preprocess it.
        
        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        
        Returns
        -------
        self : DataProcessRobustRegressor
            Fitted model instance.
        """
        X, y = self._fit_io(X, y)
        # Call the fit method from MlflowInterfaceRobustRegressor
        super()._fit(X, y)
        return self
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model after preprocessing the input data.
        
        Parameters
        ----------
        X : np.ndarray
            Input features for prediction.
        
        Returns
        -------
        y_pred : np.ndarray
            Predicted values.
        """
        X = self._predict_io(X)
        return super()._predict(X)