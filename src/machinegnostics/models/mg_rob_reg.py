import numpy as np
from machinegnostics.magcal import DataProcessRobustRegressor
from machinegnostics.metrics import robr2
from machinegnostics.magcal import disable_parent_docstring

class PolynomialRegressor(DataProcessRobustRegressor):
    """
    Polynomial Regressor model for robust regression tasks.
    
    This class extends DataProcessRobustRegressor to handle polynomial feature generation
    and scaling of input data, specifically for robust regression tasks.
    
    Parameters needed for polynomial regression:
        - degree: Degree of polynomial features to generate
        - scale: Scaling method for input data (e.g., 'auto', 'minmax', etc.)
    """
class PolynomialRegressor(DataProcessRobustRegressor):
    @disable_parent_docstring
    def __init__(
        self, 
        degree: int = 2, 
        scale: str | int | float = 'auto',
        max_iter: int = 100,
        tol: float = 1e-8,
        mg_loss: str = 'hi',
        early_stopping: bool = True,
        verbose: bool = False,
        data_form: str = 'a',
        gnostic_characteristics: bool = True,
        history: bool = True
    ):
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
        # # Optionally, set self.degree here as well for safety:
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
        self._history = []
        self.coefficients = None
        self.weights = None
    
    @disable_parent_docstring
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the polynomial regressor model to the data.
        
        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        
        Returns
        -------
        self : PolynomialRegressor
            Fitted model instance.
        """
        # Call the fit method from DataProcessRobustRegressor
        super()._fit(X, y)
        self.coefficients = self.coefficients
        self.weights = self.weights
    
    @disable_parent_docstring
    def predict(self, model_input: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted polynomial regressor model.
        
        Parameters
        ----------
        X : np.ndarray
            Input features for prediction.
        
        Returns
        -------
        y_pred : np.ndarray
            Predicted values.
        """
        # Call the predict method from DataProcessRobustRegressor
        return super()._predict(model_input)
    
    @disable_parent_docstring
    def score(self, X: np.ndarray, y: np.ndarray, case:str = 'i') -> float:
        """
        Gnostic R2 score for the polynomial regressor model.
        
        Parameters
        ----------
        X : np.ndarray
            Input features for scoring.
        y : np.ndarray
            True target values.
        
        Returns
        -------
        score : float
            Score of the model.
        """
        # prediction
        y_pred = self.predict(X)
        # Call the score method from DataProcessRobustRegressor
        r2 = robr2(y, y_pred, w=self.weights)
        return r2