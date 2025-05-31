import numpy as np
from machinegnostics.magcal import ParamRobustRegressorBase, HistoryBase
from dataclasses import dataclass

@dataclass
class ParamRecord:
    iteration: int
    h_loss: float = None
    weights: np.ndarray = None
    coefficients: np.ndarray = None
    degree: int = None
    rentropy: float = None
    fi: np.ndarray = None
    hi: np.ndarray = None
    fj: np.ndarray = None
    hj: np.ndarray = None
    infoi: dict = None
    infoj: dict = None
    pi: np.ndarray = None
    pj: np.ndarray = None
    ei: float = None
    ej: float = None

class HistoryRobustRegressor(ParamRobustRegressorBase):
    """
    History class for the Robust Regressor model.
    
    This class extends HistoryBase and ParamRobustRegressorBase to maintain a history
    of model parameters and gnostic loss values during training iterations.
    
    Parameters needed to record history:
        - h_loss: Gnostic loss value at each iteration
        - iteration: The iteration number
        - weights: Model weights at each iteration
        - coefficients: Model coefficients at each iteration
        - degree: Degree of polynomial features used in the model
        - rentropy: Entropy of the model at each iteration
        - fi, hi, fj, hj, infoi, infoj, pi, pj, ei, ej: Additional gnostic information if calculated
    """
    
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
            gnostic_characteristics=gnostic_characteristics
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
        self._history = history
        self.params = []
    
    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to the data and record history.
        
        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        """       
        # Call the parent fit method to perform fitting
        super()._fit(X, y)
        
        # Record the initial state in history
        if self.gnostic_characteristics:
            initial_record = ParamRecord(
                iteration=self._iter + 1,
                h_loss=self.loss,
                weights=self.weights.copy(),
                coefficients=self.coefficients.copy(),
                degree=self.degree,
                rentropy=self.re,
                fi= self.fi,
                hi= self.hi,
                fj= self.fj,
                hj= self.hj,
                infoi= self.infoi,
                infoj= self.infoj,
                pi= self.pi,
                pj= self.pj,
                ei= self.ei,
                ej= self.ej
            )
            self.params.append(initial_record)
        else:
            initial_record = ParamRecord(
                iteration=0,
                h_loss=None,
                weights=self.weights,
                coefficients=self.coefficients,
                degree=self.degree
            )
            self.params.append(initial_record)