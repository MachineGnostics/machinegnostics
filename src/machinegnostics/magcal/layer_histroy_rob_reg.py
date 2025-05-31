import numpy as np
from machinegnostics.magcal import ParamRobustRegressorBase, HistoryBase
from dataclasses import dataclass

@dataclass
class HistoryRecord:
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

class HistoryRobustRegressor(HistoryBase, ParamRobustRegressorBase):
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
        self._record_history = history
        self._history = []
        if not isinstance(self._record_history, bool):
            raise TypeError("history must be a boolean value")
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
    
    def _record(self):
        """
        Record the history of model parameters and gnostic loss values.
        
        Parameters
        ----------
        iteration : int
            The current iteration number.
        h_loss : float, optional
            Gnostic loss value at the current iteration.
        weights : np.ndarray, optional
            Model weights at the current iteration.
        coefficients : np.ndarray, optional
            Model coefficients at the current iteration.
        degree : int, optional
            Degree of polynomial features used in the model.
        rentropy : float, optional
            Entropy of the model at the current iteration.
        fi, hi, fj, hj : np.ndarray, optional
            Additional gnostic information if calculated.
        infoi, infoj : dict, optional
            Gnostic information if calculated.
        pi, pj : np.ndarray, optional
            Gnostic probabilities if calculated.
        ei, ej : float, optional
            Gnostic entropy if calculated.
        """
        record = HistoryRecord(iteration=self._iter,
                               h_loss=self.loss,
                               weights=self.weights.copy(),
                               coefficients=self.coefficients.copy(),
                               degree=self.degree,
                               rentropy=self.re,
                               fi=self.fi,
                               hi=self.hi,
                               fj=self.fj,
                               hj=self.hj,
                               infoi=self.infoi,
                               infoj=self.infoj,
                               pi=self.pi,
                               pj=self.pj,
                               ei=self.ei,
                               ej=self.ej)
        self.history.append(record)
        return self
    
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
        super()._fit(X, y)
        if self._record_history:
            self._record()
        return self
    
    def get_history(self, as_dict=False):
        """
        Retrieve the history of model parameters, gnostic loss values, and gnostic characteristics.

        Parameters
        ----------
        as_dict : bool, optional
            If True, return history as a list of dictionaries; otherwise, return as a list of HistoryRecord objects.
        Returns
        -------
        history : list
            List of HistoryRecord objects or dictionaries containing the recorded history.
        """
        return super().get_history(as_dict)