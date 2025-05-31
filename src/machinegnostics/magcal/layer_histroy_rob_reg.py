import numpy as np
from machinegnostics.magcal import ParamRobustRegressorBase, HistoryBase, HistoryRecord

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
    
    def __init__(self, history: bool = True):
        super().__init__(history=history)
        self._record_history = history
        self.history = []
        if not isinstance(self._record_history, bool):
            raise TypeError("history must be a boolean value")
    
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
                               h_los=self.loss,
                               weights=self.weights,
                               coefficients=self.coefficients,
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
        self._fit(X, y)
        if self._record_history:
            self._record()
        return self