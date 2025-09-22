import numpy as np
from machinegnostics.models.classification.layer_param_log_reg import ParamLogisticRegressorBase
from dataclasses import dataclass

class HistoryRobustRegressor(ParamLogisticRegressorBase):
    """
    History class for the Logistic Regressor model.
    
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
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: 'str | int | float' = 'auto',
                 data_form: str = 'a',
                 gnostic_characteristics:bool=True,
                 history: bool = True,
                 proba:str = 'gnostic'):
        super().__init__(
            degree=degree,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            verbose=verbose,
            scale=scale,
            data_form=data_form,
            gnostic_characteristics=gnostic_characteristics,
            proba=proba
        )
        
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.scale = scale
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self.history = history
        self.proba = proba
        self.params = [
            {
                'iteration': 0,
                'loss': None,
                'weights': None,
                'coefficients': None,
                'degree': self.degree,
                'rentropy': None,
                'fi': None,
                'hi': None,
                'fj': None,
                'hj': None,
                'infoi': None,
                'infoj': None,
                'pi': None,
                'pj': None,
                'ei': None,
                'ej': None
            }
        ]

        # logger
        self.logger.info("HistoryRobustRegressor initialized.")
    
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
        self.logger.info("Starting fit process for HistoryRobustRegressor.")
        # Call the parent fit method to perform fitting
        super()._fit(X, y)

        # Record the initial state in history as a dict
        params_dict = {}

        if self.gnostic_characteristics:
            params_dict['iteration'] = self._iter + 1
            params_dict['loss'] = self.loss
            params_dict['weights'] = self.weights.copy() if self.weights is not None else None
            params_dict['coefficients'] = self.coefficients.copy() if self.coefficients is not None else None
            params_dict['degree'] = self.degree
            params_dict['rentropy'] = self.re
            params_dict['fi'] = self.fi
            params_dict['hi'] = self.hi
            params_dict['fj'] = self.fj
            params_dict['hj'] = self.hj
            params_dict['infoi'] = self.infoi
            params_dict['infoj'] = self.infoj
            params_dict['pi'] = self.pi
            params_dict['pj'] = self.pj
            params_dict['ei'] = self.ei
            params_dict['ej'] = self.ej
        else:
            params_dict['iteration'] = 0
            params_dict['loss'] = None
            params_dict['weights'] = self.weights.copy() if self.weights is not None else None
            params_dict['coefficients'] = self.coefficients .copy() if self.coefficients is not None else None
            params_dict['degree'] = self.degree

        self.params.append(params_dict)