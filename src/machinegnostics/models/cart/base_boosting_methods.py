'''
BoostingMethodsBase - Base class for Machine Gnostics Boosting Methods

This class serves as the foundational base for Gnostic Boosting Regressor.

Copyright (C) Machine Gnostics
'''

import logging
from machinegnostics.magcal.util.logging import get_logger
import numpy as np
from machinegnostics.models.base_model import ModelBase
from typing import Union, Optional, Any
from machinegnostics.magcal import (DataConversion, GnosticsCharacteristics)
from machinegnostics.magcal.util.min_max_float import np_eps_float

try:
    import xgboost as xgb
except ImportError:
    xgb = None

class BoostingMethodsBase(ModelBase):
    """
    Base class for Machine Gnostics Boosting Methods.
    """
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = 6,
                 learning_rate: float = 0.3,
                 max_iter: int = 10,
                 tolerance: float = 1e-4,
                 mg_loss: str = 'hi',
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: Union[str, int, float] = 'auto',
                 history: bool = True,
                 data_form: str = 'a',
                 gnostic_characteristics: bool = False,
                 random_state: Optional[int] = None,
                 estimator_type: str = 'regressor',
                 **kwargs):
        super().__init__(verbose=verbose)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.mg_loss = mg_loss
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.scale = scale
        self.history = history
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self.random_state = random_state
        self.estimator_type = estimator_type
        self.kwargs = kwargs
        
        self.model = None
        self.weights = None
        
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)

        if self.history:
            self._history = []
        else:
            self._history = None
            
        if xgb is None:
            self.logger.warning("XGBoost is not installed. GnosticBoostingRegressor will fail at runtime.")

    def _input_checks(self):
        """Perform input validation."""
        self.logger.info("Performing input checks for arguments.")
        if not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("max_iter must be a non-negative integer.")
        if not isinstance(self.tolerance, (float, int)) or self.tolerance <= 0:
            raise ValueError("tolerance must be a positive float or int.")
        if self.mg_loss not in ['hi', 'hj']:
            raise ValueError("mg_loss must be either 'hi' or 'hj'.")
        if not isinstance(self.scale, (str, int, float)):
            raise ValueError("scale must be a string, int, or float.")
        if isinstance(self.scale, (int, float)) and (self.scale < 0 or self.scale > 2):
            raise ValueError("scale must be between 0 and 2 if it is a number.")
        if self.data_form not in ['a', 'm']:
            raise ValueError("data_form must be either 'a' (additive) or 'm' (multiplicative).")

    def _weight_init(self, n_samples: int) -> np.ndarray:
        """Initialize weights to uniform."""
        return np.ones(n_samples)

    def _fit_boosting_impl(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> Any:
        """
        Implementation of Boosting training.
        """
        if xgb is None:
            raise ImportError("XGBoost is required for this model.")
            
        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            **self.kwargs
        )
        
        model.fit(X, y, sample_weight=sample_weight)
        return model

    def _predict_boosting_impl(self, X: np.ndarray, model: Any) -> np.ndarray:
        """
        Implementation of Boosting prediction.
        """
        return model.predict(X)
    
    def _data_conversion(self, z: np.ndarray) -> np.ndarray:
        """Convert data using form."""
        dc = DataConversion()
        if self.data_form == 'a':
            return dc._convert_az(z)
        elif self.data_form == 'm':
            return dc._convert_mz(z)
        else:
            raise ValueError("data_form must be 'a' or 'm'.")

    def _compute_q(self, z, z0, s: int = 1):
        """Compute q and q1."""
        eps = np_eps_float()
        z0_safe = np.where(np.abs(z0) < eps, eps, z0)
        zz = z / z0_safe
        self.gc = GnosticsCharacteristics(zz, verbose=self.verbose)
        q, q1 = self.gc._get_q_q1(S=s)
        return q, q1

    def _gnostic_criterion(self, z: np.ndarray, z0: np.ndarray, s) -> tuple:
        """Compute gnostic criterion."""
        q, q1 = self._compute_q(z, z0, s)
        
        pi = pj = ei = ej = infoi = infoj = None

        if self.mg_loss == 'hi':
            hi = self.gc._hi(q, q1)
            fi = self.gc._fi(q, q1)
            fj = self.gc._fj(q, q1)
            re = self.gc._rentropy(fi, fj)
            if self.gnostic_characteristics:
                hj = self.gc._hj(q, q1)
            else:
                hj = None
            
            re_norm = (re - np.min(re)) / (np.max(re) - np.min(re)) if np.max(re) != np.min(re) else re
            H = np.sum(hi ** 2)
            return H, np.mean(re_norm), hi, hj, fi, fj, pi, pj, ei, ej, infoi, infoj
        elif self.mg_loss == 'hj':
             # Fallback/placeholder if needed
            hj = self.gc._hj(q, q1)
            fi = self.gc._fi(q, q1)
            fj = self.gc._fj(q, q1)
            re = self.gc._rentropy(fi, fj)
            
            re_norm = (re - np.min(re)) / (np.max(re) - np.min(re)) if np.max(re) != np.min(re) else re
            H = np.sum(hj ** 2)
            return H, np.mean(re_norm), None, hj, fi, fj, None, None, None, None, None, None
        
        return None, None, None, None, None, None, None, None, None, None, None, None

