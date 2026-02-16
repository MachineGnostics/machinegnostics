'''
BoostingCalBase - Base class for Boosting calculations

Machine Gnostics
'''

import numpy as np
from machinegnostics.models.cart.base_boosting_methods import BoostingMethodsBase
from machinegnostics.magcal import GnosticsWeights, ScaleParam

class BoostingCalBase(BoostingMethodsBase):
    """
    Base calculation class for Gnostic Boosting models.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
        # History setup is handled in base/methods if passed
        if self.early_stopping and not self.history:
            self.history = True
            
        if self.history:
             self._history = []
             self._history.append({
                'iteration': 0,
                'h_loss': None,
                'rentropy': None,
                'weights': None,
             })
        else:
             self._history = None
             
        # check input
        self._input_checks()
        self.logger.info(f"Input parameters validated successfully.")

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model using iterative gnostic weighting.
        """
        self.logger.info("Starting fit process for BoostingCalBase.")
        
        # Initialize weights
        self.weights = self._weight_init(len(y))
        
        # Iteration 0 fit
        self.model = self._fit_boosting_impl(X, y, self.weights)
        
        if self.max_iter == 0:
            return

        for self._iter in range(1, self.max_iter + 1):
            
            # Predict with current model
            y_pred = self._predict_boosting_impl(X, self.model)

            residuals = y_pred - y
            
            # Gnostic data conversion
            # Using same logic as CartCalBase (Regression)
            z_y = self._data_conversion(y)
            z_y_pred = self._data_conversion(y_pred)
            z_resid = self._data_conversion(residuals)
            
            # Compute Gnostic Weights
            gwc = GnosticsWeights()
            gw = gwc._get_gnostic_weights(z_resid)
            new_weights = self.weights * gw
            
            # Normalize
            if np.sum(new_weights) != 0:
                new_weights = new_weights / np.sum(new_weights) * len(y)
            else:
                 new_weights = np.ones(len(y))
            
            self.weights = new_weights
            
            # Re-fit model with new weights
            self.model = self._fit_boosting_impl(X, y, self.weights)
            
            # Compute loss for history/stopping
            s = gwc.s if self.scale == 'auto' else self.scale
            
            loss, re, _, _, _, _, _, _, _, _, _, _ = self._gnostic_criterion(z=z_y_pred, z0=z_y, s=s)

            if self.verbose:
                 self.logger.info(f"Iteration {self._iter}: Loss {loss}, Rentropy {re}")

            if self._history is not None:
                self._history.append({
                    'iteration': self._iter,
                    'h_loss': loss,
                    'rentropy': re,
                    'weights': self.weights.copy()
                })

            # Check convergence
            if self.early_stopping and len(self._history) > 2:
                 prev_loss = self._history[-2]['h_loss']
                 if prev_loss is not None and abs(loss - prev_loss) < self.tolerance:
                     if self.verbose:
                         self.logger.info(f"Converged at iteration {self._iter}")
                     break
