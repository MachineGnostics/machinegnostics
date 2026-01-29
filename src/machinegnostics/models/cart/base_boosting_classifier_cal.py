'''
BoostingClassifierCalBase - Base class for Boosting Classification calculations

Machine Gnostics
'''

import numpy as np
from machinegnostics.models.cart.base_boosting_classifier_methods import BoostingClassifierMethodsBase
from machinegnostics.magcal import GnosticsWeights, ScaleParam

class BoostingClassifierCalBase(BoostingClassifierMethodsBase):
    """
    Base calculation class for Gnostic Boosting Classification models.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
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

    def _one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        n_samples = len(y)
        n_classes = len(self.classes_)
        one_hot = np.zeros((n_samples, n_classes))
        # Map labels to indices
        class_map = {c: i for i, c in enumerate(self.classes_)}
        indices = np.array([class_map[label] for label in y])
        one_hot[np.arange(n_samples), indices] = 1
        return one_hot

    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model using iterative gnostic weighting.
        """
        self.logger.info("Starting fit process for BoostingClassifierCalBase.")
        
        self.classes_ = np.unique(y)
        y_encoded = self._one_hot_encode(y)
        
        # Initialize weights
        self.weights = self._weight_init(len(y))
        
        # Iteration 0 fit
        self.model = self._fit_boosting_impl(X, y, self.weights)
        
        if self.max_iter == 0:
            return

        for self._iter in range(1, self.max_iter + 1):
            
            # Predict Probabilities with current model
            proba = self._predict_proba_boosting_impl(X, self.model)

            # Compute Residuals (vector difference)
            # XGBoost might predict partial classes if n_classes in data < n_classes in problem (unlikely here as we re-fit)
            # Or if classes are skipped. Assuming alignment with self.classes_
            
            residuals = proba - y_encoded
            # Magnitude of residuals per sample
            residual_magnitude = np.linalg.norm(residuals, axis=1)
            
            # Gnostic data conversion
            z_resid = self._data_conversion(residual_magnitude)
            
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
            
            # Compute gnostic criterion for history/stopping
            # Using residuals norm as metric
            z_current = z_resid
            z_ideal = self._data_conversion(np.zeros(len(y)))
            s = gwc.s if self.scale == 'auto' else self.scale
            
            loss, re, _, _, _, _, _, _, _, _, _, _ = self._gnostic_criterion(z=z_current, z0=z_ideal, s=s)

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
