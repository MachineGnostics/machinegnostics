'''
CartClassifierCalBase - Base class for CART Classification calculations

Machine Gnostics
'''

import numpy as np
from machinegnostics.models.cart.base_cart_classifier_methods import CartClassifierMethodsBase
from machinegnostics.magcal import GnosticsWeights, ScaleParam

class CartClassifierCalBase(CartClassifierMethodsBase):
    """
    Base calculation class for Gnostic CART Classification models.
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
        self.logger.info("Starting fit process for CartClassifierCalBase.")
        
        self.classes_ = np.unique(y)
        y_encoded = self._one_hot_encode(y)
        
        # Initialize weights
        self.weights = self._weight_init(len(y))
        
        # Iteration 0 fit
        if self.estimator_type == 'forest':
            self.trees = self._fit_forest_impl(X, y, self.weights)
        else:
            self.tree = self._fit_single_tree_impl(X, y, self.weights)
        
        if self.max_iter == 0:
            return

        for self._iter in range(1, self.max_iter + 1):
            
            # Predict Probabilities
            if self.estimator_type == 'forest':
                proba = self._predict_proba_forest_impl(X, self.trees)
            else:
                proba = self._predict_proba_single_tree_impl(X, self.tree)

            # Compute Residuals (vector difference)
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
            if np.sum(new_weights) > 0:
                new_weights = new_weights / np.sum(new_weights) * len(y)
            else:
                 new_weights = np.ones(len(y))
            
            self.weights = new_weights
            
            # Compute gnostic criterion for history/stopping
            # We can use residual magnitude as 'z' for criterion, with z0=0 (ideal residual)
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
                 prev_re = self._history[-2]['rentropy']
                 if prev_loss is not None and abs(loss - prev_loss) < self.tolerance:
                     if self.verbose:
                         self.logger.info("Convergence reached.")
                     break
            
            # Re-fit
            if self.estimator_type == 'forest':
                self.trees = self._fit_forest_impl(X, y, self.weights)
            else:
                self.tree = self._fit_single_tree_impl(X, y, self.weights)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Internal predict."""
        if self.estimator_type == 'forest':
            proba = self._predict_proba_forest_impl(X, self.trees)
        else:
            proba = self._predict_proba_single_tree_impl(X, self.tree)
        return self._predict_impl(X, proba)
