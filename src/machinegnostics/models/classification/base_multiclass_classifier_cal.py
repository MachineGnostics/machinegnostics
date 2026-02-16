'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
Date: 2025-01-25

Description:
Base calculation class for multiclass classifier using gnostic weights approach.
'''
import numpy as np
from machinegnostics.magcal import GnosticsWeights
from machinegnostics.models.regression.base_regressor_methods import RegressorMethodsBase

class MulticlassClassifierCalBase(RegressorMethodsBase):
    """
    Base calculation class for the Multiclass Classifier model.
    
    This class provides core fitting and prediction logic for multiclass classification
    using gnostic weights and softmax activation.
    
    Parameters
    ----------
    degree : int, default=1
        Degree of polynomial features to use.
    max_iter : int, default=100
        Maximum number of iterations for optimization.
    tolerance : float, default=1e-1
        Convergence tolerance for stopping criteria.
    early_stopping : bool, default=True
        Whether to stop early when convergence is reached.
    verbose : bool, default=False
        Whether to print progress during training.
    scale : str | int | float, default='auto'
        Scaling parameter for gnostic calculations.
    data_form : str, default='a'
        Data form for gnostic conversion ('a' for additive).
    gnostic_characteristics : bool, default=True
        Whether to calculate gnostic characteristics during training.
    history : bool, default=True
        Whether to maintain training history.
    """
    
    def __init__(self,
                 degree: int = 1,
                 max_iter: int = 100,
                 tolerance: float = 1e-1,
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: 'str | int | float' = 'auto',
                 data_form: str = 'a',
                 gnostic_characteristics: bool = True,
                 history: bool = True):
        super().__init__(
            degree=degree,
            max_iter=max_iter,
            tolerance=tolerance,
            early_stopping=early_stopping,
            verbose=verbose,
            scale=scale,
            data_form=data_form,
            gnostic_characteristics=gnostic_characteristics,
            history=history
        )
        
        self.degree = degree
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.scale = scale
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self.history = history
        
        # Model parameters
        self.coefficients = None
        self.weights = None
        self.num_classes = None
        self.cross_entropy_loss = None

        # Auto-enable history when early_stopping is True
        if self.early_stopping and not self.history:
            self.logger.warning(
                "early_stopping=True requires history=True. Automatically enabling history."
            )
            
            self.history = True
            self.logger.info("History has been enabled.")
        
        # History tracking
        if self.history:
            self._history = []
            self._history.append({
                'iteration': 0,
                'cross_entropy_loss': None,
                'coefficients': None,
                'rentropy': None,
                'weights': None,
            })
        else:
            self._history = None
    
        # logger
        self.logger.info("MulticlassClassifierCalBase initialized.")

        # check input
        self._input_checks()
        self.logger.info("Input parameters validated successfully.")
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Compute softmax activation with numerical stability.
        
        Parameters
        ----------
        z : np.ndarray of shape (n_samples, n_classes)
            Linear predictions.
            
        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Softmax probabilities.
        """
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot_encode(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Convert class labels to one-hot encoding.
        
        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Class labels.
        num_classes : int
            Number of classes.
            
        Returns
        -------
        np.ndarray of shape (n_samples, num_classes)
            One-hot encoded labels.
        """
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y.astype(int)] = 1
        return one_hot
    
    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the multiclass classifier using gnostic weights.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training labels (class indices).
            
        Returns
        -------
        self
            Fitted classifier.
        """
        self.logger.info("Starting fit process for Multiclass Classifier.")
        
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X)
        n_samples, n_features = X_poly.shape
        
        # Determine number of classes
        self.num_classes = len(np.unique(y))
        
        # One-hot encode target
        y_encoded = self._one_hot_encode(y, self.num_classes)
        
        # Initialize weights (sample weights)
        self.weights = np.ones(n_samples)
        
        # Initialize coefficients: shape (n_features, n_classes)
        self.coefficients = np.zeros((n_features, self.num_classes))
        
        for self._iter in range(self.max_iter):
            self._iter += 1
            self._prev_coef = self.coefficients.copy()
            
            try:
                # Compute linear predictions: (n_samples, n_classes)
                linear_pred = X_poly @ self.coefficients
                
                # Compute softmax probabilities
                proba = self._softmax(linear_pred)
                
                # Compute residuals for each class
                residuals = proba - y_encoded  # shape: (n_samples, n_classes)
                
                # Calculate gnostic weights for each class and combine
                # We'll use the magnitude of residuals across all classes
                residual_magnitude = np.linalg.norm(residuals, axis=1)  # shape: (n_samples,)
                
                # MG data conversion
                z = self._data_conversion(residual_magnitude)
                
                # Calculate gnostic weights
                gwc = GnosticsWeights()
                gw = gwc._get_gnostic_weights(z)
                
                # Update weights: multiply existing weights by gnostic weights
                new_weights = self.weights * gw
                W = np.diag(new_weights)
                
                # Compute scale
                if self.scale == 'auto':
                    s = gwc.s
                else:
                    s = self.scale
                
                # Weighted least squares update for each class
                try:
                    XtW = X_poly.T @ W
                    XtWX = XtW @ X_poly + 1e-8 * np.eye(n_features)
                    
                    # Update coefficients for each class
                    for c in range(self.num_classes):
                        # For class c, solve: XtWX @ coef = XtW @ (linear_pred[:, c] - learning_rate * residuals[:, c])
                        XtWy = XtW @ (linear_pred[:, c] - residuals[:, c])
                        self.coefficients[:, c] = np.linalg.solve(XtWX, XtWy)
                        
                except np.linalg.LinAlgError:
                    for c in range(self.num_classes):
                        XtWy = XtW @ (linear_pred[:, c] - residuals[:, c])
                        self.coefficients[:, c] = np.linalg.pinv(XtWX) @ XtWy
                
                # Recompute predictions with updated coefficients
                linear_pred = X_poly @ self.coefficients
                proba = self._softmax(linear_pred)
                
                # Calculate cross-entropy loss
                proba_clipped = np.clip(proba, 1e-8, 1 - 1e-8)
                self.cross_entropy_loss = -np.mean(np.sum(y_encoded * np.log(proba_clipped), axis=1))
                
                # Calculate entropy for convergence check
                _, _, re = self._gnostic_prob(z=z)
                re_mean = np.mean(re)
                self.re = re_mean
                
                # Calculate gnostic characteristics if requested
                if self.gnostic_characteristics:
                    z_pred = self._data_conversion(np.linalg.norm(proba - y_encoded, axis=1))
                    z_true = self._data_conversion(np.zeros(n_samples))  # Perfect prediction baseline
                    self.loss, self.re, self.hi, self.hj, self.fi, self.fj, \
                    self.pi, self.pj, self.ei, self.ej, self.infoi, self.infoj = \
                        self._gnostic_criterion(z=z_pred, z0=z_true, s=s)
                
                # Update history
                if self._history is not None:
                    self._history.append({
                        'iteration': self._iter,
                        'cross_entropy_loss': self.cross_entropy_loss,
                        'coefficients': self.coefficients.copy(),
                        'rentropy': self.re,
                        'weights': self.weights.copy(),
                    })
                
                # Check convergence
                if self._iter > 0 and self.early_stopping:
                    prev_hist = self._history[-2] if len(self._history) > 1 else None
                    curr_loss = self.cross_entropy_loss
                    curr_re = self.re
                    
                    prev_loss = prev_hist['cross_entropy_loss'] if prev_hist and prev_hist['cross_entropy_loss'] is not None else None
                    prev_re_val = prev_hist['rentropy'] if prev_hist and prev_hist['rentropy'] is not None else None
                    
                    loss_converged = prev_loss is not None and np.abs(curr_loss - prev_loss) < self.tolerance
                    re_converged = prev_re_val is not None and np.abs(curr_re - prev_re_val) < self.tolerance
                    
                    if loss_converged or re_converged:
                        if self.verbose:
                            self.logger.info(f"Converged at iteration {self._iter} (early stop):")
                            if loss_converged:
                                self.logger.info(f"Cross-entropy change below tolerance: {np.abs(curr_loss - prev_loss):.6e}")
                            if re_converged:
                                self.logger.info(f"Rentropy change below tolerance: {np.abs(curr_re - prev_re_val):.6e}")
                        break
                
                if self.verbose:
                    self.logger.info(f"Iteration {self._iter}, Cross-Entropy Loss: {self.cross_entropy_loss:.6f}, mean residual entropy: {re_mean:.6f}")
                    
            except (ZeroDivisionError, np.linalg.LinAlgError) as e:
                self.coefficients = self._prev_coef
                self.weights = self.weights.copy()
                if self.verbose:
                    self.logger.error(f"Error during fitting at iteration {self._iter}: {e}")
                break
        
        self.logger.info("Fit process completed.")
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self.logger.info("Making predictions with Multiclass Classifier.")
        if self.coefficients is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")
        
        proba = self._predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.
            
        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities for each class.
        """
        self.logger.info("Calculating predicted probabilities with Multiclass Classifier.")
        if self.coefficients is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict_proba'.")
        
        X_poly = self._generate_polynomial_features(X)
        linear_pred = X_poly @ self.coefficients
        proba = self._softmax(linear_pred)
        
        return proba
