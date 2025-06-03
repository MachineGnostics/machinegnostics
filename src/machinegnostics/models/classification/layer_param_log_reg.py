'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
Date: 2025-05-31

Description:
Regressor param base class that can be used for robust classification models.
- logical regression

'''
import numpy as np
from machinegnostics.magcal import (ScaleParam, 
                                    GnosticsWeights, 
                                    ParamBase)
from machinegnostics.magcal.util.min_max_float import np_max_float
class ParamLogisticRegressorBase(ParamBase):
    """
    Parameters for the Logistic Regressor model.
    
    Attributes
    ----------
    scale_param : ScaleParam
        Scaling parameters for the model.
    gnostics_weights : GnosticsWeights
        Weights for the model.
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
                 proba: str = 'gnostic'):
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
            proba=proba
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
        self.proba = proba
        # history option
        if history:
            self._history = []
            # default history content
            self._history.append({
                'iteration': 0,
                'log_loss': None,
                'coefficients': None,
                'rentropy': None,
                'weights': None,
            })
        else:
            self._history = None
    
    def _fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to the data.
        
        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        """
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X)
        
        # Initialize weights
        self.weights = self._weight_init(d=y, like='one')
        
        # Initialize coefficients to zeros
        self.coefficients = np.zeros(X_poly.shape[1])
        
        for self._iter in range(self.max_iter):
            self._iter += 1
            self._prev_coef = self.coefficients.copy()
            
            try:
                # Weighted least squares
                self.coefficients = self._weighted_least_squares(X_poly, y, self.weights)
                
                # Update weights using gnostic approach
                y0 = X_poly @ self.coefficients
                residuals = y0 - y
                
                # mg data conversion
                z = self._data_conversion(residuals)

                # gnostic weights
                gw = GnosticsWeights()
                gw = gw._get_gnostic_weights(z)
                new_weights = self.weights * gw

                # Compute scale and loss
                if self.scale == 'auto':
                    scale = ScaleParam()
                    # local scale 
                    s = scale._gscale_loc(np.mean(2 / (z + 1/z)))
                else:
                    s = self.scale
                
                # gnostic probabilities
                if self.proba == 'gnostic':
                    # Gnostic probability calculation
                    p, info, re = self._gnostic_prob(z, s)
                elif self.proba == 'sigmoid':
                    # Sigmoid probability calculation
                    p = self._sigmoid(y0)
                    _, info, re = self._gnostic_prob(z, s)

                self.coefficients = self._wighted_least_squares_log_reg(p, y0, X_poly, y, W=new_weights)

                # --- Log loss calculation ---
                proba_pred = np.clip(p, 1e-8, 1-1e-8)
                self.log_loss = -np.mean(y * np.log(proba_pred) + (1 - y) * np.log(1 - proba_pred))
            

                if self.gnostic_characteristics:
                    self.loss, self.re, self.hi, self.hj, self.fi, self.fj, \
                    self.pi, self.pj, self.ei, self.ej, self.infoi, self.infoj  = self._gnostic_criterion(z, y0, s)

                self.weights = new_weights / np.sum(new_weights) # NOTE : Normalizing weights

                # history update for gnostic vs sigmoid
                # normalize rentropy and information
                re_norm = (re - np.min(re)) / (np.max(re) - np.min(re)) if np.max(re) != np.min(re) else re
                info_norm = (info - np.min(info)) / (np.max(info) - np.min(info)) if np.max(info) != np.min(info) else info                              
                
                # capture history and append to history
                # minimal history capture
                if self._history is not None:
                    self._history.append({
                        'iteration': self._iter +1,
                        'log_loss': self.log_loss,
                        'coefficients': self.coefficients.copy(),
                        'information': info_norm,
                        'rentropy': re_norm,
                        'weights': self.weights.copy(),
                    })

                # Check convergence with early stopping and rentropy
                # if entropy value is increasing, stop
                # --- Unified convergence check: stop if mean rentropy change is within tolerance ---

                if self._iter > 0 and self.early_stopping:
                    prev_re = self._history[-2]['rentropy'] if len(self._history) > 1 else None
                    curr_re = re_norm
                    prev_re_val = np.mean(prev_re) if prev_re is not None else None
                    if prev_re_val is not None and np.abs(curr_re - prev_re_val) < self.tol:
                        if self.verbose:
                            print(f"Converged at iteration {self._iter+ 1} (early stop): rentropy change below tolerance.")
                        break
                if self.verbose:
                    print(f"Iteration {self._iter + 1}, Log Loss: {self.log_loss:.6f}, mean residual entropy: {re_norm:.6f}")
                
            except (ZeroDivisionError, np.linalg.LinAlgError) as e:
                # Handle exceptions during fitting
                self.coefficients = self._prev_coef
                self.weights = self.weights.copy()
                if self.verbose:
                    print(f"Error during fitting: {e}")
                    print(f"Iteration {self._iter + 1}, Log Loss: {self.log_loss:.6f}")
                break
            

    def _predict(self, X: np.ndarray, threshold=0.5) -> np.ndarray:
        """
        Predict class labels for the input data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features to predict class labels for.
        threshold : float, optional (default=0.5)
            Threshold for classifying probabilities into binary classes.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        proba = self._predict_proba(X)
        return (proba >= threshold).astype(int)

      
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for the input data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features to predict probabilities for.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted probabilities.
        """
        if self.coefficients is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict_proba'.")
        
        X_poly = self._generate_polynomial_features(X)
        linear_pred = X_poly @ self.coefficients
        
        # gnostic vs sigmoid probability calculation
        if self.proba == 'gnostic':
            # Gnostic probability calculation
            proba, info, re = self._gnostic_prob(-linear_pred)
        elif self.proba == 'sigmoid':
            # Sigmoid probability calculation
            proba = self._sigmoid(linear_pred)
        else:
            raise ValueError("Invalid probability method. Must be 'gnostic' or 'sigmoid'.")
        
        return proba