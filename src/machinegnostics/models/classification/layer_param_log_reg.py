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
from machinegnostics.magcal.util.min_max_float import np_max_float, np_min_float

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
        self.proba = proba
        self.mg_loss = 'hi'
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

        # logger
        self.logger.info("ParamLogisticRegressorBase initialized.")
    
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
        self.logger.info("Starting fit process for Logistic Regressor.")
        # Generate polynomial features
        X_poly = self._generate_polynomial_features(X)

        n_samples, n_features = X_poly.shape
        
        # Initialize weights
        self.weights = np.ones(n_samples)
        
        # Initialize coefficients to zeros
        self.coefficients = np.zeros(n_features)
        
        for self._iter in range(self.max_iter):
            self._iter += 1
            self._prev_coef = self.coefficients.copy()
            
            try:
                # # Weighted least squares
                # self.coefficients = self._weighted_least_squares(X_poly, y, self.weights)
                
                # Update weights using gnostic approach
                y0 = X_poly @ self.coefficients
                residuals = y0 - y
                
                # mg data conversion
                z = self._data_conversion(residuals)
                z_y = self._data_conversion(y)
                z_y0 = self._data_conversion(y0)

                # gnostic weights
                gw = GnosticsWeights()
                gw = gw._get_gnostic_weights(z)
                new_weights = self.weights * gw
                W = np.diag(new_weights)

                # Compute scale and loss
                if self.scale == 'auto':
                    scale = ScaleParam()
                    zz = z_y0 - z_y
                    # avoid division by zero
                    zz = np.where(zz == 0, np_min_float(), zz)  # Replace zero with a very small value
                    # local scale 
                    s = scale._gscale_loc((2 / (zz + 1/zz)))
                else:
                    s = self.scale
                
                # gnostic probabilities
                if self.proba == 'gnostic':
                    # Gnostic probability calculation
                    p, info, re = self._gnostic_prob(z=z) # NOTE currently using p from local S, means ELDF. this can be improved in the future
                elif self.proba == 'sigmoid':
                    # Sigmoid probability calculation
                    p = self._sigmoid(y0)
                    _, info, re = self._gnostic_prob(z=z)

                # self.coefficients = self._wighted_least_squares_log_reg(p, 
                #                                                         y0, 
                #                                                         X_poly,
                #                                                         y, 
                #                                                         W=W, 
                #                                                         n_features=n_features, 
                #                                                         )
                # IRLS update
                try:
                    XtW = X_poly.T @ W
                    XtWX = XtW @ X_poly + 1e-8 * np.eye(n_features)
                    XtWy = XtW @ (y0 + (y - p) / (p * (1 - p) + 1e-8))
                    self.coefficients = np.linalg.solve(XtWX, XtWy)
                except np.linalg.LinAlgError:
                    self.coefficients = np.linalg.pinv(XtWX) @ XtWy

                # --- Log loss calculation ---
                proba_pred = np.clip(p, 1e-8, 1-1e-8)
                self.log_loss = -np.mean(y * np.log(proba_pred) + (1 - y) * np.log(1 - proba_pred))

                # history update for gnostic vs sigmoid
                re = np.mean(re)
                info = np.mean(info)

                if self.gnostic_characteristics:
                    self.loss, self.re, self.hi, self.hj, self.fi, self.fj, \
                    self.pi, self.pj, self.ei, self.ej, self.infoi, self.infoj  = self._gnostic_criterion(z=z_y0, z0=z_y, s=s)

                # self.weights = new_weights / np.sum(new_weights) # NOTE : Normalizing weights

                
                # capture history and append to history
                # minimal history capture
                if self._history is not None:
                    self._history.append({
                        'iteration': self._iter,
                        'log_loss': self.log_loss,
                        'coefficients': self.coefficients.copy(),
                        'rentropy': re,
                        'weights': self.weights.copy(),
                    })

                # Check convergence with early stopping and rentropy
                # if entropy value is increasing, stop
                
                # --- Unified convergence check: stop if mean rentropy or log_loss change is within tolerance ---
                if self._iter > 0 and self.early_stopping:
                    prev_hist = self._history[-2] if len(self._history) > 1 else None
                    curr_re = np.mean(re)
                    curr_log_loss = self.log_loss
                    prev_re_val = np.mean(prev_hist['rentropy']) if prev_hist and prev_hist['rentropy'] is not None else None
                    prev_log_loss_val = prev_hist['log_loss'] if prev_hist and prev_hist['log_loss'] is not None else None

                    re_converged = prev_re_val is not None and np.abs(curr_re - prev_re_val) < self.tol
                    log_loss_converged = prev_log_loss_val is not None and np.abs(curr_log_loss - prev_log_loss_val) < self.tol

                    if re_converged or log_loss_converged:
                        if self.verbose:
                            self.logger.info(f"Converged at iteration {self._iter} (early stop):")
                            if re_converged:
                                self.logger.info(f"mean rentropy change below tolerance (rentropy={np.abs(curr_re - prev_re_val):.6e}).")
                            if log_loss_converged:
                                self.logger.info(f"log_loss change below tolerance (log_loss={np.abs(curr_log_loss - prev_log_loss_val):.6e}).")
                        break
                if self.verbose:
                    self.logger.info(f"Iteration {self._iter}, Log Loss: {self.log_loss:.6f}, mean residual entropy: {np.mean(re):.6f}")

            except (ZeroDivisionError, np.linalg.LinAlgError) as e:
                # Handle exceptions during fitting
                self.coefficients = self._prev_coef
                self.weights = self.weights.copy()
                if self.verbose:
                    self.logger.error(f"Error during fitting at iteration {self._iter}: {e}")
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
        self.logger.info("Making predictions with Logistic Regressor.")
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
        self.logger.info("Calculating predicted probabilities with Logistic Regressor.")
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
            self.logger.error("Invalid probability method. Must be 'gnostic' or 'sigmoid'.")
            raise ValueError("Invalid probability method. Must be 'gnostic' or 'sigmoid'.")
        
        return proba