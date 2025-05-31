'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
Date: 2025-05-31

Description: Machine Gnostics Robust Regression Machine Learning Model
This module implements a machine learning model for robust regression using the Machine Gnostics library.
This model is designed to handle various types of data and is particularly useful for applications in machine gnostics.
'''

import numpy as np
from machinegnostics.magcal import (ScaleParam, 
                                    GnosticsWeights, 
                                    ParamBase)

class ParamRobustRegressorBase(ParamBase):
    """
    Parameters for the Robust Regressor model.
    
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
                 tol: float = 1e-8,
                 mg_loss: str = 'hi',
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: 'str | int | float' = 'auto',
                 data_form: str = 'a',
                 gnostic_characteristics:bool=True
                 ):
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
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.mg_loss = mg_loss
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.scale = scale
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self._history = []
    
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
            prev_coef = self.coefficients.copy()
            print(f'coefficients: {self.coefficients}')
            
            try:
                # Weighted least squares
                self.coefficients = self._weighted_least_squares(X_poly, y, self.weights)
                
                # Update weights using gnostic approach
                y0 = X_poly @ self.coefficients
                residuals = y - y0
                
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

                self.loss, self.re, self.hi, self.hj, self.fi, self.fj, \
                self.pi, self.pj, self.ei, self.ej, self.infoi, self.infoj  = self._gnostic_criterion(z, y0, s)

                self.weights = new_weights / np.sum(new_weights) # NOTE : Normalizing weights
                                                
                # print loss
                if self.verbose:
                    print(f'Iteration: {self._iter} - Machine Gnostic loss - {self.mg_loss} : {np.round(self.loss, 4)}, rentropy: {np.round(self.re, 4)}')

                # capture history and append to history
                # minimal history capture
                if self._history is not None:
                    self._history.append({
                        'iteration': self._iter +1,
                        'h_loss': self.loss,
                        'coefficients': self.coefficients.copy(),
                        'rentropy': self.re,
                        'weights': self.weights.copy(),
                        'fi': self.fi,
                        'fj': self.fj,
                        'hi': self.hi,
                        'hj': self.hj,
                        'pi': self.pi,
                        'pj': self.pj,
                        'ei': self.ei,
                        'ej': self.ej,
                        'infoi': self.infoi,
                        'infoj': self.infoj,
                        'S': s
                    })

                # Check convergence with early stopping and rentropy
                # if entropy value is increasing, stop
                if self.early_stopping and self._history is not None:
                    if len(self._history) > 1:
                        prev_loss = self._history[-2]['h_loss']
                        prev_re = self._history[-2]['rentropy']
                        if (prev_loss is not None) and (prev_re is not None):
                            if (np.abs(self.loss - prev_loss) < self.tol) or (np.abs(self.re - prev_re) < self.tol):
                                if self.verbose:
                                    print(f"Convergence reached at iteration {self._iter} with loss/rentropy change below tolerance.")
                                break

            except (ZeroDivisionError, np.linalg.LinAlgError) as e:
                if self.verbose:
                    print(f"Warning: {str(e)}. Using previous coefficients.")
                self.coefficients = prev_coef
                break

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Internal prediction method for base class.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features to predict for.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """ 
        # copy iteration for last iteration
        
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet.")
        
        # Process input and generate features
        X_poly = self._generate_polynomial_features(X)
        
        # Validate dimensions
        n_features_model = X_poly.shape[1]
        if n_features_model != len(self.coefficients):
            raise ValueError(
                f"Feature dimension mismatch. Model expects {len(self.coefficients)} "
                f"features but got {n_features_model} after polynomial expansion."
            )
        
        return X_poly @ self.coefficients