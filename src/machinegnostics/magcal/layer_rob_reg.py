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
                 scale: [str, int, float] = 'auto',
                 history: bool = True,
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
            history=history,
            data_form=data_form,
            gnostic_characteristics=gnostic_characteristics
        )

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
        self.weights = self._weight_init(d=X_poly, like='ones')
        
        # Initialize coefficients to zeros
        self.coefficients = np.zeros(X_poly.shape[1])
        
        for self._iter in range(self.max_iter):
            self._iter += 1
            prev_coef = self.coefficients.copy()
            
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
                if self.scale_value == 'auto':
                    scale = ScaleParam()
                    # local scale 
                    s = scale._gscale_loc(np.mean(2 / (z + 1/z)))
                else:
                    s = self.scale_value

                loss, re, pi, pj, ei, ej, infoi, infoj  = self._gnostic_criterion(z, y0, s)

                self.weights = new_weights / np.sum(new_weights) # NOTE : Normalizing weights
                                                
                # print loss
                if self.verbose:
                    print(f'Iteration: {self._iter} - Machine Gnostic loss - {self.mg_loss} : {np.round(loss, 4)}, rentropy: {np.round(re, 4)}')

                # Check convergence with early stopping and rentropy
                # if entropy value is increasing, stop
                if self.early_stopping and self._history is not None:
                    if len(self._history) > 1:
                        prev_loss = self._history[-2]['h_loss']
                        prev_re = self._history[-2]['rentropy']
                        if (prev_loss is not None) and (prev_re is not None):
                            if (np.abs(loss - prev_loss) < self.tol) or (np.abs(re - prev_re) < self.tol):
                                if self.verbose:
                                    print(f"Convergence reached at iteration {self._iter} with loss/rentropy change below tolerance.")
                                break
                        
            except (ZeroDivisionError, np.linalg.LinAlgError) as e:
                if self.verbose:
                    print(f"Warning: {str(e)}. Using previous coefficients.")
                self.coefficients = prev_coef
                break