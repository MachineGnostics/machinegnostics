'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
Date: 2025-05-31

Description:
Regressor param base class that can be used for robust regression models.
- linear regression
- polynomial regression
- logistic regression

'''
import logging
from machinegnostics.magcal.util.logging import get_logger
import numpy as np
from itertools import combinations_with_replacement
from machinegnostics.magcal import (ModelBase, GnosticCharacteristicsSample, 
                                    GnosticCriterion, 
                                    GnosticsCharacteristics, 
                                    ScaleParam,
                                    GnosticsWeights,
                                    DataConversion)
from machinegnostics.magcal.util.min_max_float import np_max_float, np_min_float, np_eps_float
from typing import Union

class ParamBase(ModelBase):
    """
    Base class for MAGCAL - Machine Gnostic Calculations.

    Calculations for fit, predict, score method
    Calculations for gnostic loss, weights, coefficients, degree, entropy, fi, hi, fj, hj, infoi, infoj, pi, pj, ei, ej
    
    """""

    def __init__(self,
                 degree: int = 1,
                 max_iter: int = 100,
                 tol: float = 1e-8,
                 mg_loss: str = 'hi',
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: Union[str, int, float] = 'auto',
                 history: bool = True,
                 data_form: str = 'a',
                 gnostic_characteristics:bool=True):
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
            history=history
        )
        """
        Initialize the ParamBase class.

        Parameters
        ----------
        gnostic_characteristics : dict, optional
            Dictionary containing gnostic characteristics like loss, weights, coefficients, etc.
        """
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.coefficients = None
        self.weights = None
        self.early_stopping = early_stopping
        self.mg_loss = mg_loss
        self.verbose = verbose
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self.scale = scale
        # --- Scale input handling ---
        # if isinstance(scale, str):
        #     if scale != 'auto':
        #         raise ValueError("scale must be 'auto' or a float between 0 and 2.")
        #     self.scale_value = 'auto'
        # elif isinstance(scale, (int, float)):
        #     if not (0 <= scale <= 2):
        #         raise ValueError("scale must be 'auto' or a float between 0 and 2.")
        #     self.scale_value = float(scale)
        # else:
        #     raise ValueError("scale must be 'auto' or a float between 0 and 2.")
        # # data form check additive or multiplicative
        # if self.data_form not in ['a', 'm']:
        #     raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative.")
        # history option
        if history:
            self._history = []
            # default history content
            self._history.append({
                'iteration': 0,
                'h_loss': None,
                'coefficients': None,
                'rentropy': None,
                'fi': None,
                'fj': None,
                'hi': None,
                'hj': None,
                'pi': None,
                'pj': None,
                'ei': None,
                'ej': None,
                'infoi': None,
                'infoj': None,
                'weights': None,
                'scale': None,
            })
        else:
            self._history = None
        
        # logger
        self.logger = get_logger(self.__class__.__name__, logging.INFO)  # Create a logger for this class
        self.logger.info("ParamBase initialized.")

    def _generate_polynomial_features(self, X:np.ndarray) -> np.ndarray:
        """
        Generate polynomial features for multivariate input up to specified degree.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input features

        Returns:
        --------
        X_poly : ndarray of shape (n_samples, n_output_features)
            Polynomial features including interaction terms
        """
        self.logger.info(f"Generating polynomial features of degree")
        n_samples, n_features = X.shape
        combinations = []
        for degree in range(self.degree + 1):
            combinations += list(combinations_with_replacement(range(n_features), degree))

        X_poly = np.ones((n_samples, len(combinations)))
        for i, comb in enumerate(combinations):
            X_poly[:, i] = np.prod(X[:, comb], axis=1)
        self.logger.info(f"Generated polynomial features shape: {X_poly.shape}")
        return X_poly
    
    def _weight_init(self, d: np.ndarray, like: str ='one') -> np.ndarray:
        """
        Initialize weights based on the input data.

        Parameters
        ----------
        d : np.ndarray
            Input data.
        like : str, optional
            Type of initialization ('one', 'zero'). Default is 'one'.

        Returns
        -------
        np.ndarray
            Initialized weights.
        """
        self.logger.info(f"Initializing weights with method: {like}")
        if like == 'one':
            return np.ones(len(d))
        elif like == 'zero':
            return np.zeros(len(d))
        # elif like == 'random':
        #     return np.random.rand(d.shape[1]).flatten()
        else:
            self.logger.error("Invalid weight initialization method.")
            raise ValueError("like must be 'one', 'zero', or 'random'.")
        
    def _weighted_least_squares(self, X_poly, y, weights):
        """
        Solve weighted least squares using normal equations.
        
        Parameters:
        -----------
        X_poly : array-like
            Polynomial features matrix
        y : array-like
            Target values
        weights : array-like
            Sample weights
            
        Returns:
        --------
        array-like
            Estimated coefficients
        """
        self.logger.info("Solving weighted least squares.")
        eps = np_eps_float()  # Small value to avoid division by zero
        # Add small regularization term
        weights = np.clip(weights, eps, None)
        W = np.diag(weights)
        XtW = X_poly.T @ W
        XtWX = XtW @ X_poly + eps * np.eye(X_poly.shape[1])
        XtWy = XtW @ y
        
        try:
            return np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse for ill-conditioned matrices
            return np.linalg.pinv(XtWX) @ XtWy
    
    def _wighted_least_squares_log_reg(self, p, y0, X_poly:np.ndarray, y:np.ndarray, W:np.ndarray, n_features) -> np.ndarray:
        """
        Solve weighted least squares for logistic regression using normal equations.
        
        Parameters:
        -----------
        X_poly : array-like
            Polynomial features matrix
        y : array-like
            Target values
        weights : array-like
            Sample weights
            
        Returns:
        --------
        array-like
            Estimated coefficients
        """
        self.logger.info("Solving weighted least squares for logistic regression.")
        try:
            XtW = X_poly.T @ W
            XtWX = XtW @ X_poly + np_min_float() * np.eye(n_features)
            XtWy = XtW @ (y0 + (y - p) / (p * (1 - p) + 1e-8))
            self.coefficients = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.pinv(XtWX) @ XtWy
        return self.coefficients
    
    def _data_conversion(self, z:np.ndarray) -> np.ndarray:
        self.logger.info(f"Converting data using form: {self.data_form}")
        dc = DataConversion()
        if self.data_form == 'a':
            return dc._convert_az(z)
        elif self.data_form == 'm':
            return dc._convert_mz(z)
        else:
            raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative.")
    
    def _gnostic_criterion(self, z:np.ndarray, z0:np.ndarray, s) -> tuple: # NOTE can be improved by connecting with GDF
        """Compute the gnostic criterion.
        
        Parameters
        ----------
        z : np.ndarray
            Input data.
        z0 : np.ndarray
            Reference data.
        s : int or np.ndarray
            Scale parameter for the gnostic criterion.
        Returns
        -------
        tuple
            Tuple containing the gnostic criterion values.
        
        NOTE:
            normalized loss and rentropy are returned.
            """
        self.logger.info("Computing gnostic criterion.")
        q, q1 = self._compute_q(z, z0, s)

        # Default values for optional outputs
        pi = pj = ei = ej = infoi = infoj = None

        if self.mg_loss == 'hi':
            self.logger.info("Computing gnostic criterion for 'hi' loss.")
            hi = self.gc._hi(q, q1)
            fi = self.gc._fi(q, q1)
            fj = self.gc._fj(q, q1)
            re = self.gc._rentropy(fi, fj)
            if self.gnostic_characteristics:
                hj = self.gc._hj(q, q1)
                pi = self.gc._idistfun(hi)
                pj = self.gc._jdistfun(hj)
                infoi = self.gc._info_i(pi)
                infoj = self.gc._info_j(pj)
                ei = self.gc._ientropy(fi)
                ej = self.gc._jentropy(fj)
            else:
                hj = pi = pj = ei = ej = infoi = infoj = None

            # normalize hi and re
            re_norm = (re - np.min(re)) / (np.max(re) - np.min(re)) if np.max(re) != np.min(re) else re
            H = np.sum(hi ** 2)
            return H, np.mean(re_norm),hi, hj, fi, fj, pi, pj, ei, ej, infoi, infoj
        elif self.mg_loss == 'hj':
            self.logger.info("Computing gnostic criterion for 'hj' loss.")
            hj = self.gc._hj(q, q1)
            fi = self.gc._fi(q, q1)
            fj = self.gc._fj(q, q1)
            re = self.gc._rentropy(fi, fj)
            if self.gnostic_characteristics:
                hi = self.gc._hi(q, q1)
                pi = self.gc._idistfun(hi)
                pj = self.gc._jdistfun(hj)
                infoi = self.gc._info_i(pi)
                infoj = self.gc._info_j(pj)
                ei = self.gc._ientropy(fi)
                ej = self.gc._jentropy(fj)
            else:
                hi = pi = pj = ei = ej = infoi = infoj = None
            # normalize hj and re
            re_norm = (re - np.min(re)) / (np.max(re) - np.min(re)) if np.max(re) != np.min(re) else re
            H = np.sum(hj ** 2)
            return H, np.mean(re_norm),hi, hj, fi, fj, pi, pj, ei, ej, infoi, infoj
    
    def _compute_q(self, z, z0, s:int = 1):
        """
        For interval use only
        Compute q and q1."""
        self.logger.info("Computing q and q1 for gnostic criterion.")
        eps = np_eps_float()  # Small value to avoid division by zero
        z0_safe = np.where(np.abs(z0) < eps, eps, z0)
        zz = z / z0_safe
        self.gc = GnosticsCharacteristics(zz, verbose=self.verbose)
        q, q1 = self.gc._get_q_q1(S=s)     
        return q, q1
    
    def _normalize_weights(self, weights):
        """
        Normalize weights to ensure they sum to 1.

        Parameters
        ----------
        weights : np.ndarray
            Weights to be normalized.

        Returns
        -------
        np.ndarray
            Normalized weights.
        """
        self.logger.info("Normalizing weights.")
        total_weight = np.sum(weights)
        if total_weight == 0:
            return np.ones_like(weights) / len(weights)
        return weights / total_weight
        
    def _fit(self, X, y):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like
            Target values.
        """
        # Placeholder for fitting logic
        pass

    def _predict(self, X):
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : array-like
            Input features for prediction.

        Returns
        -------
        y_pred : array-like
            Predicted values.
        """
        # Placeholder for prediction logic
        pass
    
    def _score(self, X, y):
        """
        Compute the score of the model.

        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like
            True values for X.

        Returns
        -------
        score : float
            Score of the model.
        """
        # Placeholder for scoring logic
        pass

    def _sigmoid(self, z):
        """
        Compute the sigmoid function for logistic regression.
        Parameters
        ----------
        z : np.ndarray
            Input array for which to compute the sigmoid function.
        Returns
        -------
        np.ndarray
            Sigmoid of the input array.
        """
        return 1 / (1 + np.exp(-z))
    
    def _gnostic_prob(self, z) -> tuple:
        """
        Compute the gnostic probabilities and characteristics.
        Parameters
        ----------
        z : np.ndarray
            Input data for which to compute gnostic probabilities.
        Returns
        -------
        tuple
            Tuple containing the gnostic probabilities, information, and normalized rentropy.
        """
        self.logger.info("Computing gnostic probabilities and characteristics.")
        zz = self._data_conversion(z)
        gc = GnosticsCharacteristics(zz, verbose=self.verbose)

        # q, q1
        q, q1 = gc._get_q_q1()
        h = gc._hi(q, q1)
        fi = gc._fi(q, q1)

        # Scale handling
        if self.scale == 'auto':
            scale = ScaleParam(verbose=self.verbose)
            s = scale._gscale_loc(np.mean(fi)) # NOTE this refer to ELDF probability. Can be improved by connecting with GDF and its PDF
        else:
            s = self.scale
            
        q, q1 = gc._get_q_q1(S=s)
        h = gc._hi(q, q1)
        fi = gc._fi(q, q1)
        fj = gc._fj(q, q1)
        p = gc._idistfun(h)
        info = gc._info_i(p)
        re = gc._rentropy(fi, fj)
        # nomalized re
        re_norm = (re - np.min(re)) / (np.max(re) - np.min(re)) if np.max(re) != np.min(re) else re
        return p, info, re_norm