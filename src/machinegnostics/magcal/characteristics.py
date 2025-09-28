'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

Author: Nirmal Parmar
'''

import numpy as np
import logging
from machinegnostics.magcal.util.logging import get_logger

class GnosticsCharacteristics:
    """
    A class containing internal functions for Machine Gnostics (MG) calculations.

    Notes
    -----
    The class takes an input matrix R = Z / Z0, where:
        - Z  : Observed data
        - Z0 : Estimated value

    Internally, it computes:
        - q  = R
        - q1 = 1 / R  (with protection against division by zero)

    The internal methods (_fi, _fj, _hi, _hj) operate on q and q1 to calculate
    various gnostic characteristics.

    Methods
    -------
    _fi(q, q1)
        Calculates the estimation weight.

    _fj(q, q1)
        Calculates the quantification weight.

    _hi(q, q1)
        Calculates the estimation relevance.

    _hj(q, q1)
        Calculates the quantification relevance.

    _rentropy(fi, fj)
        Calculates the residual entropy.

    _ientropy(fi)
        Calculates the estimating entropy.

    _jentropy(fj)
        Calculates the quantifying entropy.

    _idistfun(hi)
        Calculates the estimating distribute function function.

    _jdistfun(hj)
        Calculates the quantifying distribute function function.

    _info_i(p_i)
        Calculates the estimating information.
        
    _info_j(p_j)
        Calculates the quantifying information.
    """

    def __init__(self, 
                 R: np.ndarray,
                 eps: float = 1e-10,
                 verbose: bool = False):
        """
        Initializes the GnosticsCharacteristics class.

        Parameters
        ----------
        R : np.ndarray
            The input matrix for the gnostics calculations (R = Z / Z0).
        eps : float, default=1e-10
            Small constant for numerical stability
        """
        self.R = R
        self.eps = eps

        # logger setup
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")

    def _get_q_q1(self, S: int = 1):
        """
        Calculates the q and q1 for given z and z0
    
        For internal use only
    
        Parameters
        ----------
        R : np.ndarray
            Input values (typically residuals)
        S : int, optional
            Override for shape parameter S
    
        Returns
        -------
        tuple
            (q, q1) computed characteristic values
        """
        self.logger.info("Calculating q and q1.")
        # Add small constant to prevent division by zero
        R_safe = np.abs(self.R) + self.eps

        # avoid overflow in exponentiation
        S = np.maximum(S, 0.01)  # Ensure S is at least 0.01 to avoid division by zero
        
        # Calculate exponents
        exp_pos = 2.0 / S
        exp_neg = -2.0 / S
        
        # Use log-space calculation to determine safe limits
        log_max = np.log(np.finfo(float).max)
        
        # Safe upper limit calculation in log space
        # max_safe_value = exp(log_max / 10) to prevent overflow
        max_safe_log = log_max / 10.0
        max_safe_value = np.exp(max_safe_log)
        
        # Clip R_safe to prevent overflow
        R_safe = np.clip(R_safe, self.eps, max_safe_value)
        
        # Use log-space calculations for numerical stability
        log_R = np.log(R_safe)
        
        # Calculate in log space to avoid overflow
        log_q = exp_pos * log_R
        log_q1 = exp_neg * log_R
        
        # Safe exponential limits
        max_exp = log_max - 2.0  # Leave some headroom
        min_exp = -max_exp
        
        # Clip log values to prevent overflow/underflow
        log_q = np.clip(log_q, min_exp, max_exp)
        log_q1 = np.clip(log_q1, min_exp, max_exp)
        
        # Convert back from log space
        self.q = np.exp(log_q)
        self.q1 = np.exp(log_q1)
        
        # Final safety checks
        safe_max = np.finfo(float).max / 1e6
        self.q = np.clip(self.q, self.eps, safe_max)
        self.q1 = np.clip(self.q1, self.eps, safe_max)
        
        # Ensure no NaN or Inf values
        self.q = np.nan_to_num(self.q, nan=self.eps, posinf=safe_max, neginf=self.eps)
        self.q1 = np.nan_to_num(self.q1, nan=self.eps, posinf=safe_max, neginf=self.eps)
        
        return self.q, self.q1
        
    def _fi(self, q=None, q1=None):
        """
        Calculates the estimation weight.

        Parameters
        ----------
        q : np.ndarray or float
        q1 : np.ndarray or float

        Returns
        -------
        f : np.ndarray or float
        """
        self.logger.info("Calculating estimation weight fi.")
        if q is None:
            q = self.q
        if q1 is None:
            q1 = self.q1

        q = np.asarray(q)
        q1 = np.asarray(q1)
        if q.shape != q1.shape:
            raise ValueError("q and q1 must have the same shape")
        f = 2 / (q + q1)
        return f

    def _fj(self, q=None, q1=None):
        """
        Calculates the quantification weight.

        Parameters
        ----------
        q : np.ndarray or float
        q1 : np.ndarray or float

        Returns
        -------
        f : np.ndarray or float
        """
        self.logger.info("Calculating quantification weight fj.") 
        if q is None:
            q = self.q
        if q1 is None:
            q1 = self.q1

        q = np.asarray(q)
        q1 = np.asarray(q1)
        if q.shape != q1.shape:
            raise ValueError("q and q1 must have the same shape")
        f = (q + q1) / 2
        return f

    def _hi(self, q=None, q1=None):
        """
        Calculates the estimation relevance.
    
        Parameters
        ----------
        q : np.ndarray or float
        q1 : np.ndarray or float
    
        Returns
        -------
        h : np.ndarray or float
        """
        self.logger.info("Calculating estimation relevance hi.")
        if q is None:
            q = self.q
        if q1 is None:
            q1 = self.q1
            
        q = np.asarray(q)
        q1 = np.asarray(q1)
        if q.shape != q1.shape:
            self.logger.error("q and q1 must have the same shape")
            raise ValueError("q and q1 must have the same shape")
        
        # Handle potential overflow/underflow in q and q1
        q = np.nan_to_num(q, nan=self.eps, posinf=np.finfo(float).max / 1e6)
        q1 = np.nan_to_num(q1, nan=self.eps, posinf=np.finfo(float).max / 1e6)
        
        # Calculate numerator and denominator separately
        numerator = q - q1
        denominator = q + q1
        
        # Handle cases where denominator is very small or zero
        eps_threshold = self.eps * 1000  # Use larger threshold for stability
        denominator = np.where(np.abs(denominator) < eps_threshold, 
                              eps_threshold * np.sign(denominator), 
                              denominator)
        
        # Calculate ratio with additional safety
        with np.errstate(divide='raise', invalid='raise'):
            try:
                h = numerator / denominator
            except (FloatingPointError, RuntimeWarning):
                # Fallback calculation
                # When q >> q1 or q1 >> q, handle separately
                mask_q_large = q > 1000 * q1
                mask_q1_large = q1 > 1000 * q
                mask_normal = ~(mask_q_large | mask_q1_large)
                
                h = np.zeros_like(q)
                h[mask_q_large] = 1.0  # When q >> q1, h approaches 1
                h[mask_q1_large] = -1.0  # When q1 >> q, h approaches -1
                h[mask_normal] = numerator[mask_normal] / denominator[mask_normal]
        
        # Final clipping and NaN handling
        h = np.clip(h, -1.0, 1.0)
        h = np.nan_to_num(h, nan=0.0)
        
        return h
    
    def _hj(self, q=None, q1=None):
        """
        Calculates the quantification relevance.

        Parameters
        ----------
        q : np.ndarray or float
        q1 : np.ndarray or float

        Returns
        -------
        h : np.ndarray or float
        """
        self.logger.info("Calculating quantification relevance hj.")
        if q is None:
            q = self.q
        if q1 is None:
            q1 = self.q1
            
        q = np.asarray(q)
        q1 = np.asarray(q1)
        if q.shape != q1.shape:
            self.logger.error("q and q1 must have the same shape")
            raise ValueError("q and q1 must have the same shape")
        h = (q - q1) / 2
        return h
    
    def _rentropy(self, fi, fj):
        """
        Calculates the residual entropy.

        Parameters
        ----------
        fi : np.ndarray or float
            Estimation weight.
        fj : np.ndarray or float
            Quantification weight.

        Returns
        -------
        entropy : np.ndarray or float
            Relative entropy.
        """
        self.logger.info("Calculating residual entropy.")
        fi = np.asarray(fi)
        fj = np.asarray(fj)
        if fi.shape != fj.shape:
            self.logger.error("fi and fj must have the same shape")
            raise ValueError("fi and fj must have the same shape")
        entropy = fj - fi
        if (entropy < 0).any(): #means something is wrong
            self.logger.error("Entropy cannot be negative")
            raise ValueError("Entropy cannot be negative")
        return entropy
    
    def _ientropy(self, fi):
        """
        Calculates the estimating entropy.

        Parameters
        ----------
        fi : np.ndarray or float
            Estimation weight.

        Returns
        -------
        entropy : np.ndarray or float
            Inverse relative entropy.
        """
        self.logger.info("Calculating estimating entropy.")
        fi = np.asarray(fi)
        if fi.shape != self.q.shape:
            self.logger.error("fi and q must have the same shape")
            raise ValueError("fi and q must have the same shape")
        entropy = 1 - fi
        return entropy
    
    def _jentropy(self, fj):
        """
        Calculates the quantifying entropy.

        Parameters
        ----------
        fj : np.ndarray or float
            Quantification weight.

        Returns
        -------
        entropy : np.ndarray or float
            Relative entropy.
        """
        self.logger.info("Calculating quantifying entropy.")
        fj = np.asarray(fj)
        if fj.shape != self.q.shape:
            self.logger.error("fj and q must have the same shape")
            raise ValueError("fj and q must have the same shape")
        entropy = fj - 1
        return entropy
    
    def _idistfun(self, hi):
        """
        Calculates the estimating distribute function function.

        Parameters
        ----------
        hi : np.ndarray or float
            Estimation relevance.

        Returns
        -------
        idist : np.ndarray or float
            Inverse distance function.
        """
        self.logger.info("Calculating estimating distribute function.")
        hi = np.asarray(hi)
        if hi.shape != self.q.shape:
            self.logger.error("hi and q must have the same shape")
            raise ValueError("hi and q must have the same shape")
        p_i = np.sqrt(np.power((1 - hi) / 2, 2)) # from MGpdf
        return p_i
    
    def _jdistfun(self, hj):
        """
        Calculates the quantifying distribute function function.

        Parameters
        ----------
        hj : np.ndarray or float
            Quantification relevance.

        Returns
        -------
        jdist : np.ndarray or float
            Inverse distance function.
        """
        self.logger.info("Calculating quantifying distribute function.")
        hj = np.asarray(hj)
        if hj.shape != self.q.shape:
            self.logger.error("hj and q must have the same shape")
            raise ValueError("hj and q must have the same shape")
        p_j = np.sqrt(np.power((1 - hj) / 2, 2))
        return p_j
    
    def _info_i(self, p_i):
        """
        Calculates the estimating information.

        Parameters
        ----------
        p_i : np.ndarray or float
            Inverse distance function.

        Returns
        -------
        info : np.ndarray or float
            Estimating information.
        """
        self.logger.info("Calculating estimating information.")
        p_i = np.asarray(p_i)
        if p_i.shape != self.q.shape:
            self.logger.error("p_i and q must have the same shape")
            raise ValueError("p_i and q must have the same shape")
        epsilon = 1e-12
        # avoid log(0)
        p_i = np.clip(p_i, 0 + epsilon, 1 - epsilon)
        Ii = -p_i * np.log(p_i + epsilon) - (1 - p_i) * np.log(1 - p_i + epsilon)
        return Ii
    
    def _info_j(self, p_j):
        """
        Calculates the quantifying information.

        Parameters
        ----------
        p_j : np.ndarray or float
            Inverse distance function.

        Returns
        -------
        info : np.ndarray or float
            Quantifying information.
        """
        self.logger.info("Calculating quantifying information.")
        p_j = np.asarray(p_j)
        if p_j.shape != self.q.shape:
            self.logger.error("p_j and q must have the same shape")
            raise ValueError("p_j and q must have the same shape")
        epsilon = 1e-12
        # avoid log(0)
        p_j = np.clip(p_j, 0 + epsilon, 1 - epsilon)
        Ij = -p_j * np.log(p_j + epsilon) - (1 - p_j) * np.log(1 - p_j + epsilon)
        return Ij