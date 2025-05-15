'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.
'''

import numpy as np

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
                 eps: float = 1e-10):
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

    def _get_q_q1(self, S:int=1):
        """
        Calculates the q and q1 for given z and z0

        For internal use only

        Parameters
        ----------
        R : np.ndarray
            Input values (typically residuals)
        s : int, optional
            Override for shape parameter s

        Returns
        -------
        tuple
            (q, q1) computed characteristic values
        """
        # Add small constant to prevent division by zero
        R_safe = np.abs(self.R) + self.eps
        
        try:
            # Compute power with safety checks
            self.q = np.power(R_safe, 2/S)
            self.q1 = np.power(R_safe, -2/S)
            
            # Ensure no negative or zero values
            self.q = np.maximum(self.q, self.eps)
            self.q1 = np.maximum(self.q1, self.eps)
            
        except RuntimeWarning:
            # Handle any remaining warnings by clipping values
            self.q = np.clip(np.power(R_safe, 2/S), self.eps, None)
            self.q1 = np.clip(np.power(R_safe, -2/S), self.eps, None)
        
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
        if q is None:
            q = self.q
        if q1 is None:
            q1 = self.q1
        q = np.asarray(q)
        q1 = np.asarray(q1)
        if q.shape != q1.shape:
            raise ValueError("q and q1 must have the same shape")
        
        eps = np.finfo(float).eps
        denominator = q + q1
        denominator = np.where(denominator != 0, denominator, eps)
        
        # Calculate ratio with clipping to prevent overflow
        h = np.clip((q - q1) / denominator, -1.0, 1.0)
        
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
        if q is None:
            q = self.q
        if q1 is None:
            q1 = self.q1
            
        q = np.asarray(q)
        q1 = np.asarray(q1)
        if q.shape != q1.shape:
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
        fi = np.asarray(fi)
        fj = np.asarray(fj)
        if fi.shape != fj.shape:
            raise ValueError("fi and fj must have the same shape")
        entropy = fj - fi
        if entropy < 0: #means something is wrong
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
        fi = np.asarray(fi)
        if fi.shape != self.q.shape:
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
        fj = np.asarray(fj)
        if fj.shape != self.q.shape:
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
        hi = np.asarray(hi)
        if hi.shape != self.q.shape:
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
        hj = np.asarray(hj)
        if hj.shape != self.q.shape:
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
        p_i = np.asarray(p_i)
        if p_i.shape != self.q.shape:
            raise ValueError("p_i and q must have the same shape")
        epsilon = 1e-8
        Ii = -p_i * np.log(p_i+epsilon) - (1 - p_i) * np.log(1 - p_i)
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
        p_j = np.asarray(p_j)
        if p_j.shape != self.q.shape:
            raise ValueError("p_j and q must have the same shape")
        epsilon = 1e-8
        Ij = -p_j * np.log(p_j) - (1 - p_j) * np.log(1 - p_j)
        return Ij
    
