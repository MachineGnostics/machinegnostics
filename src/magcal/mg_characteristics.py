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
    """

    def __init__(self, R: np.ndarray):
        """
        Initializes the GnosticsCharacteristics class.

        Parameters
        ----------
        R : np.ndarray
            The input matrix for the gnostics calculations (R = Z / Z0).
        """
        eps = np.finfo(float).max
        self.R = R
        self.q = R
        # avoid division by zero
        self.q1 = np.where(R != 0, 1 / R, eps)

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
        h = (q - q1) / (q + q1)
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
