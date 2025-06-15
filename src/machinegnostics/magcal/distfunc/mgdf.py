"""
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

Author: Nirmal Parmar
License: GNU General Public License v3.0 (GPL-3.0)

# Gnostic Distribution Function Module

This module implements the Gnostic distribution functions, which are a family of robust statistical 
methods based on the principles of information theory.
"""

import numpy as np
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.scale_param import ScaleParam

class DistributionFunctions:
    """
    Gnostic Distribution Functions

    """
    def __init__(self,
                 data:np.ndarray,
                 varS:bool = False, # NOTE: in future need to add auto mode for S
                 lb:float = None,
                 ub:float = None,
                 S:float = 1,
                 lc:float = None,
                 rc:float = None,
                 data_form:str = 'a',
                 data_points:int = 1000,
                 tol:float = 1e-6,
                 homogeneous:bool = True,# NOTE: in future, auto detection and if needed conversion to homogeneous
                 homoscedasticity:bool = True):
        """
        Initialize the distribution functions with data and parameters.

        Parameters:
        - data: Input data for the distribution.
        - varS (bool): Whether to varying S (True) or with global S (False) for a given data sample.
        - lb (float): Lower bound for the distribution.
        - ub (float): Upper bound for the distribution.
        - S (float): Scale parameter.
        - lc (float): left censored bound
        - rc (float): right censored bound
        - data_form (str): form of the input data - 'a' - additive, 'm' - multiplicative, and 'None' - no transformation.
        - data_points (int): Number of points in the evaluation grid.
        - tol (float): Tolerance for numerical computations.
        - homogeneous (bool): Whether the data is homogeneous.
        - homoscedasticity (bool): Whether the data is homoscedastic.
        """
        self.data = data
        self.varS = varS
        self.lb = lb
        self.ub = ub
        self.S = S
        self.lc = lc
        self.rc = rc
        self.data_form = data_form
        self.data_points = data_points
        self.tol = tol
        self.homogeneous = homogeneous
        self.homoscedasticity = homoscedasticity

        # handling Nones
        if self.lb is None:
            self.lb = np.min(data) if data.size > 0 else 0
        if self.ub is None:
            self.ub = np.max(data) if data.size > 0 else 1

        # argument checking
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array.")
        if data.ndim != 1:
            raise ValueError("Data must be a one-dimensional array.")
        if self.data_form not in ['a', 'm', None]:
            raise ValueError("data_form must be 'a', 'm', or None.")
        if self.lb is not None and self.ub is not None and self.lb >= self.ub:
            raise ValueError("Lower bound (lb) must be less than upper bound (ub).")
        if self.lc is not None and self.rc is not None and self.lc >= self.rc:
            raise ValueError("Left censored bound (lc) must be less than right censored bound (rc).")
        if self.S is not None and (not isinstance(self.S, (float, int)) or self.S <= 0):
            raise ValueError("Scale parameter (S) must be a positive scalar. For E*DF it should be in range (0, 2].")

    def _data_transform_input(self)-> np.ndarray:
        """
        Transform the input data based on the specified data form.

        first from normal domain to standard domain, and then from finite to infinite domain.
        """
        if self.data_form == 'a':
            self.z = DataConversion._convert_az(self.data, self.lb, self.ub)
        elif self.data_form == 'm':
            self.z = DataConversion._convert_mz(self.data, self.lb, self.ub)
        elif self.data_form is None:
            self.z = self.data
        else:
            raise ValueError("Invalid data form specified. Use 'a', 'm', or None.")
        
        # bound checking for infinite domain
        if self.ilb is None:
            self.ilb = np.min(self.z) if self.z.size > 0 else 0
        if self.iub is None:
            self.iub = np.max(self.z) if self.z.size > 0 else 1
        
        # Convert to infinite domain
        if self.data_form == 'a' or self.data_form == 'm':
            self.zi = DataConversion._convert_fininf(self.z, self.ilb, self.iub)

        return self.zi

    def _data_transform_output(self, data):
        """
        Transform the output data back to the original domain based on the specified data form.
        """
        if self.data_form == 'a' or self.data_form == 'm':
            # Convert to infinite domain
            self.zf = DataConversion._convert_fininf(data, self.ilb, self.iub)
        else:
            self.zf = data

        # convert to original domain
        if self.data_form == 'a':
            self.z = DataConversion._convert_za(self.zf, self.lb, self.ub)
        elif self.data_form == 'm':
            self.z = DataConversion._convert_zm(self.zf, self.lb, self.ub)
        elif self.data_form is None:
            self.z = self.zf

        return self.z

    def _scale_param(self, F):
        """
        Calculate the scale parameter S based on the data and specified parameters.

        If varS is True, it will vary S based on the data; otherwise, it will use the provided S.
        If S is None, it will calculate S based on the data.
        Returns:
        - S (float) or (array): The scale parameter.
        """
        if self.varS and self.S is None:
            S = ScaleParam._gscale_loc(F)
        else:
            S = ScaleParam._gscale_loc(np.mean(F))
        if S:
            S = S

        return S

    def _criterion_function(self, params, Z=None):
        pass

    def _edf(self):
        pass

    def fit(self):
        pass

    def gnostic_kernel(self):
        pass

    def gnostic_density(self):
        pass
    
    def eldf(self):
        pass

    def eldf_density(self):
        pass

    def egdf(self):
        pass

    def egdf_density(self):
        pass

    def qldf(self):
        pass

    def qldf_density(self):
        pass

    def qgdf(self):
        pass

    def qgdf_density(self):
        pass
