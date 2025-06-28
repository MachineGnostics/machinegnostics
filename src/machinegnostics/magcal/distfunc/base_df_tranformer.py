"""
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

Author: Nirmal Parmar
License: GNU General Public License v3.0 (GPL-3.0)

# Base Distribution Function Transformer Module

Objective: Base class for transformations of the EGDF (Estimating Global Distribution Function).
This class can be used with other distribution functions to transform data into a standard domain.

"""
import numpy as np
from machinegnostics.magcal.distfunc.base_egdf import BaseEGDF
from machinegnostics.magcal.data_conversion import DataConversion

class BaseDistFuncTransformer(BaseEGDF):
    """
    Base class for transformations of the EGDF.
    """

    def __init__(self,
                 data,
                 DLB: float,
                 DUB: float,
                 LB: float,
                 UB: float,
                 S=1.0,
                 tolerance: float = 1e-5,
                 data_form: str = 'a',
                 n_points: int = 500,
                 homogeneous: bool = True,
                 catch: bool = True):
        super().__init__(data, S, tolerance, DLB, DUB, LB, UB, data_form, n_points, homogeneous, catch)
        
    def _transform_input(self):
        """
        Apply the data domain transformation as per MG distribution function (MGDF) requirements.
        """
        # Validate the data form
        if self.data_form not in ['a', 'm', None]:
            raise ValueError(f"Invalid data form: {self.data_form}. Must be 'a', 'm', or None.")

        # Perform the data domain transformation
        # from normal to standard domain
        if self.data_form == 'a':
            self.z = DataConversion._convert_az(self.data, lb=self.DLB, ub=self.DUB)
        elif self.data_form == 'm':
            self.z = DataConversion._convert_mz(self.data, lb=self.DLB, ub=self.DUB)
        elif self.data_form == None:
            self.z = self.data
        else:
            raise ValueError(f"Invalid data form: {self.data_form}. Must be 'a', 'm', or None.")
        # from finite to infinite domain
        if self.LB is None:
            self.LB = np.min(self.z)
        if self.UB is None:
            self.UB = np.max(self.z)
        # initial bounds
        self.LB, self.UB = self._initial_bounds()
        self.S_init = 1
        self.zi = DataConversion._convert_fininf(self.z, self.LB, self.UB)

        # store values
        # saving the initial bounds
        if self.catch:
            self.param['LB'] = self.LB
            self.param['UB'] = self.UB
            self.param['DLB'] = self.DLB
            self.param['DUB'] = self.DUB
            self.param['S'] = self.S
            self.param['zi'] = self.zi
            self.param['z'] = self.z
            self.param['data'] = self.data
            self.param['data_form'] = self.data_form
            self.param['homogeneous'] = self.homogeneous
        else:
            self.param = None

        return self
    
    def _transform_output(self):
        """
        Transform the output from the standard domain back to the original data domain.
    
        """
        # from infinite to finite domain
        self.z = DataConversion._convert_inffin(self.zi, self.LB, self.UB)
        # from standard to normal domain
        if self.data_form == 'a':
            return DataConversion._convert_za(self.z, lb=self.DLB, ub=self.DUB)
        elif self.data_form == 'm':
            return DataConversion._convert_zm(self.z, lb=self.DLB, ub=self.DUB)
        elif self.data_form is None:
            return self.z
        else:
            raise ValueError(f"Invalid data form: {self.data_form}. Must be 'a', 'm', or None.")

    def _initial_bounds(self):
        """
        Get the initial bounds for the transformation.
        
        Returns
        -------
        tuple
            Initial bounds (LB, UB) for the transformation.
        """
        # Data preprocessing
        self.sorted_data = np.sort(self.z)
        self.DLB = self.sorted_data.min()
        self.DUB = self.sorted_data.max()

        # Set default S, LB, UB if not provided
        if self.S is None:
            self.S = 1
        if self.data_form == 'a':
            if self.LB is None:
                self.LB = self.DLB - (self.DUB - self.DLB) / 2
            if self.UB is None:
                self.UB = self.DUB + (self.DUB - self.DLB) / 2
        elif self.data_form == 'm':
            if np.any(self.sorted_data <= 0):
                raise ValueError("Multiplicative data must be strictly positive")
            if self.LB is None:
                self.LB = self.DLB / np.sqrt(self.DUB / self.DLB)
            if self.UB is None:
                self.UB = self.DUB * np.sqrt(self.DUB / self.DLB)
        elif self.data_form is None:
            if self.LB is None:
                self.LB = np.min(self.sorted_data)
            if self.UB is None:
                self.UB = np.max(self.sorted_data)
        return self.LB, self.UB

    def fit(self):
        """
        Fit the distribution function transformer to the data.
        This method should be implemented in subclasses.
        """
        # Transform the input data
        self._transform_input()
        # Fit the distribution function
        super().fit()
