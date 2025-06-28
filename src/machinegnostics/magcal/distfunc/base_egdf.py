"""
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

Author: Nirmal Parmar
License: GNU General Public License v3.0 (GPL-3.0)

# Base Gnostic Distribution Function Module

EGDF - Estimating Global Distribution Function
"""

import numpy as np
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.distfunc.base_df import BaseDistFunc

class BaseEGDF(BaseDistFunc):
    """
    Estimating Global Distribution Function (EGDF) base class.
    """

    def __init__(self,
                 data,
                 DLD: float,
                 DUD: float,
                 LB: float,
                 UB: float,
                 S = 1,
                 tolerance: float = 1e-5,
                 data_form: str = 'a',
                 n_points: int = 500,
                 homogeneous: bool = True,
                 catch: bool = True
                 ):
        
        self.data = data
        self.S = S
        self.tolerance = tolerance
        self.DLD = DLD
        self.DUD = DUD
        self.LB = LB
        self.UB = UB
        self.data_form = data_form
        self.n_points = n_points
        self.homogeneous = homogeneous
        self.catch = catch
        # to store parameters
        self.param = {}

        # argument validation in main class

        # transform input data in transformer class

    def fit(self):
        """
        Fit the EGDF to the data.
        """
        pass

    def _pdf(self):
        """
        Probability Density Function (PDF) of the EGDF.
        """
        pass

    def _gdf(self):
        """
        Gnostic (cumulative) Distribution Function (GDF) of the EGDF.
        """
        pass

    def _plot(self):
        """
        Plot the EGDF.
        """
        pass


