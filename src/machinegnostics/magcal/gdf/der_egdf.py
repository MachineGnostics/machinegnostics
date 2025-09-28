'''
EGDF Derivative Module

Machine Gnostics
Author: Nirmal Parmar
'''

import numpy as np
import warnings
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.gdf.egdf import EGDF

class DerivativesEGDF:
    """
    Class for computing derivatives of the EGDF.

    For internal use only, this class provides methods to calculate
    """

    def __init__(self, egdf: EGDF):
        self.fi = egdf.fi
        self.hi = egdf.hi
        self.weights = egdf.weights
        self.S_opt = egdf.S_opt
        self.z = egdf.z
        self.di_points_n = egdf.di_points_n
        self.catch = egdf.catch
        self.LB_opt = egdf.LB_opt
        self.UB_opt = egdf.UB_opt

    def _get_egdf_first_derivative(self):
        """Calculate first derivative of EGDF (which is the PDF) from stored fidelities and irrelevances."""
        if self.fi is None or self.hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before first derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # First order moments
        f1 = np.sum(weights * self.fi, axis=0) / np.sum(weights)  # mean_fidelity
        h1 = np.sum(weights * self.hi, axis=0) / np.sum(weights)  # mean_irrelevance
        
        # Second order moments (scaled by S as in MATLAB)
        f2s = np.sum(weights * (self.fi**2 / self.S_opt), axis=0) / np.sum(weights)
        fhs = np.sum(weights * (self.fi * self.hi / self.S_opt), axis=0) / np.sum(weights)
        
        # Calculate denominator w = (f1^2 + h1^2)^(3/2)
        w = (f1**2 + h1**2)**(3/2)
        eps = np.finfo(float).eps
        w = np.where(w == 0, eps, w)
        
        # First derivative formula from MATLAB: y = (f1^2 * f2s + f1 * h1 * fhs) / w
        numerator = f1**2 * f2s + f1 * h1 * fhs
        first_derivative = numerator / w
        # first_derivative = first_derivative / self.zi
        
        if np.any(first_derivative < 0):
            warnings.warn("EGDF first derivative (PDF) contains negative values", RuntimeWarning)
        
        return first_derivative.flatten()

    def _get_egdf_second_derivative(self):
        """Calculate second derivative of EGDF from stored fidelities and irrelevances."""
        if self.fi is None or self.hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before second derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # Moment calculations
        f1 = np.sum(weights * self.fi, axis=0) / np.sum(weights)
        h1 = np.sum(weights * self.hi, axis=0) / np.sum(weights)
        f2 = np.sum(weights * self.fi**2, axis=0) / np.sum(weights)
        f3 = np.sum(weights * self.fi**3, axis=0) / np.sum(weights)
        fh = np.sum(weights * self.fi * self.hi, axis=0) / np.sum(weights)
        fh2 = np.sum(weights * self.fi * self.hi**2, axis=0) / np.sum(weights)
        f2h = np.sum(weights * self.fi**2 * self.hi, axis=0) / np.sum(weights)
        
        # Calculate components
        b = f1**2 * f2 + f1 * h1 * fh
        d = f1**2 + h1**2
        eps = np.finfo(float).eps
        d = np.where(d == 0, eps, d)
        
        # Following
        term1 = f1 * (h1 * (f3 - fh2) - f2 * fh)
        term2 = 2 * f1**2 * f2h + h1 * fh**2
        term3 = (6 * b * (f1 * fh - h1 * f2)) / d
        
        d2 = -1 / (d**(1.5)) * (2 * (term1 - term2) + term3)
        second_derivative = d2 / (self.S_opt**2)
        # second_derivative = second_derivative / self.zi**2 
        return second_derivative.flatten()

    def _get_egdf_third_derivative(self):
        """Calculate third derivative of EGDF from stored fidelities and irrelevances."""
        if self.fi is None or self.hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before third derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # All required moments
        f1 = np.sum(weights * self.fi, axis=0) / np.sum(weights)
        h1 = np.sum(weights * self.hi, axis=0) / np.sum(weights)
        f2 = np.sum(weights * self.fi**2, axis=0) / np.sum(weights)
        f3 = np.sum(weights * self.fi**3, axis=0) / np.sum(weights)
        f4 = np.sum(weights * self.fi**4, axis=0) / np.sum(weights)
        fh = np.sum(weights * self.fi * self.hi, axis=0) / np.sum(weights)
        h2 = np.sum(weights * self.hi**2, axis=0) / np.sum(weights)
        fh2 = np.sum(weights * self.fi * self.hi**2, axis=0) / np.sum(weights)
        f2h = np.sum(weights * self.fi**2 * self.hi, axis=0) / np.sum(weights)
        f2h2 = np.sum(weights * self.fi**2 * self.hi**2, axis=0) / np.sum(weights)
        f3h = np.sum(weights * self.fi**3 * self.hi, axis=0) / np.sum(weights)
        fh3 = np.sum(weights * self.fi * self.hi**3, axis=0) / np.sum(weights)
        
        # Following
        # Derivative calculations
        dh1 = -f2
        df1 = fh
        df2 = 2 * f2h
        dfh = -f3 + fh2
        dfh2 = -2 * f3h + fh3
        df3 = 3 * f3h
        df2h = -f4 + 2 * f2h2
        
        # u4 and its derivative
        u4 = h1 * f3 - h1 * fh2 - f2 * fh
        du4 = dh1 * f3 + h1 * df3 - dh1 * fh2 - h1 * dfh2 - df2 * fh - f2 * dfh
        
        # u and its derivative
        u = f1 * u4
        du = df1 * u4 + f1 * du4
        
        # v components
        v4a = (f1**2) * f2h
        dv4a = 2 * f1 * df1 * f2h + (f1**2) * df2h
        v4b = h1 * fh**2
        dv4b = dh1 * (fh**2) + 2 * h1 * fh * dfh
        
        v = 2 * v4a + v4b
        dv = 2 * dv4a + dv4b
        
        # x components
        x4a = f1**2 * f2 + f1 * h1 * fh
        dx4a = 2 * f1 * df1 * f2 + (f1**2) * df2 + df1 * h1 * fh + f1 * dh1 * fh + f1 * h1 * dfh
        x4b = f1 * fh - h1 * f2
        dx4b = df1 * fh + f1 * dfh - dh1 * f2 - h1 * df2
        
        x = 6 * x4a * x4b
        dx = 6 * (dx4a * x4b + x4a * dx4b)
        
        # d components
        d = f1**2 + h1**2
        dd = 2 * (f1 * df1 + h1 * dh1)
        eps = np.finfo(float).eps
        d = np.where(d == 0, eps, d)
        
        # Final calculation
        term1 = (du - dv) / (d**1.5) - (1.5 * (u - v)) / (d**2.5) * dd
        term2 = dx / (d**2.5) - (2.5 * x) / (d**3.5) * dd
        
        d3p = -2 * term1 - term2
        third_derivative = 2 * d3p / (self.S_opt**3)
        # third_derivative = third_derivative / (self.zi**3)
        return third_derivative.flatten()

    def _get_egdf_fourth_derivative(self):
        """Calculate fourth derivative of EGDF using numerical differentiation."""
        if self.fi is None or self.hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before fourth derivative estimation.")
        
        # For fourth derivative, use numerical differentiation as it's complex
        dz = 1e-7
        
        # Get third derivatives at slightly shifted points
        zi_plus = self.zi + dz
        zi_minus = self.zi - dz
        
        # Store original zi
        original_zi = self.zi.copy()
        
        # Calculate third derivative at zi + dz
        self.zi = zi_plus
        self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        third_plus = self._get_egdf_third_derivative()
        
        # Calculate third derivative at zi - dz  
        self.zi = zi_minus
        self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        third_minus = self._get_egdf_third_derivative()
        
        # Restore original zi and recalculate fi, hi
        self.zi = original_zi
        self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        
        # Numerical derivative
        fourth_derivative = (third_plus - third_minus) / (2 * dz) * self.zi
        
        return fourth_derivative.flatten()

    def _calculate_fidelities_irrelevances_at_given_zi(self, zi):
        """Helper method to recalculate fidelities and irrelevances for current zi."""
        # Convert to infinite domain
        zi_n = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        # is zi given then use it, else use self.zi
        if zi is None:
            zi_d = self.zi
        else:
            zi_d = zi

        # Calculate R matrix
        eps = np.finfo(float).eps
        R = zi_n.reshape(-1, 1) / (zi_d + eps).reshape(1, -1)

        # Get characteristics
        gc = GnosticsCharacteristics(R=R)
        q, q1 = gc._get_q_q1(S=self.S_opt)
        
        # Store fidelities and irrelevances
        self.fi = gc._fi(q=q, q1=q1)
        self.hi = gc._hi(q=q, q1=q1)


    def _get_pick_counts(self):
        pass

    def _get_derivatives(self):
        """Calculate all derivatives of EGDF."""
        if self.fi is None or self.hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before derivative estimation.")
        # Calculate fidelities and irrelevances at current zi     
        # Calculate all derivatives
        self.first_derivative = self._get_egdf_first_derivative()
        self.second_derivative = self._get_egdf_second_derivative()
        self.third_derivative = self._get_egdf_third_derivative()
        self.fourth_derivative = np.gradient(self.third_derivative, self.di_points_n)  # Numerical gradient for fourth derivative    

        if self.catch:
            self.params.update({
                'first_derivative': self.first_derivative,
                'second_derivative': self.second_derivative,
                'third_derivative': self.third_derivative,
                'fourth_derivative': self.fourth_derivative
            })