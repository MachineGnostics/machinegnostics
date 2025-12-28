'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

Author: Nirmal Parmar

ideas:
- LocS
- GlobS
- VarS
'''
import numpy as np
from scipy.optimize import minimize
import logging
from machinegnostics.magcal.util.logging import get_logger

class ScaleParam():
    """
    A Machine Gnostic class to compute and optimize scale parameter for different gnostic distribution functions.
    
    This class provides methods to calculate scale parameters used in gnostic analysis, including
    local scale parameters and variable scale parameters for kernel-based estimations.
    
    The scale parameter affects the shape and characteristics of gnostic distributions, controlling
    how the distributions respond to variations in the input data.
    
    Notes
    -----
    The scale parameter is a critical component in Machine Gnostics that influences the behavior
    of distribution functions, particularly their sensitivity to outliers and their overall shape.
    
    The class implements multiple scale parameter calculation strategies:
    - Local scale: Optimizes scale for individual data points
    - Variable scale: Creates a vector of scale parameters for kernel-based estimation
    """
    

    def __init__(self, verbose: bool = False):
        self.logger = get_logger('ScaleParam', level=logging.WARNING if not verbose else logging.INFO)
        self.logger.info("ScaleParam initialized.")

    def _gscale_loc(self, F):
        """
        Calculate the local scale parameter for a given fidelity parameter F.
        
        This method uses the Newton-Raphson method to solve for the scale parameter that satisfies
        the relationship between F and the scale. It supports both scalar and array-like inputs.
        
        Parameters
        ----------
        F : float or array-like
            Input parameter (e.g., fidelity of data) at Scale = 1.
            
        Returns
        -------
        float or ndarray
            The calculated local scale parameter(s). Will be the same shape as input F.
            
        Notes
        -----
        The Newton-Raphson method is used with initial values based on the magnitude of F:
        - For F < (2/π) * √2/3: Initial S = π
        - For F < 2/π: Initial S = 3π/4
        - For F < (2/π) * √2: Initial S = π/2
        - Otherwise: Initial S = π/4
        
        The method iteratively refines this estimate until convergence.
        """
        self.logger.info("Calculating local scale parameter...")
        m2pi = 2 / np.pi
        sqrt2 = np.sqrt(2)
        epsilon = 1e-5

        def _single_scale(f):
            if f < m2pi * sqrt2 / 3:
                S = np.pi
            elif f < m2pi:
                S = 3 * np.pi / 4
            elif f < m2pi * sqrt2:
                S = np.pi / 2
            else:
                S = np.pi / 4
            for _ in range(100):
                delta = (np.sin(S) - S * f) / (np.cos(S) - f)
                S -= delta
                if abs(delta) < epsilon:
                    break
            return S * m2pi
        self.logger.info("Local scale parameter calculation complete.")

        # Check if F is scalar
        if np.isscalar(F):
            return _single_scale(F)
        else:
            F = np.asarray(F)
            return np.array([_single_scale(f) for f in F])


    def estimate_global_scale_egdf(self, Fk, Ek, tolerance=0.1):
        """
        Estimate the optimal global scale parameter S_optimize to find minimum S where fidelity is maximized.
        
        Parameters
        ----------
        Fk : array-like
            Fidelity values for the data points.
        Ek : array-like
            Weighted empirical distribution function values for the data points.
        tolerance : float, optional
            Convergence tolerance for fidelity change (default is 0.01).
    
        Returns
        -------
        float
            The optimal global scale parameter S_optimize (minimum S where fidelity is maximized).
    
        Notes
        -----
        This function finds the minimum scale parameter S where fidelity is maximized,
        with early stopping when fidelity change is less than the specified tolerance.
        """
        self.logger.info("Estimating global scale parameter...")
        Fk = np.asarray(Fk)
        Ek = np.asarray(Ek)
    
        if len(Fk) != len(Ek):
            raise ValueError("Fk and Ek must have the same length.")
    
        def compute_fidelity(S):
            """Compute average fidelity for a given S"""
            # Add small epsilon to prevent division by zero
            eps = np.finfo(float).eps
            term1 = (Fk / (Ek + eps)) ** (2 / S)
            term2 = (Ek / (Fk + eps)) ** (2 / S)
            fidelities = 2 / (term1 + term2)
            return np.mean(fidelities)
    
        # Search through S values from minimum to maximum
        s_values = np.linspace(0.05, 100, 1000)  # Fine grid for accurate search
        
        max_fidelity = -np.inf
        optimal_s = None
        previous_fidelity = None
        
        for s in s_values:
            current_fidelity = compute_fidelity(s)
            
            # Check convergence condition first
            if previous_fidelity is not None:
                fidelity_change = abs(current_fidelity - previous_fidelity)
                if fidelity_change < tolerance:
                    # Converged - return the minimum S where we achieved max fidelity
                    if optimal_s is not None:
                        final_fidelity = compute_fidelity(optimal_s)
                        print(f"Converged at S={optimal_s:.4f} with fidelity={final_fidelity:.4f}")
                        return optimal_s
                    else:
                        # First iteration, use current S
                        print(f"Converged at S={s:.4f} with fidelity={current_fidelity:.4f}")
                        return s
            
            # Update maximum fidelity and optimal S (prefer minimum S for same fidelity)
            if current_fidelity > max_fidelity:
                max_fidelity = current_fidelity
                optimal_s = s
            
            previous_fidelity = current_fidelity
        self.logger.info("Global scale parameter estimation complete.")
        # If no convergence found, return the S with maximum fidelity
        if optimal_s is not None:
            final_fidelity = compute_fidelity(optimal_s)
            self.logger.warning(f"No convergence found. Returning S={optimal_s:.4f} with max fidelity={final_fidelity:.4f}")
            return optimal_s
        else:
            self.logger.error("Failed to find optimal scale parameter.")
            raise RuntimeError("Failed to find optimal scale parameter.")
        
    def estimate_varS(self, data, zi, gdf, tdf, weights, S_global: float) -> np.ndarray:
        """
        Estimate variable scale parameters for kernel-based estimation.
    
        Parameters
        ----------
        data : array-like
            Input data points.
        weights : array-like, optional
            Weights corresponding to data points (default is None, equal weights).
        S_global : float
            Global scale parameter to base variable scales on.
        gdf : Gnostic Distribution Function [ELDF/QGDF]
            The gnostic distribution function to use EGDF or QGDF calculations.
        tdf : Target Distribution Function
            The target distribution function to compare against.
    
        Returns
        -------
        varS : ndarray
            Estimated variable scale vector for each zi.
        """
        self.logger.info("Estimating variable scale parameters...")

        from machinegnostics.magcal import EGDF, QGDF
    
        data = np.asarray(data)

        if weights is None:
            weights = np.ones_like(data)
        else:
            weights = np.asarray(weights)

    
        def fidelity_loss(params):
            S0, gamma = params
            if S0 <= 0:
                return 1e8  # Penalize invalid S0
            S_vec = S0 * np.exp(gamma * zi)
            if gdf == 'EGDF':
                egdf = EGDF(verbose=False, S=S_vec)
                egdf.fit(data)
                egdf_vals = egdf.egdf
                wedf_values = tdf
            else:  # gdf is QGDF
                qgdf = QGDF(verbose=False, S=S_vec)
                qgdf.fit(data)
                qgdf_vals = qgdf.qgdf
                wedf_values = tdf
            
            # Maximum Fidelity criterion (sum of fidelities)
            f_E = 2 / ((egdf_vals / (wedf_values + 1e-12))**2 + (wedf_values / (egdf_vals + 1e-12))**2)
            loss = -np.sum(f_E * weights)
            return loss
    
        x0 = [S_global, 0.0]
        bounds = [(1e-6, 1e3), (-2, 2)]
    
        result = minimize(
            fidelity_loss,
            x0,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
    
        S0_opt, gamma_opt = result.x
        varS = S0_opt * np.exp(gamma_opt * zi)
        
        # Store optimal parameters
        self.S0 = S0_opt
        self.gamma = gamma_opt
        self.varS = varS

        self.logger.info(f"Variable scale parameter estimation complete.")
        return self.varS



# # ...existing code...
#         self.logger.info("Estimating variable scale parameters...")

#         from machinegnostics.magcal import EGDF, QGDF
    
#         data = np.asarray(data)
#         zi = np.asarray(zi)

#         if weights is None:
#             weights = np.ones_like(data)
#         else:
#             weights = np.asarray(weights)

#         # Derive adaptive bounds from zi span and desired variation K
#         zi_min, zi_max = float(np.min(zi)), float(np.max(zi))
#         zi_span = max(1e-9, zi_max - zi_min)

#         K = 5.0  # desired max multiplicative change in S across the zi span
#         gamma_bound = np.log(K) / zi_span

#         S_min = max(1e-6, S_global / 10.0)
#         S_max = S_global * 10.0

#         def fidelity_loss(params):
#             S0, gamma = params
#             if S0 <= 0:
#                 return 1e8  # Penalize invalid S0
#             # Keep S within safe range to avoid numerical issues
#             S_vec = np.clip(S0 * np.exp(gamma * zi), S_min, S_max)

#             if gdf == 'EGDF':
#                 egdf = EGDF(verbose=False, S=S_vec)
#                 egdf.fit(data)
#                 egdf_vals = egdf.egdf
#                 wedf_values = tdf
#             else:  # gdf is QGDF
#                 qgdf = QGDF(verbose=False, S=S_vec)
#                 qgdf.fit(data)
#                 qgdf_vals = qgdf.qgdf
#                 wedf_values = tdf
            
#             eps = 1e-12
#             f_E = 2.0 / ((egdf_vals / (wedf_values + eps))**2 + (wedf_values / (egdf_vals + eps))**2)
#             loss = -np.sum(f_E * weights)
#             return loss
    
#         x0 = [S_global, 0.0]
#         bounds = [(S_min, S_max), (-gamma_bound, gamma_bound)]
    
#         result = minimize(
#             fidelity_loss,
#             x0,
#             bounds=bounds,
#             method='L-BFGS-B',
#             options={'maxiter': 1000, 'ftol': 1e-3}
#         )
# # ...existing code...