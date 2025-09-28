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
from machinegnostics.magcal import GnosticsCharacteristics
from scipy.optimize import minimize_scalar
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


    # def var_s(self, Z, W=None, S=1):
    #     """
    #     Calculates vector of scale parameters for each kernel.
        
    #     Parameters:
    #     Z (array-like): Data vector
    #     W (array-like, optional): Weight vector
    #     S (float, optional): Scalar scale factor (default is 1)
        
    #     Returns:
    #     numpy.ndarray: Scale vector (same length as Z)
    #     """
    #     Z = np.asarray(Z).reshape(-1, 1)

    #     if W is None:
    #         W = np.ones_like(Z) / len(Z)
    #     else:
    #         W = np.asarray(W).reshape(-1, 1)
    #         if len(Z) != len(W):
    #             raise ValueError("Z and W must be of the same length")
    #         W = W / np.sum(W)

    #     Sz = np.zeros_like(Z, dtype=float)

    #     for k in range(len(W)):
    #         V = Z / Z[k]
    #         V = V ** (2/S) + 1.0 / (V ** (2/S))
    #         Sz[k] = self._gscale_loc(np.sum(2.0 / V * W))

    #     Sx = S * Sz / np.mean(Sz)
    #     return Sx

    def var_s(self, Z, W=None, S=1):
        """
        Calculate a vector of scale parameters for each kernel in the distribution.
        
        This method computes individualized scale parameters for each data point, allowing
        for adaptive scaling in gnostic estimations. It handles numerical edge cases to
        ensure stability.
        
        Parameters
        ----------
        Z : array-like
            Data vector containing the values for which to calculate scale parameters.
        W : array-like, optional
            Weight vector for each data point. If not provided, uniform weights are used.
        S : float, optional
            Base scalar scale factor, default is 1.
            
        Returns
        -------
        ndarray
            Vector of scale parameters, one for each element in Z.
            
        Raises
        ------
        ValueError
            If Z and W are provided but have different lengths.
            
        Notes
        -----
        The method calculates relative relationships between data points and applies
        the local scale parameter calculation for each point. For each data point k,
        it computes a ratio V of all data points relative to Z[k], then calculates
        a transformation of this ratio to determine the local scale parameter.
        
        The implementation includes safeguards against division by zero and handles
        edge cases to ensure numerical stability. In case of invalid calculations,
        it falls back to the default scale parameter.
        """
        self.logger.info("Calculating local scale parameters...")
        Z = np.asarray(Z).reshape(-1, 1)
    
        if W is None:
            W = np.ones_like(Z) / len(Z)
        else:
            W = np.asarray(W).reshape(-1, 1)
            if len(Z) != len(W):
                raise ValueError("Z and W must be of the same length")
            W = W / np.sum(W)
    
        Sz = np.zeros_like(Z, dtype=float)
        
        # Small value to prevent division by zero
        eps = np.finfo(float).eps * 100
    
        for k in range(len(W)):
            # Skip calculation if Z[k] is too close to zero
            if abs(Z[k]) < eps:
                Sz[k] = S  # Use default S value
                continue
                
            # Safe division with epsilon to prevent division by zero
            V = Z / (Z[k] + (Z[k]==0)*eps)
            V = V ** 2 + 1.0 / (V ** 2 + eps)
            
            # Calculate sum and ensure it's valid
            sum_val = np.sum(2.0 / V * W)
            if np.isnan(sum_val) or np.isinf(sum_val):
                Sz[k] = S  # Use default S value
            else:
                Sz[k] = self._gscale_loc(sum_val)
        
        # Check for any remaining NaN values and replace them
        Sz[np.isnan(Sz)] = S
        self.logger.info("Local scale parameters calculation complete.")
        return Sz

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
        
    # def _gscale_loc(self, F):
    #     '''
    #     For internal use only

    #     calculates the local scale parameter for given calculated F at Scale = 1.
    #     S with be in the same shape as F.
    #     Solve for scale parameter using Newton-Raphson."
    #     '''
    #     m2pi = 2 / np.pi
    #     sqrt2 = np.sqrt(2)
        
    #     if F < m2pi * sqrt2 / 3:
    #         S = np.pi
    #     elif F < m2pi:
    #         S = 3 * np.pi / 4
    #     elif F < m2pi * sqrt2:
    #         S = np.pi / 2
    #     else:
    #         S = np.pi / 4

    #     epsilon = 1e-5
    #     for _ in range(100):
    #         delta = (np.sin(S) - S * F) / (np.cos(S) - F)
    #         S -= delta
    #         if abs(delta) < epsilon:
    #             break
    #     return S * m2pi