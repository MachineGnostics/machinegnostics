'''
QGDF: Quantifying Global Distribution Functions

Author: Nirmal Parmar
Machine Gnostics
'''
import numpy as np
import warnings
import logging
from machinegnostics.magcal.util.logging import get_logger
from typing import Dict, Any
from machinegnostics.magcal.gdf.base_distfunc import BaseDistFuncCompute
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.gdf.z0_estimator import Z0Estimator

class BaseQGDF(BaseDistFuncCompute):
    """
    Base class for Quantifying Global Distribution Functions (QGDF).
    
    This class provides foundational methods and attributes for computing
    and analyzing global distribution functions using various techniques.
    
    Attributes:
        data (np.ndarray): Input data for distribution function computation.
        n_points (int): Number of points for evaluation.
        S (float): Smoothing parameter.
        catch (bool): Flag to enable error catching.
        verbose (bool): Flag to enable verbose output.
        params (dict): Dictionary to store parameters and results.
    """
    
    def __init__(self,
                 data: np.ndarray,
                 DLB: float = None,
                 DUB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S = 'auto',
                 z0_optimize: bool = True,
                 tolerance: float = 1e-3,
                 data_form: str = 'a',
                 n_points: int = 500,
                 homogeneous: bool = True,
                 catch: bool = True,
                 weights: np.ndarray = None,
                 wedf: bool = True,
                 opt_method: str = 'L-BFGS-B',
                 verbose: bool = False,
                 max_data_size: int = 1000,
                 flush: bool = True):
        super().__init__(data=data, 
                         DLB=DLB, 
                         DUB=DUB, 
                         LB=LB, 
                         UB=UB, 
                         S=S, 
                         z0_optimize=z0_optimize,
                         varS=False, # NOTE for QGDFF varS is always False 
                         tolerance=tolerance, 
                         data_form=data_form, 
                         n_points=n_points, 
                         homogeneous=homogeneous, 
                         catch=catch, 
                         weights=weights, 
                         wedf=wedf, 
                         opt_method=opt_method, 
                         verbose=verbose, 
                         max_data_size=max_data_size, 
                         flush=flush)

        # Store raw inputs
        self.data = data
        self.DLB = DLB
        self.DUB = DUB
        self.LB = LB
        self.UB = UB
        self.S = S
        self.z0_optimize = z0_optimize

        self.tolerance = tolerance
        self.data_form = data_form
        self.n_points = n_points
        self.homogeneous = homogeneous
        self.catch = catch
        self.weights = weights if weights is not None else np.ones_like(data)
        self.wedf = wedf
        self.opt_method = opt_method
        self.verbose = verbose
        self.max_data_size = max_data_size
        self.flush = flush
        
        # Initialize state variables
        self.params = {}
        self._fitted = False
        self._derivatives_calculated = False
        self._marginal_analysis_done = False
        
        # Initialize computation cache
        self._computation_cache = {
            'data_converter': None,
            'characteristics_computer': None,
            'weights_normalized': None,
            'smooth_curves_generated': False
        }
        
        # Store initial parameters if catching
        if self.catch:
            self._store_initial_params()

        # Validate all inputs
        self._validate_inputs()

        # logger setup
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")

    def _compute_qgdf_core(self, S, LB, UB, zi_data=None, zi_eval=None):
        """Core QGDF computation with caching."""
        self.logger.info("Computing QGDF core.")
        # Use provided data or default to instance data
        if zi_data is None:
            zi_data = self.z
        if zi_eval is None:
            zi_eval = zi_data
        
        # Convert to infinite domain
        zi_n = DataConversion._convert_fininf(zi_eval, LB, UB)
        zi_d = DataConversion._convert_fininf(zi_data, LB, UB)
        
        # Calculate R matrix with numerical stability
        R = zi_n.reshape(-1, 1) / (zi_d.reshape(1, -1) + self._NUMERICAL_EPS)
        
        # Get characteristics
        gc = GnosticsCharacteristics(R=R, verbose=self.verbose)
        q, q1 = gc._get_q_q1(S=S)
        
        # Calculate fidelities and irrelevances
        fj = gc._fj(q=q, q1=q1)
        hj = gc._hj(q=q, q1=q1)
        
        # Estimate QGDF
        return self._estimate_qgdf_from_moments(fj, hj), fj, hj

    def _estimate_qgdf_from_moments_complex(self, fidelities, irrelevances):
        """Estimate QGDF using complex number approach to handle all cases."""
        self.logger.info("Estimating QGDF using complex number approach.")

        weights = self._computation_cache['weights_normalized'].reshape(-1, 1)
        
        # Add numerical stability for both large and small values
        max_safe_value = np.sqrt(np.finfo(float).max) / 100  # More conservative
        min_safe_value = np.sqrt(np.finfo(float).eps) * 100  # Avoid very small numbers
        
        # Comprehensive clipping for extreme values (both large and small)
        def safe_clip_values(values, name="values"):
            """Safely clip values to prevent both overflow and underflow issues."""
            # Handle very small values (close to zero)
            values_magnitude = np.abs(values)
            too_small_mask = values_magnitude < min_safe_value
            
            # Handle very large values
            too_large_mask = values_magnitude > max_safe_value
            
            if np.any(too_small_mask) and self.verbose:
                small_count = np.sum(too_small_mask)
                self.logger.info(f"Warning: {small_count} very small {name} values detected (< {min_safe_value:.2e})")
            
            if np.any(too_large_mask) and self.verbose:
                large_count = np.sum(too_large_mask)
                self.logger.info(f"Warning: {large_count} very large {name} values detected (> {max_safe_value:.2e})")
            
            # Clip small values to minimum safe value (preserving sign)
            values_safe = np.where(too_small_mask, 
                                  np.sign(values) * min_safe_value, 
                                  values)
            
            # Clip large values to maximum safe value (preserving sign)
            values_safe = np.where(too_large_mask, 
                                  np.sign(values_safe) * max_safe_value, 
                                  values_safe)
            
            return values_safe
        
        # Apply safe clipping to both fidelities and irrelevances
        fidelities_safe = safe_clip_values(fidelities, "fidelity")
        irrelevances_safe = safe_clip_values(irrelevances, "irrelevance")
        
        # Calculate weighted means (f̄Q and h̄Q from equation 15.35)
        mean_fidelity = np.sum(weights * fidelities_safe, axis=0) / np.sum(weights)  # f̄Q
        mean_irrelevance = np.sum(weights * irrelevances_safe, axis=0) / np.sum(weights)  # h̄Q
        
        # Apply safe clipping to means as well
        mean_fidelity = safe_clip_values(mean_fidelity, "mean_fidelity")
        mean_irrelevance = safe_clip_values(mean_irrelevance, "mean_irrelevance")
        
        # Convert to complex for robust calculation with overflow protection
        f_complex = mean_fidelity.astype(complex)
        h_complex = mean_irrelevance.astype(complex)
        
        # Calculate the complex square root with comprehensive protection
        # Check magnitudes before squaring
        f_magnitude = np.abs(f_complex)
        h_magnitude = np.abs(h_complex)
        sqrt_max = np.sqrt(max_safe_value)
        sqrt_min = np.sqrt(min_safe_value)
        
        # Check for both very large and very small values before squaring
        f_too_large = f_magnitude > sqrt_max
        h_too_large = h_magnitude > sqrt_max
        f_too_small = f_magnitude < sqrt_min
        h_too_small = h_magnitude < sqrt_min
        
        if np.any(f_too_large) or np.any(h_too_large) or np.any(f_too_small) or np.any(h_too_small):
            if self.verbose:
                self.logger.info("Warning: Extreme values detected in complex calculation. Using scaled approach.")
            
            # Scale problematic values to safe range
            f_scaled = np.where(f_too_large, sqrt_max * (f_complex / f_magnitude), f_complex)
            f_scaled = np.where(f_too_small, sqrt_min * (f_complex / f_magnitude), f_scaled)
            
            h_scaled = np.where(h_too_large, sqrt_max * (h_complex / h_magnitude), h_complex)
            h_scaled = np.where(h_too_small, sqrt_min * (h_complex / h_magnitude), h_scaled)
            
            diff_squared_complex = f_scaled**2 - h_scaled**2
            scale_factor = 1.0
        else:
            diff_squared_complex = f_complex**2 - h_complex**2
            scale_factor = 1.0
        
        # Calculate denominator with protection against both zero and very small values
        denominator_magnitude = np.abs(diff_squared_complex)
        denominator_too_small = denominator_magnitude < min_safe_value
        
        if np.any(denominator_too_small):
            if self.verbose:
                small_denom_count = np.sum(denominator_too_small)
                self.logger.info(f"Warning: {small_denom_count} very small denominators in complex calculation.")
        
        # Use sqrt with protection
        denominator_complex = np.sqrt(diff_squared_complex)
        denominator_complex = np.where(denominator_magnitude < min_safe_value,
                                      min_safe_value + 0j, denominator_complex)
        
        # Calculate hZ,j using complex arithmetic with comprehensive protection
        h_zj_complex = h_complex / denominator_complex
        
        # **FIX THE OVERFLOW ISSUE HERE**
        # Check magnitude of h_zj_complex BEFORE any squaring operation
        h_zj_magnitude = np.abs(h_zj_complex)
        sqrt_max_for_square = np.sqrt(sqrt_max)  # Even more conservative for squaring
        
        h_zj_too_large_for_square = h_zj_magnitude > sqrt_max_for_square
        h_zj_too_small = h_zj_magnitude < sqrt_min
        
        if np.any(h_zj_too_large_for_square):
            if self.verbose:
                large_count = np.sum(h_zj_too_large_for_square)
                self.logger.info(f"Warning: {large_count} h_zj values too large for safe squaring. Using approximation.")
            
            # For very large |h_zj|, use the mathematical limit without squaring
            # When |h_zj| >> 1: h_zj / sqrt(1 + h_zj²) ≈ h_zj / |h_zj| = sign(h_zj)
            
            # Safe calculation for non-large values only
            h_zj_safe = np.where(h_zj_too_large_for_square, 0, h_zj_complex)  # Zero out large values
            h_zj_squared_safe = h_zj_safe**2  # Only square the safe values
            
            # Calculate result for safe values
            safe_result = h_zj_safe / np.sqrt(1 + h_zj_squared_safe)
            
            # Use approximation for large values
            large_result = h_zj_complex / h_zj_magnitude
            
            # Combine results
            h_gq_complex = np.where(h_zj_too_large_for_square, large_result, safe_result)

        elif np.any(h_zj_too_small):
            self.logger.info("Warning: Very small h_zj values in complex calculation.")

            # For very small |h_zj|: h_zj / sqrt(1 + h_zj²) ≈ h_zj (linear approximation)
            h_gq_complex = np.where(h_zj_too_small,
                                   h_zj_complex,  # linear approximation - no squaring!
                                   h_zj_complex / np.sqrt(1 + h_zj_complex**2))  # safe squaring only
        else:
            # All values are safe for squaring - proceed normally
            try:
                # Only square when we know it's safe
                h_zj_squared = h_zj_complex**2
                h_gq_complex = h_zj_complex / np.sqrt(1 + h_zj_squared)
            except (OverflowError, FloatingPointError, ZeroDivisionError) as e:
                # log error
                error_msg = f"Exception in h_gq calculation: {e}"
                self.params['errors'].append({
                    'method': '_calculate_pdf_from_moments',
                    'error': error_msg,
                    'exception_type': type(e).__name__
                })
                if self.verbose:
                    self.logger.info(f"Warning: Unexpected exception in h_gq calculation ({e}). Using approximation.")
                # Fallback to magnitude-based approach
                h_gq_complex = h_zj_complex / (h_zj_magnitude + min_safe_value)
        
        # Extract meaningful results from complex calculation
        h_gq_real = np.real(h_gq_complex)
        h_gq_imag = np.imag(h_gq_complex)
        h_gq_magnitude = np.abs(h_gq_complex)
        
        # Determine how to handle complex results with small value protection
        is_purely_real = np.abs(h_gq_imag) < min_safe_value
        is_real_dominant = np.abs(h_gq_real) >= np.abs(h_gq_imag)
        
        if self.verbose and not np.all(is_purely_real):
            complex_count = np.sum(~is_purely_real)
            self.logger.info(f"Info: {complex_count} points have complex intermediate results.")
        
        # Strategy for handling complex results with numerical stability
        h_gq_final = np.where(is_purely_real, 
                             h_gq_real,  # Use real part for essentially real results
                             np.where(is_real_dominant,
                                     h_gq_real,  # Use real part when real component dominates
                                     h_gq_magnitude * np.sign(h_gq_real)))  # Use magnitude with sign
        
        # Clip to reasonable range to prevent further overflow/underflow
        h_gq_final = np.clip(h_gq_final, -10, 10)
        
        # Calculate QGDF using the processed hGQ values
        qgdf_from_hgq = (1 + h_gq_final) / 2
        
        # Also calculate using direct ratio as backup with small value protection
        mean_fidelity_safe = np.where(np.abs(mean_fidelity) < min_safe_value,
                                     np.sign(mean_fidelity) * min_safe_value, mean_fidelity)
        
        ratio = mean_irrelevance / mean_fidelity_safe
        
        # Handle extreme ratios (both large and small)
        ratio_magnitude = np.abs(ratio)
        ratio_too_large = ratio_magnitude > 10
        ratio_too_small = ratio_magnitude < min_safe_value
        
        ratio_safe = np.where(ratio_too_large, 10 * np.tanh(ratio / 10), ratio)
        ratio_safe = np.where(ratio_too_small, np.sign(ratio) * min_safe_value, ratio_safe)
        
        qgdf_from_ratio = (1 - ratio_safe) / 2
        
        # Use complex method for difficult cases, ratio method for simple cases
        use_complex_method = ~is_purely_real | ratio_too_large | ratio_too_small
        
        qgdf_values = np.where(use_complex_method,
                              qgdf_from_hgq,
                              qgdf_from_ratio)
        
        # Apply final constraints
        qgdf_values = np.clip(qgdf_values, 0, 1)
        qgdf_values = np.maximum.accumulate(qgdf_values)
        
        return qgdf_values.flatten()
    
    def _estimate_qgdf_from_moments(self, fidelities, irrelevances):
        """Main QGDF estimation method with complex number fallback."""
        self.logger.info("Estimating QGDF from moments with fallback.")
        try:
            # First try the complex number approach
            return self._estimate_qgdf_from_moments_complex(fidelities, irrelevances)
        except Exception as e:
            # log error
            error_msg = f"Exception in complex QGDF estimation: {e}"
            self.logger.error(error_msg)
            if self.verbose:
                self.logger.info(f"Complex method failed: {e}. Using fallback approach.")
            self.params['errors'].append({
                'method': '_estimate_qgdf_from_moments',
                'error': error_msg,
                'exception_type': type(e).__name__
            })

            # Fallback to the robust real-number approach
            return self._estimate_qgdf_from_moments_fallback(fidelities, irrelevances)
    
    def _estimate_qgdf_from_moments_fallback(self, fidelities, irrelevances):
        """Fallback method using real numbers only."""
        self.logger.info("Estimating QGDF using fallback real-number approach.")
        weights = self._computation_cache['weights_normalized'].reshape(-1, 1)
        
        # Calculate weighted means
        mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)
        mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)
                
        # Direct ratio approach (always mathematically valid)
        mean_fidelity_safe = np.where(np.abs(mean_fidelity) < self._NUMERICAL_EPS,
                                     np.sign(mean_fidelity) * self._NUMERICAL_EPS, mean_fidelity)
        
        ratio = mean_irrelevance / mean_fidelity_safe
        ratio_limited = np.where(np.abs(ratio) > 5, 5 * np.tanh(ratio / 5), ratio)

        # hzj NOTE for QGDF book eq not working properly
        # hzj = mean_irrelevance / (np.sqrt(mean_fidelity_safe**2 + mean_irrelevance**2))

        # # hgq
        # h_gq = hzj / (np.sqrt(1 + hzj**2))

        # qgdf_values = (1 + h_gq/mean_fidelity_safe) / 2
        
        qgdf_values = (1 - ratio_limited) / 2     
        qgdf_values = np.clip(qgdf_values, 0, 1)
        qgdf_values = np.maximum.accumulate(qgdf_values)
        
        return qgdf_values.flatten()
    
    # NOTE fi and hi derivative base logic
    # this give little of PDF
    # can be improved
    # def _calculate_pdf_from_moments(self, fidelities, irrelevances):
    #     """Calculate first derivative of QGDF (which is the PDF) from stored fidelities and irrelevances."""
    #     if fidelities is None or irrelevances is None:
    #         # log error
    #         error_msg = "Fidelities and irrelevances must be calculated first"
    #         self.params['errors'].append({
    #             'method': '_calculate_pdf_from_moments',
    #             'error': error_msg,
    #             'exception_type': 'ValueError'
    #         })
    #         raise ValueError("Fidelities and irrelevances must be calculated first")
        
    #     weights = self.weights.reshape(-1, 1)
        
    #     # First order moments using QGDF's fj and hj
    #     f1 = np.sum(weights * fidelities, axis=0) / np.sum(weights)  # f̄Q
    #     h1 = np.sum(weights * irrelevances, axis=0) / np.sum(weights)  # h̄Q

    #     # Second order moments (scaled by S as in EGDF pattern)
    #     f2s = np.sum(weights * (fidelities**2 / self.S_opt), axis=0) / np.sum(weights)  # F2
    #     h2s = np.sum(weights * (irrelevances**2 / self.S_opt), axis=0) / np.sum(weights)  # H2
    #     fhs = np.sum(weights * (fidelities * irrelevances / self.S_opt), axis=0) / np.sum(weights)  # FH
        
    #     # Calculate Nj = Σ(1/f²ᵢ,ⱼ) + Σ H²ᵢ,ⱼ (from equation 10.8)
    #     eps = np.finfo(float).eps
    #     f_inv_squared = np.sum(weights * (1 / (fidelities**2 + eps)), axis=0) / np.sum(weights)
    #     h_squared = np.sum(weights * irrelevances**2, axis=0) / np.sum(weights)
    #     Nj = f_inv_squared + h_squared
    #     Nj = np.where(Nj == 0, eps, Nj)
        
    #     # Calculate denominator w = (2 * Nj)^2 for QGDF derivative
    #     w = (2 * Nj)**2
    #     w = np.where(w == 0, eps, w)
        
    #     # QGDF PDF formula: dQGDF/dZ₀ = (1/SZ₀) * (1/(2 * Nⱼ²)) * [F2 - H2 + f̄_E * h̄_E * FH]
    #     numerator = f2s - h2s + f1 * h1 * fhs
    #     first_derivative = (1 / self.S_opt) * numerator / ( Nj**2)
        
    #     return first_derivative.flatten()
    
    def _calculate_pdf_from_moments(self, fidelities, irrelevances):
        self.logger.info("Calculating PDF from moments.")
        """Calculate PDF from fidelities and irrelevances with corrected mathematical formulation."""
        self.logger.info("Calculating PDF from moments")
        if fidelities is None or irrelevances is None:
            # log error
            self.logger.error("Fidelities and irrelevances must be calculated first.")
            raise ValueError("Fidelities and irrelevances must be calculated first")
        
        weights = self._computation_cache['weights_normalized'].reshape(-1, 1)
        
        # Numerical stability constants
        max_safe_value = np.sqrt(np.finfo(float).max) / 10
        min_safe_value = np.sqrt(np.finfo(float).eps) * 100
        
        def safe_clip_for_pdf(values, name="values"):
            """Safely clip values for PDF calculations."""
            values_magnitude = np.abs(values)
            too_small_mask = values_magnitude < min_safe_value
            too_large_mask = values_magnitude > max_safe_value
            
            values_safe = np.where(too_small_mask, 
                                  np.sign(values) * min_safe_value, values)
            values_safe = np.where(too_large_mask, 
                                  np.sign(values_safe) * max_safe_value, values_safe)
            return values_safe
        
        # Apply clipping
        fidelities_safe = safe_clip_for_pdf(fidelities, "fidelity")
        irrelevances_safe = safe_clip_for_pdf(irrelevances, "irrelevance")
        
        # Calculate weighted means
        mean_fidelity = np.sum(weights * fidelities_safe, axis=0) / np.sum(weights)  # f̄Q
        mean_irrelevance = np.sum(weights * irrelevances_safe, axis=0) / np.sum(weights)  # h̄Q
        
        # Apply safety to means
        mean_fidelity = safe_clip_for_pdf(mean_fidelity, "mean_fidelity")
        mean_irrelevance = safe_clip_for_pdf(mean_irrelevance, "mean_irrelevance")
        
        # CORRECTED PDF CALCULATION FOR QGDF
        # The PDF should be the derivative of QGDF with respect to the data points
        # Based on QGDF = (1 + h_GQ)/2, where h_GQ = h̄Q/√(f̄Q² - h̄Q²)/√(1 + (h̄Q/√(f̄Q² - h̄Q²))²)
        
        S_value = self.S_opt if hasattr(self, 'S_opt') else 1.0
        
        # Calculate the denominator √(f̄Q² - h̄Q²) with protection
        mean_fidelity_safe = np.where(np.abs(mean_fidelity) < min_safe_value,
                                     np.sign(mean_fidelity) * min_safe_value, mean_fidelity)
        
        # For QGDF, the correct mathematical relationship is different from what's implemented
        # The PDF should be derived from d(QGDF)/dz, not from an empirical ratio formula
        
        # Corrected approach: Use the mathematical derivative of the QGDF equation
        # d(QGDF)/dz = (1/2) * d(h_GQ)/dz
        
        # Calculate h_Z,j = h̄Q / √(f̄Q² - h̄Q²)
        denominator_squared = mean_fidelity_safe**2 - mean_irrelevance**2
        
        # Ensure denominator is positive and safe
        denominator_squared = np.maximum(denominator_squared, min_safe_value)
        denominator = np.sqrt(denominator_squared)
        
        h_zj = mean_irrelevance / denominator
        
        # Clip h_zj to avoid overflow
        h_zj = np.clip(h_zj, 1, 1e12)
        
        # Calculate h_GQ = h_Z,j / √(1 + h_Z,j²)
        h_zj_squared = np.minimum(h_zj**2, max_safe_value)  # Prevent overflow
        h_gq_denominator = np.sqrt(1 + h_zj_squared)
        h_gq = h_zj / h_gq_denominator
        
        # For PDF calculation, we need the derivative of h_GQ with respect to z
        # This involves second-order moments which should be calculated properly
        
        # Second order moments (this is where the original method had issues)
        f2 = np.sum(weights * fidelities_safe**2, axis=0) / np.sum(weights)
        h2 = np.sum(weights * irrelevances_safe**2, axis=0) / np.sum(weights)
        fh = np.sum(weights * fidelities_safe * irrelevances_safe, axis=0) / np.sum(weights)
        
        # Apply safety to second moments
        f2 = safe_clip_for_pdf(f2, "f2")
        h2 = safe_clip_for_pdf(h2, "h2") 
        fh = safe_clip_for_pdf(fh, "fh")
        
        # Corrected PDF formula for QGDF:
        # PDF = (1/S) * derivative_term where derivative_term comes from differentiating h_GQ
        
        # This is a simplified but more mathematically sound approach
        # clip values to avoid overflow in multiplications [0, 1e12]
        mean_irrelevance = np.clip(mean_irrelevance, 1, 1e12)
        mean_fidelity = np.clip(mean_fidelity, 0, 1e12)
        fh = np.clip(fh, -1e12, 1e12)
        derivative_factor = f2 - h2 + mean_fidelity * mean_irrelevance * fh
        
        # Apply scaling and ensure positive values
        pdf_values = (1 / S_value) * np.maximum(derivative_factor, min_safe_value)
        
        # Final clipping
        pdf_values = np.clip(pdf_values, min_safe_value, max_safe_value)
        
        return pdf_values.flatten()

    def _calculate_final_results(self):
        """Calculate final QGDF and PDF with optimized parameters."""
        self.logger.info("Calculating final QGDF and PDF results.")
        # Convert to infinite domain
        # zi_n = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        zi_d = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        self.zi = zi_d

        # Calculate QGDF and get moments
        qgdf_values, fj, hj = self._compute_qgdf_core(self.S_opt, self.LB_opt, self.UB_opt)

        # Store for derivative calculations
        self.fj = fj
        self.hj = hj
        self.qgdf = qgdf_values
        self.pdf = self._calculate_pdf_from_moments(fj, hj)
        
        if self.catch:
            self.params.update({
                'qgdf': self.qgdf.copy(),
                'pdf': self.pdf.copy(),
                'zi': self.zi.copy()
            })

    def _generate_smooth_curves(self):
        """Generate smooth curves for plotting and analysis."""
        self.logger.info("Generating smooth curves for QGDF and PDF.")
        try:
            # Generate smooth QGDF and PDF
            smooth_qgdf, self.smooth_fj, self.smooth_hj = self._compute_qgdf_core(
                self.S_opt, self.LB_opt, self.UB_opt,
                zi_data=self.z_points_n, zi_eval=self.z
            )
            
            smooth_pdf = self._calculate_pdf_from_moments(self.smooth_fj, self.smooth_hj)

            self.qgdf_points = smooth_qgdf
            self.pdf_points = smooth_pdf
            
            # Store zi_n for derivative calculations
            self.zi_n = DataConversion._convert_fininf(self.z_points_n, self.LB_opt, self.UB_opt)
            
            # Mark as generated
            self._computation_cache['smooth_curves_generated'] = True
            
            if self.catch:
                self.params.update({
                    'qgdf_points': self.qgdf_points.copy(),
                    'pdf_points': self.pdf_points.copy(),
                    'zi_points': self.zi_n.copy()
                })
            
            self.logger.info(f"Generated smooth curves with {self.n_points} points.")

        except Exception as e:
            # Log the error
            error_msg = f"Could not generate smooth curves: {e}"
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_generate_smooth_curves',
                'error': error_msg,
                'exception_type': type(e).__name__
            })

            self.logger.warning(f"Could not generate smooth curves: {e}")
            # Create fallback points using original data
            self.qgdf_points = self.qgdf.copy() if hasattr(self, 'qgdf') else None
            self.pdf_points = self.pdf.copy() if hasattr(self, 'pdf') else None
            self._computation_cache['smooth_curves_generated'] = False

    def _get_results(self)-> dict:
        """Return fitting results."""
        self.logger.info("Getting results from QGDF fitting.")

        if not self._fitted:
            error_msg = "Must fit QGDF before getting results."
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_get_results',
                'error': error_msg,
                'exception_type': 'RuntimeError'
            })
            raise RuntimeError("Must fit QGDF before getting results.")

        # selected key from params if exists
        keys = ['DLB', 'DUB', 'LB', 'UB', 'S_opt', 'z0', 'qgdf', 'pdf', 
                'qgdf_points', 'pdf_points', 'zi', 'zi_points', 'weights']
        results = {key: self.params.get(key) for key in keys if key in self.params}
        return results


    def _plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = True, extra_df: bool = True, figsize: tuple = (12, 8)):
        """Enhanced plotting with better organization."""
        self.logger.info("Plotting QGDF and PDF results.")
        import matplotlib.pyplot as plt

        if plot_smooth and (len(self.data) > self.max_data_size) and self.verbose:
            self.logger.warning(f"Given data size ({len(self.data)}) exceeds max_data_size ({self.max_data_size}). For optimal compute performance, set 'plot_smooth=False', or 'max_data_size' to a larger value whichever is appropriate.")

        if not self.catch:
            self.logger.warning("Plot is not available with argument catch=False")
            return
        
        if not self._fitted:
            self.logger.error("Must fit QGDF before plotting.")
            raise RuntimeError("Must fit QGDF before plotting.")
        
        # Validate plot parameter
        if plot not in ['gdf', 'pdf', 'both']:
            self.logger.error("Invalid plot parameter.")
            raise ValueError("plot parameter must be 'gdf', 'pdf', or 'both'")
        
        # Check data availability
        if plot in ['gdf', 'both'] and self.params.get('qgdf') is None:
            self.logger.error("QGDF must be calculated before plotting GDF.")
            raise ValueError("QGDF must be calculated before plotting GDF")
        if plot in ['pdf', 'both'] and self.params.get('pdf') is None:
            self.logger.error("PDF must be calculated before plotting PDF.")
            raise ValueError("PDF must be calculated before plotting PDF.")

        # Prepare data
        x_points = self.data
        qgdf_plot = self.params.get('qgdf')
        pdf_plot = self.params.get('pdf')
        wedf = self.params.get('wedf')
        ksdf = self.params.get('ksdf')
        
        # Check smooth plotting availability
        has_smooth = (hasattr(self, 'di_points_n') and hasattr(self, 'qgdf_points') 
                    and hasattr(self, 'pdf_points') and self.di_points_n is not None
                    and self.qgdf_points is not None and self.pdf_points is not None)
        plot_smooth = plot_smooth and has_smooth
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=figsize)

        # Plot QGDF if requested
        if plot in ['gdf', 'both']:
            self._plot_qgdf(ax1, x_points, qgdf_plot, plot_smooth, extra_df, wedf, ksdf)
        
        # Plot PDF if requested
        if plot in ['pdf', 'both']:
            if plot == 'pdf':
                self._plot_pdf(ax1, x_points, pdf_plot, plot_smooth, is_secondary=False)
            else:
                ax2 = ax1.twinx()
                self._plot_pdf(ax2, x_points, pdf_plot, plot_smooth, is_secondary=True)
        
        # Add bounds and formatting
        self._add_plot_formatting(ax1, plot, bounds)
        
        # Add Z0 vertical line if available
        if hasattr(self, 'z0') and self.z0 is not None:
            ax1.axvline(x=self.z0, color='magenta', linestyle='-.', linewidth=1, 
                    alpha=0.8, label=f'Z0={self.z0:.3f}')
            # Update legend to include Z0
            ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        plt.tight_layout()
        plt.show()

    def _plot_qgdf(self, ax, x_points, qgdf_plot, plot_smooth, extra_df, wedf, ksdf):
        """Plot QGDF components."""
        self.logger.info("Plotting QGDF components.")
        if plot_smooth and hasattr(self, 'qgdf_points') and self.qgdf_points is not None:
            ax.plot(x_points, qgdf_plot, 'o', color='blue', label='QGDF', markersize=4)
            ax.plot(self.di_points_n, self.qgdf_points, color='blue', 
                   linestyle='-', linewidth=2, alpha=0.8)
        else:
            ax.plot(x_points, qgdf_plot, 'o-', color='blue', label='QGDF', 
                   markersize=4, linewidth=1, alpha=0.8)
        
        if extra_df:
            if wedf is not None:
                ax.plot(x_points, wedf, 's', color='lightblue', 
                       label='WEDF', markersize=3, alpha=0.8)
            if ksdf is not None:
                ax.plot(x_points, ksdf, 's', color='cyan', 
                       label='KS Points', markersize=3, alpha=0.8)
        
        ax.set_ylabel('QGDF', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_ylim(0, 1)

    def _plot_pdf(self, ax, x_points, pdf_plot, plot_smooth, is_secondary=False):
        """Plot PDF components."""
        self.logger.info("Plotting PDF components.")
        color = 'red'

        if plot_smooth and hasattr(self, 'pdf_points') and self.pdf_points is not None:
            ax.plot(x_points, pdf_plot, 'o', color=color, label='PDF', markersize=4)
            ax.plot(self.di_points_n, self.pdf_points, color=color, 
                   linestyle='-', linewidth=2, alpha=0.8)
            max_pdf = np.max(self.pdf_points)
        else:
            ax.plot(x_points, pdf_plot, 'o-', color=color, label='PDF', 
                   markersize=4, linewidth=1, alpha=0.8)
            max_pdf = np.max(pdf_plot)
        
        ax.set_ylabel('PDF', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim(0, max_pdf * 1.1)
        
        if is_secondary:
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

    def _add_plot_formatting(self, ax1, plot, bounds):
        """Add formatting, bounds, and legends to plot."""
        self.logger.info("Adding plot formatting and bounds.")
        ax1.set_xlabel('Data Points')
        
        # Add bounds if requested
        if bounds:
            bound_info = [
                (self.params.get('DLB'), 'green', '-', 'DLB'),
                (self.params.get('DUB'), 'orange', '-', 'DUB'),
                (self.params.get('LB'), 'purple', '--', 'LB'),
                (self.params.get('UB'), 'brown', '--', 'UB')
            ]
            
            for bound, color, style, name in bound_info:
                if bound is not None:
                    ax1.axvline(x=bound, color=color, linestyle=style, linewidth=2, 
                               alpha=0.8, label=f"{name}={bound:.3f}")
            
            # Add shaded regions
            if self.params.get('LB') is not None:
                ax1.axvspan(self.data.min(), self.params['LB'], alpha=0.15, color='purple')
            if self.params.get('UB') is not None:
                ax1.axvspan(self.params['UB'], self.data.max(), alpha=0.15, color='brown')
        
        # Set limits and add grid
        data_range = self.params['DUB'] - self.params['DLB']
        padding = data_range * 0.1
        ax1.set_xlim(self.params['DLB'] - padding, self.params['DUB'] + padding)
        
        # Set title
        titles = {
            'gdf': 'QGDF' + (' with Bounds' if bounds else ''),
            'pdf': 'PDF' + (' with Bounds' if bounds else ''),
            'both': 'QGDF and PDF' + (' with Bounds' if bounds else '')
        }
        
        ax1.set_title(titles[plot])
        ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
        ax1.grid(True, alpha=0.3)
        
    
    def _get_qgdf_second_derivative(self):
        """Calculate second derivative of QGDF with corrected mathematical formulation."""
        self.logger.info("Calculating second derivative of QGDF.")
        if self.fj is None or self.hj is None:
            self.logger.error("Fidelities and irrelevances must be calculated before second derivative estimation.")
            raise ValueError("Fidelities and irrelevances must be calculated before second derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # Calculate all required moments
        f1 = np.sum(weights * self.fj, axis=0) / np.sum(weights)  # f̄Q
        h1 = np.sum(weights * self.hj, axis=0) / np.sum(weights)  # h̄Q
        f2 = np.sum(weights * self.fj**2, axis=0) / np.sum(weights)
        h2 = np.sum(weights * self.hj**2, axis=0) / np.sum(weights)
        fh = np.sum(weights * self.fj * self.hj, axis=0) / np.sum(weights)
        
        # Additional moments for second derivative
        f3 = np.sum(weights * self.fj**3, axis=0) / np.sum(weights)
        h3 = np.sum(weights * self.hj**3, axis=0) / np.sum(weights)
        f2h = np.sum(weights * self.fj**2 * self.hj, axis=0) / np.sum(weights)
        fh2 = np.sum(weights * self.fj * self.hj**2, axis=0) / np.sum(weights)
        
        eps = np.finfo(float).eps
        f1_safe = np.where(np.abs(f1) < eps, np.sign(f1) * eps, f1)
        
        # CORRECTED: Based on the actual QGDF equation QGDF = (1 + h_GQ)/2
        # where h_GQ = h_zj / √(1 + h_zj²) and h_zj = h̄Q / √(f̄Q² - h̄Q²)
        
        # Calculate first derivatives of weighted means
        # These are derived from the variance-covariance relationships
        df1_dz = (f2 - f1**2) / self.S_opt  # Corrected: variance formula
        dh1_dz = (h2 - h1**2) / self.S_opt  # Corrected: variance formula
        
        # Calculate second derivatives
        d2f1_dz2 = (f3 - 3*f1*f2 + 2*f1**3) / (self.S_opt**2)  # Third central moment
        d2h1_dz2 = (h3 - 3*h1*h2 + 2*h1**3) / (self.S_opt**2)  # Third central moment
        
        # Calculate derivatives of h_zj = h̄Q / √(f̄Q² - h̄Q²)
        denominator_squared = f1_safe**2 - h1**2
        denominator_squared = np.maximum(denominator_squared, eps)
        denominator = np.sqrt(denominator_squared)
        
        h_zj = h1 / denominator
        
        # First derivative of h_zj using quotient rule
        d_numerator = dh1_dz
        d_denominator = (f1_safe * df1_dz - h1 * dh1_dz) / denominator
        
        dh_zj_dz = (d_numerator * denominator - h_zj * d_denominator) / denominator
        
        # Second derivative of h_zj (more complex)
        d2_numerator = d2h1_dz2
        # For d²(denominator), we need more careful calculation
        temp_term = f1_safe * d2f1_dz2 - h1 * d2h1_dz2 - df1_dz**2 - dh1_dz**2
        d2_denominator = (temp_term * denominator - d_denominator**2) / denominator
        
        d2h_zj_dz2 = ((d2_numerator * denominator - d_numerator * d_denominator) * denominator - 
                       (d_numerator * denominator - h_zj * d_denominator) * d_denominator) / (denominator**2)
        
        # Calculate derivatives of h_GQ = h_zj / √(1 + h_zj²)
        h_zj_squared = np.minimum(h_zj**2, 1e10)  # Prevent overflow
        h_gq_denominator = np.sqrt(1 + h_zj_squared)
        
        # First derivative of h_GQ
        dh_gq_dz = dh_zj_dz / (h_gq_denominator**3)
        
        # Second derivative of h_GQ
        term1 = d2h_zj_dz2 / (h_gq_denominator**3)
        term2 = -3 * dh_zj_dz**2 * h_zj / (h_gq_denominator**5)
        
        d2h_gq_dz2 = term1 + term2
        
        # Finally, second derivative of QGDF = (1/2) * d²(h_GQ)/dz²
        second_derivative = 0.5 * d2h_gq_dz2
        
        return second_derivative.flatten()

    def _get_qgdf_third_derivative(self):
        """Calculate third derivative of QGDF with corrected mathematical formulation."""
        self.logger.info("Calculating third derivative of QGDF.")
        if self.fj is None or self.hj is None:
            self.logger.error("Fidelities and irrelevances must be calculated before third derivative estimation.")
            raise ValueError("Fidelities and irrelevances must be calculated before third derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # Calculate all required moments up to 4th order
        f1 = np.sum(weights * self.fj, axis=0) / np.sum(weights)
        h1 = np.sum(weights * self.hj, axis=0) / np.sum(weights)
        f2 = np.sum(weights * self.fj**2, axis=0) / np.sum(weights)
        h2 = np.sum(weights * self.hj**2, axis=0) / np.sum(weights)
        f3 = np.sum(weights * self.fj**3, axis=0) / np.sum(weights)
        h3 = np.sum(weights * self.hj**3, axis=0) / np.sum(weights)
        f4 = np.sum(weights * self.fj**4, axis=0) / np.sum(weights)
        h4 = np.sum(weights * self.hj**4, axis=0) / np.sum(weights)
        
        eps = np.finfo(float).eps
        f1_safe = np.where(np.abs(f1) < eps, np.sign(f1) * eps, f1)
        
        # Calculate derivatives up to third order
        df1_dz = (f2 - f1**2) / self.S_opt
        dh1_dz = (h2 - h1**2) / self.S_opt
        
        d2f1_dz2 = (f3 - 3*f1*f2 + 2*f1**3) / (self.S_opt**2)
        d2h1_dz2 = (h3 - 3*h1*h2 + 2*h1**3) / (self.S_opt**2)
        
        d3f1_dz3 = (f4 - 4*f1*f3 + 6*f1**2*f2 - 3*f1**4) / (self.S_opt**3)
        d3h1_dz3 = (h4 - 4*h1*h3 + 6*h1**2*h2 - 3*h1**4) / (self.S_opt**3)
        
        # Calculate h_zj and its derivatives (simplified approach)
        denominator_squared = f1_safe**2 - h1**2
        denominator_squared = np.maximum(denominator_squared, eps)
        denominator = np.sqrt(denominator_squared)
        
        h_zj = h1 / denominator
        
        # For third derivative, use numerical differentiation as analytical form is extremely complex
        h = 1e-6 * np.std(self.data) if np.std(self.data) > 0 else 1e-6
        
        # Store original values
        original_zi = self.zi.copy()
        original_fi = self.fj.copy()
        original_hi = self.hj.copy()
        
        try:
            # Calculate second derivative at nearby points
            second_derivs = []
            points = [-h, 0, h]
            
            for delta in points:
                self.zi = original_zi + delta
                self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
                second_deriv = self._get_qgdf_second_derivative()
                second_derivs.append(second_deriv)
            
            # Use finite difference formula for third derivative
            # f'''(x) ≈ [f''(x+h) - f''(x-h)] / (2h)
            third_derivative = (second_derivs[2] - second_derivs[0]) / (2 * h)
            
            return third_derivative.flatten()
            
        finally:
            # Always restore original state
            self.zi = original_zi
            self.fj = original_fi
            self.hj = original_hi

    def _get_qgdf_fourth_derivative(self):
        """Calculate fourth derivative of QGDF using corrected numerical differentiation."""
        self.logger.info("Calculating fourth derivative of QGDF.")
        if self.fj is None or self.hj is None:
            self.logger.error("Fidelities and irrelevances must be calculated before fourth derivative estimation.")
            raise ValueError("Fidelities and irrelevances must be calculated before fourth derivative estimation.")
        
        # Use adaptive step size based on data scale
        data_scale = np.std(self.data) if np.std(self.data) > 0 else 1.0
        h = max(1e-6 * data_scale, 1e-10)
        
        # Store original state
        original_fi = self.fj.copy()
        original_hi = self.hj.copy()
        original_zi = self.zi.copy()
        
        try:
            # Use 5-point stencil for better accuracy
            # f''''(x) ≈ [f'''(x-2h) - 8f'''(x-h) + 8f'''(x+h) - f'''(x+2h)] / (12h)
            points = [-2*h, -h, 0, h, 2*h]
            third_derivatives = []
            
            for delta in points:
                self.zi = original_zi + delta
                self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
                third_deriv = self._get_qgdf_third_derivative()
                third_derivatives.append(third_deriv)
            
            # Apply 5-point finite difference formula
            fourth_derivative = (third_derivatives[0] - 8*third_derivatives[1] + 
                                8*third_derivatives[3] - third_derivatives[4]) / (12*h)
            
            # REMOVED THE INCORRECT MULTIPLICATION BY self.zi
            # The original code incorrectly multiplied by self.zi
            
            return fourth_derivative.flatten()
            
        finally:
            # Always restore original state
            self.fj = original_fi
            self.hj = original_hi  
            self.zi = original_zi

    def _calculate_fidelities_irrelevances_at_given_zi_corrected(self, zi):
        """Helper method to recalculate fidelities and irrelevances for current zi."""
        self.logger.info("Calculating fidelities and irrelevances at given zi.")
        # FIXED: Convert the data points to infinite domain, not the evaluation points
        zi_data = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)  # Data points
        zi_eval = DataConversion._convert_fininf(zi, self.LB_opt, self.UB_opt)       # Evaluation points
        
        # Calculate R matrix with proper dimensions
        eps = np.finfo(float).eps
        R = zi_eval.reshape(-1, 1) / (zi_data.reshape(1, -1) + eps)
        
        # Get characteristics
        gc = GnosticsCharacteristics(R=R, verbose=self.verbose)
        q, q1 = gc._get_q_q1(S=self.S_opt)
        
        # Store fidelities and irrelevances
        self.fj = gc._fj(q=q, q1=q1)
        self.hj = gc._hj(q=q, q1=q1)


    def _fit_qgdf(self, plot: bool = False):
        """Fit the QGDF to the data."""
        self.logger.info("Starting QGDF fitting process.")
        try:
            
            # Step 1: Data preprocessing
            self.logger.info("Preprocessing data for QGDF fitting.")
            self.data = np.sort(self.data)
            self._estimate_data_bounds()
            self._transform_data_to_standard_domain()
            self._estimate_weights()
            
            # Step 2: Bounds estimation
            self.logger.info("Estimating initial probable bounds.")
            self._estimate_initial_probable_bounds()
            self._generate_evaluation_points()
            
            # Step 3: Get distribution function values for optimization
            self.logger.info("Getting distribution function values for optimization.")
            self.df_values = self._get_distribution_function_values(use_wedf=self.wedf)
            
            # Step 4: Parameter optimization
            self.logger.info("Optimizing QGDF parameters.")
            self._determine_optimization_strategy(egdf=False)  # NOTE for QGDF egdf is False
            
            # Step 5: Calculate final QGDF and PDF
            self.logger.info("Calculating final QGDF and PDF with optimized parameters.")
            self._calculate_final_results()
            
            # Step 6: Generate smooth curves for plotting and analysis
            self.logger.info("Generating smooth curves for QGDF and PDF.")
            self._generate_smooth_curves()
            
            # Step 7: Transform bounds back to original domain
            self.logger.info("Transforming bounds back to original domain.")
            self._transform_bounds_to_original_domain()
            # Mark as fitted (Step 8 is now optional via marginal_analysis())
            self._fitted = True

            # Step 8: Z0 estimate with Z0Estimator
            self.logger.info("Estimating Z0 point with Z0Estimator.")
            self._compute_z0(optimize=self.z0_optimize) 
            # derivatives calculation
            # self._calculate_all_derivatives()
                        
            self.logger.info("QGDF fitting completed successfully.")

            if plot:
                self.logger.info("Plotting QGDF and PDF.")
                self._plot()

            # clean up computation cache
            if self.flush:  
                self.logger.info("Cleaning up computation cache.")
                self._cleanup_computation_cache()
                
        except Exception as e:
            error_msg = f"QGDF fitting failed: {e}"
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_fit_QGDF',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            
            self.logger.error(f"Error during QGDF fitting: {e}")
            raise e
        
    # z0 compute
    def _compute_z0(self, optimize: bool = None):
        """
        Compute the Z0 point where PDF is maximum using the Z0Estimator class.
        
        Parameters:
        -----------
        optimize : bool, optional
            If True, use interpolation-based methods for higher accuracy.
            If False, use simple linear search on existing points.
            If None, uses the instance's z0_optimize setting.
        """
        self.logger.info("Computing Z0 point using Z0Estimator.")

        if self.z is None:
            self.logger.error("Data must be transformed (self.z) before Z0 estimation.")
            raise ValueError("Data must be transformed (self.z) before Z0 estimation.")
        
        # Use provided optimize parameter or fall back to instance setting
        use_optimize = optimize if optimize is not None else self.z0_optimize
        
        self.logger.info('QGDF: Computing Z0 point using Z0Estimator...')

        try:
            # Create Z0Estimator instance with proper constructor signature
            z0_estimator = Z0Estimator(
                gdf_object=self,  # Pass the QGDF object itself
                optimize=use_optimize,
                verbose=self.verbose
            )
            
            # Call fit() method to estimate Z0
            self.z0 = z0_estimator.fit()
            
            # Get estimation info for debugging and storage
            if self.catch:
                estimation_info = z0_estimator.get_estimation_info()
                self.params.update({
                    'z0': float(self.z0) if self.z0 is not None else None,
                    'z0_method': estimation_info.get('z0_method', 'unknown'),
                    'z0_estimation_info': estimation_info
                })
            
            method_used = z0_estimator.get_estimation_info().get('z0_method', 'unknown')
            self.logger.info(f'QGDF: Z0 point computed successfully, (method: {method_used})')

        except Exception as e:
            # Log the error
            error_msg = f"Z0 estimation failed: {str(e)}"
            self.params['errors'].append({
                'method': '_compute_z0',
                'error': error_msg,
                'exception_type': type(e).__name__
            })

            self.logger.warning(f"Warning: Z0Estimator failed with error: {e}")
            self.logger.info("Falling back to simple maximum finding...")

            # Fallback to simple maximum finding
            self._compute_z0_fallback()
            
            if self.catch:
                self.params.update({
                    'z0': float(self.z0),
                    'z0_method': 'fallback_simple_maximum',
                    'z0_estimation_info': {'error': str(e)}
                })

    def _compute_z0_fallback(self):
        """
        Fallback method for Z0 computation using simple maximum finding.
        """
        if not hasattr(self, 'di_points_n') or not hasattr(self, 'pdf_points'):
            self.logger.error("Both 'di_points_n' and 'pdf_points' must be defined for Z0 computation.")
            raise ValueError("Both 'di_points_n' and 'pdf_points' must be defined for Z0 computation.")
    
        self.logger.info('Using fallback method for Z0 point...')
        
        # Find index with maximum PDF
        max_idx = np.argmax(self.pdf_points)
        self.z0 = self.di_points_n[max_idx]

        self.logger.info(f"Z0 point (fallback method).")

    def analyze_z0(self, figsize: tuple = (12, 6)) -> Dict[str, Any]:
        """
        Analyze and visualize Z0 estimation results.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size for the plot
            
        Returns:
        --------
        Dict[str, Any]
            Z0 analysis information
        """
        self.logger.info("Analyzing Z0 estimation results.")
        if not hasattr(self, 'z0') or self.z0 is None:
            self.logger.error("Z0 must be computed before analysis. Call fit() first.")
            raise ValueError("Z0 must be computed before analysis. Call fit() first.")
        
        # Create Z0Estimator for analysis
        z0_estimator = Z0Estimator(
            gdf_object=self,
            optimize=self.z0_optimize,
            verbose=self.verbose
        )
        
        # Re-estimate for analysis (this is safe since it's already computed)
        z0_estimator.fit()
        
        # Get detailed info
        analysis_info = z0_estimator.get_estimation_info()
        
        # Create visualization
        z0_estimator.plot_z0_analysis(figsize=figsize)
        
        return analysis_info
    
    def _calculate_all_derivatives(self):
        """Calculate all derivatives and store in params."""
        self.logger.info("Calculating all QGDF derivatives.")
        if not self._fitted:
            self.logger.error("Must fit QGDF before calculating derivatives.")
            raise RuntimeError("Must fit QGDF before calculating derivatives.")

        try:
            # Calculate derivatives using analytical methods
            second_deriv = self._get_qgdf_second_derivative()
            third_deriv = self._get_qgdf_third_derivative()
            fourth_deriv = self._get_qgdf_fourth_derivative()

            # Store in params
            if self.catch:
                self.params.update({
                    'second_derivative': second_deriv.copy(),
                    'third_derivative': third_deriv.copy(),
                    'fourth_derivative': fourth_deriv.copy()
                })
            
            self.logger.info("QGDF derivatives calculated and stored successfully.")
                
        except Exception as e:
            # Log error
            error_msg = f"Derivative calculation failed: {e}"
            self.logger.error(error_msg)
            self.params['errors'].append({
                'method': '_calculate_all_derivatives',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            self.logger.warning(f"Could not calculate derivatives: {e}")
