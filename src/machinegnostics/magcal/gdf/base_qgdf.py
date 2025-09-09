'''
QGDF: Quantifying Global Distribution Functions

Author: Nirmal Parmar
Machine Gnostics
'''
import numpy as np
import warnings
from machinegnostics.magcal.gdf.base_distfunc import BaseDistFuncCompute
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.characteristics import GnosticsCharacteristics

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

    def _compute_qgdf_core(self, S, LB, UB, zi_data=None, zi_eval=None):
        """Core QGDF computation with caching."""
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
        gc = GnosticsCharacteristics(R=R)
        q, q1 = gc._get_q_q1(S=S)
        
        # Calculate fidelities and irrelevances
        fj = gc._fj(q=q, q1=q1)
        hj = gc._hj(q=q, q1=q1)
        
        # Estimate QGDF
        return self._estimate_qgdf_from_moments(fj, hj), fj, hj

    def _estimate_qgdf_from_moments_complex(self, fidelities, irrelevances):
        """Estimate QGDF using complex number approach to handle all cases."""
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
                print(f"Warning: {small_count} very small {name} values detected (< {min_safe_value:.2e})")
            
            if np.any(too_large_mask) and self.verbose:
                large_count = np.sum(too_large_mask)
                print(f"Warning: {large_count} very large {name} values detected (> {max_safe_value:.2e})")
            
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
                print("Warning: Extreme values detected in complex calculation. Using scaled approach.")
            
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
                print(f"Warning: {small_denom_count} very small denominators in complex calculation.")
        
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
                print(f"Warning: {large_count} h_zj values too large for safe squaring. Using approximation.")
            
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
            if self.verbose:
                print("Warning: Very small h_zj values in complex calculation.")
            
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
                    print(f"Warning: Unexpected exception in h_gq calculation ({e}). Using approximation.")
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
            print(f"Info: {complex_count} points have complex intermediate results.")
        
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
    
    # def _estimate_qgdf_from_moments(self, fidelities, irrelevances):
    #     """Main QGDF estimation method with complex number fallback."""
    #     try:
    #         # First try the complex number approach
    #         return self._estimate_qgdf_from_moments_complex(fidelities, irrelevances)
    #     except Exception as e:
    #         # log error
    #         error_msg = f"Exception in complex QGDF estimation: {e}"
    #         if self.verbose:
    #             print(f"Complex method failed: {e}. Using fallback approach.")
    #         self.params['errors'].append({
    #             'method': '_estimate_qgdf_from_moments',
    #             'error': error_msg,
    #             'exception_type': type(e).__name__
    #         })

    #         # Fallback to the robust real-number approach
    #         return self._estimate_qgdf_from_moments_fallback(fidelities, irrelevances)
    
    def _estimate_qgdf_from_moments(self, fidelities, irrelevances):
        """Fallback method using real numbers only."""
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
        """Calculate PDF from fidelities and irrelevances with comprehensive numerical stability."""
        if fidelities is None or irrelevances is None:
            # log error
            error_msg = "Fidelities and irrelevances must be calculated first"
            self.params['errors'].append({
                'method': '_calculate_pdf_from_moments',
                'error': error_msg,
                'exception_type': 'ValueError'
            })
            raise ValueError("Fidelities and irrelevances must be calculated first")
        
        weights = self._computation_cache['weights_normalized'].reshape(-1, 1)
        
        # Comprehensive numerical stability for both large and small values
        max_safe_value = np.sqrt(np.finfo(float).max) / 10
        min_safe_value = np.sqrt(np.finfo(float).eps) * 100
        
        # Safe clipping function for PDF calculations
        def safe_clip_for_pdf(values, name="values"):
            """Safely clip values for PDF calculations."""
            values_magnitude = np.abs(values)
            too_small_mask = values_magnitude < min_safe_value
            too_large_mask = values_magnitude > max_safe_value
            
            if np.any(too_small_mask) and self.verbose:
                print(f"Warning: Very small {name} values in PDF calculation.")
            if np.any(too_large_mask) and self.verbose:
                print(f"Warning: Very large {name} values in PDF calculation.")
            
            # Preserve sign while ensuring safe magnitudes
            values_safe = np.where(too_small_mask, 
                                np.sign(values) * min_safe_value, 
                                values)
            values_safe = np.where(too_large_mask, 
                                np.sign(values_safe) * max_safe_value, 
                                values_safe)
            return values_safe
        
        # Apply comprehensive clipping
        fidelities_safe = safe_clip_for_pdf(fidelities, "fidelity")
        irrelevances_safe = safe_clip_for_pdf(irrelevances, "irrelevance")
        
        # Calculate weighted means with safe values
        mean_fidelity = np.sum(weights * fidelities_safe, axis=0) / np.sum(weights)  # f̄Q
        mean_irrelevance = np.sum(weights * irrelevances_safe, axis=0) / np.sum(weights)  # h̄Q
        
        # Apply safety to means
        mean_fidelity = safe_clip_for_pdf(mean_fidelity, "mean_fidelity")
        mean_irrelevance = safe_clip_for_pdf(mean_irrelevance, "mean_irrelevance")
        
        # Second order moments calculation with enhanced protection
        S_value = self.S_opt if hasattr(self, 'S_opt') else 1.0
        sqrt_max = np.sqrt(max_safe_value)
        
        # Prepare fidelities for squaring with comprehensive protection
        fidelities_for_square = np.where(np.abs(fidelities_safe) > sqrt_max, 
                                    sqrt_max * np.sign(fidelities_safe), 
                                    fidelities_safe)
        
        # Also handle very small values that might cause issues when squared
        fidelities_for_square = np.where(np.abs(fidelities_for_square) < np.sqrt(min_safe_value),
                                    np.sqrt(min_safe_value) * np.sign(fidelities_for_square),
                                    fidelities_for_square)
        
        # Calculate f2s with comprehensive error handling
        try:
            fidelities_squared = fidelities_for_square**2
            f2s = np.sum(weights * (fidelities_squared / S_value), axis=0) / np.sum(weights)
            # Ensure f2s is in safe range
            f2s = np.clip(f2s, min_safe_value, max_safe_value)
        except (OverflowError, FloatingPointError, ZeroDivisionError) as e:
            # log error
            error_msg = f"Exception in f2s calculation: {e}"
            self.params['errors'].append({
                'method': '_calculate_pdf_from_moments',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            if self.verbose:
                print(f"Warning: Exception in f2s calculation ({e}). Using fallback.")
            f2s = np.ones_like(mean_fidelity) * 1.0
        
        # Calculate PDF ratio with comprehensive protection
        mean_fidelity_safe = np.where(np.abs(mean_fidelity) < min_safe_value,
                                    np.sign(mean_fidelity) * min_safe_value, mean_fidelity)
        
        # Calculate ratio with overflow/underflow protection
        try:
            ratio = mean_irrelevance / mean_fidelity_safe
            ratio_magnitude = np.abs(ratio)
            
            # Handle extreme ratios
            ratio_too_large = ratio_magnitude > 10
            ratio_too_small = ratio_magnitude < min_safe_value
            
            ratio_for_square = np.where(ratio_too_large, 10 * np.tanh(ratio / 10), ratio)
            ratio_for_square = np.where(ratio_too_small, np.sign(ratio) * min_safe_value, ratio_for_square)
            
            # Square the ratio with protection
            if np.any(np.abs(ratio_for_square) > sqrt_max):
                ratio_squared = np.clip(ratio_for_square**2, min_safe_value, max_safe_value)
            else:
                ratio_squared = np.maximum(ratio_for_square**2, min_safe_value)
                
        except (OverflowError, FloatingPointError, ZeroDivisionError) as e:
            # log error
            error_msg = f"Exception in ratio calculation: {e}"
            self.params['errors'].append({
                'method': '_calculate_pdf_from_moments',
                'error': error_msg,
                'exception_type': type(e).__name__
            })

            if self.verbose:
                print(f"Warning: Exception in ratio calculation ({e}). Using fallback.")
            ratio_squared = np.ones_like(mean_fidelity) * 0.5
        
        # Calculate the PDF term with protection
        pdf_term = np.maximum(1 - ratio_squared, min_safe_value)  # Ensure positive and non-zero
        
        # Apply the scaling factor with comprehensive protection
        try:
            pdf_values = (1 / S_value) * pdf_term * f2s
            # Final comprehensive clipping
            pdf_values = np.clip(pdf_values, min_safe_value, max_safe_value)
        except (OverflowError, FloatingPointError, ZeroDivisionError) as e:
            # log error
            error_msg = f"Exception in final PDF calculation: {e}"
            self.params['errors'].append({
                'method': '_calculate_pdf_from_moments',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            if self.verbose:
                print(f"Warning: Exception in final PDF calculation ({e}). Using clipped result.")
            pdf_values = np.clip((1 / S_value) * pdf_term, min_safe_value, max_safe_value / 10)
        
        return pdf_values.flatten()

    def _calculate_final_results(self):
        """Calculate final QGDF and PDF with optimized parameters."""
        # Convert to infinite domain
        # zi_n = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        zi_d = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        self.zi = zi_d

        # Calculate QGDF and get moments
        qgdf_values, fi, hi = self._compute_qgdf_core(self.S_opt, self.LB_opt, self.UB_opt)

        # Store for derivative calculations
        self.fi = fi
        self.hi = hi
        self.qgdf = qgdf_values
        self.pdf = self._calculate_pdf_from_moments(fi, hi)
        
        if self.catch:
            self.params.update({
                'qgdf': self.qgdf.copy(),
                'pdf': self.pdf.copy(),
                'zi': self.zi.copy()
            })

    def _generate_smooth_curves(self):
        """Generate smooth curves for plotting and analysis."""
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
            
            if self.verbose:
                print(f"Generated smooth curves with {self.n_points} points.")
                
        except Exception as e:
            # Log the error
            error_msg = f"Could not generate smooth curves: {e}"
            self.params['errors'].append({
                'method': '_generate_smooth_curves',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            if self.verbose:
                print(f"Warning: Could not generate smooth curves: {e}")
            # Create fallback points using original data
            self.qgdf_points = self.qgdf.copy() if hasattr(self, 'qgdf') else None
            self.pdf_points = self.pdf.copy() if hasattr(self, 'pdf') else None
            self._computation_cache['smooth_curves_generated'] = False

    def _get_results(self)-> dict:
        """Return fitting results."""
        if not self._fitted:
            error_msg = "Must fit QGDF before getting results."
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
        import matplotlib.pyplot as plt

        if plot_smooth and (len(self.data) > self.max_data_size) and self.verbose:
            print(f"Warning: Given data size ({len(self.data)}) exceeds max_data_size ({self.max_data_size}). For optimal compute performance, set 'plot_smooth=False', or 'max_data_size' to a larger value whichever is appropriate.")

        if not self.catch:
            print("Plot is not available with argument catch=False")
            return
        
        if not self._fitted:
            raise RuntimeError("Must fit QGDF before plotting.")
        
        # Validate plot parameter
        if plot not in ['gdf', 'pdf', 'both']:
            raise ValueError("plot parameter must be 'gdf', 'pdf', or 'both'")
        
        # Check data availability
        if plot in ['gdf', 'both'] and self.params.get('qgdf') is None:
            raise ValueError("QGDF must be calculated before plotting GDF")
        if plot in ['pdf', 'both'] and self.params.get('pdf') is None:
            raise ValueError("PDF must be calculated before plotting PDF")
        
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
        """Calculate second derivative of QGDF from stored fidelities and irrelevances."""
        if self.fi is None or self.hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before second derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # Moment calculations using QGDF's fj and hj
        f1 = np.sum(weights * self.fi, axis=0) / np.sum(weights)  # f̄Q
        h1 = np.sum(weights * self.hi, axis=0) / np.sum(weights)  # h̄Q
        f2 = np.sum(weights * self.fi**2, axis=0) / np.sum(weights)
        f3 = np.sum(weights * self.fi**3, axis=0) / np.sum(weights)
        fh = np.sum(weights * self.fi * self.hi, axis=0) / np.sum(weights)
        fh2 = np.sum(weights * self.fi * self.hi**2, axis=0) / np.sum(weights)
        f2h = np.sum(weights * self.fi**2 * self.hi, axis=0) / np.sum(weights)
        
        # QGDF uses different base equation: QGDF = (1 - h̄Q/f̄Q) / 2
        # First derivative: dQG/dZ0 = (1/SZ0) * (1 - (h̄Q)²/(f̄Q)²)
        # For second derivative, we need to differentiate this further
        
        eps = np.finfo(float).eps
        f1_safe = np.where(f1 == 0, eps, f1)
        
        # Calculate derivatives of the ratio h̄Q/f̄Q
        # d(h̄Q)/dz = -f2*S (negative of second moment scaled)
        # d(f̄Q)/dz = fh*S (cross moment scaled)
        
        dh1 = -f2 * self.S_opt
        df1 = fh * self.S_opt
        
        # For the term (h̄Q)²/(f̄Q)²
        ratio = h1 / f1_safe
        ratio_squared = ratio**2
        
        # d/dz[(h̄Q)²/(f̄Q)²] = 2*(h̄Q/f̄Q) * d/dz(h̄Q/f̄Q)
        # d/dz(h̄Q/f̄Q) = (f̄Q*dh̄Q - h̄Q*df̄Q) / (f̄Q)²
        ratio_deriv = (f1_safe * dh1 - h1 * df1) / (f1_safe**2)
        ratio_squared_deriv = 2 * ratio * ratio_deriv
        
        # The main term is (1 - (h̄Q)²/(f̄Q)²)
        # Its derivative is -ratio_squared_deriv
        main_term_deriv = -ratio_squared_deriv
        
        # Second derivative needs the derivative of main_term_deriv
        # This involves higher order moments
        d2h1 = -2 * f2h * self.S_opt  # second derivative of h̄Q
        d2f1 = (-f3 + fh2) * self.S_opt  # second derivative of f̄Q
        
        # Second derivative of the ratio using quotient rule repeatedly
        ratio_second_deriv = ((f1_safe * d2h1 - h1 * d2f1) * (f1_safe**2) - 
                                (f1_safe * dh1 - h1 * df1) * 2 * f1_safe * df1) / (f1_safe**4)
        
        ratio_squared_second_deriv = (2 * ratio_deriv**2 + 2 * ratio * ratio_second_deriv)
        
        main_term_second_deriv = -ratio_squared_second_deriv
        
        # Apply scaling factor (1/S)
        second_derivative = (1 / self.S_opt) * main_term_second_deriv
        
        return second_derivative.flatten()

    def _get_qgdf_third_derivative(self):
        """Calculate third derivative of QGDF from stored fidelities and irrelevances."""
        if self.fi is None or self.hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before third derivative estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        # All required moments for QGDF
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
        
        eps = np.finfo(float).eps
        f1_safe = np.where(f1 == 0, eps, f1)
        
        # Derivative calculations for QGDF
        dh1 = -f2 * self.S_opt
        df1 = fh * self.S_opt
        d2h1 = -2 * f2h * self.S_opt
        d2f1 = (-f3 + fh2) * self.S_opt
        d3h1 = -3 * f3h * self.S_opt
        d3f1 = (-f4 + 2 * f2h2) * self.S_opt
        
        # Calculate third derivative of the ratio (h̄Q/f̄Q)²
        ratio = h1 / f1_safe
        
        # First derivative of ratio
        ratio_deriv = (f1_safe * dh1 - h1 * df1) / (f1_safe**2)
        
        # Second derivative of ratio
        numerator_2nd = (f1_safe * d2h1 - h1 * d2f1) * (f1_safe**2) - (f1_safe * dh1 - h1 * df1) * 2 * f1_safe * df1
        ratio_second_deriv = numerator_2nd / (f1_safe**4)
        
        # Third derivative of ratio
        term1_3rd = f1_safe * d3h1 - h1 * d3f1
        term2_3rd = 2 * (df1 * d2h1 - dh1 * d2f1)
        term3_3rd = 6 * (f1_safe * dh1 - h1 * df1) * (df1**2) / f1_safe
        
        ratio_third_deriv = (term1_3rd * (f1_safe**2) - term2_3rd * (f1_safe**3) - term3_3rd * (f1_safe**2)) / (f1_safe**6)
        
        # Third derivative of ratio²
        ratio_squared_third_deriv = (6 * ratio_deriv * ratio_second_deriv + 2 * ratio * ratio_third_deriv)
        
        # Main term third derivative
        main_term_third_deriv = -ratio_squared_third_deriv
        
        # Apply scaling factor
        third_derivative = (1 / self.S_opt) * main_term_third_deriv
        
        return third_derivative.flatten()

    def _get_qgdf_fourth_derivative(self):
        """Calculate fourth derivative of QGDF using numerical differentiation."""
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
        third_plus = self._get_qgdf_third_derivative()
        
        # Calculate third derivative at zi - dz  
        self.zi = zi_minus
        self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        third_minus = self._get_qgdf_third_derivative()
        
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
        # Use given zi if provided, else use self.zi
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
        
        # Store fidelities and irrelevances (using QGDF methods)
        self.fi = gc._fj(q=q, q1=q1)  # Note: using _fj for QGDF
        self.hi = gc._hj(q=q, q1=q1)  # Note: using _hj for QGDF


    def _fit_qgdf(self, plot: bool = False):
        try:
            if self.verbose:
                print("Starting QGDF fitting process...")
            
            # Step 1: Data preprocessing
            self.data = np.sort(self.data)
            self._estimate_data_bounds()
            self._transform_data_to_standard_domain()
            self._estimate_weights()
            
            # Step 2: Bounds estimation
            self._estimate_initial_probable_bounds()
            self._generate_evaluation_points()
            
            # Step 3: Get distribution function values for optimization
            self.df_values = self._get_distribution_function_values(use_wedf=self.wedf)
            
            # Step 4: Parameter optimization
            self._determine_optimization_strategy(egdf=False)  # NOTE for QGDF egdf is False
            
            # Step 5: Calculate final QGDF and PDF
            self._calculate_final_results()
            
            # Step 6: Generate smooth curves for plotting and analysis
            self._generate_smooth_curves()
            
            # Step 7: Transform bounds back to original domain
            self._transform_bounds_to_original_domain()
            
            # Mark as fitted (Step 8 is now optional via marginal_analysis())
            self._fitted = True

            if self.verbose:
                print("QGDF fitting completed successfully.")

            if plot:
                self._plot()

            # clean up computation cache
            if self.flush:  
                self._cleanup_computation_cache()
                
        except Exception as e:
            error_msg = f"QGDF fitting failed: {e}"
            self.params['errors'].append({
                'method': '_fit_QGDF',
                'error': error_msg,
                'exception_type': type(e).__name__
            })
            
            if self.verbose:
                print(f"Error during QGDF fitting: {e}")
            raise e


# NOTE
# PDF calculated from fi and hi derivative base logic
# this below are higher derivatives from the same logic

    # def _get_qgdf_second_derivative(self):
    #     """Calculate second derivative of QGDF from stored fidelities and irrelevances."""
    #     if self.fi is None or self.hi is None:
    #         raise ValueError("Fidelities and irrelevances must be calculated before second derivative estimation.")
        
    #     weights = self.weights.reshape(-1, 1)
        
    #     # Moment calculations using QGDF's fj and hj
    #     f1 = np.sum(weights * self.fi, axis=0) / np.sum(weights)  # f̄Q
    #     h1 = np.sum(weights * self.hi, axis=0) / np.sum(weights)  # h̄Q
    #     f2 = np.sum(weights * self.fi**2, axis=0) / np.sum(weights)
    #     f3 = np.sum(weights * self.fi**3, axis=0) / np.sum(weights)
    #     h2 = np.sum(weights * self.hi**2, axis=0) / np.sum(weights)
    #     h3 = np.sum(weights * self.hi**3, axis=0) / np.sum(weights)
    #     fh = np.sum(weights * self.fi * self.hi, axis=0) / np.sum(weights)
    #     fh2 = np.sum(weights * self.fi * self.hi**2, axis=0) / np.sum(weights)
    #     f2h = np.sum(weights * self.fi**2 * self.hi, axis=0) / np.sum(weights)
        
    #     # Calculate Nj and its derivatives
    #     eps = np.finfo(float).eps
    #     f_safe = np.where(np.abs(self.fi) < eps, eps, self.fi)
    #     f_inv_squared = np.sum(weights * (1 / f_safe**2), axis=0) / np.sum(weights)
    #     Nj = f_inv_squared + h2
    #     Nj = np.where(Nj == 0, eps, Nj)
        
    #     # QGDF second derivative components
    #     # d²(f̄Q)/dz² = 2*f2h*S (second derivative of mean fidelity)
    #     d2f1 = 2 * f2h * self.S_opt
    #     # d²(h̄Q)/dz² = 2*fh2*S (second derivative of mean irrelevance) 
    #     d2h1 = 2 * fh2 * self.S_opt
        
    #     # Second derivative of F2, H2, FH terms
    #     d2f2s = 2 * f3 * self.S_opt  # d²F2/dz²
    #     d2h2s = 2 * h3 * self.S_opt  # d²H2/dz²
    #     d2fhs = (f3 + fh2) * self.S_opt  # d²FH/dz²
        
    #     # Second derivative of Nj
    #     d2Nj = -4 * f_inv_squared / f1 * f2h * self.S_opt + 2 * h3 * self.S_opt
        
    #     # Calculate the main numerator and its second derivative
    #     numerator = f2 - h2 + f1 * h1 * fh
    #     d2_numerator = d2f2s - d2h2s + d2f1 * h1 * fh + f1 * d2h1 * fh + f1 * h1 * d2fhs
        
    #     # Second derivative formula for QGDF
    #     term1 = d2_numerator / (2 * Nj**2)
    #     term2 = -2 * numerator * d2Nj / (2 * Nj**3)
        
    #     second_derivative = (1 / self.S_opt**2) * (term1 + term2)
        
    #     return second_derivative.flatten()

    # def _get_qgdf_third_derivative(self):
    #     """Calculate third derivative of QGDF from stored fidelities and irrelevances."""
    #     if self.fi is None or self.hi is None:
    #         raise ValueError("Fidelities and irrelevances must be calculated before third derivative estimation.")
        
    #     weights = self.weights.reshape(-1, 1)
        
    #     # All required moments for QGDF
    #     f1 = np.sum(weights * self.fi, axis=0) / np.sum(weights)
    #     h1 = np.sum(weights * self.hi, axis=0) / np.sum(weights)
    #     f2 = np.sum(weights * self.fi**2, axis=0) / np.sum(weights)
    #     f3 = np.sum(weights * self.fi**3, axis=0) / np.sum(weights)
    #     f4 = np.sum(weights * self.fi**4, axis=0) / np.sum(weights)
    #     h2 = np.sum(weights * self.hi**2, axis=0) / np.sum(weights)
    #     h3 = np.sum(weights * self.hi**3, axis=0) / np.sum(weights)
    #     h4 = np.sum(weights * self.hi**4, axis=0) / np.sum(weights)
    #     fh = np.sum(weights * self.fi * self.hi, axis=0) / np.sum(weights)
    #     fh2 = np.sum(weights * self.fi * self.hi**2, axis=0) / np.sum(weights)
    #     f2h = np.sum(weights * self.fi**2 * self.hi, axis=0) / np.sum(weights)
    #     f2h2 = np.sum(weights * self.fi**2 * self.hi**2, axis=0) / np.sum(weights)
    #     f3h = np.sum(weights * self.fi**3 * self.hi, axis=0) / np.sum(weights)
    #     fh3 = np.sum(weights * self.fi * self.hi**3, axis=0) / np.sum(weights)
        
    #     eps = np.finfo(float).eps
    #     f_safe = np.where(np.abs(self.fi) < eps, eps, self.fi)
    #     f_inv_squared = np.sum(weights * (1 / f_safe**2), axis=0) / np.sum(weights)
    #     Nj = f_inv_squared + h2
    #     Nj = np.where(Nj == 0, eps, Nj)
        
    #     # Third order derivative calculations for QGDF
    #     d3f1 = 3 * f3h * self.S_opt  # third derivative of f̄Q
    #     d3h1 = 3 * fh3 * self.S_opt  # third derivative of h̄Q
        
    #     # Third derivatives of second moments
    #     d3f2s = 3 * f4 * self.S_opt
    #     d3h2s = 3 * h4 * self.S_opt  
    #     d3fhs = (f4 + 2 * f2h2) * self.S_opt
        
    #     # Third derivative of Nj (complex calculation)
    #     d3Nj = -6 * f_inv_squared / f1**2 * (f2h * self.S_opt)**2 + 3 * h4 * self.S_opt
        
    #     # Main numerator and its derivatives
    #     numerator = f2 - h2 + f1 * h1 * fh
    #     d3_numerator = d3f2s - d3h2s + d3f1 * h1 * fh + f1 * d3h1 * fh + f1 * h1 * d3fhs
        
    #     # Third derivative formula for QGDF
    #     term1 = d3_numerator / (2 * Nj**2)
    #     term2 = -6 * numerator * d3Nj / (2 * Nj**4)
        
    #     third_derivative = (1 / self.S_opt**3) * (term1 + term2)
        
    #     return third_derivative.flatten()

    # def _get_qgdf_fourth_derivative(self):
    #     """Calculate fourth derivative of QGDF using numerical differentiation."""
    #     if self.fi is None or self.hi is None:
    #         raise ValueError("Fidelities and irrelevances must be calculated before fourth derivative estimation.")
        
    #     # For fourth derivative, use numerical differentiation as it's complex
    #     dz = 1e-7
        
    #     # Get third derivatives at slightly shifted points
    #     zi_plus = self.zi + dz
    #     zi_minus = self.zi - dz
        
    #     # Store original zi
    #     original_zi = self.zi.copy()
        
    #     # Calculate third derivative at zi + dz
    #     self.zi = zi_plus
    #     self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
    #     third_plus = self._get_qgdf_third_derivative()
        
    #     # Calculate third derivative at zi - dz  
    #     self.zi = zi_minus
    #     self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
    #     third_minus = self._get_qgdf_third_derivative()
        
    #     # Restore original zi and recalculate fi, hi
    #     self.zi = original_zi
    #     self._calculate_fidelities_irrelevances_at_given_zi(self.zi)
        
    #     # Numerical derivative
    #     fourth_derivative = (third_plus - third_minus) / (2 * dz) * self.zi
        
    #     return fourth_derivative.flatten()

    # def _calculate_fidelities_irrelevances_at_given_zi(self, zi):
    #     """Helper method to recalculate fidelities and irrelevances for current zi."""
    #     # Convert to infinite domain
    #     zi_n = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
    #     # Use given zi if provided, else use self.zi
    #     if zi is None:
    #         zi_d = self.zi
    #     else:
    #         zi_d = zi

    #     # Calculate R matrix
    #     eps = np.finfo(float).eps
    #     R = zi_n.reshape(-1, 1) / (zi_d + eps).reshape(1, -1)

    #     # Get characteristics
    #     gc = GnosticsCharacteristics(R=R)
    #     q, q1 = gc._get_q_q1(S=self.S_opt)
        
    #     # Store fidelities and irrelevances (using QGDF methods)
    #     self.fi = gc._fj(q=q, q1=q1)  # Note: using _fj for QGDF
    #     self.hi = gc._hj(q=q, q1=q1)  # Note: using _hj for QGDF