"""
base class for EGDF
EGDF - Estimating Global Distribution Function.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
import warnings
from scipy.optimize import minimize
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.gdf.base_df import BaseDistFunc
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.gdf.wedf import WEDF
from machinegnostics.magcal.mg_weights import GnosticsWeights


class BaseEGDF(BaseDistFunc):
    '''
    Base class for EGDF (Estimating Global Distribution Function).
    
    '''
    def __init__(self,
                data: np.ndarray,
                DLB: float = None,
                DUB: float = None,
                LB: float = None,
                UB: float = None,
                S = 'auto',
                tolerance: float = 1e-3,
                data_form: str = 'a',
                n_points: int = 500,
                homogeneous: bool = True,
                catch: bool = True,
                weights: np.ndarray = None,
                wedf: bool = True,
                opt_method: str = 'L-BFGS-B',
                verbose: bool = False,
                max_data_size: int = 100):
        """Initialize the EGDF class."""
        
        self.data = data
        self.DLB = DLB
        self.DUB = DUB
        self.LB = LB
        self.UB = UB
        self.S = S
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
        self.params = {}

        # Validation
        self._validate_inputs()

        # Store parameters
        if self.catch:
            self._store_initial_params()

    #1
    def _validate_inputs(self):
        """Validate input arguments."""
        if not isinstance(self.data, np.ndarray):
            raise ValueError("Data must be a numpy array.")
        
        for bound, name in [(self.DLB, 'DLB'), (self.DUB, 'DUB'), (self.LB, 'LB'), (self.UB, 'UB')]:
            if bound is not None and not isinstance(bound, (int, float)):
                raise ValueError(f"{name} must be a numeric value or None.")
        
        if not isinstance(self.S, (int, float, str)):
            raise ValueError("S must be a numeric positive value or 'auto'.")
        
        if not isinstance(self.tolerance, (int, float)):
            raise ValueError("Tolerance must be a numeric value.")
        
        if self.data_form not in ['a', 'm']:
            raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative.")
        
        if not isinstance(self.n_points, int) or self.n_points <= 0:
            raise ValueError("n_points must be a positive integer.")
        
        if not isinstance(self.homogeneous, bool):
            raise ValueError("homogeneous must be a boolean value.")
        
        if not isinstance(self.catch, bool):
            raise ValueError("catch must be a boolean value.")

        if self.weights is not None:
            if not isinstance(self.weights, np.ndarray):
                raise ValueError("weights must be a numpy array.")
            if len(self.weights) != len(self.data):
                raise ValueError("Weights must have the same length as data.")
        
        if not isinstance(self.wedf, bool):
            raise ValueError("wedf must be a boolean value.")
        
        if self.opt_method not in ['L-BFGS-B', 'Nelder-Mead', 'Powell']:
            raise ValueError("opt_method must be one of 'L-BFGS-B', 'Nelder-Mead', or 'Powell'. OR a appropriate method from scipy.optimize.minimize")

        if not isinstance(self.verbose, bool):
            raise ValueError("verbose must be a boolean value.")
        

    #2
    def _store_initial_params(self):
        """Store initial parameters if catch is True."""
        self.params.update({
            'data': self.data,
            'DLB': self.DLB,
            'DUB': self.DUB,
            'LB': self.LB,
            'UB': self.UB,
            'S': self.S,
            'tolerance': self.tolerance,
            'data_form': self.data_form,
            'n_points': self.n_points,
            'homogeneous': self.homogeneous,
            'weights': self.weights
        })

    #3
    def _estimate_weights(self):
        """Estimate weights for the EGDF."""
        if self.weights is None:
            self.weights = np.ones_like(self.data)
        else:
            self.weights = np.asarray(self.weights)
            if len(self.weights) != len(self.data):
                raise ValueError("weights must have the same length as data")
        
        # Normalize weights to sum to n (number of data points)
        self.weights = self.weights / np.sum(self.weights) * len(self.weights)

        # Apply homogenization if needed
        if not self.homogeneous:
            gw = GnosticsWeights()
            self.gweights = gw._get_gnostic_weights(self.z)
            self.weights = self.gweights * self.weights
        else:
            self.weights = self.weights

        if self.catch:
            self.params['weights'] = self.weights

    #4
    def _estimate_data_bounds(self):
        """Estimate data bounds (DLB and DUB)."""
        if self.DLB is None:
            self.DLB = np.min(self.data)
        if self.DUB is None:
            self.DUB = np.max(self.data)
        
        if self.catch:
            self.params['DLB'] = self.DLB
            self.params['DUB'] = self.DUB

    #5
    def _initial_probable_bounds_estimate(self):
        """Estimate initial probable bounds (LB and UB)."""
        # Only estimate LB if it's not provided
        if self.LB is None:
            if self.data_form == 'a':
                pad = (self.DUB - self.DLB) / 2
                lb_raw = self.DLB - pad
                self.LB_init = DataConversion._convert_az(lb_raw, self.DLB, self.DUB)
            elif self.data_form == 'm':
                lb_raw = self.DLB / np.sqrt(self.DUB / self.DLB)
                self.LB_init = DataConversion._convert_mz(lb_raw, self.DLB, self.DUB)
        else:
            # Convert provided LB to appropriate form
            if self.data_form == 'a':
                self.LB_init = DataConversion._convert_az(self.LB, self.DLB, self.DUB)
            else:
                self.LB_init = DataConversion._convert_mz(self.LB, self.DLB, self.DUB)

        # Only estimate UB if it's not provided
        if self.UB is None:
            if self.data_form == 'a':
                pad = (self.DUB - self.DLB) / 2
                ub_raw = self.DUB + pad
                self.UB_init = DataConversion._convert_az(ub_raw, self.DLB, self.DUB)
            elif self.data_form == 'm':
                ub_raw = self.DUB * np.sqrt(self.DUB / self.DLB)
                self.UB_init = DataConversion._convert_mz(ub_raw, self.DLB, self.DUB)
        else:
            # Convert provided UB to appropriate form
            if self.data_form == 'a':
                self.UB_init = DataConversion._convert_az(self.UB, self.DLB, self.DUB)
            else:
                self.UB_init = DataConversion._convert_mz(self.UB, self.DLB, self.DUB)

        if self.catch:
            self.params['LB_init'] = self.LB_init
            self.params['UB_init'] = self.UB_init

    #6
    def _transform_input(self, smooth=False):
        """Transform input data to standard domain."""
        dc = DataConversion()

        if self.data_form == 'a':
            self.z = dc._convert_az(self.data, self.DLB, self.DUB)
        elif self.data_form == 'm':
            self.z = dc._convert_mz(self.data, self.DLB, self.DUB)

        # # Apply homogenization if needed
        # if not self.homogeneous:
        #     gw = GnosticsWeights()
        #     self.gweights = gw._get_gnostic_weights(self.z)
        #     self.sample = self.z * self.gweights * self.weights
        # else:
        #     self.sample = self.z * self.weights

        
        # Generate smooth points in data domain
        self.di_points_n = np.linspace(self.DLB, self.DUB, self.n_points)
        
        # Transform to z domain
        dc = DataConversion()
        if self.data_form == 'a':
            self.z_points_n = dc._convert_az(self.di_points_n, self.DLB, self.DUB)
        else:
            self.z_points_n = dc._convert_mz(self.di_points_n, self.DLB, self.DUB)

        # sample
        if smooth:
            self.sample = self.z_points_n
        else:
            self.sample = self.z

        if self.catch:
            self.params.update({'z': self.z, 'sample': self.sample})
            self.params['z_points_n'] = self.z_points_n
            self.params['di_points_n'] = self.di_points_n

    #7
    def _get_df(self, smooth=False, wedf: bool = True):
        """
        Get WEDF values for optimization.
        Weighted Emperical Distribution Function (WEDF) is used to optimize the EGDF.
        """
        if wedf:
            wedf_ = WEDF(self.data, weights=self.weights, data_lb=self.DLB, data_ub=self.DUB)
            if smooth:
                df_values = wedf_.fit(self.di_points_n)
            else:
                df_values = wedf_.fit(self.data)
    
            if self.catch:
                self.params['wedf'] = df_values
            
            if self.verbose:
                print("WEDF values computed.")
            return df_values
        else:
            # FIX: Swap the logic - use len(data) for non-smooth, n_points for smooth
            if smooth:
                df_values = self._get_ks_points(self.n_points)
            else:
                df_values = self._get_ks_points(len(self.data))
    
            if self.catch:
                self.params['ksdf'] = df_values

            if self.verbose:
                print("KSD values computed.")
            return df_values
    
    #8
    def _optimize_parameters(self):
        """Optimize S, LB, UB based on what's provided."""
        # Case 1: S='auto' and LB/UB are None - optimize all parameters
        if (isinstance(self.S, str) and self.S.lower() == 'auto') and self.LB is None and self.UB is None:
            if self.verbose:
                print("Optimizing all parameters: S, LB, UB.")
            self.S_opt, self.LB_opt, self.UB_opt = self._optimize_all_parameters()          

        # Case 2: LB and UB provided, S='auto' - optimize only S
        elif (self.LB is not None and self.UB is not None and 
              isinstance(self.S, str) and self.S.lower() == 'auto'):
            if self.verbose:
                print(f"Optimizing S with provided LB: {self.LB_opt}, UB: {self.UB_opt}.")

            self.LB_opt = self.LB
            self.UB_opt = self.UB
            self.S_opt = self._optimize_s_only(self.LB_opt, self.UB_opt)
        
        # Case 3: S provided, LB/UB not provided - optimize LB, UB
        elif (not isinstance(self.S, str) and 
              (self.LB is None or self.UB is None)):
            if self.verbose:
                print(f"Optimizing bounds with provided S: {self.S}.")

            self.S_opt = self.S
            self.S_opt, self.LB_opt, self.UB_opt = self._optimize_bounds_only(self.S_opt)
        
        # Case 4: All parameters provided - use as is
        else:
            self.S_opt = self.S if not isinstance(self.S, str) else 1.0
            self.LB_opt = self.LB
            self.UB_opt = self.UB
            if self.verbose:
                print(f"Using provided parameters: S: {self.S_opt}, LB: {self.LB_opt}, UB: {self.UB_opt}.")

    #9
    # with custom log transformation
    def _optimize_all_parameters(self):
        """Optimize with normalized parameter space [0,1] for each parameter."""
        # Parameter bounds for optimization
        S_MIN, S_MAX = 0.05, 100.0
        LB_MIN, LB_MAX = 1e-6, np.exp(-1.00001)
        UB_MIN, UB_MAX = np.exp(1.00001), 1e6
        
        def normalize_params(s, lb, ub):
            """Normalize parameters to [0,1] space."""
            s_norm = (s - S_MIN) / (S_MAX - S_MIN)
            lb_norm = (lb - LB_MIN) / (LB_MAX - LB_MIN)
            ub_norm = (ub - UB_MIN) / (UB_MAX - UB_MIN)
            return s_norm, lb_norm, ub_norm
        
        def denormalize_params(s_norm, lb_norm, ub_norm):
            """Denormalize parameters from [0,1] space."""
            s = S_MIN + s_norm * (S_MAX - S_MIN)
            lb = LB_MIN + lb_norm * (LB_MAX - LB_MIN)
            ub = UB_MIN + ub_norm * (UB_MAX - UB_MIN)
            return s, lb, ub
        
        def loss_function(norm_params):
            s_norm, lb_norm, ub_norm = norm_params
            try:
                s, lb, ub = denormalize_params(s_norm, lb_norm, ub_norm)
                
                # Ensure valid parameter ranges
                if s <= 0 or ub <= lb:
                    return 1e6
                    
                egdf_values = self._compute_egdf(s, lb, ub)
                
                # Primary loss
                diff = np.mean(np.abs(egdf_values - self.df_values) * self.weights)
                
                # Regularization in normalized space (prefer smaller normalized values)
                s_reg = s_norm**2  # Prefer smaller S
                lb_reg = (lb_norm)**2 # Prefer smaller |LB|
                ub_reg = (ub_norm)**2  # Prefer smaller UB

                total_loss = diff + s_reg + lb_reg + ub_reg
                
                if self.verbose:
                    # Print detailed loss information
                    print(f"Loss: {diff:.6f}, Total: {total_loss:.6f}, S: {s:.3f}, LB: {lb:.3f}, UB: {ub:.3f}")
                return total_loss
                
            except Exception as e:
                return 1e6

        # Initial values (normalized)
        s_init = 0.1  # Default S value
        lb_init = self.LB_init if self.LB_init is not None else LB_MIN
        ub_init = self.UB_init if self.UB_init is not None else UB_MAX

        s_norm_init, lb_norm_init, ub_norm_init = normalize_params(s_init, lb_init, ub_init)
        initial_params = [s_norm_init, lb_norm_init, ub_norm_init]
        
        # All bounds are [0, 1] in normalized space
        norm_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

        try:
            result = minimize(loss_function, 
                              initial_params, 
                              method=self.opt_method, 
                              bounds=norm_bounds,
                              options={'maxiter': 10000, 'ftol': self.tolerance}, 
                              tol=self.tolerance)
            
            s_opt, lb_opt, ub_opt = denormalize_params(*result.x)
            
            if lb_opt >= ub_opt:
                print("Warning: Optimized LB >= UB, using initial values")
                return s_init, lb_init, ub_init
                
            return s_opt, lb_opt, ub_opt
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return s_init, lb_init, ub_init
            

    #10
    def _optimize_s_only(self, lb, ub):
        """Optimize only S parameter for given LB and UB."""
        def loss_function(s):
            try:
                egdf_values = self._compute_egdf(s[0], lb, ub)
                diff = np.mean(np.abs(egdf_values - self.df_values) * self.weights)
                if self.verbose:
                    # Print detailed loss information
                    print(f"Loss: {diff:.6f}, S: {s[0]:.3f}, LB: {lb:.3f}, UB: {ub:.3f}")
                return diff
            except Exception:
                return 1e6
        
        try:
            result = minimize(loss_function, [1.0], bounds=[(0.05, 100.0)],
                            method=self.opt_method, options={'maxiter': 1000})
            return result.x[0]
        except Exception:
            return 1.0


    #11
    def _optimize_bounds_only(self, s):
        """Optimize only LB and UB for given S with normalized parameter space [0,1]."""
        # Parameter bounds for optimization
        LB_MIN, LB_MAX = 1e-6, np.exp(-1.00001)
        UB_MIN, UB_MAX = np.exp(1.00001), 1e6
        
        def normalize_bounds(lb, ub):
            """Normalize LB and UB to [0,1] space."""
            lb_norm = (lb - LB_MIN) / (LB_MAX - LB_MIN)
            ub_norm = (ub - UB_MIN) / (UB_MAX - UB_MIN)
            return lb_norm, ub_norm
        
        def denormalize_bounds(lb_norm, ub_norm):
            """Denormalize LB and UB from [0,1] space."""
            lb = LB_MIN + lb_norm * (LB_MAX - LB_MIN)
            ub = UB_MIN + ub_norm * (UB_MAX - UB_MIN)
            return lb, ub
        
        def loss_function(norm_params):
            lb_norm, ub_norm = norm_params
            try:
                lb, ub = denormalize_bounds(lb_norm, ub_norm)
                
                # Ensure valid parameter ranges
                if lb <= 0 or ub <= lb:
                    return 1e6
                    
                egdf_values = self._compute_egdf(s, lb, ub)
                
                # Primary loss
                diff = np.mean(np.abs(egdf_values - self.df_values) * self.weights)
                
                # Regularization in normalized space (prefer smaller normalized values)
                lb_reg = lb_norm**2  # Prefer smaller |LB|
                ub_reg = ub_norm**2  # Prefer smaller UB

                total_loss = diff + lb_reg + ub_reg
                
                if self.verbose:
                    # Print detailed loss information
                    print(f"Loss: {diff:.6f}, Total: {total_loss:.6f}, S: {s:.3f}, LB: {lb:.6f}, UB: {ub:.3f}")
                return total_loss
                
            except Exception as e:
                print(f"Error in loss function: {e}")
                return 1e6
        
        # Initial values
        lb_init = self.LB_init if self.LB_init is not None else LB_MIN
        ub_init = self.UB_init if self.UB_init is not None else UB_MIN

        # Ensure initial values are within bounds
        lb_init = np.clip(lb_init, LB_MIN, LB_MAX)
        ub_init = np.clip(ub_init, UB_MIN, UB_MAX)
        
        # Ensure lb < ub
        if lb_init >= ub_init:
            lb_init = LB_MIN / 10
            ub_init = UB_MIN * 10
        
        # Normalize initial values
        lb_norm_init, ub_norm_init = normalize_bounds(lb_init, ub_init)
        initial_params = [lb_norm_init, ub_norm_init]
        
        # All bounds are [0, 1] in normalized space
        norm_bounds = [(0.0, 1.0), (0.0, 1.0)]
        
        try:
            result = minimize(loss_function, 
                              initial_params, 
                              method=self.opt_method,
                              bounds=norm_bounds,
                              options={'maxiter': 10000, 'ftol': self.tolerance}, 
                              tol=self.tolerance)
            
            lb_opt, ub_opt = denormalize_bounds(*result.x)
            
            # Validate final results
            if lb_opt >= ub_opt:
                print("Warning: Optimized LB >= UB, using initial values")
                return s, lb_init, ub_init
                
            return s, lb_opt, ub_opt
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return s, self.LB, self.UB
        
    #12
    def _compute_egdf(self, S, LB, UB):
        """Core EGDF computation logic."""
        # Convert to infinite domain
        zi_n = DataConversion._convert_fininf(self.z, LB, UB)
        zi_d = DataConversion._convert_fininf(self.sample, LB, UB)
        
        # Calculate R matrix
        eps = np.finfo(float).eps
        R = zi_n.reshape(-1, 1) / (zi_d + eps).reshape(1, -1)

        # Get characteristics
        gc = GnosticsCharacteristics(R=R)
        q, q1 = gc._get_q_q1(S=S)
        
        # Calculate fidelities and irrelevances
        fi = gc._fi(q=q, q1=q1)
        hi = gc._hi(q=q, q1=q1)
        
        # Estimate EGDF
        return self._estimate_egdf(fi, hi)
    
    #13
    def _estimate_egdf(self, fidelities, irrelevances):
        """Estimate EGDF from fidelities and irrelevances."""
        weights = self.weights.reshape(-1, 1)

        # shape check
        # print(f"Fidelities shape: {fidelities.shape}, Irrelevances shape: {irrelevances.shape}, Weights shape: {weights.shape}")
        
        mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)
        mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)
        
        M_zi = np.sqrt(mean_fidelity**2 + mean_irrelevance**2)
        eps = np.finfo(float).eps
        M_zi = np.where(M_zi == 0, eps, M_zi)
        
        egdf_values = (1 - mean_irrelevance / M_zi) / 2
        egdf_values = np.maximum.accumulate(egdf_values)
        egdf_values = np.clip(egdf_values, 0, 1)
        
        return egdf_values.flatten()
    
    #14
    def _calculate_final_egdf_pdf(self):
        """Calculate final EGDF and PDF with optimized parameters."""
        # Store zi for later use
        # Convert to infinite domain
        zi_n = DataConversion._convert_fininf(self.z, self.LB_opt, self.UB_opt)
        zi_d = DataConversion._convert_fininf(self.sample, self.LB_opt, self.UB_opt)
        # self.zi = DataConversion._convert_fininf(self.sample, self.LB_opt, self.UB_opt)
        self.zi = zi_d
        # Calculate R matrix
        eps = np.finfo(float).eps
        R = zi_n.reshape(-1, 1) / (zi_d + eps).reshape(1, -1)

        # Get characteristics
        gc = GnosticsCharacteristics(R=R)
        q, q1 = gc._get_q_q1(S=self.S_opt)
        
        # Store fidelities and irrelevances for PDF calculation
        self.fi = gc._fi(q=q, q1=q1)
        self.hi = gc._hi(q=q, q1=q1)
        
        # Calculate EGDF and PDF
        self.egdf = self._estimate_egdf(self.fi, self.hi)
        self.pdf = self._get_pdf()
        
        if self.catch:
            self.params.update({
                'egdf': self.egdf,
                'pdf': self.pdf,
                'zi': self.zi
            })
    #15
    def _get_pdf(self):
        """Calculate PDF from stored fidelities and irrelevances."""
        if self.fi is None or self.hi is None:
            raise ValueError("Fidelities and irrelevances must be calculated before PDF estimation.")
        
        weights = self.weights.reshape(-1, 1)
        
        mean_fidelity = np.sum(weights * self.fi, axis=0) / np.sum(weights)
        mean_irrelevance = np.sum(weights * self.hi, axis=0) / np.sum(weights)
        
        F2 = np.sum(weights * self.fi**2, axis=0) / np.sum(weights)
        FH = np.sum(weights * self.fi * self.hi, axis=0) / np.sum(weights)
        
        M_zi = np.sqrt(mean_fidelity**2 + mean_irrelevance**2)
        eps = np.finfo(float).eps
        M_zi = np.where(M_zi == 0, eps, M_zi)
        M_zi_cubed = M_zi**3

        numerator = ((mean_fidelity**2) * F2) + (mean_fidelity * mean_irrelevance * FH)
        density = (1 / (self.S_opt * self.zi)) * (numerator / M_zi_cubed) # NOTE devide by ZO
        # density = (1 / (self.S_opt)) * (numerator / M_zi_cubed)

        if np.any(density < 0):
            warnings.warn("EGDF density contains negative values, which may indicate non-homogeneous data", RuntimeWarning)
        
        return density.flatten()
    
    #16
    def _transform_bounds_back(self):
        """Transform optimized bounds back to original domain."""
        if self.data_form == 'a':
            self.LB = DataConversion._convert_za(self.LB_opt, self.DLB, self.DUB)
            self.UB = DataConversion._convert_za(self.UB_opt, self.DLB, self.DUB)
        else:
            self.LB = DataConversion._convert_zm(self.LB_opt, self.DLB, self.DUB)
            self.UB = DataConversion._convert_zm(self.UB_opt, self.DLB, self.DUB)
    

    #17
    def _plot(self, plot_smooth: bool = True, plot: str = 'both', bounds: bool = True, extra_df: bool = True):
        """
        Plot EGDF, WEDF, and/or PDF.
        
        Parameters:
        -----------
        plot_smooth : bool, default True
            Whether to plot smooth curves if available
        plot : str, default 'both'
            What to plot: 'gdf' for EGDF only, 'pdf' for PDF only, 'both' for both
        bounds : bool, default True
            Whether to display LB, UB, DLB, DUB bounds on the plot
        """
        import matplotlib.pyplot as plt
        
        if not self.catch:
            print("Plot is not available with argument catch=False")
            return
        
        # Validate plot parameter
        if plot not in ['gdf', 'pdf', 'both']:
            raise ValueError("plot parameter must be 'gdf', 'pdf', or 'both'")
        
        # Check required data availability
        if plot in ['gdf', 'both'] and self.params.get('egdf') is None:
            raise ValueError("EGDF must be calculated before plotting GDF")
        if plot in ['pdf', 'both'] and self.params.get('pdf') is None:
            raise ValueError("PDF must be calculated before plotting PDF")
    
        # Use original data points for plotting
        x_points = self.data
        egdf_plot = self.params.get('egdf')
        pdf_plot = self.params.get('pdf')
        wedf = self.params.get('wedf')
        ksdf = self.params.get('ksdf')
        
        # Check smooth plotting availability
        has_smooth = (hasattr(self, 'di_points_n') and hasattr(self, 'egdf_points') 
                     and hasattr(self, 'pdf_points') and self.di_points_n is not None)
        plot_smooth = plot_smooth and has_smooth
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot EGDF (GDF) if requested
        if plot in ['gdf', 'both']:
            if plot_smooth and hasattr(self, 'egdf_points'):
                # Plot smooth EGDF
                ax1.plot(x_points, egdf_plot, 'o', color='blue', label='EGDF', markersize=4)
                ax1.plot(self.di_points_n, self.egdf_points, color='blue', 
                        linestyle='-', linewidth=2, alpha=0.8)
            else:
                # Plot with connecting lines when smooth is False
                ax1.plot(x_points, egdf_plot, 'o-', color='blue', label='EGDF', 
                        markersize=4, linewidth=1, alpha=0.8)
                        
            if extra_df:
                # Plot WEDF if available
                if wedf is not None:
                    ax1.plot(x_points, wedf, 's', color='lightblue', 
                            label='WEDF', markersize=3, alpha=0.8)
                    
                # Plot KSDF if available
                if ksdf is not None:
                    ax1.plot(x_points, ksdf, 's', color='cyan', 
                            label='KS Points', markersize=3, alpha=0.8)
            
            ax1.set_ylabel('EGDF', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim(0, 1)
        
        # Plot PDF if requested
        if plot in ['pdf', 'both']:
            if plot == 'pdf':
                # PDF only - use primary axis
                if plot_smooth and hasattr(self, 'pdf_points'):
                    # Plot smooth PDF
                    ax1.plot(x_points, pdf_plot, 'o', color='red', label='PDF', markersize=4)
                    ax1.plot(self.di_points_n, self.pdf_points, color='red', 
                            linestyle='-', linewidth=2, alpha=0.8)
                else:
                    # Plot with connecting lines when smooth is False
                    ax1.plot(x_points, pdf_plot, 'o-', color='red', label='PDF', 
                            markersize=4, linewidth=1, alpha=0.8)
                
                ax1.set_ylabel('PDF', color='red')
                ax1.tick_params(axis='y', labelcolor='red')
                max_pdf = np.max(self.pdf_points) if (plot_smooth and hasattr(self, 'pdf_points') and self.pdf_points is not None) else np.max(pdf_plot)
                ax1.set_ylim(0, max_pdf * 1.1)
                ax_pdf = ax1
            else:
                # Both - use secondary axis for PDF
                ax2 = ax1.twinx()
                if plot_smooth and hasattr(self, 'pdf_points'):
                    # Plot smooth PDF
                    ax2.plot(x_points, pdf_plot, 'o', color='red', label='PDF', markersize=4)
                    ax2.plot(self.di_points_n, self.pdf_points, color='red', 
                            linestyle='-', linewidth=2, alpha=0.8)
                else:
                    # Plot with connecting lines when smooth is False
                    ax2.plot(x_points, pdf_plot, 'o-', color='red', label='PDF', 
                            markersize=4, linewidth=1, alpha=0.8)
                
                ax2.set_ylabel('PDF', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                max_pdf = np.max(self.pdf_points) if (plot_smooth and hasattr(self, 'pdf_points') and self.pdf_points is not None) else np.max(pdf_plot)
                ax2.set_ylim(0, max_pdf * 1.1)
                ax_pdf = ax2
        
        # Common settings
        ax1.set_xlabel('Data Points')
        
        # Add bounds only if bounds=True
        if bounds:
            # Add bound lines (only for primary axis to avoid duplication)
            for bound, color, style, name in [
                (self.params.get('DLB'), 'green', '-', 'DLB'),
                (self.params.get('DUB'), 'orange', '-', 'DUB'),
                (self.params.get('LB'), 'purple', '--', 'LB'),
                (self.params.get('UB'), 'brown', '--', 'UB')
            ]:
                if bound is not None:
                    ax1.axvline(x=bound, color=color, linestyle=style, linewidth=2, 
                               alpha=0.8, label=f"{name}={bound:.3f}")
            
            # Add shaded regions for probable bounds
            if self.params.get('LB') is not None:
                ax1.axvspan(x_points.min(), self.params['LB'], alpha=0.15, color='purple')
            if self.params.get('UB') is not None:
                ax1.axvspan(self.params['UB'], x_points.max(), alpha=0.15, color='brown')
        
        # Set x-axis limits
        data_range = self.params['DUB'] - self.params['DLB']
        padding = data_range * 0.1
        ax1.set_xlim(self.params['DLB'] - padding, self.params['DUB'] + padding)
        
        # Add legends
        ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
        if plot == 'both':
            ax_pdf.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        # Set title based on what's being plotted
        if plot == 'gdf':
            title = 'EGDF' + (' with Bounds' if bounds else '')
        elif plot == 'pdf':
            title = 'PDF' + (' with Bounds' if bounds else '')
        else:
            title = 'EGDF and PDF' + (' with Bounds' if bounds else '')
        
        plt.title(title)
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
    
    #18
    def _get_ks_points(self, N):
        """
        Generate Kolmogorov-Smirnov points for the EGDF.

        Parameters:
        N (int): Number of points to generate.

        Returns:
            np.ndarray: The KS points.
        """
        if N <= 0:
            raise ValueError("N must be a positive integer.")

        # Generate n values from 1 to N
        n = np.arange(1, N + 1)

        # Apply the KS-points formula: (2n-1)/(2N)
        self.ks_points = (2 * n - 1) / (2 * N)

        if self.catch:
            self.params['ks_points'] = self.ks_points

        return self.ks_points
    
    #19
    def _generate_smooth_egdf(self):
        """Generate smooth EGDF with n_points for plotting."""
        try:
            # Generate smooth points in data domain
            self.di_points_n = np.linspace(self.DLB, self.DUB, self.n_points)
            
            # Transform to z domain
            dc = DataConversion()
            if self.data_form == 'a':
                self.z_points_n = dc._convert_az(self.di_points_n, self.DLB, self.DUB)
            else:
                self.z_points_n = dc._convert_mz(self.di_points_n, self.DLB, self.DUB)

            # CRITICAL FIX: For smooth evaluation points, use the SAME transformation
            # approach as the original data but WITHOUT applying original data weights
            # to smooth points. The smooth points are just evaluation locations.
            
            # # Apply the same transformation logic as original data
            # if not self.homogeneous:
            #     # Apply gnostic weights to smooth evaluation points
            #     gw = GnosticsWeights()
            #     gweights_n = gw._get_gnostic_weights(self.z_points_n) 
            #     sample_n = self.z_points_n * gweights_n
            # else:
            #     # For homogeneous case, no gnostic weights needed
            #     sample_n = self.z_points_n

            sample_n = self.z_points_n  
            # Convert to infinite domain
            self.zi_n = DataConversion._convert_fininf(sample_n, self.LB_opt, self.UB_opt)

            # CORRECT APPROACH: Use original data zi for rows, smooth self.zi_n for columns
            # This evaluates EGDF/PDF at smooth points based on original data
            eps = np.finfo(float).eps
            
            # R matrix: original data points (rows) vs smooth evaluation points (columns)
            R_n = self.zi.reshape(-1, 1) / (self.zi_n.reshape(1, -1) + eps)
            
            gc_n = GnosticsCharacteristics(R=R_n)
            q_n, q1_n = gc_n._get_q_q1(S=self.S_opt)
            
            fi_n = gc_n._fi(q=q_n, q1=q1_n)
            hi_n = gc_n._hi(q=q_n, q1=q1_n)
            
            # Use original data weights (for weighting the original data contributions)
            weights_matrix = self.weights.reshape(-1, 1)
            
            # Calculate EGDF at smooth points
            mean_fidelity = np.sum(weights_matrix * fi_n, axis=0) / np.sum(weights_matrix)
            mean_irrelevance = np.sum(weights_matrix * hi_n, axis=0) / np.sum(weights_matrix)
            
            M_zi = np.sqrt(mean_fidelity**2 + mean_irrelevance**2)
            M_zi = np.where(M_zi == 0, eps, M_zi)
            
            self.egdf_points = (1 - mean_irrelevance / M_zi) / 2
            self.egdf_points = np.maximum.accumulate(self.egdf_points)
            self.egdf_points = np.clip(self.egdf_points, 0, 1).flatten()
            
            # Calculate PDF at smooth points
            F2 = np.sum(weights_matrix * fi_n**2, axis=0) / np.sum(weights_matrix)
            FH = np.sum(weights_matrix * fi_n * hi_n, axis=0) / np.sum(weights_matrix)
            
            M_zi_cubed = M_zi**3
            numerator = (mean_fidelity**2) * F2 + mean_fidelity * mean_irrelevance * FH
            self.pdf_points = (1 / (self.S_opt * self.zi_n)) * (numerator / M_zi_cubed) # NOTE divide by NO
            # self.pdf_points = (1 / (self.S_opt )) * (numerator / M_zi_cubed)
            self.pdf_points = self.pdf_points.flatten()

            # # pdf with gradient
            # self.pdf_points = np.gradient(self.egdf_points, self.di_points_n)

            if np.any(self.pdf_points < 0):
                warnings.warn("EGDF density contains negative values, which may indicate non-homogeneous data sample!", RuntimeWarning)

            if self.catch:
                self.params.update({
                    'di_points_n': self.di_points_n,
                    'egdf_points': self.egdf_points,
                    'pdf_points': self.pdf_points,
                    'self.zi_points': self.zi_n
                })
        except Exception as e:
            # If smooth generation fails, just skip it
            print(f"Warning: Could not generate smooth n_points: {e}")
            if self.catch:
                self.params.update({
                    'di_points_n': None,
                    'egdf_points': None,
                    'pdf_points': None,
                    'self.zi_points': None
                })


    #20
    def _fit(self):
        """Fit the EGDF model to the data."""
        SMOOTH = False # NOTE do not use as argument, it is used for internal processing
        if self.verbose:
            print("Fitting EGDF model to data...")
        # Initial processing
        self.data = np.sort(self.data)
        self._estimate_data_bounds()
        
        # Store parameters
        if self.catch:
            self._store_initial_params()

        # Step 1: Transform input data
        self._transform_input(smooth=SMOOTH)
        self._estimate_weights()
        
        # Step 2: Initial probable bounds estimation
        self._initial_probable_bounds_estimate()
        
        # Step 3: Get WEDF/KS points for optimization
        self.df_values = self._get_df(smooth=SMOOTH, wedf=self.wedf)

        # Step 4: Determine optimization strategy based on provided parameters
        self._optimize_parameters()
        
        # Step 5: Calculate final EGDF and PDF with optimized parameters
        self._calculate_final_egdf_pdf()
        
        # Step 6: Generate smooth n_points for plotting
        if len(self.data) < self.max_data_size:
            self._generate_smooth_egdf()
            if self.verbose:
                print("Generated smooth EGDF and PDF for plotting. Total points:", len(self.di_points_n))
        else:
            self.di_points_n = None
            self.egdf_points = None
            self.pdf_points = None
            if self.verbose:
                print("Data size too large for smooth EGDF generation, skipping smooth points to avoid excessive memory usage.")

        # Step 7: Transform bounds back to original domain
        self._transform_bounds_back()
        
        # Step 8: Store final parameters
        if self.catch:
            self.params.update({
                'LB': self.LB,
                'UB': self.UB,
                'S_opt': self.S_opt
            })
        if self.verbose:
            print(f"Fitting completed. Calculated parameters: \nS: {self.S_opt}, \nLB: {self.LB}, \nUB: {self.UB}")
        # Step 9: Check homogeneity if needed
        # will add later if needed



# -- previous code was not used (functions 9 and 11) --

    #9
    # def _optimize_all_parameters(self):
    #     """Optimize S, LB, and UB simultaneously."""
    #     # Parameter bounds for optimization
    #     S_MIN, S_MAX = 0.05, 100.0
    #     LB_MIN, LB_MAX = 1e-10, np.exp(-1.00001)
    #     UB_MIN, UB_MAX = np.exp(1.00001), 1e10
        
    #     def transform_to_opt_space(s, lb, ub):
    #         s_mz = DataConversion._convert_mz(s, S_MIN, S_MAX)
    #         lb_mz = DataConversion._convert_mz(lb, LB_MIN, LB_MAX)
    #         ub_mz = DataConversion._convert_mz(ub, UB_MIN, UB_MAX)
    #         return np.log(s_mz), np.log(lb_mz), np.log(ub_mz)
        
    #     def transform_from_opt_space(s_opt, lb_opt, ub_opt):
    #         s_mz = np.exp(s_opt)
    #         lb_mz = np.exp(lb_opt)
    #         ub_mz = np.exp(ub_opt)
    #         s = DataConversion._convert_zm(s_mz, S_MIN, S_MAX)
    #         lb = DataConversion._convert_zm(lb_mz, LB_MIN, LB_MAX)
    #         ub = DataConversion._convert_zm(ub_mz, UB_MIN, UB_MAX)
    #         return s, lb, ub
        
    #     def loss_function(opt_params):
    #         s_opt, lb_opt, ub_opt = opt_params
    #         try:
    #             s, lb, ub = transform_from_opt_space(s_opt, lb_opt, ub_opt)
    #             egdf_values = self._compute_egdf(s, lb, ub)
    #             diff = np.mean(np.abs(egdf_values - self.df_values) * self.weights)
    #             print(f"Optimization difference: {diff}, S: {s}, LB: {lb}, UB: {ub}")
    #             return diff
    #         except Exception:
    #             return 1e6
        
    #     # Initial values
    #     s_init = 1.0
    #     lb_init = self.LB if self.LB is not None else LB_MIN * 10  # Use a reasonable default
    #     ub_init = self.UB if self.UB is not None else UB_MIN * 10  # Use a reasonable default
        
    #     # Transform initial values to optimization space
    #     s_opt_init, lb_opt_init, ub_opt_init = transform_to_opt_space(s_init, lb_init, ub_init)
        
    #     # Optimization bounds in transformed space (log space typically ranges from -10 to 10)
    #     opt_bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    #     initial_params = [s_opt_init, lb_opt_init, ub_opt_init]
        
    #     # Clip initial parameters to bounds
    #     initial_params = [
    #         np.clip(s_opt_init, -10.0, 10.0),
    #         np.clip(lb_opt_init, -10.0, 10.0),
    #         np.clip(ub_opt_init, -10.0, 10.0)
    #     ]
    
    #     try:
    #         result = minimize(loss_function, 
    #                           initial_params, 
    #                           method=self.opt_method, 
    #                           bounds=opt_bounds,
    #                           options={'maxiter': 1000, 'ftol': self.tolerance}, 
    #                           tol=self.tolerance,
    #                         )
    #         s_opt_final, lb_opt_final, ub_opt_final = result.x
    #         return transform_from_opt_space(s_opt_final, lb_opt_final, ub_opt_final)
    #     except Exception as e:
    #         print(f"Optimization failed: {e}")
    #         return s_init, lb_init, ub_init

    # alternative way without transformation
    # def _optimize_all_parameters(self):
    #     """Optimize S, LB, and UB simultaneously with preference for smaller values."""
    #     # Parameter bounds for optimization (direct bounds without transformation)
    #     S_MIN, S_MAX = 0.05, 100.0
    #     LB_MIN, LB_MAX = np.finfo(float).eps, np.exp(-1.00001)
    #     UB_MIN, UB_MAX = np.exp(1.00001), np.finfo(float).max
    
    #     def loss_function(params):
    #         s, lb, ub = params
    #         try:
    #             # Ensure valid parameter ranges
    #             if s <= 0 or lb <= 0 or ub <= lb:
    #                 return 1e6
                    
    #             egdf_values = self._compute_egdf(s, lb, ub)
                
    #             # Primary loss: difference between EGDF and target values
    #             diff = np.mean(np.abs(egdf_values - self.df_values) * self.weights)
                
    #             # Regularization terms to prefer smaller values
    #             # Scale regularization based on parameter ranges
    #             s_reg = 0.01 * (s - S_MIN) / (S_MAX - S_MIN)  # Prefer smaller S
    #             lb_reg = 0.001 * np.abs(lb) / LB_MAX  # Prefer smaller |LB|
    #             ub_reg = 0.001 * (ub - UB_MIN) / (UB_MAX - UB_MIN)  # Prefer smaller UB
                
    #             total_loss = diff + s_reg + lb_reg + ub_reg
                
    #             print(f"Loss: {diff:.6f}, S_reg: {s_reg:.6f}, Total: {total_loss:.6f}, S: {s:.3f}, LB: {lb:.6f}, UB: {ub:.3f}")
    #             return total_loss
                
    #         except Exception as e:
    #             print(f"Error in loss function: {e}")
    #             return 1e6
        
    #     # Initial values - start with smaller values
    #     s_init = 0.1  # Start with smaller S
    #     lb_init = self.LB if self.LB is not None else LB_MIN * 2
    #     ub_init = self.UB if self.UB is not None else UB_MIN * 2
    
    #     # Ensure initial values are within bounds
    #     s_init = np.clip(s_init, S_MIN, S_MAX)
    #     lb_init = np.clip(lb_init, LB_MIN, LB_MAX)
    #     ub_init = np.clip(ub_init, UB_MIN, UB_MAX)
        
    #     # Ensure lb < ub
    #     if lb_init >= ub_init:
    #         lb_init = LB_MIN * 2
    #         ub_init = UB_MIN * 2
        
    #     initial_params = [s_init, lb_init, ub_init]
        
    #     # Direct optimization bounds
    #     opt_bounds = [(S_MIN, S_MAX), (LB_MIN, LB_MAX), (UB_MIN, UB_MAX)]
        
    #     try:
    #         result = minimize(loss_function, 
    #                           initial_params, 
    #                           method=self.opt_method, 
    #                           bounds=opt_bounds,
    #                           options={'maxiter': 1000, 'ftol': self.tolerance}, 
    #                           tol=self.tolerance)
            
    #         s_opt_final, lb_opt_final, ub_opt_final = result.x
            
    #         # Validate final results
    #         if lb_opt_final >= ub_opt_final:
    #             print("Warning: Optimized LB >= UB, using initial values")
    #             return s_init, lb_init, ub_init
                
    #         return s_opt_final, lb_opt_final, ub_opt_final
            
    #     except Exception as e:
    #         print(f"Optimization failed: {e}")
    #         return s_init, lb_init, ub_init

        #11
    # def _optimize_bounds_only(self, s):
    #     """Optimize only LB and UB for given S."""
    #     LB_MIN, LB_MAX = 1e-10, np.exp(-1.00001)
    #     UB_MIN, UB_MAX = np.exp(1.00001), 1e10

    #     def transform_bounds_to_opt_space(lb, ub):
    #         lb_mz = DataConversion._convert_mz(lb, LB_MIN, LB_MAX)
    #         ub_mz = DataConversion._convert_mz(ub, UB_MIN, UB_MAX)
    #         return np.log(lb_mz), np.log(ub_mz)
        
    #     def transform_bounds_from_opt_space(lb_opt, ub_opt):
    #         lb_mz = np.exp(lb_opt)
    #         ub_mz = np.exp(ub_opt)
    #         lb = DataConversion._convert_zm(lb_mz, LB_MIN, LB_MAX)
    #         ub = DataConversion._convert_zm(ub_mz, UB_MIN, UB_MAX)
    #         return lb, ub
        
    #     def loss_function(opt_params):
    #         lb_opt, ub_opt = opt_params
    #         try:
    #             lb, ub = transform_bounds_from_opt_space(lb_opt, ub_opt)
    #             egdf_values = self._compute_egdf(s, lb, ub)
    #             # # fidelity
    #             # R_ = egdf_values / (self.df_values + 1e-10)
    #             # gc = GnosticsCharacteristics(R=R_)
    #             # q, q1 = gc._get_q_q1(S=s)
    #             # fi = np.mean(gc._fi(q=q, q1=q1))
    #             # print(f"Fidelity: {fi}, S: {s}, LB: {lb}, UB: {ub}")  # Debugging output
    #             # return -fi
    #             diff = np.mean(np.abs(egdf_values - self.df_values) * self.weights)
    #             print(f"Optimization difference: {diff}, S: {s}, LB: {lb}, UB: {ub}")
    #             return diff
    #         except Exception:
    #             return 1e6
        
    #     lb_opt_init, ub_opt_init = transform_bounds_to_opt_space(self.LB, self.UB)
    #     initial_params = [
    #         np.clip(lb_opt_init, -1.0, 1.0),
    #         np.clip(ub_opt_init, -1.0, 1.0)
    #     ]
        
    #     try:
    #         result = minimize(loss_function, initial_params, method=self.opt_method,
    #                         bounds=[(-1.0, 1.0), (-1.0, 1.0)], options={'maxiter': 1000})
    #         lb_opt_final, ub_opt_final = result.x
    #         lb_final, ub_final = transform_bounds_from_opt_space(lb_opt_final, ub_opt_final)
    #         return s, lb_final, ub_final
    #     except Exception:
    #         return s, self.LB, self.UB

    # def _optimize_bounds_only(self, s):
    #     """Optimize only LB and UB for given S without bound transformation."""
    #     # Parameter bounds for optimization (direct bounds without transformation)
    #     LB_MIN, LB_MAX = 1e-10, np.exp(-1.00001)
    #     UB_MIN, UB_MAX = np.exp(1.00001), 1e10
        
    #     def loss_function(params):
    #         lb, ub = params
    #         try:
    #             # Ensure valid parameter ranges
    #             if lb <= 0 or ub <= lb:
    #                 return 1e6
                    
    #             egdf_values = self._compute_egdf(s, lb, ub)
    #             diff = np.mean(np.abs(egdf_values - self.df_values) * self.weights)
    #             print(f"Optimization difference: {diff}, S: {s}, LB: {lb}, UB: {ub}")
    #             return diff
    #         except Exception as e:
    #             print(f"Error in loss function: {e}")
    #             return 1e6
        
    #     # Initial values
    #     lb_init = self.LB if self.LB is not None else LB_MIN * 10
    #     ub_init = self.UB if self.UB is not None else UB_MIN * 10
        
    #     # Ensure initial values are within bounds
    #     lb_init = np.clip(lb_init, LB_MIN, LB_MAX)
    #     ub_init = np.clip(ub_init, UB_MIN, UB_MAX)
        
    #     # Ensure lb < ub
    #     if lb_init >= ub_init:
    #         lb_init = LB_MIN * 10
    #         ub_init = UB_MIN * 10
        
    #     initial_params = [lb_init, ub_init]
        
    #     # Direct optimization bounds
    #     opt_bounds = [(LB_MIN, LB_MAX), (UB_MIN, UB_MAX)]
        
    #     try:
    #         result = minimize(loss_function, 
    #                         initial_params, 
    #                         method=self.opt_method,
    #                         bounds=opt_bounds,
    #                         options={'maxiter': 1000, 'ftol': self.tolerance}, 
    #                         tol=self.tolerance)
            
    #         lb_opt_final, ub_opt_final = result.x
            
    #         # Validate final results
    #         if lb_opt_final >= ub_opt_final:
    #             print("Warning: Optimized LB >= UB, using initial values")
    #             return s, lb_init, ub_init
                
    #         return s, lb_opt_final, ub_opt_final
            
    #     except Exception as e:
    #         print(f"Optimization failed: {e}")
    #         return s, self.LB, self.UB