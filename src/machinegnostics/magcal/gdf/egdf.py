"""
EGDF - Estimating Global Distribution Function.

Author: Nirmal Parmar
Machine Gnostics
"""

import numpy as np
import warnings
from machinegnostics.magcal.scale_param import ScaleParam
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.gdf.base_df import BaseDistFunc
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.gdf.wedf import WEDF
from machinegnostics.magcal.gdf.homogeneity import DataHomogeneity
from machinegnostics.magcal.gdf.bound_estimator import BoundEstimator

class EGDF(BaseDistFunc, DataHomogeneity, BoundEstimator):
    """
    EGDF - A class for estimating the global distribution function.
    """
    def __init__(self,
                 data: np.ndarray,
                 DLB: float = None,
                 DUB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S = 'auto',
                 tolerance: float = 1e-3,
                 data_form: str = 'a',
                 n_points: int = 1000,
                 homogeneous: bool = True,
                 catch: bool = True,
                 weights: np.ndarray = None):
        
        """
        Initialize the EGDF class.

        Parameters:
        data (np.ndarray): Input data for the EGDF.
        DLB (float): Lower bound for the data.
        DUB (float): Upper bound for the data.
        LB (float): Lower (Probable) Bound.
        UB (float): Upper (Probable) Bound.
        S (float): Scale parameter.
        tolerance (float): Tolerance for convergence.
        data_form (str): Form of the data ('a' for additive, 'm' for multiplicative).
        n_points (int): Number of points in the distribution function.
        homogeneous (bool): Whether given data is homogeneous, True by default. if False, in that case data will be homogenized.
        catch (bool): To catch calculated values or not, True by default.
        weights (np.ndarray): Priory Weights for the data points.
        data_pad (float): Padding for the data range.
        """
        
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
        self.params = {}
        
        # argument validation
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array.")
        
        # Validate only if not None
        if DLB is not None and not isinstance(DLB, (int, float)):
            raise ValueError("DLB must be a numeric value or None.")
        if DUB is not None and not isinstance(DUB, (int, float)):
            raise ValueError("DUB must be a numeric value or None.")
            raise ValueError("USB must be a numeric value or None.")
        if LB is not None and not isinstance(LB, (int, float)):
            raise ValueError("LB must be a numeric value or None.")
        if UB is not None and not isinstance(UB, (int, float)):
            raise ValueError("UB must be a numeric value or None.")
        
        # S can be int, float, or str 'auto' to automatically determine the scale parameter
        if not isinstance(S, (int, float, str)):
            raise ValueError("S must be a numeric positive value or 'auto'.")
        
        if not isinstance(tolerance, (int, float)):
            raise ValueError("Tolerance must be a numeric value.")
        
        if data_form not in ['a', 'm']:
            raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative.")
        
        if not isinstance(n_points, int) or n_points <= 0:
            raise ValueError("n_points must be a positive integer.")
        
        if not isinstance(homogeneous, bool):
            raise ValueError("homogeneous must be a boolean value. It indicates whether the input data is homogeneous or not.")
        
        if not isinstance(catch, bool):
            raise ValueError("catch must be a boolean value. It indicates whether to catch calculated values or not.")

        if weights is not None and not isinstance(weights, np.ndarray):
            raise ValueError("weights must be a numpy array.")
        if weights is not None and len(weights) != len(data):
            raise ValueError("Weights must have the same length as data.")
        
        # initial estimates and processing
        # data sorting
        self.data = np.sort(data)
        # DLB, DUB bounds
        self._estimate_data_bounds()
        # weights validation
        self._estimate_weights()

        # store if arguments are available
        if self.catch:
            self.params['data'] = self.data
            self.params['DLB'] = self.DLB if self.DLB is not None else None
            self.params['DUB'] = self.DUB if self.DUB is not None else None
            self.params['LB'] = self.LB if self.LB is not None else None
            self.params['UB'] = self.UB if self.UB is not None else None
            self.params['S'] = self.S if self.S is not None else None
            self.params['tolerance'] = self.tolerance if self.tolerance is not None else None
            self.params['data_form'] = self.data_form if self.data_form is not None else None
            self.params['n_points'] = self.n_points if self.n_points is not None else None
            self.params['homogeneous'] = self.homogeneous if self.homogeneous is not None else None
            self.params['weights'] = self.weights if self.weights is not None else None
        else:
            self.params = {}

        # initialize parent class
        DataHomogeneity.__init__(self, params=self.params, data=self.data, catch=self.catch)
        BoundEstimator.__init__(self, params=self.params, data=self.data, catch=self.catch)

    def _estimate_weights(self):
        """
        Estimate weights for the EGDF.
        
        This method can be overridden in subclasses to provide custom weight estimation logic.
        """
        # Default implementation uses uniform weights
        if self.weights is None:
            self.weights = np.ones_like(self.data)
        else:
            self.weights = np.asarray(self.weights)
            if len(self.weights) != len(self.data):
                raise ValueError("weights must have the same length as data")
        # Normalize weights to sum to n (number of data points)
        self.weights = self.weights / np.sum(self.weights) * len(self.weights)

        # store weights to param
        if self.catch:
            self.params['weights'] = self.weights
        else:
            self.params['weights'] = None

    def _estimate_data_bounds(self):
        """
        Estimate data bounds based on the EGDF.

        DLB and DUB are the data bounds where samples are expected.
        """
        # Estimate data bounds
        if self.DLB is None:
            self.DLB = np.min(self.data)
        if self.DUB is None:
            self.DUB = np.max(self.data)
        if self.catch:
            self.params['DLB'] = self.DLB
            self.params['DUB'] = self.DUB
        else:
            self.params['DLB'] = None
            self.params['DUB'] = None

    def _initial_probable_bounds_estimate(self):
        """
        Estimate probable bounds based on the EGDF.
    
        LB and UB are the probable bounds where samples are expected.
    
        This is first estimate, and then optimized later.
        Only estimates bounds if they are not already provided.
        """
        # Only estimate LB if it's not provided
        if self.LB is None:
            if self.data_form == 'a':
                # For additive form: LB should be less than data minimum
                pad = (self.DUB - self.DLB) / 2
                lb_raw = self.DLB - pad
                self.LB = DataConversion._convert_az(lb_raw, self.DLB, self.DUB)
            elif self.data_form == 'm':
                # For multiplicative form: LB should be less than data minimum
                lb_raw = self.DLB / np.sqrt(self.DUB / self.DLB)
                self.LB = DataConversion._convert_mz(lb_raw, self.DLB, self.DUB)
            else:
                raise ValueError(f"Unknown data form: {self.data_form}, expected 'a' or 'm'.")
    
        # Only estimate UB if it's not provided
        if self.UB is None:
            if self.data_form == 'a':
                # For additive form: UB should be greater than data maximum
                pad = (self.DUB - self.DLB) / 2
                ub_raw = self.DUB + pad
                self.UB = DataConversion._convert_az(ub_raw, self.DLB, self.DUB)
            elif self.data_form == 'm':
                # For multiplicative form: UB should be greater than data maximum
                ub_raw = self.DUB * np.sqrt(self.DUB / self.DLB)
                self.UB = DataConversion._convert_mz(ub_raw, self.DLB, self.DUB)
            else:
                raise ValueError(f"Unknown data form: {self.data_form}, expected 'a' or 'm'.")
    
        # Store initial estimates
        if self.catch:
            self.params['LB_init'] = self.LB
            self.params['UB_init'] = self.UB
        else:
            self.params['LB_init'] = None
            self.params['UB_init'] = None

    def fit(self):
        """
        Fit the EGDF model to the data.
    
        This method applies the transformation and prepares the data for further analysis.
        """
        # transform data from standard domain to normal domain
        self._transform_input()
    
        # LB and UB initial estimate
        self._initial_probable_bounds_estimate()
    
        # get z points
        self._get_z_points()
        
        # estimate wedf
        self.wedf = self._get_wedf()
    
        # optimize scale parameter S, LB and UB for minimum difference
        if isinstance(self.S, str) and self.S.lower() == 'auto':
            self.S_opt, self.LB_opt, self.UB_opt = self._get_optimized_bounds()
            self._egdf(S=self.S_opt, LB=self.LB_opt, UB=self.UB_opt)
        else:
            # Use provided S value and calculate final EGDF
            self.S_opt = self.S
            self.LB_opt = self.LB
            self.UB_opt = self.UB
            self._egdf(S=self.S_opt, LB=self.LB_opt, UB=self.UB_opt)

        # transform back bounds to normal domain
        if self.data_form == 'a':
            self.LB = DataConversion._convert_za(self.LB_opt, self.DLB, self.DUB)
            self.UB = DataConversion._convert_za(self.UB_opt, self.DLB, self.DUB)
        else:
            self.LB = DataConversion._convert_zm(self.LB_opt, self.DLB, self.DUB)
            self.UB = DataConversion._convert_zm(self.UB_opt, self.DLB, self.DUB)

        if self.catch:
            self.params['LB'] = self.LB
            self.params['UB'] = self.UB
    
        # calculate final PDF
        self.pdf = self._get_pdf()
    
        # if data was not homogeneous, and checking that is homogenized or not
        if self.homogeneous == False and self.catch == True:
            self._is_homogeneous()


    def _egdf(self, S, LB, UB):
        """
        Estimate the Estimating Global Distribution Function (EGDF).

        convert to infinite domain
        estimate egdf at s=1
        find LB, UB, and S where difference of wedf and egdf is minimized
        """
        # Convert to infinite domain
        self.zi = DataConversion._convert_fininf(self.sample, LB, UB)
        self.zi_points = np.linspace(LB, UB, self.n_points)

        # R, q and q1
        eps = np.finfo(float).eps  # small value to avoid division by zero
        R = self.zi.reshape(-1, 1) / (self.zi_points + eps).reshape(1, -1)
        gc = GnosticsCharacteristics(R=R)
        q, q1 = gc._get_q_q1(S=S)

        # fi and hi
        self.fi = gc._fi(q=q, q1=q1)
        self.hi = gc._hi(q=q, q1=q1)

        # estimate egdf
        self.egdf = self._estimate_egdf(self.fi, self.hi)

        # param store
        if self.catch:
            self.params['zi'] = self.zi
        else:
            self.params['zi'] = None

    def _get_optimized_bounds(self):
        """
        Optimize scale parameter S, LB and UB bounds to minimize difference between EGDF and WEDF.
        
        Bounds logic: 0 < LB < min(zi) < zi_values < max(zi) < UB < ∞
        """
        from scipy.optimize import minimize
        import warnings
        import numpy as np
        
        # First, we need to get zi values to set proper bounds
        # Convert current sample to infinite domain using initial LB, UB estimates
        zi_temp = DataConversion._convert_fininf(self.sample, self.LB, self.UB)
        zi_min, zi_max = zi_temp.min(), zi_temp.max()
        zi_range = zi_max - zi_min
        
        # Get initial bounds based on data form
        sx = 1.0  # Scale parameter initial value
        lbx = self.LB  # Initial LB estimate
        ubx = self.UB  # Initial UB estimate
        initial_guess = [sx, lbx, ubx]
    
        # Define optimization ranges for parameters in infinite domain
        # S: scale parameter (positive)
        s_min, s_max = 0.05, 100.0
        
        # LB: must be positive and less than min(zi)
        # Add some margin to ensure LB < min(zi)
        lb_min = np.finfo(float).eps  # Very small positive number
        lb_max = zi_min # Ensure LB < min(zi)
        if lb_max <= lb_min:
            lb_max = zi_min * 0.9  # Fallback: 90% of min zi value
        
        # UB: must be greater than max(zi)
        # Add some margin to ensure UB > max(zi)
        ub_min = zi_max # Ensure UB > max(zi)
        ub_max = np.finfo(float).max # Upper limit, but not infinity to avoid numerical issues
        
        # Ensure bounds are valid
        if lb_max <= lb_min:
            lb_min = np.finfo(float).eps
            lb_max = zi_min * 0.5
        
        if ub_min >= ub_max:
            ub_min = zi_max * 1.1
            ub_max = zi_max * 10.0
        
        # Update initial guess to be within bounds
        if lbx <= lb_min or lbx >= lb_max:
            lbx = (lb_min + lb_max) / 2
        if ubx <= ub_min or ubx >= ub_max:
            ubx = (ub_min + ub_max) / 2
        
        initial_guess = [sx, lbx, ubx]
        opt_bounds = [(s_min, s_max), (lb_min, lb_max), (ub_min, ub_max)]
    
        def objective_function(params):
            """
            Objective function to minimize difference between EGDF and WEDF.
            """
            try:
                sx_opt, lbx_opt, ubx_opt = params
                
                # Validate bounds constraints: LB < min(zi) < max(zi) < UB
                zi_temp_check = DataConversion._convert_fininf(self.sample, lbx_opt, ubx_opt)
                zi_min_check, zi_max_check = zi_temp_check.min(), zi_temp_check.max()
                
                if not (lbx_opt < zi_min_check and zi_max_check < ubx_opt):
                    return 1e10
                
                # Convert normalized data to infinite domain
                zi_temp = DataConversion._convert_fininf(self.sample, lbx_opt, ubx_opt)
                zi_points_temp = np.linspace(lbx_opt, ubx_opt, self.n_points)
    
                # Calculate R matrix
                eps = np.finfo(float).eps
                R_temp = zi_temp.reshape(-1, 1) / (zi_points_temp.reshape(1, -1) + eps)
                
                # Get characteristics
                gc_temp = GnosticsCharacteristics(R=R_temp)
                q_temp, q1_temp = gc_temp._get_q_q1(S=sx_opt)
    
                # Calculate fidelities and irrelevances
                fi_temp = gc_temp._fi(q=q_temp, q1=q1_temp)
                hi_temp = gc_temp._hi(q=q_temp, q1=q1_temp)
    
                # Calculate EGDF values using temporary estimation
                egdf_vals = self._estimate_egdf(fi_temp, hi_temp)
                
                # Calculate weighted absolute difference with WEDF
                if len(egdf_vals) != len(self.wedf):
                    return 1e10
                    
                diff = np.abs(egdf_vals - self.wedf)
                objective_val = np.mean(diff)
                
                return objective_val
                    
            except Exception as e:
                return 1e10
        
        # Perform optimization
        try:
            result = minimize(
                objective_function,
                initial_guess,
                bounds=opt_bounds,
                method='L-BFGS-B',
                options={
                    'maxiter': 1000,
                    'ftol': 1e-9,
                    'gtol': 1e-9,
                    'disp': False
                }
            )
            
            if result.success and result.fun < 1e9:
                sx_opt, lbx_opt, ubx_opt = result.x
                self.S_opt = sx_opt
                self.LB_opt = lbx_opt
                self.UB_opt = ubx_opt
                optimization_success = True
                min_diff = result.fun
            else:
                # Fallback to initial values
                self.S_opt = sx
                self.LB_opt = lbx
                self.UB_opt = ubx
                optimization_success = False
                min_diff = objective_function(initial_guess)
                warnings.warn("Bound optimization failed, using initial estimates")
                    
        except Exception as e:
            # Fallback to initial values
            self.S_opt = sx
            self.LB_opt = lbx
            self.UB_opt = ubx
            optimization_success = False
            min_diff = float('inf')
            warnings.warn(f"Bound optimization failed with error: {e}, using initial estimates")
        
        # Update instance variables
        self.S = self.S_opt
        self.LB = self.LB_opt
        self.UB = self.UB_opt
        
        # Store optimization results
        if self.catch:
            self.params.update({
                'S_opt': self.S_opt,
                'LB_opt': self.LB_opt,
                'UB_opt': self.UB_opt,
                'optimization_success': optimization_success,
                'min_difference': min_diff,
                'zi_bounds': {
                    'zi_min': zi_min,
                    'zi_max': zi_max,
                    'zi_range': zi_range
                },
                'optimization_bounds': {
                    'S_range': [s_min, s_max],
                    'LB_range': [lb_min, lb_max], 
                    'UB_range': [ub_min, ub_max]
                }
            })
        
        return self.S_opt, self.LB_opt, self.UB_opt
    
    def plot(self):
        """
        plot EGDF and PDF.

        EGDF is in blue line on y1 axis and PDF is in red line on y2 axis.
        Optimized bounds LB and UB are shown as vertical lines.
        """
        import matplotlib.pyplot as plt
        if self.catch:
            if self.params.get('egdf') is None or self.params.get('pdf') is None:
                raise ValueError("EGDF and PDF must be calculated before plotting.")
            egdf = self.params['egdf']
            pdf = self.params['pdf']
        if self.catch == False:
            # raise warning only and exit function
            return print("Plot is not available with argument catch=False")

        # Use the original di_points (data domain) for plotting instead of zi_points
        x_points = self.di_points

        fig, ax1 = plt.subplots()

        # Plot EGDF
        ax1.plot(x_points, egdf, color='blue', label='EGDF', linewidth=2)
        ax1.set_xlabel('Data Points')
        ax1.set_ylabel('EGDF', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        # Most robust version
        lb = self.params['DLB']
        ub = self.params['DUB']

        # Calculate range and apply 2% padding
        range_val = ub - lb
        padding = range_val * 0.1

        x_min = lb - padding
        x_max = ub + padding

        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(0, 1)
 
        # Create a second y-axis for the PDF
        ax2 = ax1.twinx()
        ax2.plot(x_points, pdf, color='red', label='PDF', linewidth=2)
        ax2.set_ylabel('PDF', color='red')
        ax2.set_ylim(0, np.max(pdf) * 1.1)  # Adjust y-limits for PDF
        ax2.tick_params(axis='y', labelcolor='red')

        # Add data points as vertical lines
        for point in self.data:
            ax1.axvline(x=point, color='gray', linestyle='--', alpha=0.5)

        # Add optimized bounds if available
        if self.params.get('LB') is not None:
            ax1.axvline(x=self.params['LB'], color='black', linestyle='--', linewidth=1.5, 
                    alpha=0.7)
            # Add pink shaded area to the left of LB
            ax1.axvspan(x_points.min(), self.params['LB'], alpha=0.2, color='pink', label=f"LB={self.params['LB']:.3f}")

        if self.params.get('UB') is not None:
            ax1.axvline(x=self.params['UB'], color='black', linestyle='--', linewidth=1.5, 
                    alpha=0.7)
            # Add green shaded area to the right of UB
            ax1.axvspan(self.params['UB'], x_points.max(), alpha=0.2, color='lightgreen', label=f"UB={self.params['UB']:.3f}")

        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.title('EGDF and PDF')
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()
    
    def _transform_input(self):
        """
        Transform input data to the standard domain.

        This method can be overridden in subclasses to provide custom transformation logic.
        """
        dc = DataConversion()

        if self.data_form == 'a':
            self.z = dc._convert_az(self.data, self.DLB, self.DUB)
        elif self.data_form == 'm':
            self.z = dc._convert_mz(self.data, self.DLB, self.DUB)
        else:
            raise ValueError(f"Unknown data form: {self.data_form}, expected 'a' or 'm'.")
        
        # homogenize
        if not self.homogeneous:
            self._homogenize()
            self.sample = self.z * self.weights
        else:
            self.sample = self.z * self.weights

        # store
        if self.catch:
            self.params['zf'] = self.z
            self.params['sample'] = self.sample
        else:
            self.params['zf'] = None
            self.params['sample'] = None       

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
    
    def _get_z_points(self):
        """
        Get the z points for the EGDF.

        Returns:
            np.ndarray: The z points.
        """
        # generate di_points
        self.di_points = np.linspace(self.DLB, self.DUB, self.n_points)

        # data transformation
        dc = DataConversion()
        if self.data_form == 'a':
            self.z_points = dc._convert_az(self.di_points, self.DLB, self.DUB)
        elif self.data_form == 'm':
            self.z_points = dc._convert_mz(self.di_points, self.DLB, self.DUB)
        else:
            raise ValueError(f"Unknown data form: {self.data_form}, expected 'a' or 'm'.")

        if self.catch:
            self.params['di_points'] = self.di_points
            self.params['z_points'] = self.z_points
        else:
            self.params['di_points'] = None
            self.params['z_points'] = None

    def _get_wedf(self):
        """
        Get the WEDF for the EGDF.

        Returns:
            WEDF: The WEDF object.
        """
        wedf = WEDF(self.data, weights=self.weights, data_lb=self.DLB, data_ub=self.DUB)

        wedf_values = wedf.fit(self.di_points) # NOTE need to find logic when to use data vs di_points

        if self.catch:
            self.params['wedf'] = wedf_values
        else:
            self.params['wedf'] = None
        return wedf_values


    def _estimate_egdf(self, fidelities, irrelevances):
        """
        Estimate the EGDF based on the fidelities and irrelevances.
        
        Parameters:
        fidelities : np.ndarray
            Fidelity values for each data point at each evaluation point
        irrelevances : np.ndarray
            Irrelevance values for each data point at each evaluation point
        
        Returns:
            np.ndarray: The estimated EGDF values.
        """
        # Calculate weighted means using equation 15.31
        weights = self.weights.reshape(-1, 1)  # Shape: (n_data, 1)
        
        mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)  # f̄_E
        mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)  # h̄_E
        
        # Calculate estimating modulus using equation 15.28
        M_zi = np.sqrt(mean_fidelity**2 + mean_irrelevance**2)
        
        # Avoid division by zero
        eps = np.finfo(float).eps
        M_zi = np.where(M_zi == 0, eps, M_zi)
        
        # Calculate EGDF using equation 15.29
        # Fix: Use correct formula - should be (1 + mean_irrelevance / M_zi) / 2
        egdf_values = (1 - mean_irrelevance / M_zi) / 2
        
        # Ensure EGDF is monotonically non-decreasing
        egdf_values = np.maximum.accumulate(egdf_values)
        
        # Ensure EGDF is bounded between 0 and 1
        egdf_values = np.clip(egdf_values, 0, 1)
        
        # Flatten the result to ensure it's 1D
        egdf_values = egdf_values.flatten()
    
        if self.catch:
            self.params['egdf'] = egdf_values
            # self.params['mean_fidelity'] = mean_fidelity
            # self.params['mean_irrelevance'] = mean_irrelevance
            # self.params['M_zi'] = M_zi
        else:
            self.params['egdf'] = None
            # self.params['mean_fidelity'] = None
            # self.params['mean_irrelevance'] = None
            # self.params['M_zi'] = None
    
        return egdf_values

    
    def _get_pdf(self):
        """
        Get the PDF for the EGDF using fidelity and irrelevance calculations.

        Returns:
            np.ndarray: The PDF values.
        """
        # Initialize output array with correct shape
        density = np.zeros_like(self.zi.flatten(), dtype=float)
        
        # Handle empty data case
        if len(self.zi) == 0:
            return density
        
        # Get fidelities and irrelevances from stored params
        fidelities = self.fi  # Use stored fi from _get_egdf
        irrelevances = self.hi  # Use stored hi from _get_egdf
        
        if fidelities is None or irrelevances is None:
            raise ValueError("Fidelities and irrelevances must be calculated before PDF estimation.")
        
        # Reshape weights for broadcasting
        weights = self.weights.reshape(-1, 1)  # Shape: (n_data, 1)
        
        # Calculate weighted means using equation 15.31
        mean_fidelity = np.sum(weights * fidelities, axis=0) / np.sum(weights)  # f̄_E
        mean_irrelevance = np.sum(weights * irrelevances, axis=0) / np.sum(weights)  # h̄_E

        # Calculate F2 and FH using equation 15.31
        F2 = np.sum(weights * fidelities**2, axis=0) / np.sum(weights)
        FH = np.sum(weights * fidelities * irrelevances, axis=0) / np.sum(weights)
        
        # Calculate estimating modulus using equation 15.28
        M_zi = np.sqrt(mean_fidelity**2 + mean_irrelevance**2)
        
        # Avoid division by zero
        eps = np.finfo(float).eps
        M_zi = np.where(M_zi == 0, eps, M_zi)
        M_zi_cubed = M_zi**3
        
        # Calculate PDF using equation 15.30
        self.sparam = self.S_opt if hasattr(self, 'S_opt') and self.S_opt is not None else 1.0
        numerator = (mean_fidelity**2) * F2 + mean_fidelity * mean_irrelevance * FH
        density = (1 / self.sparam) * (numerator / M_zi_cubed)
        
        # Flatten the result to ensure it's 1D
        density = density.flatten()
        
        # Handle negative density values
        if np.any(density < 0):
            warnings.warn("EGDF density contains negative values, which may indicate non-homogeneous data", RuntimeWarning)
        
        if self.catch:
            self.params['pdf'] = density
            # self.params['F2'] = F2
            # self.params['FH'] = FH
        else:
            self.params['pdf'] = None

        return density

    
    def _homogenize(self):
        """
        Homogenize the data if it is not homogeneous.
        
        This method can be overridden in subclasses to provide custom homogenization logic.
        """
        self.weights = self.homogenize()
        
    def _is_homogeneous(self):
        """
        Check if the data is homogeneous.
        
        This method can be overridden in subclasses to provide custom homogeneity checks.
        """
        self.is_homo, self.params = self.is_homogeneous()

        if self.catch:
            self.params['is_homogeneous'] = self.is_homo