"""
Bound Estimator for EGDF and other GDFs.

Machine Gnostics
Author: Nirmal Parmar
"""
import numpy as np
from scipy.optimize import minimize
from machinegnostics.magcal.data_conversion import DataConversion

class BoundEstimator:
    """
    Class for estimating bounds for the EGDF and other GDFs.
    """
    def __init__(self, params, data, catch=True):
        self.params = params
        self.data = data
        self.catch = catch
        # if self catch is False, many functions will not be available
        if self.catch == False:
            print("Warning: Catch is set to False, some functions may not be available.")


    def estimate_bounds(self):
        """
        Estimate the bounds for the data.
        """
        # estimate probable bounds (LB, UB)

        # estimate data support bounds (LSB, USB)

        # estimate egdf location parameters (mean, median, mode)
        pass


    def _get_derivative(self, pdf=None):
        """
        Get the derivatives of the EGDF.
        
        Since PDF = dP/dZ₀ (first derivative of EGDF), we have:
        - PDF = dP/dZ₀ (first derivative of EGDF)
        - Second derivative of EGDF = d²P/dZ₀² = d(PDF)/dZ₀
        - Third derivative of EGDF = d³P/dZ₀³ = d²(PDF)/dZ₀²
        
        Returns:
        tuple: (first_derivative, second_derivative, third_derivative) of EGDF
               where first_derivative is the PDF itself
        """
        if 'pdf' not in self.params and pdf is None:
            raise ValueError("PDF not found in params. Please fit the model first.") 
        
        if pdf is None:
            pdf = self.params['pdf']
    
        # PDF is already the first derivative of EGDF
        first_derivative = pdf
        
        # Second derivative of EGDF = gradient of PDF
        second_derivative = np.gradient(pdf)
        
        # Third derivative of EGDF = gradient of second derivative
        third_derivative = np.gradient(second_derivative)
        
        return first_derivative, second_derivative, third_derivative
    
    def _initial_probable_bounds_estimate(self):
        """Estimate initial probable bounds (LB and UB)."""
        # Only estimate LB if it's not provided
        if self.LB is None:
            if self.data_form == 'a':
                pad = (self.DUB - self.DLB) / 2
                lb_raw = self.DLB - pad
                self.LB = DataConversion._convert_az(lb_raw, self.DLB, self.DUB)
            elif self.data_form == 'm':
                lb_raw = self.DLB / np.sqrt(self.DUB / self.DLB)
                self.LB = DataConversion._convert_mz(lb_raw, self.DLB, self.DUB)
        else:
            # Convert provided LB to appropriate form
            if self.data_form == 'a':
                self.LB = DataConversion._convert_az(self.LB, self.DLB, self.DUB)
            else:
                self.LB = DataConversion._convert_mz(self.LB, self.DLB, self.DUB)

        # Only estimate UB if it's not provided
        if self.UB is None:
            if self.data_form == 'a':
                pad = (self.DUB - self.DLB) / 2
                ub_raw = self.DUB + pad
                self.UB = DataConversion._convert_az(ub_raw, self.DLB, self.DUB)
            elif self.data_form == 'm':
                ub_raw = self.DUB * np.sqrt(self.DUB / self.DLB)
                self.UB = DataConversion._convert_mz(ub_raw, self.DLB, self.DUB)
        else:
            # Convert provided UB to appropriate form
            if self.data_form == 'a':
                self.UB = DataConversion._convert_az(self.UB, self.DLB, self.DUB)
            else:
                self.UB = DataConversion._convert_mz(self.UB, self.DLB, self.DUB)

        if self.catch:
            self.params['LB_init'] = self.LB
            self.params['UB_init'] = self.UB
    
    def _optimize_all_parameters(self):
        """Optimize S, LB, and UB simultaneously."""
        # Parameter bounds for optimization
        S_MIN, S_MAX = 0.05, 100.0
        LB_MIN, LB_MAX = 1e-10, np.exp(-1.00001)
        UB_MIN, UB_MAX = np.exp(1.00001), 1e10
        
        def transform_to_opt_space(s, lb, ub):
            s_mz = DataConversion._convert_mz(s, S_MIN, S_MAX)
            lb_mz = DataConversion._convert_mz(lb, LB_MIN, LB_MAX)
            ub_mz = DataConversion._convert_mz(ub, UB_MIN, UB_MAX)
            return np.log(s_mz), np.log(lb_mz), np.log(ub_mz)
        
        def transform_from_opt_space(s_opt, lb_opt, ub_opt):
            s_mz = np.exp(s_opt)
            lb_mz = np.exp(lb_opt)
            ub_mz = np.exp(ub_opt)
            s = DataConversion._convert_zm(s_mz, S_MIN, S_MAX)
            lb = DataConversion._convert_zm(lb_mz, LB_MIN, LB_MAX)
            ub = DataConversion._convert_zm(ub_mz, UB_MIN, UB_MAX)
            return s, lb, ub
        
        def loss_function(opt_params):
            s_opt, lb_opt, ub_opt = opt_params
            try:
                s, lb, ub = transform_from_opt_space(s_opt, lb_opt, ub_opt)
                egdf_values = self._compute_egdf(s, lb, ub)
                return np.mean(np.abs(egdf_values - self.wedf_values) * self.weights) 
            except Exception:
                return 1e6
        
        # Initial values
        s_init = 1.0
        lb_init = self.LB
        ub_init = self.UB
        
        s_opt_init, lb_opt_init, ub_opt_init = transform_to_opt_space(s_init, lb_init, ub_init)
        
        # Optimization bounds
        opt_bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
        initial_params = [
            np.clip(s_opt_init, -1.0, 1.0),
            np.clip(lb_opt_init, -1.0, 1.0),
            np.clip(ub_opt_init, -1.0, 1.0)
        ]
        
        try:
            result = minimize(loss_function, initial_params, method='L-BFGS-B', bounds=opt_bounds,
                            options={'maxiter': 1000, 'ftol': 1e-9})
            s_opt_final, lb_opt_final, ub_opt_final = result.x
            return transform_from_opt_space(s_opt_final, lb_opt_final, ub_opt_final)
        except Exception:
            return s_init, lb_init, ub_init

    #10
    def _optimize_s_only(self, lb, ub):
        """Optimize only S parameter for given LB and UB."""
        def loss_function(s):
            try:
                egdf_values = self._compute_egdf(s[0], lb, ub)
                return np.mean(np.abs(egdf_values - self.wedf_values) * self.weights) 
            except Exception:
                return 1e6
        
        try:
            result = minimize(loss_function, [1.0], bounds=[(0.05, 100.0)],
                            method='L-BFGS-B', options={'maxiter': 1000})
            return result.x[0]
        except Exception:
            return 1.0

    #11
    def _optimize_bounds_only(self, s):
        """Optimize only LB and UB for given S."""
        LB_MIN, LB_MAX = 1e-10, np.exp(-1.00001)
        UB_MIN, UB_MAX = np.exp(1.00001), 1e10

        def transform_bounds_to_opt_space(lb, ub):
            lb_mz = DataConversion._convert_mz(lb, LB_MIN, LB_MAX)
            ub_mz = DataConversion._convert_mz(ub, UB_MIN, UB_MAX)
            return np.log(lb_mz), np.log(ub_mz)
        
        def transform_bounds_from_opt_space(lb_opt, ub_opt):
            lb_mz = np.exp(lb_opt)
            ub_mz = np.exp(ub_opt)
            lb = DataConversion._convert_zm(lb_mz, LB_MIN, LB_MAX)
            ub = DataConversion._convert_zm(ub_mz, UB_MIN, UB_MAX)
            return lb, ub
        
        def loss_function(opt_params):
            lb_opt, ub_opt = opt_params
            try:
                lb, ub = transform_bounds_from_opt_space(lb_opt, ub_opt)
                egdf_values = self._compute_egdf(s, lb, ub)
                return np.mean(np.abs(egdf_values - self.wedf_values) * self.weights) 
            except Exception:
                return 1e6
        
        lb_opt_init, ub_opt_init = transform_bounds_to_opt_space(self.LB, self.UB)
        initial_params = [
            np.clip(lb_opt_init, -1.0, 1.0),
            np.clip(ub_opt_init, -1.0, 1.0)
        ]
        
        try:
            result = minimize(loss_function, initial_params, method='L-BFGS-B',
                            bounds=[(-1.0, 1.0), (-1.0, 1.0)], options={'maxiter': 1000})
            lb_opt_final, ub_opt_final = result.x
            lb_final, ub_final = transform_bounds_from_opt_space(lb_opt_final, ub_opt_final)
            return s, lb_final, ub_final
        except Exception:
            return s, self.LB, self.UB