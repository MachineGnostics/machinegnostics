"""
Bound Estimator for EGDF and other GDFs.

Machine Gnostics
Author: Nirmal Parmar
"""
import numpy as np

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

    def _find_optimized_probable_bounds(self, target_cdf_lower=0.00001, target_cdf_upper=0.99999):
        """
        Estimate LB and UB (probable bounds) for EGDF.
        LB and UB are the lower and upper bounds of the EGDF at min and max of the EGDF values.

        Find optimized bounds using interpolation, ensuring bounds are outside original data range.
        
        Parameters:
        target_cdf_lower (float): Target CDF value for lower bound
        target_cdf_upper (float): Target CDF value for upper bound
        """
        egdf_values = self.params['egdf']
        di_points = self.params['di_points']
        data_min = self.data.min()
        data_max = self.data.max()
        
        # Interpolate to find exact points where EGDF = target values
        from scipy.interpolate import interp1d
        
        # Create interpolation function (EGDF -> di_points)
        # Remove any duplicate EGDF values for interpolation
        unique_indices = np.unique(egdf_values, return_index=True)[1]
        egdf_unique = egdf_values[unique_indices]
        di_unique = di_points[unique_indices]
        
        # Sort by EGDF values for interpolation
        sort_indices = np.argsort(egdf_unique)
        egdf_sorted = egdf_unique[sort_indices]
        di_sorted = di_unique[sort_indices]
        
        if len(egdf_sorted) > 1:
            interp_func = interp1d(egdf_sorted, di_sorted, 
                                bounds_error=False, fill_value='extrapolate')
            
            # Find bounds
            optimized_lower = float(interp_func(target_cdf_lower))
            optimized_upper = float(interp_func(target_cdf_upper))
            
            # Ensure bounds are outside data range
            # Lower bound must be < data.min()
            if optimized_lower >= data_min:
                # Find the rightmost di_point that is < data_min with lowest CDF
                valid_lower_points = di_points[di_points < data_min]
                if len(valid_lower_points) > 0:
                    # Get the index of the rightmost valid point
                    valid_lower_idx = np.where(di_points < data_min)[0][-1]
                    optimized_lower = di_points[valid_lower_idx]
                else:
                    # If no valid points, use minimum di_point
                    optimized_lower = di_points[0]
            
            # Upper bound must be > data.max()
            if optimized_upper <= data_max:
                # Find the leftmost di_point that is > data_max with highest CDF
                valid_upper_points = di_points[di_points > data_max]
                if len(valid_upper_points) > 0:
                    # Get the index of the leftmost valid point
                    valid_upper_idx = np.where(di_points > data_max)[0][0]
                    optimized_upper = di_points[valid_upper_idx]
                else:
                    # If no valid points, use maximum di_point
                    optimized_upper = di_points[-1]
                    
        else:
            # Fallback to edge values
            optimized_lower = di_points[0]
            optimized_upper = di_points[-1]
        
        # Store optimized bounds in params
        if self.catch:
            self.params['LB'] = optimized_lower
            self.params['UB'] = optimized_upper
        else:
            self.params['LB'] = None
            self.params['UB'] = None

    def _optimize_probable_bounds(self):
        """
        Optimize the probable bounds (LB, UB) and S for EGDF.

        criteria function is: minimize LB, UB and S for maximum fidelity.
        """
        # current fidelity
        fi = self.params['fidelity']
        fi_mean = np.mean(fi)

        # bound search range
        z = self.params['z']

        # initial bounds
        lb_init = self.params['LB_init']
        ub_init = self.params['UB_init']

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
    
    def _find_data_support_bounds(self):
        pass