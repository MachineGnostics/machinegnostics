'''
EGDF Marginal Analysis Module

Machine Gnostics
Author: Nirmal Parmar
'''

import numpy as np
import warnings
from machinegnostics.magcal.gdf.egdf import EGDF
from machinegnostics.magcal.gdf.homogeneity import DataHomogeneity

class BaseMarginalAnalysisEGDF:
    """
    Base class for Marginal Analysis EGDF.
    This class provides the basic structure and methods for marginal analysis.
    """

    def __init__(self,
                data: np.ndarray,
                sample_bound_tolerance: float = 0.1,
                max_iterations: int = 10000,
                early_stopping_steps: int = 10,
                estimating_rate: float = 0.1,
                cluster_threshold: float = 0.05,
                get_clusters: bool = True,
                DLB: float = None,
                DUB: float = None,
                LB: float = None,
                UB: float = None,
                S = 'auto',
                tolerance: float = 1e-6,
                data_form: str = 'a',
                n_points: int = 1000,
                homogeneous: bool = True,
                catch: bool = True,
                weights: np.ndarray = None,
                wedf: bool = True,
                opt_method: str = 'L-BFGS-B',
                verbose: bool = False,
                max_data_size: int = 1000,
                flush: bool = False):

        self.data = data
        self.sample_bound_tolerance = sample_bound_tolerance  # derivative sample bound tolerance
        self.max_iterations = max_iterations  # maximum iterations for optimization
        self.early_stopping_steps = early_stopping_steps
        self.estimating_rate = estimating_rate
        self.get_clusters = get_clusters
        self.cluster_threshold = cluster_threshold  # threshold for PDF-based clustering
        self.DLB = DLB  # lower bound for data
        self.DUB = DUB  # upper bound for data
        self.LB = LB  # lower bound for EGDF
        self.UB = UB  # upper bound for EGDF
        self.S = S  # scaling factor for EGDF
        self.tolerance = tolerance  # tolerance for calculations
        self.data_form = data_form  # format of the data
        self.n_points = n_points  # number of points for calculations
        self.homogeneous = homogeneous  # whether the data is homogeneous
        self.catch = catch  # whether to catch exceptions
        self.weights = weights  # weights for the data
        self.wedf = wedf  # whether to use WEDF
        self.opt_method = opt_method  # optimization method
        self.verbose = verbose  # verbosity of the output
        self.max_data_size = max_data_size  # maximum size of the data
        self.flush = flush  # whether to flush the output
        self.params = {}

        # vars for marginal analysis

        # derivative sample bound tolerance
        self._TOLERANCE = sample_bound_tolerance
        self._MAX_ITERATIONS = max_iterations
        self._EARLY_STOPPING_STEPS = early_stopping_steps
        self._ESTIMATING_RATE = estimating_rate
        self._EARLY_STOPPING_THRESHOLD = 0.01  # threshold for early stopping

        # validate input parameters
        self._input_validation()


    def _input_validation(self):
        """Validate input parameters for EGDF."""

        # bound tolerance
        if not isinstance(self.sample_bound_tolerance, (int, float)):
            raise ValueError(f"Bound tolerance must be a number. Current type: {type(self.sample_bound_tolerance)}.")
        
        if self.sample_bound_tolerance <= 0:
            raise ValueError(f"Bound tolerance must be greater than 0. Current value: {self.sample_bound_tolerance}.")
        
        # max iterations
        if not isinstance(self.max_iterations, int):
            raise ValueError(f"Maximum iterations must be an integer. Current type: {type(self.max_iterations)}.")

        if self.max_iterations <= 0:
            raise ValueError(f"Maximum iterations must be greater than 0. Current value: {self.max_iterations}.")
        
        # early stopping steps
        if not isinstance(self.early_stopping_steps, int):
            raise ValueError(f"Early stopping steps must be an integer. Current type: {type(self.early_stopping_steps)}.")

        if self.early_stopping_steps <= 0:
            raise ValueError(f"Early stopping steps must be greater than 0. Current value: {self.early_stopping_steps}.")

        # estimating rate
        if not isinstance(self.estimating_rate, (int, float)):
            raise ValueError(f"Estimating rate must be a number. Current type: {type(self.estimating_rate)}.")

        if self.estimating_rate <= 0:
            raise ValueError(f"Estimating rate must be greater than 0. Current value: {self.estimating_rate}.")
        
        # get clusters
        if not isinstance(self.get_clusters, bool):
            raise ValueError(f"Get clusters must be a boolean. Current type: {type(self.get_clusters)}.")
    
        # data
        if not isinstance(self.data, np.ndarray):
            raise ValueError(f"Data must be a numpy array. Current type: {type(self.data)}.")

        if self.data.size == 0:
            raise ValueError("Data cannot be empty.")
        
        if self.data.ndim != 1:
            raise ValueError(f"Data must be a 1-dimensional array. Current dimensions: {self.data.ndim}.")

        # if self.distribution not in ['egdf', 'eldf', 'qgdf', 'qldf']:
        #     raise ValueError(f"Unsupported Machine Gnostics distribution type: {self.distribution}. Supported types are 'EGDF', 'ELDF', 'QGDF', and 'QLDF'.")
        
        # varS
        # # varS only True for ELDF and QLDF
        # if self.varS:
        #     if self.distribution not in ['eldf', 'qldf']:
        #         raise ValueError(f"Variable S (varS) can only be 'TRUE' for 'ELDF' and 'QLDF' distributions. Current distribution: {self.distribution}.")
    
    def _get_initial_egdf(self):
        """Get initial EGDF based on the distribution."""
        
        # Initialize BaseEGDF with the provided parameters and base data
        self.init_egdf = EGDF(
            data=self.data,
            DLB=self.DLB,
            DUB=self.DUB,
            LB=self.LB,
            UB=self.UB,
            S=self.S,
            tolerance=self.tolerance,
            data_form=self.data_form,
            n_points=self.n_points,
            homogeneous=self.homogeneous,
            catch=self.catch,
            weights=self.weights,
            wedf=self.wedf,
            opt_method=self.opt_method,
            verbose=self.verbose,
            max_data_size=self.max_data_size,
            flush=self.flush
        )
        # fitting the initial EGDF
        self.init_egdf.fit()

        # saving bounds from initial EGDF
        self.LB = self.init_egdf.LB
        self.UB = self.init_egdf.UB
        self.DLB = self.init_egdf.DLB
        self.DUB = self.init_egdf.DUB
        self.S_opt = self.init_egdf.S_opt
        # store if catch is True
        if self.catch:
            self.params = self.init_egdf.params.copy()
        

    def _get_data_sample_bounds(self):
        """
        Estimate Lower Sample Bound (LSB) and Upper Sample Bound (USB) using iterative optimization.
        """
        # Calculate LSB
        self.LSB = self._estimate_sample_bound(bound_type='lower')
        
        # Calculate USB  
        self.USB = self._estimate_sample_bound(bound_type='upper')

        # NOTE: This Sample Bound estimation can be improved using Newton's method (or any other) for faster convergence.

        # # newton method for LSB and USB
        # self.LSB = self._estimate_sample_bound_newton(bound_type='lower')
        # self.USB = self._estimate_sample_bound_newton(bound_type='upper')

        if self.catch:
            self.params['LSB'] = float(self.LSB)
            self.params['USB'] = float(self.USB)
    
    def _estimate_sample_bound(self, bound_type='lower'):
        """
        Estimate either LSB or USB using gradient descent optimization.
        
        Parameters:
        -----------
        bound_type : str
            Either 'lower' or 'upper' to specify which bound to estimate
            
        Returns:
        --------
        float : The estimated bound value
        """
        if bound_type not in ['lower', 'upper']:
            raise ValueError("bound_type must be either 'lower' or 'upper'")
        
        # Initialize starting point
        datum = self.init_egdf.DLB if bound_type == 'lower' else self.init_egdf.DUB
        loss_history = []
        direction = -1 if bound_type == 'lower' else 1  # LSB moves left, USB moves right
        
        for i in range(self._MAX_ITERATIONS):
            previous_datum = datum
            
            # Create extended dataset and fit EGDF
            egdf_extended = self._create_extended_egdf(datum)
            
            # Get derivatives at the extended point
            derivatives = self._get_derivatives_at_point(egdf_extended, datum)
            d1, d2, d3 = derivatives['first'], derivatives['second'], derivatives['third']
            
            # Calculate loss and update datum
            loss = np.abs(d2 + d3)
            datum = datum + direction * (d1 * self._ESTIMATING_RATE)
            
            loss_history.append(loss)
            
            # Verbose output
            if self.verbose:
                bound_name = "LSB" if bound_type == 'lower' else "USB"
                print(f"Iteration {i}: {bound_name}_datum = {datum:.6f}, "
                      f"First_derivative = {d1:.6f}, Second_derivative = {d2:.6f}, "
                      f"Third_derivative = {d3:.6f}, Loss = {loss:.6f}")
            
            # Check convergence conditions
            if self._check_convergence(i, loss_history, derivatives, egdf_extended, datum, bound_type):
                if self.verbose:
                    bound_name = "LSB" if bound_type == 'lower' else "USB"
                    print(f"{bound_name} convergence reached at iteration {i} with loss {loss:.6f}")
                break
        
        return previous_datum
    
    def _create_extended_egdf(self, datum):
        """Create EGDF with extended data including the given datum."""
        data_extended = np.append(self.init_egdf.data, datum)
        
        egdf_extended = EGDF(
            data=data_extended,
            tolerance=self.tolerance,
            data_form=self.data_form,
            n_points=self.n_points,
            homogeneous=self.homogeneous,
            catch=False,
            weights=self.weights,
            wedf=self.wedf,
            opt_method=self.opt_method,
            verbose=False,
            max_data_size=self.max_data_size,
            flush=False
        )
        
        egdf_extended.fit()
        return egdf_extended
    
    def _get_derivatives_at_point(self, egdf_extended, datum):
        """Get first, second, and third derivatives at the specified datum point."""
        # Find index of the datum in the extended data
        idx = np.where(egdf_extended.data == datum)[0][0]
        
        # Calculate derivatives
        d1 = egdf_extended.pdf[idx]
        d2 = egdf_extended._get_egdf_second_derivative()[idx]
        d3 = egdf_extended._get_egdf_third_derivative()[idx]
        
        return {
            'first': d1,
            'second': d2,
            'third': d3,
            'index': idx
        }
    
    def _check_convergence(self, iteration, loss_history, derivatives, egdf_extended, datum, bound_type):
        """
        Check various convergence conditions for the optimization.
        
        Parameters:
        -----------
        iteration : int
            Current iteration number
        loss_history : list
            History of loss values
        derivatives : dict
            Dictionary containing derivative values
        egdf_extended : EGDF
            Extended EGDF object
        datum : float
            Current datum value
        bound_type : str
            'lower' or 'upper' bound type
            
        Returns:
        --------
        bool : True if convergence criteria are met
        """
        d1, d2, d3 = derivatives['first'], derivatives['second'], derivatives['third']
        idx = derivatives['index']
        
        # Calculate current loss (d2 + d3)
        current_loss = np.abs(d2 + d3)
        
        # Adaptive tolerance based on data scale and initial loss
        adaptive_tolerance = self._calculate_adaptive_tolerance(loss_history, current_loss)
        
        if self.verbose and iteration % 10 == 0:  # Print every 10 iterations to avoid spam
            print(f"Iteration {iteration}: Current loss = {current_loss:.8f}, "
                  f"Adaptive tolerance = {adaptive_tolerance:.8f}")
        
        # Check boundary constraints first (hard stops)
        if self._check_boundary_constraints(datum, bound_type):
            return True
        
        # 1. Primary convergence: d2 + d3 < adaptive_tolerance
        if current_loss < adaptive_tolerance:
            if self.verbose:
                print(f"Convergence: d2+d3 ({current_loss:.8f}) < tolerance ({adaptive_tolerance:.8f})")
            return True
        
        # 2. Secondary convergence: no change in d2 + d3 (plateau detection)
        if self._check_loss_plateau(loss_history, iteration):
            if self.verbose:
                print(f"Convergence: Loss plateau detected")
            return True
        
        # 3. Additional stability checks
        if self._check_numerical_stability(derivatives, egdf_extended, idx):
            if self.verbose:
                print(f"Convergence: Numerical instability detected")
            return True
        
        return False
    
    def _calculate_adaptive_tolerance(self, loss_history, current_loss):
        """
        Calculate adaptive tolerance based on loss history and data characteristics.
        
        Parameters:
        -----------
        loss_history : list
            History of loss values
        current_loss : float
            Current loss value
            
        Returns:
        --------
        float : Adaptive tolerance value
        """
        base_tolerance = self._TOLERANCE
        
        # If we have loss history, adapt based on initial loss magnitude
        if len(loss_history) > 0:
            initial_loss = loss_history[0]
            
            # Scale tolerance relative to initial loss magnitude
            if initial_loss > 0:
                # Use relative tolerance: base_tolerance as percentage of initial loss
                relative_tolerance = base_tolerance * initial_loss
                
                # But don't let it get too small or too large
                min_tolerance = base_tolerance * 0.01  # At least 1% of base tolerance
                max_tolerance = base_tolerance * 100   # At most 100x base tolerance
                
                adaptive_tolerance = np.clip(relative_tolerance, min_tolerance, max_tolerance)
            else:
                adaptive_tolerance = base_tolerance
        else:
            # Fallback: scale based on current loss magnitude
            if current_loss > 0:
                adaptive_tolerance = max(base_tolerance, current_loss * 0.01)  # 1% of current loss
            else:
                adaptive_tolerance = base_tolerance
        
        return adaptive_tolerance
    
    def _check_boundary_constraints(self, datum, bound_type):
        """
        Check if datum has moved outside valid boundaries.
        
        Parameters:
        -----------
        datum : float
            Current datum value
        bound_type : str
            'lower' or 'upper' bound type
            
        Returns:
        --------
        bool : True if boundary constraint violated
        """
        if bound_type == 'lower':
            if hasattr(self.init_egdf, 'LB') and self.init_egdf.LB is not None:
                if datum < self.init_egdf.LB:
                    if self.verbose:
                        print(f"Lower bound {datum:.6f} is below LB {self.init_egdf.LB:.6f}. Stopping.")
                    return True
        
        if bound_type == 'upper':
            if hasattr(self.init_egdf, 'UB') and self.init_egdf.UB is not None:
                if datum > self.init_egdf.UB:
                    if self.verbose:
                        print(f"Upper bound {datum:.6f} is above UB {self.init_egdf.UB:.6f}. Stopping.")
                    return True
        
        return False
    
    def _check_loss_plateau(self, loss_history, iteration):
        """
        Check if loss has plateaued (no significant change over recent iterations).
        
        Parameters:
        -----------
        loss_history : list
            History of loss values
        iteration : int
            Current iteration number
            
        Returns:
        --------
        bool : True if plateau detected
        """
        # Need sufficient history to detect plateau
        min_plateau_steps = max(self._EARLY_STOPPING_STEPS, 5)
        
        if iteration < min_plateau_steps:
            return False
        
        # Check if recent losses are nearly constant
        recent_losses = loss_history[-min_plateau_steps:]
        
        if len(recent_losses) < min_plateau_steps:
            return False
        
        # Calculate relative change over plateau window
        max_recent = np.max(recent_losses)
        min_recent = np.min(recent_losses)
        
        if max_recent == 0:
            return True  # All recent losses are zero
        
        relative_change = (max_recent - min_recent) / max_recent
        plateau_threshold = self._EARLY_STOPPING_THRESHOLD
        
        # Also check if loss is increasing (divergence)
        if len(recent_losses) >= 3:
            recent_trend = np.mean(np.diff(recent_losses[-3:]))
            if recent_trend > plateau_threshold:  # Loss increasing
                if self.verbose:
                    print(f"Loss diverging: recent trend = {recent_trend:.8f}")
                return True
        
        if relative_change < plateau_threshold:
            if self.verbose:
                print(f"Plateau detected: relative change = {relative_change:.8f} < {plateau_threshold:.8f}")
            return True
        
        return False
    
    def _check_numerical_stability(self, derivatives, egdf_extended, idx):
        """
        Check for numerical instability conditions.
        
        Parameters:
        -----------
        derivatives : dict
            Dictionary containing derivative values
        egdf_extended : EGDF
            Extended EGDF object
        idx : int
            Index of current datum
            
        Returns:
        --------
        bool : True if numerical instability detected
        """
        d1, d2, d3 = derivatives['first'], derivatives['second'], derivatives['third']
        
        # Check for NaN or infinite values
        if not np.isfinite([d1, d2, d3]).all():
            if self.verbose:
                print("Numerical instability: NaN or infinite derivatives detected")
            return True
        
        # Check for negative PDF values (should not happen in valid EGDF)
        if hasattr(egdf_extended, 'pdf') and egdf_extended.pdf is not None:
            if np.any(egdf_extended.pdf < 0):
                if self.verbose:
                    print("Numerical instability: Negative PDF values detected")
                return True
        
        # Check for EGDF values outside [0,1] range
        if hasattr(egdf_extended, 'egdf') and egdf_extended.egdf is not None:
            egdf_val = egdf_extended.egdf[idx] if idx < len(egdf_extended.egdf) else 0
            if egdf_val < -self._TOLERANCE or egdf_val > (1 + self._TOLERANCE):
                if self.verbose:
                    print(f"Numerical instability: EGDF value {egdf_val:.6f} outside [0,1] range")
                return True
        
        return False

        
    def _is_homogeneous(self):
        """
        Check if the data is homogeneous.
        Returns True if homogeneous, False otherwise.
        """
        ih = DataHomogeneity(self.init_egdf, catch=self.catch, verbose=self.verbose)
        is_homogeneous = ih.test_homogeneity()

        if self.catch:
            self.params['is_homogeneous'] = is_homogeneous
        return is_homogeneous
    
    
    def _get_data_sample_clusters(self):
        """
        Identify data clusters based on PDF characteristics with focus on global maxima.
        Simplified version that tries pdf_points first, then falls back to pdf.
        Includes EGDF convergence validation logic.
        
        Returns:
        --------
        dict: Dictionary containing the three clusters and critical points
        """
        # Try to get PDF data - prefer smooth pdf_points, fallback to discrete pdf
        pdf_data = None
        data_points = None
        
        if hasattr(self.init_egdf, 'pdf_points') and self.init_egdf.pdf_points is not None:
            # Use smooth PDF curve
            pdf_data = self.init_egdf.pdf_points
            data_points = self.init_egdf.di_points_n
            if self.verbose:
                print("Using smooth PDF points for clustering")
        elif hasattr(self.init_egdf, 'pdf') and self.init_egdf.pdf is not None:
            # Use discrete PDF
            pdf_data = self.init_egdf.pdf
            data_points = self.init_egdf.data
            if self.verbose:
                print("Using discrete PDF for clustering")
        else:
            if self.verbose:
                print("Warning: Neither PDF points nor PDF data available for clustering")
            return {}
        
        # Get EGDF data - prefer smooth egdf_points, fallback to discrete egdf
        egdf_data = None
        egdf_data_points = None
        
        if hasattr(self.init_egdf, 'egdf_points') and self.init_egdf.egdf_points is not None:
            # Use smooth EGDF curve
            egdf_data = self.init_egdf.egdf_points
            egdf_data_points = self.init_egdf.di_points_n
            if self.verbose:
                print("Using smooth EGDF points for validation")
        elif hasattr(self.init_egdf, 'egdf') and self.init_egdf.egdf is not None:
            # Use discrete EGDF
            egdf_data = self.init_egdf.params.get('egdf', self.init_egdf.egdf)
            egdf_data_points = self.init_egdf.data
            if self.verbose:
                print("Using discrete EGDF for validation")
        
        # Get sorted data and corresponding PDF values
        sorted_indices = np.argsort(data_points)
        sorted_data = data_points[sorted_indices]
        sorted_pdf = pdf_data[sorted_indices]
        
        # Sort EGDF data if available
        sorted_egdf = None
        if egdf_data is not None:
            if np.array_equal(data_points, egdf_data_points):
                # Same data points, use same sorting
                sorted_egdf = egdf_data[sorted_indices]
            else:
                # Different data points, need to interpolate or find closest matches
                egdf_sorted_indices = np.argsort(egdf_data_points)
                sorted_egdf_data_points = egdf_data_points[egdf_sorted_indices]
                sorted_egdf_values = egdf_data[egdf_sorted_indices]
                # Interpolate EGDF values at PDF data points
                sorted_egdf = np.interp(sorted_data, sorted_egdf_data_points, sorted_egdf_values)
        
        # Step 1: Find PDF global maxima
        global_max_idx = np.argmax(sorted_pdf)
        global_max_value = sorted_pdf[global_max_idx]
        global_max_point = sorted_data[global_max_idx]
        
        if self.verbose:
            print(f"Global PDF maximum: {global_max_value:.6f} at data point {global_max_point:.3f}")
        
        # Step 2: Calculate thresholds (simplified)
        noise_threshold = global_max_value * 0.05  # 5% of max for noise filtering
        slope_threshold = global_max_value * self.cluster_threshold  # User configurable
        
        # Calculate gradient for slope detection
        pdf_gradient = np.gradient(sorted_pdf)
        gradient_std = np.std(pdf_gradient)
        significant_gradient = gradient_std * 0.3  # 30% of gradient std
        
        if self.verbose:
            print(f"Thresholds - Noise: {noise_threshold:.6f}, Slope: {slope_threshold:.6f}, Gradient: {significant_gradient:.6f}")
        
        # Step 3: Find onset start point (where upward slope begins)
        onset_start_idx = 0
        for i in range(global_max_idx, 0, -1):
            if (sorted_pdf[i] < slope_threshold and 
                pdf_gradient[i] < significant_gradient):
                onset_start_idx = i
                break
        
        # Step 4: Find stop point (where downward slope ends)
        stop_end_idx = len(sorted_data) - 1
        for i in range(global_max_idx, len(sorted_pdf) - 1):
            if (sorted_pdf[i] < slope_threshold and 
                abs(pdf_gradient[i]) < significant_gradient):
                stop_end_idx = i
                break
        
        # Step 5: EGDF Convergence Validation Logic
        if sorted_egdf is not None:
            egdf_convergence_tolerance = 0.1  # 10% tolerance for EGDF convergence check
            
            # Check onset point: EGDF should be close to 0
            onset_egdf_value = sorted_egdf[onset_start_idx]
            onset_convergence_valid = onset_egdf_value <= egdf_convergence_tolerance
            
            # Check stop point: EGDF should be close to 1
            stop_egdf_value = sorted_egdf[stop_end_idx]
            stop_convergence_valid = stop_egdf_value >= (1 - egdf_convergence_tolerance)
            
            # if self.verbose:
            #     print(f"EGDF Convergence Validation:")
            #     print(f"  Onset point EGDF: {onset_egdf_value:.6f} (should be ≈ 0) - {'✓' if onset_convergence_valid else '✗'}")
            #     print(f"  Stop point EGDF: {stop_egdf_value:.6f} (should be ≈ 1) - {'✓' if stop_convergence_valid else '✗'}")
            
            # Refine boundaries if convergence is not valid
            if not onset_convergence_valid:
                # Search for better onset point closer to EGDF ≈ 0
                for i in range(onset_start_idx, global_max_idx):
                    if sorted_egdf[i] <= egdf_convergence_tolerance:
                        onset_start_idx = i
                        if self.verbose:
                            print(f"  Refined onset point to index {i} with EGDF: {sorted_egdf[i]:.6f}")
                        break
            
            if not stop_convergence_valid:
                # Search for better stop point closer to EGDF ≈ 1
                for i in range(stop_end_idx, 0, -1):
                    if sorted_egdf[i] >= (1 - egdf_convergence_tolerance):
                        stop_end_idx = i
                        if self.verbose:
                            print(f"  Refined stop point to index {i} with EGDF: {sorted_egdf[i]:.6f}")
                        break
        else:
            if self.verbose:
                print("EGDF data not available for convergence validation")
        
        # Step 6: Define cluster boundaries
        onset_point = sorted_data[onset_start_idx]
        stop_point = sorted_data[stop_end_idx]
        
        # Create clusters based on identified boundaries using original data
        lower_cluster_mask = self.init_egdf.data < onset_point
        main_cluster_mask = (self.init_egdf.data >= onset_point) & (self.init_egdf.data <= stop_point)
        upper_cluster_mask = self.init_egdf.data > stop_point
        
        # perform clustering based on masks
        if self.get_clusters:
            # Extract actual data points for each cluster
            lower_cluster = self.init_egdf.data[lower_cluster_mask]
            main_cluster = self.init_egdf.data[main_cluster_mask]
            upper_cluster = self.init_egdf.data[upper_cluster_mask]
            
            # Calculate cluster statistics using original discrete PDF
            if hasattr(self.init_egdf, 'pdf') and self.init_egdf.pdf is not None:
                lower_pdf_avg = np.mean(self.init_egdf.pdf[lower_cluster_mask]) if len(lower_cluster) > 0 else 0
                main_pdf_avg = np.mean(self.init_egdf.pdf[main_cluster_mask]) if len(main_cluster) > 0 else 0
                upper_pdf_avg = np.mean(self.init_egdf.pdf[upper_cluster_mask]) if len(upper_cluster) > 0 else 0
            else:
                lower_pdf_avg = main_pdf_avg = upper_pdf_avg = 0
            
            # Calculate EGDF values at boundary points for validation
            onset_egdf_final = None
            stop_egdf_final = None
            if sorted_egdf is not None:
                onset_egdf_final = sorted_egdf[onset_start_idx]
                stop_egdf_final = sorted_egdf[stop_end_idx]
        
        # clusters = {
        #     'lower_cluster': lower_cluster,
        #     'main_cluster': main_cluster,
        #     'upper_cluster': upper_cluster,
        #     'global_max_point': global_max_point,
        #     'global_max_value': global_max_value,
        #     'onset_start_point': onset_point,
        #     'stop_end_point': stop_point,
        #     'slope_threshold': slope_threshold,
        #     'noise_threshold': noise_threshold,
        #     'lower_pdf_avg': lower_pdf_avg,
        #     'main_pdf_avg': main_pdf_avg,
        #     'upper_pdf_avg': upper_pdf_avg,
        #     'onset_egdf_value': onset_egdf_final,
        #     'stop_egdf_value': stop_egdf_final
        # }
        
            if self.verbose:
                print(f"Data sample clustering completed.")
            
            self.CLB = onset_point
            self.CUB = stop_point
        
            if self.catch:
                self.params.update({
                    'lower_cluster': lower_cluster,
                    'main_cluster': main_cluster, 
                    'upper_cluster': upper_cluster,
                })
        # bound update        
        if self.catch:
            self.params.update({
                'CLB': onset_point,
                'CUB': stop_point,
            })


    def _estimate_sample_bound_newton(self, bound_type='lower'):
        """
        Newton-Raphson method for quadratic convergence.
        """
        # Initial guess
        x = self.init_egdf.DLB if bound_type == 'lower' else self.init_egdf.DUB
        
        for i in range(min(20, self._MAX_ITERATIONS)):  # Newton converges very fast
            egdf_extended = self._create_extended_egdf(x)
            derivatives = self._get_derivatives_at_point(egdf_extended, x)
            
            # f(x) = d2 + d3 (we want this to be zero)
            f_val = derivatives['second'] + derivatives['third']
            
            # f'(x) = d3 + d4 (derivative of our target function)
            # For simplicity, approximate f'(x) using numerical differentiation
            h = 1e-6
            egdf_h = self._create_extended_egdf(x + h)
            derivatives_h = self._get_derivatives_at_point(egdf_h, x + h)
            f_prime = (derivatives_h['second'] + derivatives_h['third'] - f_val) / h
            
            if np.abs(f_prime) < 1e-12:  # Avoid division by zero
                break
                
            # Newton update
            x_new = x - f_val / f_prime
            
            # Check convergence
            if np.abs(f_val) < self._TOLERANCE:
                if self.verbose:
                    print(f"Newton method converged at iteration {i} with loss {np.abs(f_val):.8f}")
                return x_new
                
            if self.verbose:
                print(f"Newton iteration {i}: x = {x:.6f}, f(x) = {f_val:.8f}")
                
            x = x_new

        return x

    def _get_z0(self, egdf: EGDF):
        """
        Find Z0 point where:
        1. PDF is at global maximum 
        2. EGDF is at 0.5
        3. Second derivative (d2) approaches zero
        
        Returns:
        --------
        float: The Z0 value
        """
        # Start with median of the data as initial estimate
        zo_est = np.median(egdf.data)
        loss_history = []
        
        if self.verbose:
            print(f"\nFinding Z0 starting from median: {zo_est:.6f}")
        
        for iteration in range(self._MAX_ITERATIONS):
            # Create extended EGDF with the current estimate
            egdf_extended = self._create_extended_egdf(zo_est)
            
            # Get derivatives at the Z0 estimate point
            derivatives = self._get_derivatives_at_point(egdf_extended, zo_est)
            d1, d2, d3 = derivatives['first'], derivatives['second'], derivatives['third']
            z0_idx = derivatives['index']
            
            # Get EGDF value at Z0 estimate
            egdf_at_z0 = egdf_extended.egdf[z0_idx] if hasattr(egdf_extended, 'egdf') else 0.5
            
            # Calculate multi-objective loss:
            # 1. d2 should be close to 0 (PDF at maximum)
            # 2. EGDF should be close to 0.5
            # 3. d2/d1 ratio for numerical stability
            
            d2_loss = np.abs(d2)  # Second derivative should be zero at maximum
            egdf_loss = np.abs(egdf_at_z0 - 0.5)  # EGDF should be 0.5
            ratio_loss = np.abs(d2 / d1) if np.abs(d1) > 1e-12 else 0  # Stability measure
            
            # Combined loss with weights
            loss = d2_loss + 2.0 * egdf_loss + 0.5 * ratio_loss
            loss_history.append(loss)
            
            if self.verbose:
                print(f"Iteration {iteration}: z0_est = {zo_est:.6f}, "
                      f"d1(PDF) = {d1:.6f}, d2 = {d2:.6f}, d3 = {d3:.6f}")
                print(f"  EGDF at z0 = {egdf_at_z0:.6f}, d2_loss = {d2_loss:.6f}, "
                      f"egdf_loss = {egdf_loss:.6f}, total_loss = {loss:.6f}")
            
            # Check convergence
            if loss < self._TOLERANCE:
                if self.verbose:
                    print(f"Z0 convergence reached at iteration {iteration} with loss {loss:.6f}")
                break
            
            # Update z0_est using gradient descent with adaptive step
            if iteration > 0 and len(loss_history) >= 2:
                # Adaptive learning rate based on loss history
                if loss_history[-1] > loss_history[-2]:
                    # Loss is increasing, reduce step size
                    step_size = self._ESTIMATING_RATE * 0.5
                else:
                    # Loss is decreasing, maintain or slightly increase step size
                    step_size = self._ESTIMATING_RATE
            else:
                step_size = self._ESTIMATING_RATE
            
            # Update z0_est using Newton's method when possible
            if np.abs(d1) > 1e-12:
                # Newton step for d2 = 0 (finding PDF maximum)
                newton_step = d2 / d3 if np.abs(d3) > 1e-12 else 0
                
                # EGDF correction step (move towards EGDF = 0.5)
                egdf_step = (egdf_at_z0 - 0.5) * step_size
                
                # Combined update
                zo_est = zo_est - newton_step - egdf_step
            else:
                # Fallback to simple gradient descent
                zo_est = zo_est - loss * step_size
            
            # Boundary constraints to keep z0 within reasonable range
            data_range = egdf.DUB - egdf.DLB
            zo_est = np.clip(zo_est, 
                            egdf.DLB - 0.1 * data_range,
                            egdf.DUB + 0.1 * data_range)

            # Early stopping if loss plateaus
            if iteration > self._EARLY_STOPPING_STEPS:
                recent_losses = loss_history[-self._EARLY_STOPPING_STEPS:]
                if len(recent_losses) >= self._EARLY_STOPPING_STEPS:
                    loss_change = np.std(recent_losses) / np.mean(recent_losses)
                    if loss_change < 0.01:  # 1% relative change
                        if self.verbose:
                            print(f"Z0 early stopping at iteration {iteration} due to loss plateau")
                        break
        
        # Final validation and refinement
        final_egdf_extended = self._create_extended_egdf(zo_est)
        final_derivatives = self._get_derivatives_at_point(final_egdf_extended, zo_est)
        final_z0_idx = final_derivatives['index']
        final_egdf_at_z0 = final_egdf_extended.egdf[final_z0_idx] if hasattr(final_egdf_extended, 'egdf') else 0.5
        
        # Additional validation: check if this is actually near PDF maximum
        pdf_at_z0 = final_derivatives['first']
        max_pdf_in_data = np.max(self.init_egdf.pdf) if hasattr(self.init_egdf, 'pdf') else pdf_at_z0
        
        if self.verbose:
            print(f"\nZ0 Final Results:")
            print(f"  Z0 value: {zo_est:.6f}")
        
        # Store Z0 and related information
        if self.catch:
            self.params.update({
                'z0': float(zo_est)
            })
        
        # Validate results
        if abs(final_egdf_at_z0 - 0.5) > 0.001:  # 0.1% tolerance for EGDF = 0.5
            if self.verbose:
                print(f"Warning: Z0 EGDF value {final_egdf_at_z0:.6f} is far from 0.5")

        if abs(final_derivatives['second']) > 0.001:  # 0.1% tolerance for second derivative ≈ 0
            if self.verbose:
                print(f"Warning: Z0 second derivative {final_derivatives['second']:.6f} is not close to 0")
                
        self.z0 = float(zo_est)
        return self.z0

    
    def _add_marginal_points(self, ax, bounds=True):
        """Add marginal analysis points to plot."""
        marginal_info = []
        
        # Always add Z0 regardless of bounds setting
        if hasattr(self, 'params') and 'z0' in self.params:
            marginal_info.append((self.params['z0'], 'magenta', '-.', 'Z0'))
        
        # Only add other marginal points if bounds=True
        if bounds:
            if hasattr(self, 'LSB') and self.LSB is not None:
                marginal_info.append((self.LSB, 'darkred', ':', 'LSB'))
            if hasattr(self, 'USB') and self.USB is not None:
                marginal_info.append((self.USB, 'darkblue', ':', 'USB'))
            
            # Add CLB and CUB (Cluster Lower Bound and Cluster Upper Bound)
            if hasattr(self, 'CLB') and self.CLB is not None:
                marginal_info.append((self.CLB, 'orange', '--', 'CLB'))
            if hasattr(self, 'CUB') and self.CUB is not None:
                marginal_info.append((self.CUB, 'orange', '--', 'CUB'))
    
        for point, color, style, name in marginal_info:
            # Make CLB, CUB, and Z0 lines very thin as requested
            linewidth = 1 if name in ['CLB', 'CUB', 'Z0'] else 2
            alpha = 0.6 if name in ['CLB', 'CUB', 'Z0'] else 0.8
            
            ax.axvline(x=point, color=color, linestyle=style, linewidth=linewidth, 
                    alpha=alpha, label=f"{name}={point:.3f}")
    
    def _add_bounds(self, ax):
        """Add bound lines to plot."""
        bound_info = []
        
        if hasattr(self.init_egdf, 'DLB') and self.init_egdf.DLB is not None:
            bound_info.append((self.init_egdf.DLB, 'green', '-', 'DLB'))
        if hasattr(self.init_egdf, 'DUB') and self.init_egdf.DUB is not None:
            bound_info.append((self.init_egdf.DUB, 'orange', '-', 'DUB'))
        if hasattr(self.init_egdf, 'LB') and self.init_egdf.LB is not None:
            bound_info.append((self.init_egdf.LB, 'purple', '--', 'LB'))
        if hasattr(self.init_egdf, 'UB') and self.init_egdf.UB is not None:
            bound_info.append((self.init_egdf.UB, 'brown', '--', 'UB'))
        
        for bound, color, style, name in bound_info:
            ax.axvline(x=bound, color=color, linestyle=style, linewidth=1, 
                      alpha=0.6, label=f"{name}={bound:.3f}")
    
    def _plot_egdf(self, plot_type: str = 'marginal', plot_smooth: bool = True, bounds: bool = True, derivatives: bool = False, figsize: tuple = (12, 8)):
        """
        Enhanced plotting for marginal analysis with LSB, USB, and clustering visualization.
        
        Parameters:
        -----------
        plot_type : str, default='marginal'
            Type of plot: 'marginal', 'egdf', 'pdf', 'both', 'clusters'
        plot_smooth : bool, default=True
            Whether to plot smooth curves
        bounds : bool, default=True
            Whether to show bounds (LB, UB, DLB, DUB, LSB, USB, CLB, CUB)
        derivatives : bool, default=False
            Whether to show derivative analysis plot
        figsize : tuple, default=(12, 8)
            Figure size
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not self.catch:
            print("Plot is not available with argument catch=False")
            return
        
        if not hasattr(self.init_egdf, '_fitted') or not self.init_egdf._fitted:
            raise RuntimeError("Must fit marginal analysis before plotting.")
        
        # Validate plot_type
        valid_types = ['marginal', 'egdf', 'pdf', 'both']
        if plot_type not in valid_types:
            raise ValueError(f"plot_type must be one of {valid_types}")
        
        if derivatives:
            self._plot_derivatives(figsize=figsize)
            return
        
        # Create single plot with dual y-axes
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Get data
        x_points = self.init_egdf.data
        egdf_vals = self.init_egdf.params.get('egdf')
        pdf_vals = self.init_egdf.params.get('pdf')
        wedf_vals = self.init_egdf.params.get('wedf')
        
        # Plot EGDF on primary y-axis
        if plot_type in ['marginal', 'egdf', 'both']:
            if plot_smooth and hasattr(self.init_egdf, 'egdf_points') and self.init_egdf.egdf_points is not None:
                ax1.plot(x_points, egdf_vals, 'o', color='blue', label='EGDF', markersize=4)
                ax1.plot(self.init_egdf.di_points_n, self.init_egdf.egdf_points, 
                        color='blue', linestyle='-', linewidth=2, alpha=0.8)
            else:
                ax1.plot(x_points, egdf_vals, 'o-', color='blue', label='EGDF', 
                        markersize=4, linewidth=2, alpha=0.8)
        
        # Plot WEDF if available
        if wedf_vals is not None and plot_type in ['marginal', 'egdf', 'both']:
            ax1.plot(x_points, wedf_vals, 's', color='lightblue', 
                    label='WEDF', markersize=3, alpha=0.8)
        
        ax1.set_xlabel('Data Points')
        ax1.set_ylabel('EGDF', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(0, 1)
        
        # Create secondary y-axis for PDF
        ax2 = ax1.twinx()
        
        if plot_type in ['marginal', 'pdf', 'both']:
            if plot_smooth and hasattr(self.init_egdf, 'pdf_points') and self.init_egdf.pdf_points is not None:
                ax2.plot(x_points, pdf_vals, 'o', color='red', label='PDF', markersize=4)
                ax2.plot(self.init_egdf.di_points_n, self.init_egdf.pdf_points, 
                        color='red', linestyle='-', linewidth=2, alpha=0.8)
                max_pdf = np.max(self.init_egdf.pdf_points)
            else:
                ax2.plot(x_points, pdf_vals, 'o-', color='red', label='PDF', 
                        markersize=4, linewidth=2, alpha=0.8)
                max_pdf = np.max(pdf_vals) if pdf_vals is not None else 1
        
        ax2.set_ylabel('PDF', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        if 'max_pdf' in locals():
            ax2.set_ylim(0, max_pdf * 1.1)
        
        # Add bounds only if bounds=True
        if bounds:
            self._add_bounds(ax1)
        
        # Add marginal points (Z0 always, others only if bounds=True)
        self._add_marginal_points(ax1, bounds=bounds)
        
        # Set xlim to DLB-DUB range
        if hasattr(self.init_egdf, 'DLB') and hasattr(self.init_egdf, 'DUB'):
            # 5% data pad on either side
            pad = (self.init_egdf.DUB - self.init_egdf.DLB) * 0.05
            ax1.set_xlim(self.init_egdf.DLB - pad, self.init_egdf.DUB + pad)
            ax2.set_xlim(self.init_egdf.DLB - pad, self.init_egdf.DUB + pad)
    
        # Add shaded regions for bounds only if bounds=True
        if bounds and hasattr(self.init_egdf, 'LB') and hasattr(self.init_egdf, 'UB'):
            if self.init_egdf.LB is not None:
                ax1.axvspan(self.init_egdf.DLB, self.init_egdf.LB, alpha=0.15, color='purple')
            if self.init_egdf.UB is not None:
                ax1.axvspan(self.init_egdf.UB, self.init_egdf.DUB, alpha=0.15, color='brown')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Set title
        plt.title('EGDF and PDF with Bounds and Marginal Points', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_derivatives(self, figsize: tuple = (14, 10)):
        """Plot EGDF derivatives for marginal analysis visualization."""
        import matplotlib.pyplot as plt
        
        if not self.catch:
            print("Plot is not available with argument catch=False")
            return
            
        if not hasattr(self.init_egdf, '_fitted') or not self.init_egdf._fitted:
            raise RuntimeError("Must fit EGDF before plotting derivatives.")
        
        # Get derivatives from init_egdf
        try:
            d1_vals = self.init_egdf.pdf
            d2_vals = self.init_egdf._get_egdf_second_derivative()
            d3_vals = self.init_egdf._get_egdf_third_derivative()
        except:
            print("Error: Could not calculate derivatives")
            return
        
        # Create subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot derivatives
        ax1.plot(self.init_egdf.data, d1_vals, 'b-', linewidth=2, label='1st Derivative (PDF)')
        ax1.set_title('First Derivative (PDF)')
        ax1.set_ylabel('PDF Value')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(self.init_egdf.data, d2_vals, 'r-', linewidth=2, label='2nd Derivative')
        ax2.set_title('Second Derivative')
        ax2.set_ylabel('2nd Derivative Value')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3.plot(self.init_egdf.data, d3_vals, 'g-', linewidth=2, label='3rd Derivative')
        ax3.set_title('Third Derivative')
        ax3.set_xlabel('Data Points')
        ax3.set_ylabel('3rd Derivative Value')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Combined derivatives for boundary analysis
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Derivative Analysis for LSB/USB Detection')
        ax4.set_xlabel('Data Points')
        ax4.set_ylabel('Derivative Value')
        ax4.grid(True, alpha=0.3)
        
        # For derivatives plot, always show all marginal points
        for ax in [ax1, ax2, ax3, ax4]:
            self._add_marginal_points(ax, bounds=True)
        
        plt.suptitle('EGDF Derivative Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def _fit_egdf(self, plot=True):
        try:
            if self.verbose:
                print("\n\nFitting EGDF Marginal Analysis...")

            # get initial EGDF
            self._get_initial_egdf()

            # get Z0 of the base sample
            self.z0 = self._get_z0(self.init_egdf)

            # homogeneous check
            self.h = self._is_homogeneous()
            if self.h:
                if self.verbose:
                    print("Data is homogeneous. Using homogeneous data for Marginal Analysis.")
            else:
                if self.verbose:
                    print("Data is heterogeneous. Need to estimate cluster bounds to find main cluster.")
            
            # h check
            if self.h == False and self.get_clusters == False:
                warnings.warn("Data is heterogeneous but get_clusters is False. "
                            "Consider setting 'get_clusters=True' to find main cluster bounds.")

            # optional data sampling bounds
            self._get_data_sample_bounds()

            # cluster bounds
            self._get_data_sample_clusters() # if get_clusters is True, it will estimate cluster bounds

            if self.verbose:
                print(f"Marginal Analysis completed. Z0: {self.z0:.6f}, Homogeneous: {self.h}")

            if plot:
                self._plot_egdf()

        except Exception as e:
            if self.verbose:
                print(f"Error occurred during fitting: {e}")
