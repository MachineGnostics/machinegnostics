'''
EGDF Marginal Analysis Module

Machine Gnostics
Author: Nirmal Parmar
'''

import numpy as np
from machinegnostics.magcal.gdf.egdf import EGDF
from machinegnostics.magcal.gdf.homogeneity import DataHomogeneity

class BaseMarginalAnalysisEGDF:
    """
    Base class for Marginal Analysis EGDF.
    This class provides the basic structure and methods for marginal analysis.
    """

    def __init__(self,
                data: np.ndarray,
                bound_tolerance: float = 0.1,
                max_iterations: int = 10000,
                early_stopping_steps: int = 10,
                estimating_rate: float = 0.1,
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
        self.bound_tolerance = bound_tolerance  # derivative sample bound tolerance
        self.max_iterations = max_iterations  # maximum iterations for optimization
        self.early_stopping_steps = early_stopping_steps
        self.estimating_rate = estimating_rate
        self.get_clusters = get_clusters
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
        self._TOLERANCE = bound_tolerance
        self._MAX_ITERATIONS = max_iterations
        self._EARLY_STOPPING_STEPS = early_stopping_steps
        self._ESTIMATING_RATE = estimating_rate
        self._EARLY_STOPPING_THRESHOLD = 0.01  # threshold for early stopping

        # validate input parameters
        self._input_validation()


    def _input_validation(self):
        """Validate input parameters for EGDF."""

        # bound tolerance
        if not isinstance(self.bound_tolerance, (int, float)):
            raise ValueError(f"Bound tolerance must be a number. Current type: {type(self.bound_tolerance)}.")
        
        if self.bound_tolerance <= 0:
            raise ValueError(f"Bound tolerance must be greater than 0. Current value: {self.bound_tolerance}.")
        
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
        '''
        Given LSB and USB estimate get clusters

        if data is non-homogeneous, then use the LSB and USB to estimate clusters.

        lower cluster = LB < data < LSB
        upper cluster = USB < data < UB
        main cluster = LSB < data < USB
        '''
        # if non-homogeneous, then use the LSB and USB to estimate clusters
        if not self.h:
            # lower cluster
            lower_cluster = self.init_egdf.data[(self.init_egdf.data > self.init_egdf.LB) & (self.init_egdf.data < self.LSB)]
            # upper cluster
            upper_cluster = self.init_egdf.data[(self.init_egdf.data > self.USB) & (self.init_egdf.data < self.init_egdf.UB)]
            # main cluster
            main_cluster = self.init_egdf.data[(self.init_egdf.data > self.LSB) & (self.init_egdf.data < self.USB)]

            print(f"Lower cluster: {lower_cluster}")
            print(f"Upper cluster: {upper_cluster}")
            print(f"Main cluster: {main_cluster}")

            if self.verbose:
                print("Data sample grouped into clusters.")

            if self.catch:
                self.params['lower_cluster'] = lower_cluster
                self.params['upper_cluster'] = upper_cluster
                self.params['main_cluster'] = main_cluster

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

    def _get_z0(self):
        pass

    def plot(self, plot_type: str = 'marginal', plot_smooth: bool = True, bounds: bool = True, derivatives: bool = False, figsize: tuple = (12, 8)):
        """
        Enhanced plotting for marginal analysis with LSB, USB, and clustering visualization.
        
        Parameters:
        -----------
        plot_type : str, default='marginal'
            Type of plot: 'marginal', 'egdf', 'pdf', 'both', 'clusters'
        plot_smooth : bool, default=True
            Whether to plot smooth curves
        bounds : bool, default=True
            Whether to show bounds (LB, UB, DLB, DUB)
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
            self.plot_derivatives(figsize=figsize)
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
        
        # Add bounds
        if bounds:
            self._add_bounds(ax1)
        
        # Add marginal points (LSB, USB)
        self._add_marginal_points(ax1)
        
        # Set xlim to DLB-DUB range
        if hasattr(self.init_egdf, 'DLB') and hasattr(self.init_egdf, 'DUB'):
            # 5% data pad on either side
            pad = (self.init_egdf.DUB - self.init_egdf.DLB) * 0.05
            ax1.set_xlim(self.init_egdf.DLB - pad, self.init_egdf.DUB + pad)
            ax2.set_xlim(self.init_egdf.DLB - pad, self.init_egdf.DUB + pad)

        # Add shaded regions for bounds
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
    
    def _add_marginal_points(self, ax):
        """Add marginal analysis points to plot."""
        marginal_info = []
        
        # Add Z0 if available
        if hasattr(self, 'params') and 'z0' in self.params:
            marginal_info.append((self.params['z0'], 'magenta', '-.', 'Z0'))
        
        if hasattr(self, 'LSB') and self.LSB is not None:
            marginal_info.append((self.LSB, 'darkred', ':', 'LSB'))
        if hasattr(self, 'USB') and self.USB is not None:
            marginal_info.append((self.USB, 'darkblue', ':', 'USB'))
        
        for point, color, style, name in marginal_info:
            ax.axvline(x=point, color=color, linestyle=style, linewidth=2, 
                      alpha=0.8, label=f"{name}={point:.3f}")
    
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
    
    def plot_derivatives(self, figsize: tuple = (14, 10)):
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
        
        # Add marginal points to all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            self._add_marginal_points(ax)
        
        plt.suptitle('EGDF Derivative Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def fit(self, plot=True):

        # get initial EGDF
        self._get_initial_egdf()

        if self.verbose:
            print("\n\nFitting EGDF Marginal Analysis...")

        # get data sample bounds
        self._get_data_sample_bounds()

        # get Z0 of the base sample
        
        # homogeneous check
        self.h = self._is_homogeneous()

        # data sample clusters
        if self.get_clusters:
            self._get_data_sample_clusters()

        if plot:
            self.plot()
