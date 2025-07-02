"""
Machine Gnostics - Joint Parameter Optimizer
Copyright (C) 2025  Machine Gnostics Team

Author: Nirmal Parmar
License: GNU General Public License v3.0 (GPL-3.0)

Joint optimization of Scale Parameter (S), Lower Sample Boundary (LSB), and Upper Sample Boundary (USB)
for EGDF fitting following equation (21.6) from the theoretical framework.

This module implements two-stage optimization:
1. Simultaneous optimization of S, LSB, USB until S reaches a fidelity plateau
2. Freeze S and optimize only LSB, USB for fine-tuning boundaries

Terminology:
- S: Scale parameter for fidelity calculation  
- LSB/USB: Sample boundaries (inner bounds where samples are expected)
- LB/UB: Outer bounds (data transformation bounds, LB < LSB < LP < USB < UB)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, dual_annealing
from scipy.stats import kstest
import warnings
warnings.filterwarnings('ignore')

class JointParameterOptimizer:
    """
    Two-stage optimization of Scale Parameter (S), Lower Sample Boundary (LSB), and Upper Sample Boundary (USB)
    for EGDF fitting based on equation (21.6).
    
    Stage 1: Simultaneous optimization of S, LSB, USB until S reaches fidelity plateau
    Stage 2: Freeze S and optimize only LSB, USB for boundary refinement
    
    This approach ensures robust scale parameter estimation followed by precise boundary tuning.
    """
    
    def __init__(self, data, egdf_cdf, wedf_values, z_points, data_lb=None, data_ub=None, verbose=True):
        """
        Initialize the joint optimizer.
        
        Parameters:
        -----------
        data : array-like
            Original data points
        egdf_cdf : array-like
            EGDF CDF values
        wedf_values : array-like
            WEDF values
        z_points : array-like
            Z points where distributions are evaluated
        data_lb, data_ub : float, optional
            Outer bounds for data transformation (distinct from LSB/USB sample boundaries)
        verbose : bool
            Whether to print detailed information
        """
        self.data = np.array(data)
        self.egdf_cdf = np.array(egdf_cdf)
        self.wedf_values = np.array(wedf_values)
        self.z_points = np.array(z_points)
        self.verbose = verbose
        
        # Data preprocessing
        self.data_min = self.data.min()
        self.data_max = self.data.max()
        self.data_range = self.data_max - self.data_min
        
        # Outer bounds for data transformation (if provided)
        self.data_lb = data_lb if data_lb is not None else self.data_min - self.data_range
        self.data_ub = data_ub if data_ub is not None else self.data_max + self.data_range
        
        # Clean data for optimization
        self._clean_data()
        
        # Optimization history and plateau detection
        self.optimization_history = []
        self.best_result = None
        self.plateau_detected = False
        self.plateau_tolerance = 0.001  # Relative change threshold for plateau detection
        self.plateau_window = 10  # Number of iterations to check for plateau
        
        if self.verbose:
            print("JointParameterOptimizer initialized successfully")
            print(f"Data range: [{self.data_min:.6f}, {self.data_max:.6f}]")
            print(f"Data bounds: LB={self.data_lb:.6f}, UB={self.data_ub:.6f}")
            print(f"Valid data points: {len(self.egdf_clean)} / {len(self.egdf_cdf)}")
            print("Two-stage optimization: S/LSB/USB → freeze S → LSB/USB refinement")
    
    def _clean_data(self):
        """Clean and validate input data."""
        # Create mask for valid data points
        mask = (~np.isnan(self.egdf_cdf) & ~np.isnan(self.wedf_values) & 
                ~np.isinf(self.egdf_cdf) & ~np.isinf(self.wedf_values) &
                (self.egdf_cdf > 0) & (self.wedf_values > 0) &
                (self.egdf_cdf <= 1) & (self.wedf_values <= 1))
        
        self.egdf_clean = self.egdf_cdf[mask]
        self.wedf_clean = self.wedf_values[mask]
        self.z_clean = self.z_points[mask]
        
        if len(self.egdf_clean) < 10:
            raise ValueError("Too few valid data points for reliable optimization")
    
    def _calculate_fidelity(self, params):
        """
        Calculate fidelity for given parameters following equation (21.6).
        
        Parameters:
        -----------
        params : tuple
            (S, LSB, USB) parameters
            
        Returns:
        --------
        float
            Negative fidelity (for minimization)
        """
        try:
            S, LSB, USB = params
            
            # Parameter validation
            if S <= 0 or S > 100:
                return 1e10
            if LSB >= USB:
                return 1e10
            if USB - LSB < self.data_range * 0.1:  # Minimum reasonable range
                return 1e10
            
            # Check if sample boundaries are within outer bounds
            if LSB < self.data_lb or USB > self.data_ub:
                return 1e10
            
            # Check if sample boundaries contain the data reasonably
            if LSB > self.data_min or USB < self.data_max:
                return 1e10
            
            # Safety parameters
            eps = 1e-12
            
            # Calculate ratios with sample boundaries consideration
            # Adjust EGDF based on new sample boundaries (simple linear scaling)
            z_normalized = (self.z_clean - LSB) / (USB - LSB)
            z_normalized = np.clip(z_normalized, eps, 1 - eps)
            
            # Recalculate EGDF with new sample boundaries
            egdf_adjusted = z_normalized  # Simplified adjustment
            
            # Calculate fidelity components
            ratio = egdf_adjusted / (self.wedf_clean + eps)
            ratio = np.clip(ratio, eps, 1/eps)
            
            # qk calculation with scale parameter
            qk = ratio ** (2 / S)
            qk = np.clip(qk, eps, 1/eps)
            
            qk1 = 1 / qk
            fi = 2 / (qk + qk1)
            
            # Remove any non-finite values
            fi_valid = fi[np.isfinite(fi)]
            
            if len(fi_valid) == 0:
                return 1e10
            
            # Mean fidelity
            fidelity = np.mean(fi_valid)
            
            # Add penalty for extreme parameter values
            penalty = 0
            if S < 0.1 or S > 50:
                penalty += abs(S - 1) * 0.1
            
            # Sample boundary penalty
            range_ratio = (USB - LSB) / self.data_range
            if range_ratio < 1.5 or range_ratio > 10:
                penalty += abs(range_ratio - 3) * 0.01
            
            # Return negative fidelity for minimization
            result = -fidelity + penalty
            
            # Store in history
            self.optimization_history.append({
                'S': S, 'LSB': LSB, 'USB': USB, 
                'fidelity': fidelity, 'objective': result
            })
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"Error in fidelity calculation: {e}")
            return 1e10
    
    def _get_parameter_bounds(self, stage='full'):
        """
        Get reasonable bounds for optimization parameters.
        
        Parameters:
        -----------
        stage : str
            'full' for full optimization, 'boundaries_only' for LSB/USB refinement
        """
        if stage == 'full':
            # Scale parameter bounds
            S_bounds = (0.1, 20.0)
            
            # LSB bounds - should be between data_lb and data_min
            LSB_bounds = (self.data_lb, self.data_min)
            
            # USB bounds - should be between data_max and data_ub  
            USB_bounds = (self.data_max, self.data_ub)
            
            return [S_bounds, LSB_bounds, USB_bounds]
        
        elif stage == 'boundaries_only':
            # Only LSB and USB bounds for boundary refinement
            LSB_bounds = (self.data_lb, self.data_min)
            USB_bounds = (self.data_max, self.data_ub)
            
            return [LSB_bounds, USB_bounds]
    
    def _get_initial_guess(self, stage='full'):
        """
        Get reasonable initial guess for parameters.
        
        Parameters:
        -----------
        stage : str
            'full' for full optimization, 'boundaries_only' for LSB/USB refinement
        """
        if stage == 'full':
            S_init = 1.0  # Start with neutral scale
            LSB_init = self.data_min - self.data_range * 0.5  # Extend below data
            USB_init = self.data_max + self.data_range * 0.5  # Extend above data
            
            return [S_init, LSB_init, USB_init]
        
        elif stage == 'boundaries_only':
            LSB_init = self.data_min - self.data_range * 0.5
            USB_init = self.data_max + self.data_range * 0.5
            
            return [LSB_init, USB_init]
    
    def _detect_plateau(self):
        """
        Detect if scale parameter S has reached a plateau in optimization.
        
        Returns:
        --------
        bool
            True if plateau detected, False otherwise
        """
        if len(self.optimization_history) < self.plateau_window:
            return False
        
        # Get recent S values
        recent_S = [h['S'] for h in self.optimization_history[-self.plateau_window:]]
        
        # Calculate relative change in S
        S_min = min(recent_S)
        S_max = max(recent_S)
        
        if S_min == 0:
            return False
        
        relative_change = (S_max - S_min) / S_min
        
        # Check if change is below threshold
        plateau_detected = relative_change < self.plateau_tolerance
        
        if plateau_detected and self.verbose:
            print(f"Plateau detected: S varies only {relative_change:.6f} over last {self.plateau_window} iterations")
            print(f"S range: [{S_min:.6f}, {S_max:.6f}]")
        
        return plateau_detected
    
    def _calculate_fidelity_boundaries_only(self, params, fixed_S):
        """
        Calculate fidelity for LSB/USB optimization with fixed S.
        
        Parameters:
        -----------
        params : tuple
            (LSB, USB) parameters
        fixed_S : float
            Fixed scale parameter
            
        Returns:
        --------
        float
            Negative fidelity (for minimization)
        """
        LSB, USB = params
        return self._calculate_fidelity((fixed_S, LSB, USB))
    
    def optimize_joint_parameters(self, method='differential_evolution'):
        """
        Perform two-stage optimization of S, LSB, and USB parameters.
        
        Stage 1: Simultaneous optimization until S plateau is detected
        Stage 2: Freeze S and optimize only LSB, USB for refinement
        
        Parameters:
        -----------
        method : str
            Optimization method ('differential_evolution', 'dual_annealing', 'nelder_mead')
            
        Returns:
        --------
        dict
            Optimization results including both stages
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("TWO-STAGE JOINT OPTIMIZATION")
            print('='*60)
            print("Stage 1: Simultaneous S, LSB, USB optimization")
            print("Stage 2: Fixed S, LSB/USB refinement")
            print(f"Method: {method}")
        
        # STAGE 1: Full optimization with plateau detection
        stage1_result = self._optimize_stage1(method)
        
        if not stage1_result['success']:
            return stage1_result
        
        # STAGE 2: Boundary refinement with fixed S
        stage2_result = self._optimize_stage2(stage1_result, method)
        
        # Combine results
        final_result = {
            'success': True,
            'method': method,
            'stage1_result': stage1_result,
            'stage2_result': stage2_result,
            'S_optimal': stage2_result['S_optimal'],
            'LSB_optimal': stage2_result['LSB_optimal'], 
            'USB_optimal': stage2_result['USB_optimal'],
            'fidelity_optimal': stage2_result['fidelity_optimal'],
            'plateau_detected': self.plateau_detected,
            'total_function_evaluations': stage1_result.get('function_evaluations', 0) + stage2_result.get('function_evaluations', 0)
        }
        
        self.best_result = final_result
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("TWO-STAGE OPTIMIZATION COMPLETED")
            print('='*60)
            print(f"Final parameters:")
            print(f"  S = {final_result['S_optimal']:.6f}")
            print(f"  LSB = {final_result['LSB_optimal']:.6f}")
            print(f"  USB = {final_result['USB_optimal']:.6f}")
            print(f"  Fidelity = {final_result['fidelity_optimal']:.6f}")
            print(f"  Plateau detected: {final_result['plateau_detected']}")
            print(f"  Total evaluations: {final_result['total_function_evaluations']}")
        
        return final_result
    
    def _optimize_stage1(self, method):
        """Stage 1: Simultaneous S, LSB, USB optimization until plateau."""
        bounds = self._get_parameter_bounds('full')
        initial_guess = self._get_initial_guess('full')
        
        if self.verbose:
            print(f"\nStage 1 - Parameter bounds: S{bounds[0]}, LSB{bounds[1]}, USB{bounds[2]}")
            print(f"Initial guess: S={initial_guess[0]:.3f}, LSB={initial_guess[1]:.3f}, USB={initial_guess[2]:.3f}")
        
        self.optimization_history = []
        self.plateau_detected = False
        
        try:
            if method == 'differential_evolution':
                # Custom callback to check for plateau
                def callback(xk, convergence):
                    self.plateau_detected = self._detect_plateau()
                    return self.plateau_detected  # Stop if plateau detected
                
                result = differential_evolution(
                    self._calculate_fidelity,
                    bounds=bounds,
                    maxiter=200,
                    popsize=15,
                    tol=1e-6,
                    seed=42,
                    callback=callback,
                    disp=False
                )
                
            elif method == 'dual_annealing':
                # For dual_annealing, we need to manually check plateau
                max_iter = 500
                for i in range(max_iter):
                    if i == 0:
                        result = dual_annealing(
                            self._calculate_fidelity,
                            bounds=bounds,
                            maxiter=50,  # Smaller chunks
                            initial_temp=5230,
                            seed=42
                        )
                    else:
                        # Continue from previous result
                        result = dual_annealing(
                            self._calculate_fidelity,
                            bounds=bounds,
                            maxiter=50,
                            x0=result.x,
                            seed=42
                        )
                    
                    if self._detect_plateau():
                        self.plateau_detected = True
                        break
                
            elif method == 'nelder_mead':
                # For Nelder-Mead, we can't easily interrupt, so just run normally
                result = minimize(
                    self._calculate_fidelity,
                    x0=initial_guess,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'disp': False}
                )
                # Check if we reached plateau at the end
                self.plateau_detected = self._detect_plateau()
                
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            if result.success or self.plateau_detected:
                S_opt, LSB_opt, USB_opt = result.x
                fidelity_opt = -result.fun
                
                stage1_result = {
                    'success': True,
                    'method': method,
                    'S_optimal': S_opt,
                    'LSB_optimal': LSB_opt, 
                    'USB_optimal': USB_opt,
                    'fidelity_optimal': fidelity_opt,
                    'plateau_detected': self.plateau_detected,
                    'function_evaluations': result.nfev if hasattr(result, 'nfev') else len(self.optimization_history),
                    'message': f"Stage 1 completed - {'plateau detected' if self.plateau_detected else 'converged'}"
                }
                
                if self.verbose:
                    print(f"Stage 1 completed: {'Plateau detected' if self.plateau_detected else 'Converged'}")
                    print(f"  S = {S_opt:.6f}, LSB = {LSB_opt:.6f}, USB = {USB_opt:.6f}")
                    print(f"  Fidelity = {fidelity_opt:.6f}")
                
                return stage1_result
                
            else:
                return {
                    'success': False,
                    'method': method,
                    'message': f"Stage 1 failed: {result.message if hasattr(result, 'message') else 'Unknown error'}"
                }
                
        except Exception as e:
            if self.verbose:
                print(f"Stage 1 threw exception: {e}")
            
            return {
                'success': False,
                'method': method,
                'error': str(e)
            }
    
    def _optimize_stage2(self, stage1_result, method):
        """Stage 2: Boundary refinement with fixed S."""
        fixed_S = stage1_result['S_optimal']
        bounds = self._get_parameter_bounds('boundaries_only')
        initial_guess = self._get_initial_guess('boundaries_only')
        
        if self.verbose:
            print(f"\nStage 2 - Boundary refinement with fixed S = {fixed_S:.6f}")
            print(f"LSB bounds: {bounds[0]}, USB bounds: {bounds[1]}")
            print(f"Initial guess: LSB={initial_guess[0]:.3f}, USB={initial_guess[1]:.3f}")
        
        # Store stage 1 history
        stage1_history = self.optimization_history.copy()
        self.optimization_history = []
        
        try:
            if method == 'differential_evolution':
                result = differential_evolution(
                    lambda params: self._calculate_fidelity_boundaries_only(params, fixed_S),
                    bounds=bounds,
                    maxiter=100,
                    popsize=10,
                    tol=1e-6,
                    seed=42,
                    disp=False
                )
                
            elif method == 'dual_annealing':
                result = dual_annealing(
                    lambda params: self._calculate_fidelity_boundaries_only(params, fixed_S),
                    bounds=bounds,
                    maxiter=200,
                    initial_temp=5230,
                    seed=42
                )
                
            elif method == 'nelder_mead':
                result = minimize(
                    lambda params: self._calculate_fidelity_boundaries_only(params, fixed_S),
                    x0=initial_guess,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'disp': False}
                )
                
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            if result.success:
                LSB_opt, USB_opt = result.x
                fidelity_opt = -result.fun
                
                stage2_result = {
                    'success': True,
                    'method': method,
                    'S_optimal': fixed_S,  # Unchanged from stage 1
                    'LSB_optimal': LSB_opt, 
                    'USB_optimal': USB_opt,
                    'fidelity_optimal': fidelity_opt,
                    'function_evaluations': result.nfev if hasattr(result, 'nfev') else len(self.optimization_history),
                    'message': "Stage 2 boundary refinement completed"
                }
                
                # Combine optimization history
                self.optimization_history = stage1_history + self.optimization_history
                
                if self.verbose:
                    print(f"Stage 2 completed successfully")
                    print(f"  Final LSB = {LSB_opt:.6f}, USB = {USB_opt:.6f}")
                    print(f"  Final fidelity = {fidelity_opt:.6f}")
                    
                    # Show improvement
                    stage1_fidelity = stage1_result['fidelity_optimal']
                    improvement = fidelity_opt - stage1_fidelity
                    print(f"  Fidelity improvement: {improvement:.6f} ({improvement/stage1_fidelity*100:.2f}%)")
                
                return stage2_result
                
            else:
                # Fall back to stage 1 results if stage 2 fails
                if self.verbose:
                    print(f"Stage 2 failed, using Stage 1 results")
                
                return {
                    'success': True,
                    'method': method,
                    'S_optimal': stage1_result['S_optimal'],
                    'LSB_optimal': stage1_result['LSB_optimal'], 
                    'USB_optimal': stage1_result['USB_optimal'],
                    'fidelity_optimal': stage1_result['fidelity_optimal'],
                    'function_evaluations': 0,
                    'message': f"Stage 2 failed, used Stage 1 results: {result.message if hasattr(result, 'message') else 'Unknown error'}"
                }
                
        except Exception as e:
            if self.verbose:
                print(f"Stage 2 threw exception: {e}, using Stage 1 results")
            
            return {
                'success': True,
                'method': method,
                'S_optimal': stage1_result['S_optimal'],
                'LSB_optimal': stage1_result['LSB_optimal'], 
                'USB_optimal': stage1_result['USB_optimal'],
                'fidelity_optimal': stage1_result['fidelity_optimal'],
                'function_evaluations': 0,
                'error': str(e)
            }
    
    def optimize_all_methods(self):
        """
        Try multiple optimization methods and return the best result.
        
        Returns:
        --------
        dict
            Best optimization results across all methods
        """
        methods = ['differential_evolution', 'dual_annealing', 'nelder_mead']
        results = {}
        best_fidelity = -np.inf
        best_method = None
        
        for method in methods:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"TRYING METHOD: {method.upper()}")
                print('='*60)
            
            result = self.optimize_joint_parameters(method=method)
            results[method] = result
            
            if result['success'] and result['fidelity_optimal'] > best_fidelity:
                best_fidelity = result['fidelity_optimal']
                best_method = method
                self.best_result = result
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("OPTIMIZATION SUMMARY")
            print('='*60)
            
            for method, result in results.items():
                if result['success']:
                    print(f"{method:20}: SUCCESS - Fidelity = {result['fidelity_optimal']:.6f}")
                else:
                    print(f"{method:20}: FAILED")
            
            if best_method:
                print(f"\nBest method: {best_method}")
                print(f"Best fidelity: {best_fidelity:.6f}")
        
        return {
            'all_results': results,
            'best_method': best_method,
            'best_result': self.best_result
        }
    
    def evaluate_quality(self, S, LSB, USB):
        """
        Evaluate the quality of fitted parameters using multiple metrics.
        
        Parameters:
        -----------
        S, LSB, USB : float
            Parameters to evaluate
            
        Returns:
        --------
        dict
            Quality metrics
        """
        try:
            # Calculate fidelity
            fidelity = -self._calculate_fidelity((S, LSB, USB))
            
            # Calculate other quality metrics
            z_normalized = (self.z_clean - LSB) / (USB - LSB)
            z_normalized = np.clip(z_normalized, 1e-12, 1 - 1e-12)
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = kstest(z_normalized, 'uniform')
            
            # Coverage analysis
            data_coverage = np.sum((self.data >= LSB) & (self.data <= USB)) / len(self.data)
            
            # Range efficiency
            range_efficiency = self.data_range / (USB - LSB)
            
            return {
                'fidelity': fidelity,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'data_coverage': data_coverage,
                'range_efficiency': range_efficiency,
                'parameter_summary': {
                    'S': S,
                    'LSB': LSB,
                    'USB': USB,
                    'range_width': USB - LSB,
                    'scale_category': 'narrow' if S < 0.5 else 'wide' if S > 2 else 'moderate'
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'fidelity': np.nan
            }
    
    def plot_optimization_progress(self, figsize=(15, 10)):
        """
        Plot the optimization progress and results.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        if not self.optimization_history:
            print("No optimization history available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Extract history
        iterations = range(len(self.optimization_history))
        S_vals = [h['S'] for h in self.optimization_history]
        LSB_vals = [h['LSB'] for h in self.optimization_history]
        USB_vals = [h['USB'] for h in self.optimization_history]
        fidelity_vals = [h['fidelity'] for h in self.optimization_history]
        objective_vals = [h['objective'] for h in self.optimization_history]
        
        # Plot S evolution
        axes[0,0].plot(iterations, S_vals, 'b-', alpha=0.7)
        axes[0,0].set_title('Scale Parameter (S) Evolution')
        axes[0,0].set_xlabel('Iteration')
        axes[0,0].set_ylabel('S')
        axes[0,0].grid(True, alpha=0.3)
        
        # Mark plateau detection region if available
        if self.plateau_detected and len(S_vals) >= self.plateau_window:
            plateau_start = len(S_vals) - self.plateau_window
            axes[0,0].axvspan(plateau_start, len(S_vals)-1, alpha=0.2, color='yellow', label='Plateau Region')
        
        # Plot LSB evolution
        axes[0,1].plot(iterations, LSB_vals, 'r-', alpha=0.7)
        axes[0,1].axhline(self.data_min, color='k', linestyle='--', label='Data Min')
        axes[0,1].set_title('Lower Sample Boundary (LSB) Evolution')
        axes[0,1].set_xlabel('Iteration')
        axes[0,1].set_ylabel('LSB')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # Plot USB evolution
        axes[0,2].plot(iterations, USB_vals, 'g-', alpha=0.7)
        axes[0,2].axhline(self.data_max, color='k', linestyle='--', label='Data Max')
        axes[0,2].set_title('Upper Sample Boundary (USB) Evolution')
        axes[0,2].set_xlabel('Iteration')
        axes[0,2].set_ylabel('USB')
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].legend()
        
        # Plot fidelity evolution
        axes[1,0].plot(iterations, fidelity_vals, 'purple', alpha=0.7)
        axes[1,0].set_title('Fidelity Evolution')
        axes[1,0].set_xlabel('Iteration')
        axes[1,0].set_ylabel('Fidelity')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot objective function
        axes[1,1].plot(iterations, objective_vals, 'orange', alpha=0.7)
        axes[1,1].set_title('Objective Function (Negative Fidelity)')
        axes[1,1].set_xlabel('Iteration')
        axes[1,1].set_ylabel('Objective Value')
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot parameter correlation
        axes[1,2].scatter(S_vals, fidelity_vals, alpha=0.6, c=iterations, cmap='viridis')
        axes[1,2].set_xlabel('Scale Parameter (S)')
        axes[1,2].set_ylabel('Fidelity')
        axes[1,2].set_title('S vs Fidelity')
        axes[1,2].grid(True, alpha=0.3)
        
        if self.best_result and self.best_result['success']:
            # Mark best result
            best_S = self.best_result['S_optimal']
            best_LSB = self.best_result['LSB_optimal']
            best_USB = self.best_result['USB_optimal']
            best_fidelity = self.best_result['fidelity_optimal']
            
            axes[0,0].axhline(best_S, color='red', linestyle=':', label=f'Optimal: {best_S:.3f}')
            axes[0,1].axhline(best_LSB, color='red', linestyle=':', label=f'Optimal: {best_LSB:.3f}')
            axes[0,2].axhline(best_USB, color='red', linestyle=':', label=f'Optimal: {best_USB:.3f}')
            axes[1,0].axhline(best_fidelity, color='red', linestyle=':', label=f'Optimal: {best_fidelity:.3f}')
            
            # Add annotation for two-stage optimization if applicable
            if 'stage1_result' in self.best_result:
                # Mark transition between stages (approximate)
                stage1_evals = self.best_result['stage1_result'].get('function_evaluations', 0)
                if stage1_evals < len(iterations):
                    for ax in axes.flat:
                        ax.axvline(stage1_evals, color='blue', linestyle='--', alpha=0.5, label='Stage 1→2')
            
            for ax in axes.flat:
                ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Example usage and testing
if __name__ == "__main__":
    print("JointParameterOptimizer module loaded successfully!")
