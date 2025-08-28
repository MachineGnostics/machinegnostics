'''
base ELDF class
Estimating Local Distribution Functions

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
import warnings
from scipy.optimize import minimize
from machinegnostics.magcal.characteristics import GnosticsCharacteristics
from machinegnostics.magcal.gdf.base_df import BaseDistFunc
from machinegnostics.magcal.data_conversion import DataConversion
from machinegnostics.magcal.gdf.wedf import WEDF
from machinegnostics.magcal.mg_weights import GnosticsWeights
from machinegnostics.magcal.gdf.base_egdf import BaseEGDF
from machinegnostics.magcal.gdf.base_distfunc import BaseDistFuncCompute

class BaseELDF(BaseDistFuncCompute):
    '''Base ELDF class'''
    def __init__(self,
                 data: np.ndarray,
                 DLB: float = None,
                 DUB: float = None,
                 LB: float = None,
                 UB: float = None,
                 S = 'auto',
                 varS: bool = False,
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
                         varS=varS, 
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
        self.varS = varS
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

        # Validate all inputs
        self._validate_inputs()
        
        # Store initial parameters if catching
        if self.catch:
            self._store_initial_params()

    def _fit_eldf(self):
        """Fit the ELDF model to the data."""
    #     try:
        if self.verbose:
            print("Starting ELDF fitting process...")

        # Step 1: Data preprocessing
        self.data = np.sort(self.data)
        self._estimate_data_bounds()
        self._transform_data_to_standard_domain()
        self._estimate_weights()
        
        # Step 2: Bounds estimation
        self._estimate_initial_probable_bounds()
        self._generate_evaluation_points()
        
#         # Step 3: Get distribution function values for optimization
#         self.df_values = self._get_distribution_function_values(use_wedf=self.wedf)
        
#         # Step 4: Parameter optimization
#         self._determine_optimization_strategy()
        
#         # Step 5: Calculate final EGDF and PDF
#         self._calculate_final_results()
        
#         # Step 6: Generate smooth curves for plotting and analysis
#         self._generate_smooth_curves()
        
#         # Step 7: Transform bounds back to original domain
#         self._transform_bounds_to_original_domain()
        
#         # Mark as fitted (Step 8 is now optional via marginal_analysis())
#         self._fitted = True
        
#         if self.verbose:
#             print("EGDF fitting completed successfully.")
        
#         # clean up computation cache
#         if self.flush:  
#             self._cleanup_computation_cache()
                
    #     except Exception as e:
    #         if self.verbose:
    #             print(f"Error during EGDF fitting: {e}")
    #         raise e

    def _plot_eldf(self):
        """Plot the ELDF model."""
        # Implement plotting logic here
        pass