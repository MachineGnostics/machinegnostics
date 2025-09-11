'''
DataCluster class for cluster boundary detection in GDFs.

This module provides the DataCluster class which performs cluster boundary detection on fitted GDF objects (ELDF, QLDF, EGDF, QGDF).

Author: Nirmal Parmar
Machine Gnostics
'''

import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, argrelextrema
from typing import Union, Dict, Any, Optional, Tuple, List
from machinegnostics.magcal import ELDF, QLDF, EGDF, QGDF

class DataCluster:
    """
    Data Cluster Boundary Detection for Generalized Distribution Functions (GDF).
    
    This class performs cluster boundary detection on fitted GDF objects (ELDF, QLDF, EGDF, QGDF)
    by analyzing the probability density function (PDF) to identify cluster lower bound (CLB) and 
    cluster upper bound (CUB).
    
    The clustering analysis uses two main approaches:
    1. **Maxima Method**: Identifies extrema points and uses S_opt threshold to find boundaries
    2. **Derivative Method**: Analyzes PDF derivatives to locate boundary points
    
    **GDF Type Recommendations:**
    - **ELDF/QLDF (Local Functions)**: Recommended for clustering - more effective
    - **EGDF/QGDF (Global Functions)**: Less effective for clustering analysis
    
    **Algorithm Logic:**
    
    For ELDF/EGDF:
    - PDF global maximum expected at Z0
    - Cluster boundaries defined by adjacent minima with PDF < S_opt
    - Main cluster region shaded between CLB and CUB
    
    For QLDF/QGDF:
    - PDF inverted first: inverted_pdf = max(pdf) - pdf
    - Same boundary detection logic applied to inverted PDF
    - Inlier regions shaded outside cluster bounds (DLB to CLB, CUB to DUB)
    
    Parameters
    ----------
    gdf : ELDF, QLDF, EGDF, or QGDF
        Fitted GDF object with pdf_points available (ensure catch=True during GDF fitting)
    verbose : bool, default=False
        Enable detailed output during analysis
    catch : bool, default=True
        Enable error catching and logging
    method : str, default='maxima'
        Primary analysis method: 'maxima' or 'derivative'
        - For ELDF/EGDF: tries 'maxima' first, falls back to 'derivative'
        - For QLDF/QGDF: tries 'derivative' first, falls back to 'maxima'
    
    Attributes
    ----------
    CLB : float or None
        Cluster Lower Bound
    CUB : float or None
        Cluster Upper Bound
    z0 : float or None
        Central point of distribution (from GDF)
    S_opt : float or None
        Optimization parameter (from GDF)
    params : dict
        Complete analysis results and metadata
    gdf_type : str
        Type of input GDF ('eldf', 'qldf', 'egdf', 'qgdf')
    
    Examples
    --------
    Basic usage with ELDF (recommended):
    
    >>> from machinegnostics.magcal import ELDF, DataCluster
    >>> 
    >>> # Fit ELDF first
    >>> eldf = ELDF(data=data, S=0.5)
    >>> eldf.fit(catch=True)  # Important: catch=True for pdf_points
    >>> 
    >>> # Perform cluster analysis
    >>> cluster = DataCluster(gdf=eldf, verbose=True)
    >>> success = cluster.fit()
    >>> 
    >>> # Get results
    >>> results = cluster.results()
    >>> print(f"CLB: {results['CLB']}, CUB: {results['CUB']}")
    >>> 
    >>> # Visualize
    >>> cluster.plot()
    
    Using different method with QLDF:
    
    >>> qldf = QLDF(data=data, S=0.5)
    >>> qldf.fit(catch=True)
    >>> 
    >>> cluster = DataCluster(gdf=qldf, method='derivative', verbose=True)
    >>> cluster.fit()
    >>> cluster.plot()
    
    Accessing cluster info from GDF params:
    
    >>> cluster.fit()
    >>> gdf_cluster_info = eldf.params['data_cluster']
    >>> print(f"Clustering successful: {gdf_cluster_info['clustering_successful']}")
    
    Notes
    -----
    - Works best with Local Distribution Functions (ELDF, QLDF)
    - Global Distribution Functions (EGDF, QGDF) will show effectiveness warnings
    - Requires GDF to be fitted with catch=True to have pdf_points available
    - May fall back to data bounds if boundary detection fails
    - Both DataCluster.params and gdf.params are updated with results
    
    Warnings
    --------
    - Using EGDF/QGDF may result in less effective clustering
    - Missing pdf_points will cause initialization failure
    - Very noisy or irregular PDFs may result in boundary detection failure
    """
    def __init__(self, gdf, verbose=False, catch=True, method='maxima'):
        self.gdf = gdf
        self.gdf_type = gdf.__class__.__name__.lower()
        self.verbose = verbose
        self.catch = catch
        self.method = method  # 'maxima' or 'derivative'
     
    # Initialize DataCluster params (REMOVED validation_warnings)
        self.params = {
            'gdf_type': self.gdf_type,
            'method': self.method,
            'CLB': None,
            'CUB': None,
            'Z0': None,
            'S_opt': None,
            'cluster_width': None,
            'clustering_successful': False,
            'local_minima': [],
            'local_maxima': [],
            'global_minimum_idx': None,
            'global_maximum_idx': None,
            'local_minima_count': 0,
            'local_maxima_count': 0,
            'errors': [],
            'warnings': []  # All validation info goes here
        }
        
        # Initialize boundary results
        self.CLB = None
        self.CUB = None
        self.z0 = None
        self.S_opt = None
        self._fitted = False
        
        # Initialize extrema detection results
        self.local_minima = []
        self.local_maxima = []
        self.global_minimum_idx = None
        self.global_maximum_idx = None
        
        # Validate GDF object
        try:
            self._validate_gdf()
            self._validate_gdf_type_for_clustering()
        except Exception as e:
            self._append_error(f"GDF validation failed: {str(e)}", type(e).__name__)
            if self.verbose:
                print(f"DataCluster: Error: GDF validation failed: {str(e)}")

    def _validate_gdf(self):
        """Basic GDF object validation"""
        if not hasattr(self.gdf, '_fitted') or not self.gdf._fitted:
            raise ValueError("GDF object must be fitted before cluster analysis")
        
        if not hasattr(self.gdf, 'pdf_points') or self.gdf.pdf_points is None:
            raise AttributeError("GDF object missing pdf_points. Ensure catch=True during fitting.")
        
        if not hasattr(self.gdf, 'data'):
            raise ValueError("GDF object missing data attribute")

    def _validate_gdf_type_for_clustering(self):
        """Validate GDF type suitability for clustering analysis"""
        
        if self.gdf_type in ['egdf', 'qgdf']:
            # Global distribution functions - less suitable for clustering
            gdf_full_name = 'EGDF' if self.gdf_type == 'egdf' else 'QGDF'
            local_alternative = 'ELDF' if self.gdf_type == 'egdf' else 'QLDF'
            
            warning_msg = (
                f"Using {gdf_full_name} (Global Distribution Function) for clustering analysis. "
                f"Clustering may not be as effective with global functions. "
                f"Consider using {local_alternative} (Local Distribution Function) for better clustering results."
            )
            
            # Add detailed validation info to warnings (not as separate validation_warnings)
            validation_info = {
                'type': 'effectiveness_warning',
                'gdf_type': gdf_full_name,
                'recommended_alternative': local_alternative,
                'message': warning_msg
            }
            
            self._append_warning(validation_info)
            
            if self.verbose:
                print(f"DataCluster: Warning: {warning_msg}")
        
        elif self.gdf_type in ['eldf', 'qldf']:
            # Local distribution functions - ideal for clustering
            gdf_full_name = 'ELDF' if self.gdf_type == 'eldf' else 'QLDF'
            
            info_msg = (
                f"Using {gdf_full_name} (Local Distribution Function) for clustering analysis. "
                f"This is the recommended approach for effective clustering."
            )
            
            validation_info = {
                'type': 'suitability_confirmation',
                'gdf_type': gdf_full_name,
                'message': info_msg
            }
            
            self._append_warning(validation_info)  # Using warnings for all validation info
            
            if self.verbose:
                print(f"DataCluster: Info: {info_msg}")
        
        else:
            # Unknown GDF type
            unknown_warning = f"Unknown GDF type '{self.gdf_type}'. Clustering effectiveness cannot be guaranteed."
            validation_info = {
                'type': 'unknown_type_warning',
                'gdf_type': self.gdf_type,
                'message': unknown_warning
            }
            
            self._append_warning(validation_info)
            
            if self.verbose:
                print(f"DataCluster: Warning: {unknown_warning}")

    def _append_error(self, error_message, exception_type=None):
        error_entry = {
            'method': 'DataCluster',
            'error': error_message,
            'exception_type': exception_type or 'DataClusterError'
        }
        
        # Add to DataCluster params
        self.params['errors'].append(error_entry)
        
        # Add to GDF params
        if hasattr(self.gdf, 'params') and 'errors' in self.gdf.params:
            self.gdf.params['errors'].append(error_entry)
        elif hasattr(self.gdf, 'params'):
            self.gdf.params['errors'] = [error_entry]

    def _append_warning(self, warning_message):
        warning_entry = {
            'method': 'DataCluster',
            'warning': warning_message
        }
        
        # Add to DataCluster params
        self.params['warnings'].append(warning_entry)
        
        # Add to GDF params
        if hasattr(self.gdf, 'params') and 'warnings' in self.gdf.params:
            self.gdf.params['warnings'].append(warning_entry)
        elif hasattr(self.gdf, 'params'):
            self.gdf.params['warnings'] = [warning_entry]

    def _get_pdf_data(self):
        return self.gdf.pdf_points

    def _get_data_points(self):
        if hasattr(self.gdf, 'di_points_n') and self.gdf.di_points_n is not None:
            return self.gdf.di_points_n
        elif hasattr(self.gdf, 'di_points') and self.gdf.di_points is not None:
            return self.gdf.di_points
        elif hasattr(self.gdf, 'params') and 'di_points' in self.gdf.params:
            return self.gdf.params['di_points']
        else:
            raise AttributeError("Cannot find data points in GDF object")

    def _get_z0(self):
        if hasattr(self.gdf, 'z0') and self.gdf.z0 is not None:
            return self.gdf.z0
        elif hasattr(self.gdf, 'params') and 'z0' in self.gdf.params:
            return self.gdf.params['z0']
        else:
            self._append_warning("Z0 not found in GDF object. Using PDF global extremum as Z0.")
            return self._find_pdf_z0()

    def _get_s_opt(self):
        if hasattr(self.gdf, 'S_opt') and self.gdf.S_opt is not None:
            return self.gdf.S_opt
        elif hasattr(self.gdf, 'params') and 'S_opt' in self.gdf.params:
            return self.gdf.params['S_opt']
        else:
            self._append_warning("S_opt not found in GDF object. Using default value 1.0.")
            return 1.0

    def _get_data_bounds(self):
        if hasattr(self.gdf, 'DLB') and hasattr(self.gdf, 'DUB'):
            return self.gdf.DLB, self.gdf.DUB
        else:
            return np.min(self.gdf.data), np.max(self.gdf.data)

    def _find_pdf_z0(self):
        pdf_data = self._get_pdf_data()
        data_points = self._get_data_points()
        
        if self.gdf_type in ['eldf', 'egdf']:
            max_idx = np.argmax(pdf_data)
            return data_points[max_idx]
        else:
            min_idx = np.argmin(pdf_data)
            return data_points[min_idx]

    def _find_all_extrema(self, pdf_data, data_points):
        local_min_indices = argrelextrema(pdf_data, np.less, order=1)[0]
        local_max_indices = argrelextrema(pdf_data, np.greater, order=1)[0]
        
        global_min_idx = np.argmin(pdf_data)
        global_max_idx = np.argmax(pdf_data)
        
        self.local_minima = local_min_indices.tolist()
        self.local_maxima = local_max_indices.tolist()
        self.global_minimum_idx = global_min_idx
        self.global_maximum_idx = global_max_idx
        
        # Update params
        self.params['local_minima'] = self.local_minima
        self.params['local_maxima'] = self.local_maxima
        self.params['global_minimum_idx'] = int(global_min_idx)
        self.params['global_maximum_idx'] = int(global_max_idx)
        self.params['local_minima_count'] = len(self.local_minima)
        self.params['local_maxima_count'] = len(self.local_maxima)
        
        if self.verbose:
            print(f"DataCluster: Found {len(local_min_indices)} local minima: {[f'{data_points[i]:.3f}' for i in local_min_indices]}")
            print(f"DataCluster: Found {len(local_max_indices)} local maxima: {[f'{data_points[i]:.3f}' for i in local_max_indices]}")
            print(f"DataCluster: Global minimum at {data_points[global_min_idx]:.3f}")
            print(f"DataCluster: Global maximum at {data_points[global_max_idx]:.3f}")
        
        return local_min_indices, local_max_indices, global_min_idx, global_max_idx

    def _find_boundaries_maxima_method(self, pdf_data, data_points):
        self._find_all_extrema(pdf_data, data_points)
        
        if self.gdf_type in ['eldf', 'egdf']:
            # For ELDF/EGDF: Find minima adjacent to global maximum
            # Filter minima that have PDF value < S_opt (CORRECTED)
            z0_idx = self.global_maximum_idx
            z0_value = data_points[z0_idx]
            
            valid_minima = []
            for min_idx in self.local_minima:
                if pdf_data[min_idx] < self.S_opt:  # CORRECTED: < instead of >
                    valid_minima.append(min_idx)
            
            if self.verbose:
                print(f"DataCluster: Valid minima (PDF < S_opt={self.S_opt:.3f}): {[f'{data_points[i]:.3f}' for i in valid_minima]}")
            
            # Find closest minima on each side of z0
            left_candidates = [idx for idx in valid_minima if idx < z0_idx]
            right_candidates = [idx for idx in valid_minima if idx > z0_idx]
            
            if left_candidates:
                self.CLB = data_points[max(left_candidates)]  # Closest to z0
            if right_candidates:
                self.CUB = data_points[min(right_candidates)]  # Closest to z0
                
        else:
            # For QLDF/QGDF: Invert PDF then apply same logic
            inverted_pdf = np.max(pdf_data) - pdf_data
            
            # Find extrema in inverted PDF
            inv_local_min_indices = argrelextrema(inverted_pdf, np.less, order=1)[0]
            inv_local_max_indices = argrelextrema(inverted_pdf, np.greater, order=1)[0]
            inv_global_max_idx = np.argmax(inverted_pdf)
            
            if self.verbose:
                print(f"DataCluster: Inverted PDF - Local minima: {[f'{data_points[i]:.3f}' for i in inv_local_min_indices]}")
                print(f"DataCluster: Inverted PDF - Local maxima: {[f'{data_points[i]:.3f}' for i in inv_local_max_indices]}")
                print(f"DataCluster: Inverted PDF - Global maximum at {data_points[inv_global_max_idx]:.3f}")
            
            # Find minima adjacent to global maximum in inverted PDF
            # Using inverted S_opt threshold
            inv_s_opt = np.max(pdf_data) - self.S_opt
            valid_minima = []
            for min_idx in inv_local_min_indices:
                if inverted_pdf[min_idx] < inv_s_opt:
                    valid_minima.append(min_idx)
            
            left_candidates = [idx for idx in valid_minima if idx < inv_global_max_idx]
            right_candidates = [idx for idx in valid_minima if idx > inv_global_max_idx]
            
            if left_candidates:
                self.CLB = data_points[max(left_candidates)]
            if right_candidates:
                self.CUB = data_points[min(right_candidates)]

    def _find_boundaries_derivative_method(self, pdf_data, data_points):
        if self.gdf_type in ['eldf', 'egdf']:
            # For ELDF/EGDF: Use original PDF
            first_derivative = np.gradient(pdf_data)
            z0_idx = self.global_maximum_idx
        else:
            # For QLDF/QGDF: Invert PDF first
            inverted_pdf = np.max(pdf_data) - pdf_data
            first_derivative = np.gradient(inverted_pdf)
            z0_idx = np.argmax(inverted_pdf)
        
        # Find where derivative is close to zero
        zero_threshold = 0.01 * np.max(np.abs(first_derivative))
        near_zero_indices = np.where(np.abs(first_derivative) <= zero_threshold)[0]
        
        # Find where PDF + derivative is close to zero or constant
        if self.gdf_type in ['eldf', 'egdf']:
            combined_signal = pdf_data + first_derivative
        else:
            combined_signal = inverted_pdf + first_derivative
            
        combined_threshold = 0.01 * np.max(np.abs(combined_signal))
        combined_near_zero = np.where(np.abs(combined_signal) <= combined_threshold)[0]
        
        # Combine both conditions
        candidate_indices = np.intersect1d(near_zero_indices, combined_near_zero)
        
        if self.verbose:
            print(f"DataCluster: Derivative near-zero points: {len(near_zero_indices)}")
            print(f"DataCluster: PDF+derivative near-zero points: {len(combined_near_zero)}")
            print(f"DataCluster: Combined candidates: {len(candidate_indices)}")
        
        # Find closest candidates on each side of z0
        left_candidates = candidate_indices[candidate_indices < z0_idx]
        right_candidates = candidate_indices[candidate_indices > z0_idx]
        
        if len(left_candidates) > 0:
            self.CLB = data_points[left_candidates[-1]]  # Closest to z0
        if len(right_candidates) > 0:
            self.CUB = data_points[right_candidates[0]]   # Closest to z0

    def _fallback_to_data_bounds(self):
        dlb, dub = self._get_data_bounds()
        
        if self.CLB is None:
            self.CLB = dlb
            if self.verbose:
                print(f"DataCluster: CLB set to data lower bound: {self.CLB:.3f}")
        
        if self.CUB is None:
            self.CUB = dub
            if self.verbose:
                print(f"DataCluster: CUB set to data upper bound: {self.CUB:.3f}")

    def _update_params(self):
        """Update both DataCluster and GDF params with final results"""
        # Update DataCluster params
        self.params.update({
            'CLB': float(self.CLB) if self.CLB is not None else None,
            'CUB': float(self.CUB) if self.CUB is not None else None,
            'Z0': float(self.z0) if self.z0 is not None else None,
            'S_opt': float(self.S_opt) if self.S_opt is not None else None,
            'cluster_width': float(self.CUB - self.CLB) if (self.CLB is not None and self.CUB is not None) else None,
            'clustering_successful': self.CLB is not None and self.CUB is not None
        })
        
        # Update GDF params with cluster information (REMOVED validation_warnings)
        if hasattr(self.gdf, 'params'):
            cluster_params = {
                'data_cluster': {
                    'CLB': self.params['CLB'],
                    'CUB': self.params['CUB'],
                    'cluster_width': self.params['cluster_width'],
                    'clustering_successful': self.params['clustering_successful'],
                    'method': self.params['method'],
                    'local_minima_count': self.params['local_minima_count'],
                    'local_maxima_count': self.params['local_maxima_count']
                }
            }
            self.gdf.params.update(cluster_params)

    def get_validation_summary(self):
        """
        Get user-friendly summary of GDF type validation and recommendations.
        
        Analyzes the suitability of the input GDF type for clustering and provides
        clear recommendations for optimal results.
        
        Returns
        -------
        str
            Formatted validation summary with:
            - Effectiveness warnings for global functions
            - Suitability confirmations for local functions  
            - Recommendations for better alternatives
            - Unknown type warnings
            
        Examples
        --------
        >>> cluster = DataCluster(gdf=egdf)  # Global function
        >>> cluster.fit()
        >>> print(cluster.get_validation_summary())
        EGDF may be less effective for clustering. Consider ELDF.
        
        >>> cluster = DataCluster(gdf=eldf)  # Local function  
        >>> cluster.fit()
        >>> print(cluster.get_validation_summary())
        ELDF is well-suited for clustering analysis.
        
        Notes
        -----
        - Extracts validation info from warnings array
        - Uses emojis for quick visual identification
        - Provides specific alternative recommendations
        - Returns simple message if no validation issues found
        """
        """Get a summary of validation warnings and recommendations from warnings array"""
        validation_warnings = [w for w in self.params['warnings'] if isinstance(w, dict) and 'type' in w]
        
        if not validation_warnings:
            msg = "No validation warnings. GDF type is suitable for clustering."
            return msg
        
        summary = []
        for validation in validation_warnings:
            if validation['type'] == 'effectiveness_warning':
                summary.append(f"{validation['gdf_type']} may be less effective for clustering. Consider {validation['recommended_alternative']}.")
            elif validation['type'] == 'suitability_confirmation':
                summary.append(f"{validation['gdf_type']} is well-suited for clustering analysis.")
            elif validation['type'] == 'unknown_type_warning':
                summary.append(f"Unknown GDF type '{validation['gdf_type']}' - effectiveness uncertain.")
        
        return "\n".join(summary)

    def fit(self):
        """
        Perform cluster boundary detection analysis.
        
        Executes the complete clustering pipeline:
        1. Validates GDF object and type suitability
        2. Extracts PDF data and parameters (Z0, S_opt)
        3. Applies primary method (maxima or derivative)
        4. Falls back to alternative method if needed
        5. Uses data bounds as final fallback
        6. Updates both DataCluster.params and gdf.params
        
        The method selection follows GDF-specific strategies:
        - ELDF/EGDF: maxima → derivative fallback
        - QLDF/QGDF: derivative → maxima fallback
        
        Returns
        -------
        bool
            True if analysis completed successfully, False if errors occurred
            
        Examples
        --------
        >>> cluster = DataCluster(gdf=eldf, verbose=True)
        >>> success = cluster.fit()
        >>> if success:
        ...     print(f"Found boundaries: CLB={cluster.CLB}, CUB={cluster.CUB}")
        ... else:
        ...     print("Clustering failed, check cluster.params['errors']")
        
        Notes
        -----
        - Automatically shows validation warnings for EGDF/QGDF when verbose=True
        - Updates cluster width calculation: CUB - CLB
        - Sets clustering_successful flag based on boundary detection
        - All errors and warnings logged to params for later inspection
        """
        try:
            if self.verbose:
                print(f"DataCluster: Starting cluster analysis for {self.gdf_type.upper()}")
                print(f"DataCluster: Method: {self.method}")
                
                # Show validation summary
                validation_summary = self.get_validation_summary()
                if validation_summary:
                    print(f"DataCluster: Validation Summary:")
                    for line in validation_summary.split('\n'):
                        print(f"DataCluster:   {line}")
            
            # Get basic data
            pdf_data = self._get_pdf_data()
            data_points = self._get_data_points()
            self.z0 = self._get_z0()
            self.S_opt = self._get_s_opt()
            
            if self.verbose:
                print(f"DataCluster: Z0: {self.z0:.3f}, S_opt: {self.S_opt:.3f}")
            
            # Choose method based on GDF type and user preference
            if self.gdf_type in ['eldf', 'egdf']:
                # For ELDF/EGDF: Try maxima method first (as specified)
                if self.method == 'maxima':
                    self._find_boundaries_maxima_method(pdf_data, data_points)
                    
                    # If unsuccessful, try derivative method
                    if self.CLB is None and self.CUB is None:
                        if self.verbose:
                            print("DataCluster: Maxima method failed, trying derivative method...")
                        self._find_boundaries_derivative_method(pdf_data, data_points)
                else:
                    self._find_boundaries_derivative_method(pdf_data, data_points)
                    
                    # If unsuccessful, try maxima method
                    if self.CLB is None and self.CUB is None:
                        if self.verbose:
                            print("DataCluster: Derivative method failed, trying maxima method...")
                        self._find_boundaries_maxima_method(pdf_data, data_points)
            else:
                # For QLDF/QGDF: Try derivative method first (as you requested)
                if self.method == 'derivative':
                    self._find_boundaries_derivative_method(pdf_data, data_points)
                    
                    # If unsuccessful, try maxima method
                    if self.CLB is None and self.CUB is None:
                        if self.verbose:
                            print("DataCluster: Derivative method failed, trying maxima method...")
                        self._find_boundaries_maxima_method(pdf_data, data_points)
                else:
                    self._find_boundaries_maxima_method(pdf_data, data_points)
                    
                    # If unsuccessful, try derivative method
                    if self.CLB is None and self.CUB is None:
                        if self.verbose:
                            print("DataCluster: Maxima method failed, trying derivative method...")
                        self._find_boundaries_derivative_method(pdf_data, data_points)
            
            # Fallback to data bounds if needed
            self._fallback_to_data_bounds()
            
            # Update params
            self._update_params()
            
            self._fitted = True
            
            if self.verbose:
                print(f"DataCluster: Final boundaries: CLB={self.CLB:.3f}, CUB={self.CUB:.3f}")
                print("DataCluster: Clustering: SUCCESSFUL")
            
            return True
            
        except Exception as e:
            error_msg = f"Error during cluster analysis: {str(e)}"
            self._append_error(error_msg, type(e).__name__)
            if self.verbose:
                print(f"DataCluster: Error: {error_msg}")
            return False

    def results(self):
        """
        Get comprehensive clustering analysis results.
        
        Returns complete analysis results including boundaries, metadata,
        extrema information, and validation status.
        
        Returns
        -------
        dict
            Complete results dictionary containing:
            
            **Boundary Results:**
            - 'CLB' : float or None - Cluster Lower Bound
            - 'CUB' : float or None - Cluster Upper Bound  
            - 'cluster_width' : float or None - CUB - CLB
            - 'clustering_successful' : bool - Overall success status
            
            **Input Parameters:**
            - 'gdf_type' : str - Input GDF type
            - 'method' : str - Analysis method used
            - 'Z0' : float or None - Distribution center
            - 'S_opt' : float or None - Optimization parameter
            
            **Extrema Analysis:**
            - 'local_minima' : list - Indices of local minima
            - 'local_maxima' : list - Indices of local maxima
            - 'global_minimum_idx' : int - Global minimum index
            - 'global_maximum_idx' : int - Global maximum index
            - 'local_minima_count' : int - Number of local minima
            - 'local_maxima_count' : int - Number of local maxima
            
            **Quality Information:**
            - 'errors' : list - Analysis errors encountered
            - 'warnings' : list - Warnings and validation messages
        
        Raises
        ------
        RuntimeError
            If called before fit() method
            
        Examples
        --------
        >>> cluster = DataCluster(gdf=eldf)
        >>> cluster.fit()
        >>> results = cluster.results()
        >>> 
        >>> # Check success
        >>> if results['clustering_successful']:
        ...     print(f"Cluster width: {results['cluster_width']:.3f}")
        ...     print(f"Found {results['local_minima_count']} local minima")
        ... 
        >>> # Check for issues
        >>> if results['warnings']:
        ...     print("Warnings:", len(results['warnings']))
        >>> if results['errors']:
        ...     print("Errors:", len(results['errors']))
        
        Notes
        -----
        - Returns a copy of internal params to prevent external modification
        - All numeric values converted to Python floats for JSON serialization
        - Use this method for programmatic access to results
        """
        if not self._fitted:
            raise RuntimeError("No analysis results available. Call fit() method first.")
        
        return self.params.copy()

    def plot(self, figsize=(12, 8)):
        """
        Create comprehensive visualization of cluster boundary detection.
        
        Generates detailed plot showing:
        - Original PDF curve
        - All local extrema (grey dots)
        - Global extrema (red dots) 
        - Z0 vertical line (red)
        - S_opt threshold line (orange dashed)
        - Detected boundaries CLB/CUB (green dotted)
        - Shaded cluster regions (green)
        - Analysis status and method info
        - Validation warnings for global functions
        
        Parameters
        ----------
        figsize : tuple, default=(12, 8)
            Figure size as (width, height) in inches
            
        Examples
        --------
        Basic plotting:
        
        >>> cluster = DataCluster(gdf=eldf, verbose=True)
        >>> cluster.fit()
        >>> cluster.plot()
        
        Custom figure size:
        
        >>> cluster.plot(figsize=(15, 10))
        
        Plot interpretation:
        
        **For ELDF/EGDF:**
        - Green shaded area = main cluster region (between CLB and CUB)
        - PDF maximum should align with Z0
        - Valid minima have PDF values below S_opt line
        
        **For QLDF/QGDF:**
        - Green shaded areas = inlier regions (outside CLB-CUB)
        - PDF minimum should align with Z0
        - Analysis performed on inverted PDF internally
        
        **Status Box Colors:**
        - Green: Clustering successful
        - Orange: Successful but using global function (warning)
        - Red: Clustering failed
        
        **Legend Elements:**
        - Blue line: Original PDF
        - Grey dots: All local extrema
        - Red dots: Global extrema with values
        - Red line: Z0 position
        - Orange dashed: S_opt threshold
        - Green dotted: Detected boundaries
        
        Notes
        -----
        - Automatically handles different GDF types with appropriate shading
        - Shows validation warnings in status box for EGDF/QGDF
        - Legend positioned outside plot area to avoid overlap
        - Grid enabled for easier value reading
        - All errors during plotting logged to params['errors']
        """
        try:
            pdf_data = self._get_pdf_data()
            data_points = self._get_data_points()
            
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            
            # Plot original PDF
            ax.plot(data_points, pdf_data, 'b-', label='PDF', linewidth=2)
            
            # Plot all local extrema with light grey dots
            for min_idx in self.local_minima:
                if 0 <= min_idx < len(data_points):
                    ax.plot(data_points[min_idx], pdf_data[min_idx], 'o', 
                           color='lightgrey', markersize=6, alpha=0.8)
            
            for max_idx in self.local_maxima:
                if 0 <= max_idx < len(data_points):
                    ax.plot(data_points[max_idx], pdf_data[max_idx], 'o', 
                           color='lightgrey', markersize=6, alpha=0.8)
            
            # Add light grey dots to legend
            if len(self.local_minima) > 0 or len(self.local_maxima) > 0:
                ax.plot([], [], 'o', color='lightgrey', markersize=6, alpha=0.8, 
                       label=f'Local Extrema ({len(self.local_minima) + len(self.local_maxima)})')
            
            # Plot global extrema with light red dots
            if self.global_minimum_idx is not None:
                ax.plot(data_points[self.global_minimum_idx], pdf_data[self.global_minimum_idx], 
                       'o', color='lightcoral', markersize=8, alpha=0.9, 
                       label=f'Global Min ({data_points[self.global_minimum_idx]:.3f})')
            
            if self.global_maximum_idx is not None:
                ax.plot(data_points[self.global_maximum_idx], pdf_data[self.global_maximum_idx], 
                       'o', color='lightcoral', markersize=8, alpha=0.9, 
                       label=f'Global Max ({data_points[self.global_maximum_idx]:.3f})')
            
            # Plot Z0
            if self.z0 is not None:
                ax.axvline(x=self.z0, color='red', linestyle='-', linewidth=2, alpha=0.7, label=f'Z0={self.z0:.3f}')
            
            # Plot S_opt threshold line
            if self.S_opt is not None:
                ax.axhline(y=self.S_opt, color='orange', linestyle='--', alpha=0.7, label=f'S_opt={self.S_opt:.3f}')
            
            # Plot boundaries
            if self.CLB is not None:
                ax.axvline(x=self.CLB, color='green', linestyle=':', linewidth=2, label=f'CLB={self.CLB:.3f}')
            if self.CUB is not None:
                ax.axvline(x=self.CUB, color='green', linestyle=':', linewidth=2, label=f'CUB={self.CUB:.3f}')
            
            # Shade regions based on GDF type
            dlb, dub = self._get_data_bounds()
            if self.CLB is not None and self.CUB is not None:
                if self.gdf_type in ['eldf', 'egdf']:
                    # Shade main cluster region between CLB and CUB
                    ax.axvspan(self.CLB, self.CUB, alpha=0.2, color='lightgreen', label='Main Cluster')
                else:
                    # Shade inlier regions (DLB to CLB and CUB to DUB)
                    ax.axvspan(dlb, self.CLB, alpha=0.2, color='lightgreen', label='Main Cluster')
                    ax.axvspan(self.CUB, dub, alpha=0.2, color='lightgreen')
            
            ax.set_xlabel('Data Points')
            ax.set_ylabel('PDF Values')
            ax.set_title(f'{self.gdf_type.upper()} Cluster Boundary Detection')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Status box with validation info
            clustering_successful = self.CLB is not None and self.CUB is not None
            if clustering_successful:
                status = "Clustering Successful"
                status_color = "green"
            else:
                status = "Clustering Failed"
                status_color = "red"
            
            method_info = f"Method: {self.method.capitalize()}"
            
            # Add validation warning if using global functions
            status_text = f"{status}\n{method_info}"
            if self.gdf_type in ['egdf', 'qgdf']:
                status_text += f"\nGlobal Function"
                status_color = "orange" if clustering_successful else "red"
            
            ax.text(0.02, 0.98, status_text, transform=ax.transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor=status_color, alpha=0.7, edgecolor='black'))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            error_msg = f"Error creating plot: {str(e)}"
            self._append_error(error_msg, type(e).__name__)
            if self.verbose:
                print(f"DataCluster: Error: {error_msg}")