import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, argrelextrema
from typing import Union, Dict, Any, Optional, Tuple, List

class DataCluster:
    """
    Advanced cluster boundary detection for Gnostic Distribution Functions (GDFs).
    
    The DataCluster class identifies main cluster boundaries (CLB and CUB) from probability 
    density functions of four types of Gnostic Distribution Functions: ELDF, EGDF, QLDF, and QGDF.
    It uses normalized PDF analysis with derivative-based methods and shape detection algorithms
    to precisely locate cluster boundaries.
    
    Clustering Performance by GDF Type:
    - **Local Functions (ELDF, QLDF)**: Excellent clustering performance due to unlimited 
      flexibility controlled by scale parameter
    - **Global Functions (EGDF, QGDF)**: Limited clustering effectiveness due to constrained 
      flexibility and uniqueness assumptions
    
    Key Features:
    - PDF normalization for consistent analysis across all GDF types
    - QLDF W-shape vs U-shape detection for accurate valley boundary identification
    - Derivative-based boundary detection with adaptive thresholds
    - Multiple fallback methods for robust cluster identification
    - Comprehensive error handling and validation
    
    Parameters
    ----------
    gdf : ELDF, EGDF, QLDF, or QGDF
        A fitted Gnostic Distribution Function object with pdf_points available.
        Must have been fitted with catch=True to ensure pdf_points are stored.
    verbose : bool, default=False
        Enable detailed progress reporting and diagnostic output.
    catch : bool, default=True
        Enable error catching and graceful degradation (inherited from GDF conventions).
    derivative_threshold : float, default=0.01
        Threshold for ELDF/EGDF boundary detection. Points where (PDF + 1st_derivative) 
        falls below this threshold are considered boundary candidates.
    slope_percentile : int, default=70
        Percentile threshold for QLDF/QGDF slope-based boundary detection. Higher values
        create more conservative (narrower) cluster boundaries.
    
    Attributes
    ----------
    CLB : float or None
        Cluster Lower Boundary - left boundary of the main cluster
    CUB : float or None
        Cluster Upper Boundary - right boundary of the main cluster
    z0 : float or None
        Characteristic point of the distribution (from GDF object)
    S_opt : float or None
        Optimal scale parameter (from GDF object)
    pdf_normalized : ndarray or None
        Min-max normalized PDF values [0,1] used for analysis
    pdf_original : ndarray or None
        Original PDF values before normalization
    params : dict
        Complete analysis results including boundaries, methods used, and diagnostics
    
    Methods
    -------
    fit()
        Perform cluster boundary detection analysis
    plot(figsize=(12, 8))
        Visualize PDF, boundaries, and derivative analysis
    results()
        Return comprehensive analysis results dictionary
    
    Algorithm Details
    ----------------
    **ELDF/EGDF (Estimating Distribution Functions):**
    - PDF has global maximum at z0 (characteristic point)
    - Boundaries found where (PDF + 1st_derivative) ≤ derivative_threshold
    - Main cluster region is BETWEEN CLB and CUB (shaded green)
    - Works best with local ELDF due to flexible scale parameter control
    
    **QLDF (Quantifying Local Distribution Function):**
    - **W-shape detection**: Identifies peaks between boundary extremes
      - 1 internal peak → W-shape → Find valley minima as boundaries
      - 0 internal peaks → U-shape → Use slope transition method
      - 2+ internal peaks → Heterogeneous data warning
    - **Valley detection**: Uses scipy.signal.argrelextrema for precise minima
    - Main cluster region is OUTSIDE CLB and CUB boundaries (shaded green)
    
    **QGDF (Quantifying Global Distribution Function):**
    - Uses slope transition detection with percentile-based thresholds
    - Limited effectiveness due to global function constraints
    - Fallback to curvature analysis when slope detection fails
    - Main cluster region is OUTSIDE CLB and CUB boundaries (shaded green)
    
    Normalization Strategy
    ---------------------
    All PDFs are normalized to [0,1] range using min-max normalization:
    - Ensures consistent threshold application across different GDF types
    - Enables robust derivative analysis regardless of original PDF scale
    - Maintains relative shape characteristics while standardizing magnitude
    
    Error Handling
    -------------
    - Validates GDF object fitness and required attributes
    - Warns when using global functions (EGDF/QGDF) for clustering
    - Provides fallback to data bounds when boundary detection fails
    - Comprehensive error logging with method traceability
    
    Examples
    --------
    >>> # Basic usage with QLDF
    >>> from machinegnostics.magcal import QLDF
    >>> from machinegnostics.magcal import DataCluster
    >>> 
    >>> # Fit QLDF first
    >>> qldf = QLDF(data=your_data, catch=True)
    >>> qldf.fit()
    >>> 
    >>> # Perform cluster analysis
    >>> cluster = DataCluster(gdf=qldf, verbose=True)
    >>> cluster.fit()
    >>> cluster.plot()
    >>> 
    >>> # Get results
    >>> results = cluster.results()
    >>> print(f"CLB: {results['CLB']}, CUB: {results['CUB']}")
    >>> print(f"Cluster width: {results['cluster_width']}")
    >>> print(f"PDF shape: {results['pdf_shape']}")  # For QLDF
    
    >>> # Advanced usage with custom thresholds
    >>> cluster = DataCluster(
    ...     gdf=eldf, 
    ...     derivative_threshold=0.005,  # More sensitive
    ...     slope_percentile=80,         # More conservative
    ...     verbose=True
    ... )
    >>> cluster.fit()
    
    Notes
    -----
    - Clustering works best with local distribution functions (ELDF, QLDF)
    - Global functions (EGDF, QGDF) have limited clustering effectiveness due to 
      their uniqueness constraints and automatic parameter optimization
    - QLDF W-shape detection is particularly effective for data with central clusters
      between outlying regions
    - For heterogeneous data with multiple clusters, consider data splitting before analysis
    
    References
    ----------
    Based on Gnostic Distribution Function theory and cluster analysis methods
    as described in mathematical gnostics literature.
    """
    def __init__(self, gdf, verbose=False, catch=True, derivative_threshold=0.01, slope_percentile=70):
        """
        Initialize DataCluster for boundary detection analysis.
        
        Parameters
        ----------
        gdf : ELDF, EGDF, QLDF, or QGDF
            A fitted Gnostic Distribution Function object. Must have pdf_points 
            available (fitted with catch=True).
        verbose : bool, default=False
            Enable detailed progress reporting and diagnostic messages.
        catch : bool, default=True
            Enable error catching and graceful degradation.
        derivative_threshold : float, default=0.01
            Threshold for ELDF/EGDF boundary detection. Lower values create 
            wider cluster boundaries, higher values create narrower boundaries.
        slope_percentile : int, default=70
            Percentile threshold (0-100) for QLDF/QGDF slope detection. 
            Higher values create more conservative cluster boundaries.
        
        Raises
        ------
        ValueError
            If GDF object is not fitted or missing required attributes.
        AttributeError
            If GDF object is missing pdf_points (ensure catch=True during fitting).
        """
        self.gdf = gdf
        self.gdf_type = gdf.__class__.__name__.lower()
        self.verbose = verbose
        self.catch = catch
        self.derivative_threshold = derivative_threshold
        self.slope_percentile = slope_percentile
        
        self.params = {
            'gdf_type': self.gdf_type,
            'derivative_threshold': self.derivative_threshold,
            'slope_percentile': self.slope_percentile,
            'CLB': None,
            'CUB': None,
            'Z0': None,
            'S_opt': None,
            'cluster_width': None,
            'clustering_successful': False,
            'method_used': None,
            'normalization_method': None,
            'pdf_shape': None,
            'errors': [],
            'warnings': []
        }
        
        self.CLB = None
        self.CUB = None
        self.z0 = None
        self.S_opt = None
        self._fitted = False
        
        self.pdf_normalized = None
        self.pdf_original = None
        
        try:
            self._validate_gdf()
            self._validate_gdf_type_for_clustering()
        except Exception as e:
            self._append_error(f"GDF validation failed: {str(e)}", type(e).__name__)
            if self.verbose:
                print(f"DataCluster: Error: GDF validation failed: {str(e)}")

    def _validate_gdf(self):
        if not hasattr(self.gdf, '_fitted') or not self.gdf._fitted:
            raise ValueError("GDF object must be fitted before cluster analysis")
        
        if not hasattr(self.gdf, 'pdf_points') or self.gdf.pdf_points is None:
            raise AttributeError("GDF object missing pdf_points. Ensure catch=True during fitting.")
        
        if not hasattr(self.gdf, 'data'):
            raise ValueError("GDF object missing data attribute")

    def _validate_gdf_type_for_clustering(self):
        if self.gdf_type in ['egdf', 'qgdf']:
            gdf_full_name = 'EGDF' if self.gdf_type == 'egdf' else 'QGDF'
            local_alternative = 'ELDF' if self.gdf_type == 'egdf' else 'QLDF'
            
            warning_msg = (
                f"Using {gdf_full_name} (Global Distribution Function) for clustering analysis. "
                f"Clustering may not be as effective with global functions. "
                f"Consider using {local_alternative} (Local Distribution Function) for better clustering results."
            )
            
            self._append_warning(warning_msg)
            
            if self.verbose:
                print(f"DataCluster: Warning: {warning_msg}")

    def _append_error(self, error_message, exception_type=None):
        error_entry = {
            'method': 'DataCluster',
            'error': error_message,
            'exception_type': exception_type or 'DataClusterError'
        }
        
        self.params['errors'].append(error_entry)
        
        if hasattr(self.gdf, 'params') and 'errors' in self.gdf.params:
            self.gdf.params['errors'].append(error_entry)
        elif hasattr(self.gdf, 'params'):
            self.gdf.params['errors'] = [error_entry]

    def _append_warning(self, warning_message):
        warning_entry = {
            'method': 'DataCluster',
            'warning': warning_message
        }
        
        self.params['warnings'].append(warning_entry)
        
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

    def _normalize_pdf(self, pdf_data):
        self.pdf_original = pdf_data.copy()
        
        pdf_min = np.min(pdf_data)
        pdf_max = np.max(pdf_data)
        
        if pdf_max == pdf_min:
            normalized_pdf = np.ones_like(pdf_data) * 0.5
            self.params['normalization_method'] = 'constant_pdf'
        else:
            normalized_pdf = (pdf_data - pdf_min) / (pdf_max - pdf_min)
            self.params['normalization_method'] = 'min_max_normalization'
        
        if self.verbose:
            print(f"DataCluster: PDF normalization: {self.params['normalization_method']}")
            print(f"DataCluster: Original PDF range: [{pdf_min:.3f}, {pdf_max:.3f}]")
            print(f"DataCluster: Normalized PDF range: [{np.min(normalized_pdf):.3f}, {np.max(normalized_pdf):.3f}]")
        
        return normalized_pdf

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
        data_points = self._get_data_points()
        
        if self.gdf_type in ['eldf', 'egdf']:
            max_idx = np.argmax(self.pdf_normalized)
            return data_points[max_idx]
        else:
            min_idx = np.argmin(self.pdf_normalized)
            return data_points[min_idx]

    def _find_z0_index(self, data_points):
        z0_idx = np.argmin(np.abs(data_points - self.z0))
        return z0_idx

    def _detect_qldf_shape_and_boundaries(self, pdf_normalized, data_points):
        z0_idx = self._find_z0_index(data_points)
        
        # Find all peaks with lower sensitivity to catch all significant peaks
        peaks, peak_properties = find_peaks(pdf_normalized, 
                                           height=0.05,     
                                           distance=5,      
                                           prominence=0.05) 
        
        # Exclude boundary peaks (first and last 10% of data)
        boundary_margin = len(data_points) // 10
        internal_peaks = peaks[(peaks > boundary_margin) & (peaks < len(data_points) - boundary_margin)]
        
        if self.verbose:
            print(f"DataCluster: Found {len(internal_peaks)} internal peaks at indices: {internal_peaks}")
            if len(internal_peaks) > 0:
                peak_values = [f'{data_points[p]:.1f}' for p in internal_peaks]
                print(f"DataCluster: Internal peak values: {peak_values}")
        
        # Determine shape based on number of internal peaks
        if len(internal_peaks) == 1:
            # W-shape: One peak between extremes
            self.params['pdf_shape'] = 'W-shape'
            return self._find_w_shape_valley_boundaries(pdf_normalized, data_points, internal_peaks[0])
            
        elif len(internal_peaks) == 0:
            # U-shape: No peaks between extremes
            self.params['pdf_shape'] = 'U-shape'
            return self._find_u_shape_slope_boundaries(pdf_normalized, data_points)
            
        else:
            # Heterogeneous: Multiple peaks (2+)
            self.params['pdf_shape'] = 'Heterogeneous'
            warning_msg = f"QLDF detected {len(internal_peaks)} internal peaks. Data may be heterogeneous. Consider splitting the dataset."
            self._append_warning(warning_msg)
            if self.verbose:
                print(f"DataCluster: Warning: {warning_msg}")
            # Fallback to slope method
            return self._find_u_shape_slope_boundaries(pdf_normalized, data_points)

    def _find_w_shape_valley_boundaries(self, pdf_normalized, data_points, central_peak_idx):
        z0_idx = self._find_z0_index(data_points)
        central_peak_value = data_points[central_peak_idx]
        
        if self.verbose:
            print(f"DataCluster: W-shape detected with central peak at {central_peak_value:.3f}")
            print(f"DataCluster: Z0 at {data_points[z0_idx]:.3f}")
        
        left_candidates = []
        right_candidates = []
        
        # Method 1: Find actual minima using scipy
        minima_indices = argrelextrema(pdf_normalized, np.less, order=3)[0]  
        
        # Filter minima and find those on left and right of central peak
        left_minima = [m for m in minima_indices if m < central_peak_idx and m > len(data_points)//10]
        right_minima = [m for m in minima_indices if m > central_peak_idx and m < len(data_points)*9//10]
        
        if self.verbose:
            print(f"DataCluster: Found {len(left_minima)} left minima, {len(right_minima)} right minima")
        
        # Take the closest minima to the central peak
        if left_minima:
            closest_left_min = max(left_minima)  # Closest to central peak from left
            left_candidates.append(closest_left_min)
            if self.verbose:
                print(f"DataCluster: Left valley minimum at {data_points[closest_left_min]:.3f}")
        
        if right_minima:
            closest_right_min = min(right_minima)  # Closest to central peak from right
            right_candidates.append(closest_right_min)
            if self.verbose:
                print(f"DataCluster: Right valley minimum at {data_points[closest_right_min]:.3f}")
        
        # Method 2: If no clear minima found, use regional minimum search
        if not left_candidates or not right_candidates:
            if self.verbose:
                print("DataCluster: No clear minima found, using regional minimum search")
            
            # Define search regions around the central peak
            search_radius = (len(data_points) // 4)
            
            # Left region: from start to central peak
            left_start = max(0, central_peak_idx - search_radius)
            left_end = central_peak_idx
            if not left_candidates and left_end > left_start:
                left_region = pdf_normalized[left_start:left_end]
                local_min_idx = np.argmin(left_region) + left_start
                left_candidates.append(local_min_idx)
                if self.verbose:
                    print(f"DataCluster: Left regional minimum at {data_points[local_min_idx]:.3f}")
            
            # Right region: from central peak to end
            right_start = central_peak_idx
            right_end = min(len(pdf_normalized), central_peak_idx + search_radius)
            if not right_candidates and right_end > right_start:
                right_region = pdf_normalized[right_start:right_end]
                local_min_idx = np.argmin(right_region) + right_start
                right_candidates.append(local_min_idx)
                if self.verbose:
                    print(f"DataCluster: Right regional minimum at {data_points[local_min_idx]:.3f}")
        
        # Method 3: Enhanced valley detection using percentile approach
        if not left_candidates or not right_candidates:
            if self.verbose:
                print("DataCluster: Using percentile-based valley detection")
            
            # Find points in bottom 20% of PDF values
            valley_threshold = np.percentile(pdf_normalized, 20)
            valley_indices = np.where(pdf_normalized <= valley_threshold)[0]
            
            # Split valleys by central peak
            left_valleys = [v for v in valley_indices if v < central_peak_idx]
            right_valleys = [v for v in valley_indices if v > central_peak_idx]
            
            if left_valleys and not left_candidates:
                # Take valley closest to central peak
                left_candidates.append(max(left_valleys))
                if self.verbose:
                    print(f"DataCluster: Left percentile valley at {data_points[max(left_valleys)]:.3f}")
            
            if right_valleys and not right_candidates:
                # Take valley closest to central peak
                right_candidates.append(min(right_valleys))
                if self.verbose:
                    print(f"DataCluster: Right percentile valley at {data_points[min(right_valleys)]:.3f}")
        
        return left_candidates, right_candidates

    def _find_u_shape_slope_boundaries(self, pdf_normalized, data_points):
        z0_idx = self._find_z0_index(data_points)
        
        if self.verbose:
            print("DataCluster: U-shape detected, using slope transition method")
        
        # Use existing slope detection logic
        first_derivative = np.gradient(pdf_normalized)
        deriv_abs = np.abs(first_derivative)
        slope_threshold = np.percentile(deriv_abs, self.slope_percentile)
        
        left_candidates = []
        right_candidates = []
        
        search_radius = min(20, len(data_points) // 4)
        
        # Search for slope transitions
        for i in range(z0_idx - search_radius, -1, -1):
            if i >= 0 and deriv_abs[i] > slope_threshold:
                left_candidates.append(i)
                break
        
        for i in range(z0_idx + search_radius, len(deriv_abs)):
            if deriv_abs[i] > slope_threshold:
                right_candidates.append(i)
                break
        
        return left_candidates, right_candidates

    def _find_boundaries_normalized_method(self, pdf_normalized, data_points):
        z0_idx = self._find_z0_index(data_points)
        
        # Calculate derivatives on normalized PDF
        first_derivative = np.gradient(pdf_normalized)
        second_derivative = np.gradient(first_derivative)
        
        if self.gdf_type in ['eldf', 'egdf']:
            # ELDF/EGDF: Find where pdf + derivative falls below threshold
            combined_signal = pdf_normalized + first_derivative
            
            left_candidates = []
            right_candidates = []
            
            # Search outward from Z0
            for i in range(z0_idx - 1, -1, -1):
                if combined_signal[i] <= self.derivative_threshold:
                    left_candidates.append(i)
                    break
            
            for i in range(z0_idx + 1, len(combined_signal)):
                if combined_signal[i] <= self.derivative_threshold:
                    right_candidates.append(i)
                    break
            
            if left_candidates:
                self.CLB = data_points[left_candidates[0]]
            if right_candidates:
                self.CUB = data_points[right_candidates[0]]
                
            self.params['method_used'] = 'normalized_derivative_eldf_egdf'
            
        elif self.gdf_type == 'qldf':
            # QLDF: Use shape-based detection strategy
            left_candidates, right_candidates = self._detect_qldf_shape_and_boundaries(pdf_normalized, data_points)
            
            if left_candidates:
                self.CLB = data_points[left_candidates[0]]
            if right_candidates:
                self.CUB = data_points[right_candidates[0]]
                
            shape = self.params.get('pdf_shape', 'unknown')
            self.params['method_used'] = f'qldf_{shape.lower()}_valley_detection'
            
        else:
            # QGDF: Use slope transition method
            deriv_abs = np.abs(first_derivative)
            slope_threshold = np.percentile(deriv_abs, self.slope_percentile)
            
            left_candidates = []
            right_candidates = []
            
            search_radius = min(20, len(data_points) // 4)
            
            for i in range(z0_idx - search_radius, -1, -1):
                if i >= 0 and deriv_abs[i] > slope_threshold:
                    left_candidates.append(i)
                    break
            
            for i in range(z0_idx + search_radius, len(deriv_abs)):
                if deriv_abs[i] > slope_threshold:
                    right_candidates.append(i)
                    break
            
            if not left_candidates or not right_candidates:
                if self.verbose:
                    print("DataCluster: Using normalized curvature-based detection")
                
                curvature_threshold = np.std(second_derivative) * 0.7
                
                for i in range(z0_idx - 1, -1, -1):
                    if abs(second_derivative[i]) > curvature_threshold:
                        if not left_candidates:
                            left_candidates.append(i)
                        break
                
                for i in range(z0_idx + 1, len(second_derivative)):
                    if abs(second_derivative[i]) > curvature_threshold:
                        if not right_candidates:
                            right_candidates.append(i)
                        break
                
                self.params['method_used'] = 'normalized_curvature_qgdf'
            else:
                self.params['method_used'] = 'normalized_slope_transition_qgdf'
            
            if left_candidates:
                self.CLB = data_points[left_candidates[0]]
            if right_candidates:
                self.CUB = data_points[right_candidates[0]]

        if self.verbose:
            method = self.params['method_used']
            print(f"DataCluster: Using method: {method}")
            if hasattr(self, 'params') and 'pdf_shape' in self.params:
                print(f"DataCluster: PDF shape: {self.params['pdf_shape']}")
            if self.CLB is not None:
                print(f"DataCluster: Found CLB at {self.CLB:.3f}")
            if self.CUB is not None:
                print(f"DataCluster: Found CUB at {self.CUB:.3f}")

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
        self.params.update({
            'CLB': float(self.CLB) if self.CLB is not None else None,
            'CUB': float(self.CUB) if self.CUB is not None else None,
            'Z0': float(self.z0) if self.z0 is not None else None,
            'S_opt': float(self.S_opt) if self.S_opt is not None else None,
            'cluster_width': float(self.CUB - self.CLB) if (self.CLB is not None and self.CUB is not None) else None,
            'clustering_successful': self.CLB is not None and self.CUB is not None
        })
        
        if hasattr(self.gdf, 'params'):
            cluster_params = {
                'data_cluster': {
                    'CLB': self.params['CLB'],
                    'CUB': self.params['CUB'],
                    'cluster_width': self.params['cluster_width'],
                    'clustering_successful': self.params['clustering_successful'],
                    'method_used': self.params['method_used'],
                    'derivative_threshold': self.params['derivative_threshold'],
                    'slope_percentile': self.params['slope_percentile'],
                    'normalization_method': self.params['normalization_method'],
                    'pdf_shape': self.params.get('pdf_shape', None)
                }
            }
            self.gdf.params.update(cluster_params)

    def fit(self):
        """
        Perform cluster boundary detection analysis on the GDF.
        
        Executes the complete clustering pipeline:
        1. Validates GDF object and extracts PDF data
        2. Normalizes PDF for consistent analysis
        3. Applies GDF-specific boundary detection algorithms
        4. Implements fallback strategies if needed
        5. Updates all parameters and results
        
        The method automatically selects the appropriate algorithm based on GDF type:
        - **ELDF/EGDF**: Derivative threshold method
        - **QLDF**: Shape detection (W-shape vs U-shape) with valley finding
        - **QGDF**: Slope transition detection with curvature fallback
        
        Returns
        -------
        bool
            True if clustering analysis completed successfully, False if errors occurred.
            Check self.params['errors'] for detailed error information.
        
        Side Effects
        ------------
        - Sets self.CLB and self.CUB with detected boundaries
        - Updates self.params with complete analysis results
        - Stores normalized and original PDF data
        - Adds cluster parameters to original GDF object
        
        Examples
        --------
        >>> cluster = DataCluster(gdf=qldf, verbose=True)
        >>> success = cluster.fit()
        >>> if success:
        ...     print(f"Boundaries: CLB={cluster.CLB:.3f}, CUB={cluster.CUB:.3f}")
        ... else:
        ...     print("Clustering failed:", cluster.params['errors'])
        """
        try:
            if self.verbose:
                print(f"DataCluster: Starting normalized cluster analysis for {self.gdf_type.upper()}")
                print(f"DataCluster: Derivative threshold: {self.derivative_threshold}")
                print(f"DataCluster: Slope percentile: {self.slope_percentile}")
            
            # Get basic data
            pdf_data = self._get_pdf_data()
            data_points = self._get_data_points()
            
            # Normalize PDF for consistent processing
            self.pdf_normalized = self._normalize_pdf(pdf_data)
            
            # Get Z0 and S_opt
            self.z0 = self._get_z0()
            self.S_opt = self._get_s_opt()
            
            if self.verbose:
                print(f"DataCluster: Z0: {self.z0:.3f}, S_opt: {self.S_opt:.3f}")
            
            # Apply normalized clustering method
            self._find_boundaries_normalized_method(self.pdf_normalized, data_points)
            
            # Fallback to data bounds if needed
            if self.CLB is None or self.CUB is None:
                if self.verbose:
                    print("DataCluster: Normalized method incomplete, using data bounds as fallback")
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
        Return comprehensive cluster analysis results dictionary.
        
        Provides complete analysis results including boundaries, cluster characteristics,
        method diagnostics, and error information.
        
        Returns
        -------
        dict
            Complete analysis results with the following keys:
            
            **Boundary Results:**
            - 'CLB' : float or None - Cluster Lower Boundary
            - 'CUB' : float or None - Cluster Upper Boundary  
            - 'cluster_width' : float or None - Distance between boundaries
            - 'clustering_successful' : bool - Overall success status
            
            **GDF Information:**
            - 'gdf_type' : str - Type of GDF ('eldf', 'egdf', 'qldf', 'qgdf')
            - 'Z0' : float or None - Characteristic point from GDF
            - 'S_opt' : float or None - Optimal scale parameter from GDF
            
            **Method Details:**
            - 'method_used' : str - Specific algorithm used for boundary detection
            - 'normalization_method' : str - PDF normalization approach
            - 'pdf_shape' : str or None - Detected shape for QLDF ('W-shape', 'U-shape', 'Heterogeneous')
            
            **Parameters:**
            - 'derivative_threshold' : float - Threshold used for ELDF/EGDF
            - 'slope_percentile' : int - Percentile used for QLDF/QGDF
            
            **Diagnostics:**
            - 'errors' : list - Any errors encountered during analysis
            - 'warnings' : list - Warning messages (e.g., global function usage)
        
        Raises
        ------
        RuntimeError
            If fit() method has not been called successfully.
        
        Examples
        --------
        >>> cluster = DataCluster(gdf=qldf)
        >>> cluster.fit()
        >>> results = cluster.results()
        >>> 
        >>> # Access boundary information
        >>> print(f"Lower boundary: {results['CLB']}")
        >>> print(f"Upper boundary: {results['CUB']}")
        >>> print(f"Cluster width: {results['cluster_width']}")
        >>> 
        >>> # Check method and shape information
        >>> print(f"Method used: {results['method_used']}")
        >>> if results['pdf_shape']:
        ...     print(f"PDF shape: {results['pdf_shape']}")
        >>> 
        >>> # Verify success and check for issues
        >>> if results['clustering_successful']:
        ...     print("Clustering completed successfully")
        >>> else:
        ...     print("Issues found:", results['errors'])
        """
        if not self._fitted:
            raise RuntimeError("No analysis results available. Call fit() method first.")
        
        return self.params.copy()

    def plot(self, figsize=(12, 8)):
        """
        Create comprehensive visualization of cluster boundary detection results.
        
        Generates a two-panel plot showing:
        1. **Top panel**: Original PDF with detected boundaries, Z0, and cluster regions
        2. **Bottom panel**: Derivative analysis with thresholds and boundary markers
        
        Visualization Features:
        - Original PDF curve with detected CLB/CUB boundaries (green dotted lines)
        - Z0 characteristic point (red solid line)
        - Cluster region shading (light green):
          - ELDF/EGDF: Between CLB and CUB
          - QLDF/QGDF: Outside CLB and CUB boundaries
        - First and second derivatives for boundary detection analysis
        - Threshold lines and slope indicators
        - QLDF shape information (W-shape, U-shape, Heterogeneous) in title
        
        Parameters
        ----------
        figsize : tuple, default=(12, 8)
            Figure size as (width, height) in inches.
        
        Raises
        ------
        RuntimeError
            If fit() has not been called successfully before plotting.
        
        Notes
        -----
        - Requires successful completion of fit() method
        - Automatically adjusts visualization based on GDF type
        - For QLDF, includes PDF shape detection results in title
        - Derivative plots help understand boundary detection mechanism
        - Green shaded regions indicate the main cluster areas
        
        Examples
        --------
        >>> cluster = DataCluster(gdf=qldf)
        >>> cluster.fit()
        >>> cluster.plot()  # Standard plot
        >>> cluster.plot(figsize=(15, 10))  # Larger plot
        """
        try:
            data_points = self._get_data_points()
            
            # Calculate derivatives for plotting
            first_derivative = np.gradient(self.pdf_normalized)
            second_derivative = np.gradient(first_derivative)
            combined_signal = self.pdf_normalized + first_derivative
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 2])
            
            # Top plot: Original PDF and boundaries
            ax1.plot(data_points, self.pdf_original, 'b-', label='Original PDF', linewidth=2)
            
            # Plot Z0
            if self.z0 is not None:
                ax1.axvline(x=self.z0, color='red', linestyle='-', linewidth=2, alpha=0.7, label=f'Z0={self.z0:.3f}')
            
            # Plot boundaries
            if self.CLB is not None:
                ax1.axvline(x=self.CLB, color='green', linestyle=':', linewidth=2, label=f'CLB={self.CLB:.3f}')
            if self.CUB is not None:
                ax1.axvline(x=self.CUB, color='green', linestyle=':', linewidth=2, label=f'CUB={self.CUB:.3f}')
            
            # Shade regions based on GDF type
            dlb, dub = self._get_data_bounds()
            if self.CLB is not None and self.CUB is not None:
                if self.gdf_type in ['eldf', 'egdf']:
                    ax1.axvspan(self.CLB, self.CUB, alpha=0.2, color='lightgreen', label='Main Cluster')
                else:
                    ax1.axvspan(dlb, self.CLB, alpha=0.2, color='lightgreen', label='Main Cluster')
                    ax1.axvspan(self.CUB, dub, alpha=0.2, color='lightgreen')
            
            # Add shape info to title for QLDF
            title = f'{self.gdf_type.upper()} Normalized Cluster Detection'
            if self.gdf_type == 'qldf' and 'pdf_shape' in self.params:
                title += f' ({self.params["pdf_shape"]})'
            
            ax1.set_ylabel('PDF Values')
            ax1.set_title(title)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: Derivatives and thresholds
            ax2.plot(data_points, first_derivative, 'orange', label='1st Derivative', alpha=0.7)
            ax2.plot(data_points, combined_signal, 'purple', label='PDF + 1st Derivative', linewidth=2)
            
            # Plot threshold lines
            if self.gdf_type in ['eldf', 'egdf']:
                ax2.axhline(y=self.derivative_threshold, color='red', linestyle='--', alpha=0.7, 
                           label=f'Threshold={self.derivative_threshold}')
            else:
                # For QLDF/QGDF, show slope threshold
                deriv_abs = np.abs(first_derivative)
                slope_threshold = np.percentile(deriv_abs, self.slope_percentile)
                ax2.plot(data_points, deriv_abs, 'brown', label='|1st Derivative|', alpha=0.7)
                ax2.axhline(y=slope_threshold, color='red', linestyle='--', alpha=0.7, 
                           label=f'Slope Threshold ({self.slope_percentile}%)')
                ax2.plot(data_points, second_derivative, 'gray', label='2nd Derivative', alpha=0.5)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Zero Line')
            
            # Plot boundaries on derivative plot
            if self.CLB is not None:
                ax2.axvline(x=self.CLB, color='green', linestyle=':', linewidth=2, alpha=0.7)
            if self.CUB is not None:
                ax2.axvline(x=self.CUB, color='green', linestyle=':', linewidth=2, alpha=0.7)
            
            # Plot Z0 on derivative plot
            if self.z0 is not None:
                ax2.axvline(x=self.z0, color='red', linestyle='-', linewidth=2, alpha=0.7)
            
            ax2.set_xlabel('Data Points')
            ax2.set_ylabel('Derivative Values')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            error_msg = f"Error creating plot: {str(e)}"
            self._append_error(error_msg, type(e).__name__)
            if self.verbose:
                print(f"DataCluster: Error: {error_msg}")