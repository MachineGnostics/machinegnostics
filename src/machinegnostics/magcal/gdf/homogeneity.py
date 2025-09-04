import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from typing import Union, Dict, Any, Optional, Tuple, List
from machinegnostics.magcal import ELDF, EGDF

class DataHomogeneity:
    """
    Analyze data homogeneity and perform automatic cluster detection for GDF objects.
    
    This class provides comprehensive homogeneity analysis for EGDF and ELDF objects by:
    1. Detecting peaks and valleys in the probability density function (PDF)
    2. Identifying cluster boundaries using S_opt-based threshold filtering
    3. Validating cluster consistency and warning about potential S parameter issues
    4. Providing visualization tools for cluster analysis
    
    The class uses S_opt (optimal smoothing parameter) as a reference threshold to filter
    out shallow minima that have high PDF values, focusing on deep valleys that represent
    true cluster boundaries between different data populations.
    
    Key Features:
    - Automatic peak and valley detection in PDF
    - S_opt-based minima filtering for robust cluster boundary detection
    - Z0 validation to detect artificial cluster fragmentation
    - Comprehensive visualization with color-coded peaks, valleys, and boundaries
    - Support for both EGDF and ELDF objects
    
    Parameters
    ----------
    gdf : Union[EGDF, ELDF]
        A fitted GDF object (EGDF or ELDF) with catch=True. The object must be
        fitted before homogeneity analysis and should have pdf_points available.
        
    verbose : bool, default=True
        If True, prints detailed information about the analysis process including
        peak/valley detection, cluster boundaries, and validation results.
        
    catch : bool, default=True
        If True, stores analysis results in the params dictionary for later access.
        Recommended to keep True for result persistence.
        
    smoothing_sigma : float, default=1.0
        Gaussian smoothing parameter applied to PDF before peak detection.
        Higher values provide more smoothing but may merge distinct peaks.
        Typical range: 0.5-2.0.
        
    min_height_ratio : float, default=0.01
        Minimum height ratio for peak detection relative to the global maximum.
        Peaks below this threshold (as fraction of max height) are ignored.
        Typical range: 0.001-0.1.
        
    min_distance : Optional[int], default=None
        Minimum distance between peaks/valleys in array indices. If None,
        automatically calculated as len(pdf_data) // 20. Prevents detection
        of very close peaks that may be noise.
        
    s_threshold_factor : float, default=2.0
        Multiplier applied to S_opt to create PDF threshold for minima filtering.
        Only minima with PDF values below (S_opt * s_threshold_factor) are
        considered valid cluster boundaries. Higher values include more minima,
        lower values are more restrictive. Typical range: 1.5-3.0.
    
    Attributes
    ----------
    CLB : float or None
        Cluster Lower Bound - position of the left boundary minimum
        
    CUB : float or None
        Cluster Upper Bound - position of the right boundary minimum
        
    z0 : float or None
        The Z0 value (global optimum) from the GDF object
        
    clusters : List[Dict]
        List of detected clusters with their properties
        
    maxima_indices : np.ndarray or None
        Array indices of all detected maxima in the PDF
        
    minima_indices : np.ndarray or None
        Array indices of all detected minima in the PDF
        
    global_max_idx : int or None
        Index of the global maximum (corresponding to Z0)
        
    global_min_idx : int or None
        Index of the global minimum in the PDF
        
    fitted : bool
        Property indicating if the analysis has been completed
        
    cluster_bounds : Tuple[float, float] or None
        Property returning (CLB, CUB) if both bounds are available
        
    num_clusters : int
        Property returning the number of detected clusters
        
    z0_validation : Dict or None
        Property returning Z0 validation results if available
    
    Examples
    --------
    Basic usage with ELDF:
    
    >>> import numpy as np
    >>> from machinegnostics.magcal import ELDF
    >>> from machinegnostics.magcal import DataHomogeneity
    >>> 
    >>> # Prepare data and fit ELDF
    >>> data = np.array([1, 2, 3, 15, 16, 17, 30, 31, 32])
    >>> eldf = ELDF(data=data, catch=True)
    >>> eldf.fit()
    >>> 
    >>> # Analyze homogeneity and detect clusters
    >>> homogeneity = DataHomogeneity(eldf, verbose=True)
    >>> is_homogeneous = homogeneity.fit(estimate_cluster_bounds=True, plot=True)
    >>> 
    >>> print(f"Data is homogeneous: {is_homogeneous}")
    >>> print(f"Cluster bounds: CLB={homogeneity.CLB:.3f}, CUB={homogeneity.CUB:.3f}")
    
    Advanced usage with custom parameters:
    
    >>> # More restrictive minima filtering
    >>> homogeneity = DataHomogeneity(
    ...     eldf, 
    ...     s_threshold_factor=1.5,  # More restrictive threshold
    ...     smoothing_sigma=0.8,     # Less smoothing
    ...     min_height_ratio=0.05    # Higher peak threshold
    ... )
    >>> homogeneity.fit(plot=True)
    >>> 
    >>> # Extract cluster data
    >>> lower, main, upper = homogeneity.get_cluster_data()
    >>> print(f"Lower cluster: {len(lower)} points")
    >>> print(f"Main cluster: {len(main)} points")  
    >>> print(f"Upper cluster: {len(upper)} points")
    
    Accessing detailed results:
    
    >>> # Get validation results
    >>> validation = homogeneity.z0_validation
    >>> if validation and not validation['validation_passed']:
    ...     print("Warning: Consider increasing S parameter")
    >>> 
    >>> # Get all parameters
    >>> params = homogeneity.get_homogeneity_params()
    >>> print(f"Number of peaks detected: {params['num_peaks']}")
    >>> print(f"Has negative PDF: {params['has_negative_pdf']}")
    
    Notes
    -----
    1. **S Parameter Guidance**: If Z0 validation warnings appear (PDF-derived Z0 falls
       outside cluster bounds), consider increasing the S parameter in your GDF object 
       to reduce spurious local maxima.
       
    2. **Threshold Tuning**: The s_threshold_factor parameter controls the sensitivity
       of cluster boundary detection. Lower values (1.5-2.0) are more conservative,
       higher values (2.5-3.0) detect more boundaries.
       
    3. **Data Requirements**: The GDF object must be fitted with catch=True to generate
       the pdf_points required for analysis.
       
    4. **Validation Logic**: The validation only checks if the PDF-derived Z0 falls within
       the estimated cluster bounds [CLB, CUB]. This ensures the global maximum is 
       contained within the main cluster region.
    
    Warnings
    --------
    - "Marginal Clustering Warning": Indicates potential issues with S parameter
    - "PDF-derived Z0 falls outside cluster bounds": Indicates artificial fragmentation
    
    See Also
    --------
    ELDF : Empirical Log Density Function for univariate data
    EGDF : Empirical Generalized Density Function for univariate data
    MarginalAnalysisELDF : High-level interface for ELDF with automatic clustering
    """
    
    def __init__(self, gdf: Union[EGDF, ELDF], verbose=True, catch=True, smoothing_sigma=1.0, 
                 min_height_ratio=0.01, min_distance=None, s_threshold_factor=2.0):
        self.gdf = gdf
        self.gdf_type = self._detect_gdf_type()
        self.verbose = verbose
        self.catch = catch
        self.params = {}
        self._fitted = False

        self.smoothing_sigma = smoothing_sigma
        self.min_height_ratio = min_height_ratio
        self.min_distance = min_distance
        self.s_threshold_factor = s_threshold_factor

        self.clusters = []
        self.main_cluster_idx = None
        self.CLB = None
        self.CUB = None
        self.z0 = None

        self.maxima_indices = None
        self.minima_indices = None
        self.global_max_idx = None
        self.global_min_idx = None
        self.left_boundary_min = None
        self.right_boundary_min = None

        self._gdf_obj_validation()

    def _detect_gdf_type(self):
        """Detect whether the GDF object is EGDF or ELDF type."""
        class_name = self.gdf.__class__.__name__
        if 'EGDF' in class_name:
            return 'egdf'
        elif 'ELDF' in class_name:
            return 'eldf'
        else:
            if hasattr(self.gdf, '_fit_egdf'):
                return 'egdf'
            elif hasattr(self.gdf, '_fit_eldf'):
                return 'eldf'
            else:
                raise ValueError("Cannot determine GDF type. Object must be EGDF or ELDF.")

    def _gdf_obj_validation(self):
        """Validate that the GDF object meets requirements for homogeneity analysis."""
        if not hasattr(self.gdf, '_fitted'):
            raise ValueError("GDF object must have _fitted attribute")
        
        if not self.gdf._fitted:
            raise ValueError("GDF object must be fitted before homogeneity checking")
        
        required_attrs = ['data']
        for attr in required_attrs:
            if not hasattr(self.gdf, attr):
                raise ValueError(f"GDF object missing required attribute: {attr}")
        
        if not (hasattr(self.gdf, 'pdf_points') and self.gdf.pdf_points is not None):
            if hasattr(self.gdf, 'catch') and not self.gdf.catch:
                raise AttributeError(f"{self.gdf_type.upper()} object must have catch=True to generate pdf_points required for homogeneity analysis.")
            else:
                raise AttributeError(f"{self.gdf_type.upper()} object is missing 'pdf_points'. Please ensure catch=True when fitting {self.gdf_type.upper()}.")

    def _get_s_opt(self):
        """
        Get S_opt (optimal smoothing parameter) from the GDF object.
        
        Falls back to S parameter if S_opt not available, or uses default value.
        """
        if hasattr(self.gdf, 'S_opt') and self.gdf.S_opt is not None:
            return self.gdf.S_opt
        elif hasattr(self.gdf, 'params') and 'S_opt' in self.gdf.params:
            return self.gdf.params['S_opt']
        elif hasattr(self.gdf, 'S') and self.gdf.S is not None:
            if self.verbose:
                print("Warning: S_opt not found. Using S parameter as reference.")
            return self.gdf.S
        else:
            if self.verbose:
                print("Warning: Neither S_opt nor S found. Using default threshold of 0.05")
            return 0.05

    def _validate_cluster_consistency(self):
        """
        Validate cluster consistency by checking if PDF-derived Z0 falls within cluster bounds.
        
        Issues warnings if:
        - PDF-derived Z0 falls outside estimated cluster bounds [CLB, CUB]
        
        This condition suggests the S parameter may be too low, causing artificial
        cluster fragmentation where the global maximum appears outside the main cluster.
        
        Returns
        -------
        dict
            Validation results including boundary check and validation status with any warnings issued.
        """
        if self.global_max_idx is None:
            return
        
        original_z0 = self._get_z0()
        data_points = self._get_data_points()
        pdf_z0 = data_points[self.global_max_idx]
        
        is_outside_bounds = False
        if self.CLB is not None and self.CUB is not None:
            is_outside_bounds = (original_z0 < self.CLB) or (original_z0 > self.CUB)

        warnings_issued = []
        
        if is_outside_bounds:
            warning_msg = (f"Marginal Clustering Warning: Gnostic Mode Z0 ({original_z0:.6f}) falls outside "
                          f"cluster bounds [CLB={self.CLB:.6f}, CUB={self.CUB:.6f}]. "
                          f"S parameter may be too low, creating artificial cluster fragmentation.")
            warnings_issued.append(warning_msg)
            
            if self.verbose:
                print(f"⚠️  {warning_msg}")
                warnings.warn(warning_msg)
            else:
                warnings.warn(warning_msg)
    
        validation_results = {
            'original_z0': float(original_z0),
            'pdf_derived_z0': float(pdf_z0),
            'is_outside_bounds': bool(is_outside_bounds),
            'validation_passed': not is_outside_bounds,
            'warnings': warnings_issued
        }
        
        if self.catch:
            self.params['z0_validation'] = validation_results
        
        if hasattr(self.gdf, 'catch') and self.gdf.catch and hasattr(self.gdf, 'params'):
            self.gdf.params['z0_validation'] = validation_results
        
        if self.verbose and not warnings_issued:
            print(f"✅ Z0 validation passed: Gnostic Mode Z0 ({original_z0:.6f}) falls within cluster bounds [CLB={self.CLB:.6f}, CUB={self.CUB:.6f}]")

        return validation_results

    def fit(self, estimate_cluster_bounds=True, plot=False):
        """
        Perform comprehensive homogeneity analysis and cluster detection.
        
        This is the main method that orchestrates the entire analysis pipeline:
        1. Homogeneity testing (negative PDF check, peak counting)
        2. Peak and valley detection using smoothed PDF
        3. Cluster boundary identification using S_opt-based filtering
        4. Cluster definition and main cluster identification
        5. Z0 validation to detect potential S parameter issues
        
        Parameters
        ----------
        estimate_cluster_bounds : bool, default=True
            If True, performs cluster boundary estimation and identification.
            If False, only performs basic homogeneity testing.
            
        plot : bool, default=False
            If True, displays a comprehensive visualization plot after analysis.
            Shows PDF, thresholds, peaks, valleys, and cluster regions.
        
        Returns
        -------
        bool
            True if data is homogeneous (single peak, no negative PDF values),
            False otherwise.
        
        Notes
        -----
        The method sets the following instance attributes upon completion:
        - _fitted: True
        - CLB, CUB: Cluster bounds if estimate_cluster_bounds=True
        - maxima_indices, minima_indices: Peak and valley locations
        - clusters: List of detected cluster information
        - z0: Global optimum value
        
        Examples
        --------
        Basic homogeneity check:
        
        >>> homogeneity = DataHomogeneity(gdf_object)
        >>> is_homogeneous = homogeneity.fit(estimate_cluster_bounds=False)
        
        Full analysis with visualization:
        
        >>> is_homogeneous = homogeneity.fit(estimate_cluster_bounds=True, plot=True)
        >>> print(f"Cluster bounds: [{homogeneity.CLB:.3f}, {homogeneity.CUB:.3f}]")
        """
        if self.verbose:
            print(f"Starting homogeneity and cluster analysis for {self.gdf_type.upper()} data...")
        
        if self.min_distance is None:
            pdf_data = self._get_pdf_data()
            self.min_distance = len(pdf_data) // 20
        
        is_homogeneous = self._is_homogeneous()
        
        if estimate_cluster_bounds:
            self._find_peaks_and_valleys()
            self._find_cluster_boundaries()
            self._define_clusters()
            self._identify_main_cluster()
            self._set_cluster_bounds()
            self._validate_cluster_consistency()
        
        self._fitted = True
        
        if self.verbose:
            print(f"Data-Homogeneity analysis completed for {self.gdf_type.upper()}.")
            if estimate_cluster_bounds:
                print(f"Total maxima detected: {len(self.maxima_indices) if self.maxima_indices is not None else 0}")
                print(f"Total minima detected: {len(self.minima_indices) if self.minima_indices is not None else 0}")
                if self.main_cluster_idx is not None:
                    print(f"Main cluster bounds: CLB={self.CLB:.6f}, CUB={self.CUB:.6f}")
        
        if plot:
            self.plot()
            
        return is_homogeneous

    def test_homogeneity(self, estimate_cluster_bounds=True):
        """
        Legacy method for homogeneity testing. Use fit() instead.
        
        Parameters
        ----------
        estimate_cluster_bounds : bool, default=True
            Whether to estimate cluster boundaries.
            
        Returns
        -------
        bool
            True if data is homogeneous, False otherwise.
        """
        return self.fit(estimate_cluster_bounds=estimate_cluster_bounds)

    def _smooth_pdf(self):
        """Apply Gaussian smoothing to PDF for robust peak detection."""
        pdf_data = self._get_pdf_data()
        return gaussian_filter1d(pdf_data, sigma=self.smoothing_sigma)

    def _find_peaks_and_valleys(self):
        """
        Detect all peaks and valleys in the smoothed PDF.
        
        Uses scipy.signal.find_peaks to identify:
        - Maxima: Peaks above min_height_ratio threshold
        - Minima: Peaks in the inverted (negative) PDF
        
        Also identifies global maximum and minimum, with global maximum
        determined by proximity to Z0 if available, or highest peak otherwise.
        """
        smoothed_pdf = self._smooth_pdf()
        data_points = self._get_data_points()
        
        min_height = np.max(smoothed_pdf) * self.min_height_ratio
        maxima_idx, _ = find_peaks(smoothed_pdf, 
                                   height=min_height,
                                   distance=self.min_distance)
        
        inverted_pdf = -smoothed_pdf
        minima_idx, _ = find_peaks(inverted_pdf, distance=self.min_distance)
        
        self.maxima_indices = maxima_idx
        self.minima_indices = minima_idx
        
        z0_value = self._get_z0()
        if z0_value is not None and len(maxima_idx) > 0:
            max_positions = data_points[maxima_idx]
            closest_idx = np.argmin(np.abs(max_positions - z0_value))
            self.global_max_idx = maxima_idx[closest_idx]
        elif len(maxima_idx) > 0:
            max_heights = smoothed_pdf[maxima_idx]
            self.global_max_idx = maxima_idx[np.argmax(max_heights)]
        
        if len(minima_idx) > 0:
            min_heights = smoothed_pdf[minima_idx]
            self.global_min_idx = minima_idx[np.argmin(min_heights)]

        if self.verbose:
            print(f"Found {len(maxima_idx)} maxima and {len(minima_idx)} minima in PDF")
            
            s_opt = self._get_s_opt()
            pdf_threshold = s_opt * self.s_threshold_factor
            valid_minima = sum(1 for idx in minima_idx if smoothed_pdf[idx] <= pdf_threshold)
            print(f"S_opt={s_opt:.6f}, threshold={pdf_threshold:.6f} -> {valid_minima}/{len(minima_idx)} minima are valid")

    def _find_cluster_boundaries(self):
        """
        Find cluster boundaries using S_opt-based threshold filtering.
        
        This method implements the core logic for robust cluster boundary detection:
        1. Filters minima based on PDF threshold (S_opt * s_threshold_factor)
        2. Separates candidates into left and right of global maximum
        3. Selects closest valid minima on each side as cluster boundaries
        
        Only minima with PDF values below the threshold are considered valid,
        as they represent true valleys between clusters rather than shallow
        local minima within a single cluster.
        """
        if self.global_max_idx is None or self.minima_indices is None:
            return
        
        data_points = self._get_data_points()
        pdf_data = self._get_pdf_data()
        global_max_idx = self.global_max_idx
        
        s_opt = self._get_s_opt()
        pdf_threshold = s_opt * self.s_threshold_factor
        
        if self.verbose:
            print(f"Using S_opt={s_opt:.6f}, PDF threshold={pdf_threshold:.6f} for minima filtering")
        
        left_minima_candidates = []
        right_minima_candidates = []
        
        for min_idx in self.minima_indices:
            min_pdf_value = pdf_data[min_idx]
            min_position = data_points[min_idx]
            
            if min_pdf_value <= pdf_threshold:
                if min_idx < global_max_idx:
                    left_minima_candidates.append((min_idx, min_position, min_pdf_value))
                elif min_idx > global_max_idx:
                    right_minima_candidates.append((min_idx, min_position, min_pdf_value))
        
        if self.verbose:
            print(f"Found {len(left_minima_candidates)} valid left minima and {len(right_minima_candidates)} valid right minima below PDF threshold")
        
        if left_minima_candidates:
            left_minima_candidates.sort(key=lambda x: abs(x[0] - global_max_idx))
            self.left_boundary_min = left_minima_candidates[0][0]
            
            if self.verbose:
                left_pos = left_minima_candidates[0][1]
                left_pdf = left_minima_candidates[0][2]
                print(f"Left boundary minimum: index {self.left_boundary_min}, position {left_pos:.3f}, PDF {left_pdf:.6f}")
        else:
            if self.verbose:
                print("Warning: No valid left minima found below PDF threshold")
        
        if right_minima_candidates:
            right_minima_candidates.sort(key=lambda x: abs(x[0] - global_max_idx))
            self.right_boundary_min = right_minima_candidates[0][0]
            
            if self.verbose:
                right_pos = right_minima_candidates[0][1]
                right_pdf = right_minima_candidates[0][2]
                print(f"Right boundary minimum: index {self.right_boundary_min}, position {right_pos:.3f}, PDF {right_pdf:.6f}")
        else:
            if self.verbose:
                print("Warning: No valid right minima found below PDF threshold")

    def _define_clusters(self):
        """
        Define cluster structure based on detected boundaries.
        
        Creates the main cluster spanning from left boundary minimum to
        right boundary minimum, containing the global maximum (Z0).
        """
        if self.global_max_idx is None:
            return
            
        data_points = self._get_data_points()
        
        left_bound = (data_points[self.left_boundary_min] 
                     if self.left_boundary_min is not None 
                     else data_points[0])
        right_bound = (data_points[self.right_boundary_min] 
                      if self.right_boundary_min is not None 
                      else data_points[-1])
        
        self.clusters = [{
            'cluster_id': 0,
            'peak_position': data_points[self.global_max_idx],
            'onset_position': left_bound,
            'offset_position': right_bound,
            'is_main_cluster': True
        }]

    def _is_homogeneous(self):
        """
        Determine if data is homogeneous based on PDF characteristics.
        
        Data is considered homogeneous if:
        1. PDF has no negative values (valid probability distribution)
        2. Exactly one significant peak is detected
        
        Returns
        -------
        bool
            True if data is homogeneous, False otherwise.
        """
        peaks_info = self._detect_peaks()
        num_peaks = len(peaks_info)
        has_negative_pdf = np.any(self._get_pdf_data() < 0)

        is_homogeneous = not has_negative_pdf and num_peaks == 1

        if self.verbose:
            if not is_homogeneous:
                reasons = []
                if has_negative_pdf:
                    reasons.append("PDF has negative values")
                if num_peaks > 1:
                    reasons.append(f"multiple peaks [{num_peaks}] detected")
                elif num_peaks == 0:
                    reasons.append("no significant peaks detected")
                print(f"{self.gdf_type.upper()} data is not homogeneous: {', '.join(reasons)}.")
            else:
                print(f"{self.gdf_type.upper()} data is homogeneous: PDF has no negative values and exactly one peak detected.")

        if self.catch:
            self.params.update({
                'is_homogeneous': is_homogeneous,
                'has_negative_pdf': has_negative_pdf,
                'num_peaks': num_peaks,
                'gdf_type': self.gdf_type,
                'homogeneity_fitted': True
            })
        
        if hasattr(self.gdf, 'catch') and self.gdf.catch and hasattr(self.gdf, 'params'):
            self.gdf.params.update({
                'is_homogeneous': is_homogeneous,
                'has_negative_pdf': has_negative_pdf,
                'num_peaks': num_peaks,
                'homogeneity_checked': True,
                'homogeneity_fitted': True
            })
            
            if self.verbose:
                print(f"Homogeneity results written to {self.gdf_type.upper()} params dictionary.")
        
        return is_homogeneous

    def _get_pdf_data(self):
        """Get PDF values from the GDF object."""
        if hasattr(self.gdf, 'pdf_points') and self.gdf.pdf_points is not None:
            return self.gdf.pdf_points
        else:
            raise AttributeError("PDF points not available. Ensure GDF object was fitted with catch=True.")

    def _get_data_points(self):
        """Get data point positions from the GDF object."""
        if hasattr(self.gdf, 'di_points_n') and self.gdf.di_points_n is not None:
            return self.gdf.di_points_n
        else:
            raise AttributeError("Data points (di_points_n) not available. Ensure GDF object was fitted with catch=True.")

    def _get_z0(self):
        """Get Z0 (global optimum) value from the GDF object."""
        if hasattr(self.gdf, 'z0') and self.gdf.z0 is not None:
            return self.gdf.z0
        elif hasattr(self.gdf, 'params') and 'z0' in self.gdf.params:
            return self.gdf.params['z0']
        else:
            pdf_data = self._get_pdf_data()
            data_points = self._get_data_points()
            max_idx = np.argmax(pdf_data)
            if self.verbose:
                print("Warning: Z0 not found in GDF object. Using PDF global maximum as Z0.")
            return data_points[max_idx]

    def _detect_peaks(self):
        """
        Detect all peaks in the PDF for homogeneity assessment.
        
        Returns
        -------
        list
            List of dictionaries containing peak information with keys:
            'index', 'value', 'position', 'is_global_max'
        """
        pdf_data = self._get_pdf_data()
        data_points = self._get_data_points()
        
        if len(pdf_data) < 3:
            if len(pdf_data) > 0 and np.max(pdf_data) > 0:
                global_max_idx = np.argmax(pdf_data)
                return [{
                    'index': global_max_idx, 
                    'value': pdf_data[global_max_idx], 
                    'position': data_points[global_max_idx],
                    'is_global_max': True
                }]
            else:
                return []
        
        peaks = []
        global_max_idx = np.argmax(pdf_data)
        global_max_value = pdf_data[global_max_idx]
        
        for i in range(1, len(pdf_data) - 1):
            if (pdf_data[i] > pdf_data[i-1] and pdf_data[i] > pdf_data[i+1]):
                is_global = (i == global_max_idx)
                peaks.append({
                    'index': i,
                    'value': pdf_data[i],
                    'position': data_points[i],
                    'is_global_max': is_global
                })
        
        if len(pdf_data) > 1:
            if pdf_data[0] > pdf_data[1]:
                is_global = (0 == global_max_idx)
                peaks.append({
                    'index': 0,
                    'value': pdf_data[0],
                    'position': data_points[0],
                    'is_global_max': is_global
                })
            
            last_idx = len(pdf_data) - 1
            if pdf_data[last_idx] > pdf_data[last_idx-1]:
                is_global = (last_idx == global_max_idx)
                peaks.append({
                    'index': last_idx,
                    'value': pdf_data[last_idx],
                    'position': data_points[last_idx],
                    'is_global_max': is_global
                })
        
        global_max_found = any(peak['is_global_max'] for peak in peaks)
        if not global_max_found:
            peaks.append({
                'index': global_max_idx,
                'value': global_max_value,
                'position': data_points[global_max_idx],
                'is_global_max': True
            })
        
        peaks.sort(key=lambda x: (not x['is_global_max'], -x['value']))
        
        return peaks

    def _identify_main_cluster(self):
        """Identify the main cluster containing the global maximum (Z0)."""
        if not self.clusters:
            if self.verbose:
                print("No clusters detected for main cluster identification")
            return
        
        self.z0 = self._get_z0()
        self.main_cluster_idx = 0
        
        if self.verbose:
            print(f"Main cluster identified: Contains Z0 ({self.z0:.6f})")

    def _set_cluster_bounds(self):
        """
        Set cluster bounds (CLB, CUB) and store results in params dictionaries.
        
        Stores comprehensive results in both the instance params and the
        original GDF object's params (if catch=True).
        """
        if not self.clusters or self.global_max_idx is None:
            if self.verbose:
                print("Cannot set cluster bounds: no main cluster identified")
            return
        
        main_cluster = self.clusters[0]
        self.CLB = main_cluster['onset_position']
        self.CUB = main_cluster['offset_position']
        
        if self.catch:
            self.params.update({
                'CLB': float(self.CLB),
                'CUB': float(self.CUB),
                'z0': float(self.z0),
                'main_cluster_id': 0,
                'total_clusters': 1,
                'cluster_bounds_estimated': True,
                'clusters_info': self.clusters,
                'maxima_indices': self.maxima_indices.tolist() if self.maxima_indices is not None else [],
                'minima_indices': self.minima_indices.tolist() if self.minima_indices is not None else [],
                'global_max_idx': int(self.global_max_idx) if self.global_max_idx is not None else None,
                'global_min_idx': int(self.global_min_idx) if self.global_min_idx is not None else None
            })
        
        if hasattr(self.gdf, 'catch') and self.gdf.catch and hasattr(self.gdf, 'params'):
            self.gdf.params.update({
                'CLB': float(self.CLB),
                'CUB': float(self.CUB),
                'z0': float(self.z0),
                'main_cluster_id': 0,
                'total_clusters': 1,
                'cluster_bounds_estimated': True
            })
            
            if self.verbose:
                print(f"Cluster bounds written to {self.gdf_type.upper()} params dictionary.")

    def plot(self, figsize=(12, 8), title=None):
        """
        Create comprehensive visualization of the homogeneity analysis results.
        
        The plot includes:
        - PDF curve (blue line)
        - S_opt threshold line (orange dashed)
        - Main cluster region (light green fill)
        - Global maximum and minimum (magenta dashed lines with circles/squares)
        - Selected boundary minima (red lines with squares)
        - Valid minima below threshold (green lines with squares)
        - Invalid minima above threshold (grey lines with squares)
        - Cluster boundaries CLB and CUB (dark red dotted lines)
        - Original vs PDF-derived Z0 comparison (cyan dotted line if different)
        - Validation warning indicator (yellow box if issues detected)
        
        Parameters
        ----------
        figsize : tuple, default=(12, 8)
            Figure size in inches (width, height).
            
        title : str, optional
            Custom title for the plot. If None, uses default descriptive title.
        
        Raises
        ------
        RuntimeError
            If called before fit() method has been executed.
        
        Examples
        --------
        >>> homogeneity = DataHomogeneity(gdf_object)
        >>> homogeneity.fit()
        >>> homogeneity.plot(figsize=(14, 10), title="Custom Analysis Results")
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before plotting. Run fit() method first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        pdf_data = self._get_pdf_data()
        data_points = self._get_data_points()
        
        ax.plot(data_points, pdf_data, 'b-', linewidth=2, label='PDF')
        
        s_opt = self._get_s_opt()
        pdf_threshold = s_opt * self.s_threshold_factor
        ax.axhline(pdf_threshold, color='orange', linestyle='--', linewidth=1, 
                   alpha=0.7, label=f'PDF Threshold ({self.s_threshold_factor}×S_opt={pdf_threshold:.6f})')
        
        if self.CLB is not None and self.CUB is not None:
            main_mask = (data_points >= self.CLB) & (data_points <= self.CUB)
            if np.any(main_mask):
                ax.fill_between(data_points[main_mask], 
                               pdf_data[main_mask],
                               alpha=0.3, color='lightgreen', 
                               label='Main Cluster')
        
        if self.maxima_indices is not None and len(self.maxima_indices) > 0:
            max_positions = data_points[self.maxima_indices]
            max_values = pdf_data[self.maxima_indices]
            
            for i, (pos, val) in enumerate(zip(max_positions, max_values)):
                if self.maxima_indices[i] != self.global_max_idx:
                    ax.axvline(pos, color='grey', linestyle='-', linewidth=1, alpha=0.7)
                    ax.plot(pos, val, 'o', color='grey', markersize=6)
            
            if self.global_max_idx is not None:
                global_pos = data_points[self.global_max_idx]
                global_val = pdf_data[self.global_max_idx]
                ax.axvline(global_pos, color='magenta', linestyle='--', linewidth=2)
                ax.plot(global_pos, global_val, 'o', color='magenta', markersize=8, 
                       label=f'Global Maximum (PDF Z0={global_pos:.3f})')
                
                original_z0 = self._get_z0()
                if abs(original_z0 - global_pos) > 0.001:
                    ax.axvline(original_z0, color='cyan', linestyle=':', linewidth=2, alpha=0.8,
                              label=f'Original Z0={original_z0:.3f}')
        
        if self.minima_indices is not None and len(self.minima_indices) > 0:
            min_positions = data_points[self.minima_indices]
            min_values = pdf_data[self.minima_indices]
            
            for i, (pos, val) in enumerate(zip(min_positions, min_values)):
                is_global_min = (self.minima_indices[i] == self.global_min_idx)
                is_boundary = (self.minima_indices[i] == self.left_boundary_min or 
                              self.minima_indices[i] == self.right_boundary_min)
                is_valid = val <= pdf_threshold
                
                if is_global_min:
                    ax.axvline(pos, color='magenta', linestyle='--', linewidth=2)
                    ax.plot(pos, val, 's', color='magenta', markersize=8, label='Global Minimum')
                elif is_boundary:
                    ax.axvline(pos, color='red', linestyle='-', linewidth=2, alpha=0.8)
                    ax.plot(pos, val, 's', color='red', markersize=7, label='Boundary Minimum')
                elif is_valid:
                    ax.axvline(pos, color='green', linestyle='-', linewidth=1, alpha=0.6)
                    ax.plot(pos, val, 's', color='green', markersize=5)
                else:
                    ax.axvline(pos, color='grey', linestyle='-', linewidth=1, alpha=0.4)
                    ax.plot(pos, val, 's', color='grey', markersize=4, alpha=0.6)
        
        if self.CLB is not None:
            ax.axvline(self.CLB, color='darkred', linestyle=':', linewidth=2, alpha=0.8, 
                      label=f'CLB={self.CLB:.3f}')
        if self.CUB is not None:
            ax.axvline(self.CUB, color='darkred', linestyle=':', linewidth=2, alpha=0.8, 
                      label=f'CUB={self.CUB:.3f}')
        
        if hasattr(self, 'params') and 'z0_validation' in self.params:
            validation = self.params['z0_validation']
            if not validation['validation_passed']:
                ax.text(0.02, 0.98, '⚠️ Marginal Clustering Warning', 
                       transform=ax.transAxes, fontsize=10, color='red',
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('Data Points')
        ax.set_ylabel('PDF Values')
        
        if title is None:
            title = f"{self.gdf_type.upper()} PDF Analysis with S_opt-based Cluster Detection"
        ax.set_title(title)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def get_cluster_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract original data points separated into clusters.
        
        Divides the original data into three clusters based on the estimated
        cluster boundaries (CLB and CUB):
        - Lower cluster: Data below CLB
        - Main cluster: Data between CLB and CUB (inclusive)
        - Upper cluster: Data above CUB
        
        Returns
        -------
        tuple of np.ndarray
            (lower_cluster, main_cluster, upper_cluster) containing the
            original data points assigned to each cluster.
        
        Raises
        ------
        RuntimeError
            If called before fit() or if cluster bounds weren't estimated.
        
        Examples
        --------
        >>> lower, main, upper = homogeneity.get_cluster_data()
        >>> print(f"Cluster sizes: {len(lower)}, {len(main)}, {len(upper)}")
        >>> print(f"Main cluster range: [{np.min(main):.3f}, {np.max(main):.3f}]")
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before extracting cluster data")
        
        if self.CLB is None or self.CUB is None:
            raise RuntimeError("Cluster bounds not estimated. Call fit(estimate_cluster_bounds=True)")
        
        data = self.gdf.data
        
        lower_cluster = data[data < self.CLB]
        main_cluster = data[(data >= self.CLB) & (data <= self.CUB)]
        upper_cluster = data[data > self.CUB]
        
        if self.verbose:
            print(f"Clustered data: Lower={len(lower_cluster)}, Main={len(main_cluster)}, Upper={len(upper_cluster)}")
        
        return lower_cluster, main_cluster, upper_cluster

    def get_all_clusters_data(self) -> List[Dict]:
        """
        Get detailed information about all detected clusters.
        
        Returns
        -------
        list of dict
            List of dictionaries containing cluster information with keys:
            - 'cluster_id': Unique cluster identifier
            - 'data': Original data points in this cluster
            - 'peak_position': Position of the cluster's peak
            - 'boundaries': Tuple of (onset_position, offset_position)
            - 'size': Number of data points in cluster
            - 'is_main_cluster': Boolean indicating if this is the main cluster
        
        Raises
        ------
        RuntimeError
            If called before fit() method has been executed.
        
        Examples
        --------
        >>> clusters = homogeneity.get_all_clusters_data()
        >>> for cluster in clusters:
        ...     print(f"Cluster {cluster['cluster_id']}: {cluster['size']} points")
        ...     print(f"  Peak at: {cluster['peak_position']:.3f}")
        ...     print(f"  Bounds: {cluster['boundaries']}")
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before extracting cluster data")
        
        if not self.clusters:
            return []
        
        data = self.gdf.data
        clusters_data = []
        
        for cluster in self.clusters:
            cluster_data = data[(data >= cluster['onset_position']) & 
                              (data <= cluster['offset_position'])]
            
            cluster_info = {
                'cluster_id': cluster['cluster_id'],
                'data': cluster_data,
                'peak_position': cluster['peak_position'],
                'boundaries': (cluster['onset_position'], cluster['offset_position']),
                'size': len(cluster_data),
                'is_main_cluster': cluster['cluster_id'] == self.main_cluster_idx
            }
            clusters_data.append(cluster_info)
        
        return clusters_data

    def get_homogeneity_params(self):
        """
        Get comprehensive analysis parameters and results.
        
        Returns
        -------
        dict
            Dictionary containing all analysis results including:
            - is_homogeneous: Boolean homogeneity status
            - has_negative_pdf: Boolean indicating negative PDF values
            - num_peaks: Number of detected peaks
            - CLB, CUB: Cluster bounds
            - z0: Global optimum value
            - clusters_info: Detailed cluster information
            - maxima_indices, minima_indices: Peak and valley locations
            - z0_validation: Validation results if available (boundary check only)
        
        Raises
        ------
        RuntimeError
            If called before fit() or if catch=False during initialization.
        
        Examples
        --------
        >>> params = homogeneity.get_homogeneity_params()
        >>> print(f"Homogeneous: {params['is_homogeneous']}")
        >>> print(f"Number of peaks: {params['num_peaks']}")
        >>> if 'z0_validation' in params:
        ...     validation = params['z0_validation']
        ...     print(f"Z0 validation passed: {validation['validation_passed']}")
        """
        if not self._fitted:
            raise RuntimeError("No analysis parameters available. Call fit() method first.")
        
        if not self.params:
            raise RuntimeError("No parameters stored. Ensure catch=True during initialization.")
        
        return self.params.copy()

    @property
    def fitted(self):
        """
        bool: True if the analysis has been completed, False otherwise.
        
        This property indicates whether the fit() method has been called
        and the analysis pipeline has been executed successfully.
        """
        return self._fitted

    @property
    def cluster_bounds(self) -> Optional[Tuple[float, float]]:
        """
        tuple or None: Cluster bounds (CLB, CUB) if available.
        
        Returns a tuple (CLB, CUB) containing the lower and upper bounds
        of the main cluster, or None if bounds haven't been estimated.
        
        Examples
        --------
        >>> bounds = homogeneity.cluster_bounds
        >>> if bounds:
        ...     clb, cub = bounds
        ...     print(f"Main cluster spans [{clb:.3f}, {cub:.3f}]")
        """
        if self.CLB is not None and self.CUB is not None:
            return (self.CLB, self.CUB)
        return None

    @property
    def num_clusters(self) -> int:
        """
        int: Number of detected clusters.
        
        Currently returns the number of identified clusters. In the current
        implementation, this is typically 1 (the main cluster) but the
        framework supports multiple cluster detection.
        """
        return len(self.clusters)

    @property
    def z0_validation(self) -> Optional[Dict]:
        """
        dict or None: Z0 validation results if available.
        
        Returns validation results comparing original Z0 with PDF-derived Z0,
        or None if validation hasn't been performed. The dictionary contains:
        - original_z0: Original Z0 value from GDF
        - pdf_derived_z0: Z0 derived from PDF global maximum
        - is_outside_bounds: Whether PDF Z0 falls outside cluster bounds
        - validation_passed: Overall validation status
        - warnings: List of any warning messages
        
        Examples
        --------
        >>> validation = homogeneity.z0_validation
        >>> if validation and not validation['validation_passed']:
        ...     print("Consider increasing S parameter:")
        ...     for warning in validation['warnings']:
        ...         print(f"  - {warning}")
        """
        if hasattr(self, 'params') and 'z0_validation' in self.params:
            return self.params['z0_validation']
        return None