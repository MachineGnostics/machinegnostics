"""
Data homogeneity check for GDF (Gnostic Distribution Functions) calculations.

Machine Gnostics
Author: Nirmal Parmar
"""

import numpy as np
from typing import Union, Dict, Any, Optional, Tuple
from machinegnostics.magcal import ELDF, EGDF

class DataHomogeneity:
    """
    A comprehensive class for checking data homogeneity and performing cluster analysis for GDF objects.
    
    This class analyzes the probability density function (PDF) of EGDF or ELDF objects to determine if the 
    underlying data is homogeneous and provides cluster boundary estimation when data is heterogeneous.
    
    Data is considered homogeneous when it has:
    1. No negative PDF values (indicating proper probability distribution)
    2. Exactly one peak (indicating a single, coherent data distribution)
    
    When data is heterogeneous (multiple peaks), the class can identify cluster boundaries
    to separate the main cluster from outlying clusters.
    
    Features:
    - Homogeneity analysis with peak detection
    - Cluster boundary estimation (CLB, CUB) 
    - EGDF convergence validation for cluster bounds
    - Automatic writing of results to GDF params
    - Support for both EGDF and ELDF objects
    
    Attributes:
        gdf (EGDF or ELDF): The input GDF object containing data and computed PDF
        gdf_type (str): Detected type of the GDF object ('egdf' or 'eldf')
        verbose (bool): Controls detailed output during analysis
        catch (bool): Controls whether results are stored in internal parameters dictionary
        cluster_threshold (float): Threshold for PDF-based clustering (default: 0.05)
        params (dict): Dictionary storing analysis results when catch=True
        _fitted (bool): Indicates whether analysis has been performed
    """

    def __init__(self, gdf: Union[EGDF, ELDF], verbose=True, catch=True, cluster_threshold=0.05):
        """
        Initialize the DataHomogeneity class.
        
        Parameters:
            gdf (EGDF or ELDF): GDF object containing data and computed PDF.
                               Must be fitted before homogeneity checking.
            verbose (bool, optional): If True, prints detailed information. Defaults to True.
            catch (bool, optional): If True, stores results in params dictionary. Defaults to True.
            cluster_threshold (float, optional): Threshold for PDF-based clustering. Defaults to 0.05.
        """
        self.gdf = gdf
        self.gdf_type = self._detect_gdf_type()
        self.verbose = verbose
        self.catch = catch
        self.cluster_threshold = cluster_threshold
        self.params = {}
        self._fitted = False

        # Cluster boundaries (will be set during analysis)
        self.CLB = None  # Cluster Lower Bound
        self.CUB = None  # Cluster Upper Bound

        # validation
        self._gdf_obj_validation()

    def _detect_gdf_type(self):
        """Detect whether the object is EGDF or ELDF."""
        class_name = self.gdf.__class__.__name__
        if 'EGDF' in class_name:
            return 'egdf'
        elif 'ELDF' in class_name:
            return 'eldf'
        else:
            # Try to detect by checking for specific methods
            if hasattr(self.gdf, '_fit_egdf'):
                return 'egdf'
            elif hasattr(self.gdf, '_fit_eldf'):
                return 'eldf'
            else:
                raise ValueError("Cannot determine GDF type. Object must be EGDF or ELDF.")

    def _gdf_obj_validation(self):
        """Validate that the input is a fitted GDF object with required attributes."""
        # Check if object has _fitted attribute and is fitted
        if not hasattr(self.gdf, '_fitted'):
            raise ValueError("GDF object must have _fitted attribute")
        
        if not self.gdf._fitted:
            raise ValueError("GDF object must be fitted before homogeneity checking")
        
        # Check for required attributes
        required_attrs = ['data']
        for attr in required_attrs:
            if not hasattr(self.gdf, attr):
                raise ValueError(f"GDF object missing required attribute: {attr}")
        
        # Check for PDF data
        if not hasattr(self.gdf, 'pdf') or self.gdf.pdf is None:
            raise AttributeError(f"{self.gdf_type.upper()} object is missing 'pdf' attribute or PDF is None. Please fit {self.gdf_type.upper()} before using DataHomogeneity.")

    def fit(self, estimate_cluster_bounds=True):
        """
        Perform comprehensive homogeneity and cluster analysis on the GDF data.
        
        This method performs:
        1. Homogeneity analysis (peak detection and negative value checking)
        2. Cluster boundary estimation (if estimate_cluster_bounds=True)
        3. Automatic writing of results to GDF params
        
        Parameters:
        -----------
        estimate_cluster_bounds : bool, default=True
            Whether to estimate cluster boundaries (CLB, CUB) when data is heterogeneous
        
        Returns:
        --------
        bool: True if data is homogeneous, False otherwise
        """
        if self.verbose:
            print(f"Starting homogeneity and cluster analysis for {self.gdf_type.upper()} data...")
        
        # Perform homogeneity analysis
        is_homogeneous = self._is_homogeneous()
        
        # Estimate cluster bounds if requested and data is heterogeneous
        if estimate_cluster_bounds:
            if not is_homogeneous and self.verbose:
                print("Data is heterogeneous. Estimating cluster boundaries...")
            
            self._estimate_cluster_bounds()
        
        self._fitted = True
        
        if self.verbose:
            print(f"Analysis completed for {self.gdf_type.upper()}.")
            if hasattr(self, 'CLB') and hasattr(self, 'CUB'):
                print(f"Cluster boundaries: CLB={self.CLB}, CUB={self.CUB}")
        
        return is_homogeneous

    def test_homogeneity(self, estimate_cluster_bounds=True):
        """
        Legacy method for backward compatibility.
        
        Parameters:
        -----------
        estimate_cluster_bounds : bool, default=True
            Whether to estimate cluster boundaries
            
        Returns:
        --------
        bool: True if data is homogeneous, False otherwise
        """
        if self.verbose:
            print("Note: test_homogeneity() is deprecated. Use fit() method instead.")
        
        return self.fit(estimate_cluster_bounds=estimate_cluster_bounds)

    def _is_homogeneous(self):
        """Internal method to check if the data is homogeneous."""
        num_peaks = self._get_peaks()
        has_negative_pdf = np.any(self.gdf.pdf < 0)

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

        # Store in internal params if catch=True
        if self.catch:
            self.params.update({
                'is_homogeneous': is_homogeneous,
                'has_negative_pdf': has_negative_pdf,
                'num_peaks': num_peaks,
                'gdf_type': self.gdf_type,
                'homogeneity_fitted': True
            })
        
        # Write to GDF object's params if GDF has catch=True
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
        elif self.verbose:
            if not hasattr(self.gdf, 'catch') or not self.gdf.catch:
                print(f"Note: {self.gdf_type.upper()} object has catch=False. Results not written to GDF params.")
            
        return is_homogeneous

    def _get_peaks(self):
        """Estimate number of significant peaks (local maxima) in the PDF."""
        if not hasattr(self.gdf, 'pdf') or self.gdf.pdf is None:
            if self.verbose:
                print(f"Warning: PDF not available for peak detection in {self.gdf_type.upper()}.")
            return 0
        
        try:
            pdf = self.gdf.pdf
            
            # Handle edge cases
            if len(pdf) < 3:
                return 1 if len(pdf) > 0 and np.max(pdf) > 0 else 0
            
            # Find local maxima
            peaks = []
            
            # Check each point to see if it's a local maximum
            for i in range(1, len(pdf) - 1):
                # A point is a local maximum if it's greater than both neighbors
                if pdf[i] > pdf[i-1] and pdf[i] > pdf[i+1]:
                    # Only consider significant peaks (> 1% of maximum PDF value)
                    if pdf[i] > np.max(pdf) * 0.01:
                        peaks.append(i)
            
            # If no significant peaks found but PDF has positive values, assume one peak
            if len(peaks) == 0 and np.max(pdf) > 0:
                # Find the global maximum as the single peak
                max_idx = np.argmax(pdf)
                if pdf[max_idx] > 0:
                    peaks = [max_idx]
            
            peak_count = len(peaks)
            
            if self.verbose:
                print(f"Detected {peak_count} peaks in {self.gdf_type.upper()} PDF")
            
            return peak_count
            
        except Exception as e:
            if self.verbose:
                print(f"Error in peak detection for {self.gdf_type.upper()}: {e}")
            return 0

    def _estimate_cluster_bounds(self):
        """
        Estimate cluster boundaries (CLB, CUB) based on PDF characteristics.
        
        This method identifies the main cluster boundaries by:
        1. Finding the global PDF maximum
        2. Identifying onset and stop points where significant slopes begin/end
        3. Validating boundaries using EGDF convergence criteria
        4. Setting CLB (Cluster Lower Bound) and CUB (Cluster Upper Bound)
        """
        # Try to get PDF data - prefer smooth pdf_points, fallback to discrete pdf
        pdf_data = None
        data_points = None
        
        if hasattr(self.gdf, 'pdf_points') and self.gdf.pdf_points is not None:
            # Use smooth PDF curve
            pdf_data = self.gdf.pdf_points
            data_points = self.gdf.di_points_n
            if self.verbose:
                print("Using smooth PDF points for clustering")
        elif hasattr(self.gdf, 'pdf') and self.gdf.pdf is not None:
            # Use discrete PDF
            pdf_data = self.gdf.pdf
            data_points = self.gdf.data
            if self.verbose:
                print("Using discrete PDF for clustering")
        else:
            if self.verbose:
                print("Warning: Neither PDF points nor PDF data available for clustering")
            return
        
        # Get EGDF data - prefer smooth egdf_points, fallback to discrete egdf
        egdf_data = None
        egdf_data_points = None
        
        if hasattr(self.gdf, 'egdf_points') and self.gdf.egdf_points is not None:
            # Use smooth EGDF curve
            egdf_data = self.gdf.egdf_points
            egdf_data_points = self.gdf.di_points_n
            if self.verbose:
                print("Using smooth EGDF points for validation")
        elif hasattr(self.gdf, 'egdf') and self.gdf.egdf is not None:
            # Use discrete EGDF
            egdf_data = self.gdf.params.get('egdf', self.gdf.egdf) if hasattr(self.gdf, 'params') else self.gdf.egdf
            egdf_data_points = self.gdf.data
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
                # Different data points, interpolate EGDF values at PDF data points
                egdf_sorted_indices = np.argsort(egdf_data_points)
                sorted_egdf_data_points = egdf_data_points[egdf_sorted_indices]
                sorted_egdf_values = egdf_data[egdf_sorted_indices]
                sorted_egdf = np.interp(sorted_data, sorted_egdf_data_points, sorted_egdf_values)
        
        # Step 1: Find PDF global maxima
        global_max_idx = np.argmax(sorted_pdf)
        global_max_value = sorted_pdf[global_max_idx]
        global_max_point = sorted_data[global_max_idx]
        
        if self.verbose:
            print(f"Global PDF maximum: {global_max_value:.6f} at data point {global_max_point:.3f}")
        
        # Step 2: Calculate thresholds
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
        
        # Step 6: Set cluster boundaries
        self.CLB = sorted_data[onset_start_idx]
        self.CUB = sorted_data[stop_end_idx]
        
        # Store results
        if self.catch:
            self.params.update({
                'CLB': float(self.CLB),
                'CUB': float(self.CUB),
                'global_max_point': float(global_max_point),
                'global_max_value': float(global_max_value),
                'cluster_bounds_estimated': True
            })
        
        # Write to GDF object's params if GDF has catch=True
        if hasattr(self.gdf, 'catch') and self.gdf.catch and hasattr(self.gdf, 'params'):
            self.gdf.params.update({
                'CLB': float(self.CLB),
                'CUB': float(self.CUB),
                'cluster_bounds_estimated': True
            })
            
            if self.verbose:
                print(f"Cluster bounds written to {self.gdf_type.upper()} params dictionary.")
        
        if self.verbose:
            print(f"Cluster boundaries estimated: CLB={self.CLB:.6f}, CUB={self.CUB:.6f}")

    def get_cluster_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract clustered data based on estimated cluster bounds.
        
        Returns:
        --------
        tuple: (lower_cluster, main_cluster, upper_cluster)

        - lower_cluster: Data points < CLB
        - main_cluster: Data points between CLB and CUB (inclusive)
        - upper_cluster: Data points > CUB
        
        Raises:
        -------
        RuntimeError: If cluster bounds have not been estimated yet
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before extracting cluster data")
        
        if self.CLB is None or self.CUB is None:
            raise RuntimeError("Cluster bounds not estimated. Call fit(estimate_cluster_bounds=True)")
        
        data = self.gdf.data
        
        # Split data based on cluster bounds
        lower_cluster = data[data < self.CLB]
        main_cluster = data[(data >= self.CLB) & (data <= self.CUB)]
        upper_cluster = data[data > self.CUB]
        
        if self.verbose:
            print(f"Clustered data: Lower={len(lower_cluster)}, Main={len(main_cluster)}, Upper={len(upper_cluster)}")
        
        return lower_cluster, main_cluster, upper_cluster

    def get_homogeneity_params(self):
        """Returns dictionary of stored analysis parameters."""
        if not self._fitted:
            raise RuntimeError("No analysis parameters available. Call fit() method first.")
        
        if not self.params:
            raise RuntimeError("No parameters stored. Ensure catch=True during initialization.")
        
        return self.params.copy()

    @property
    def fitted(self):
        """Check if analysis has been performed."""
        return self._fitted

    @property
    def cluster_bounds(self) -> Optional[Tuple[float, float]]:
        """
        Get cluster bounds as a tuple.
        
        Returns:
        --------
        tuple or None: (CLB, CUB) if bounds have been estimated, None otherwise
        """
        if self.CLB is not None and self.CUB is not None:
            return (self.CLB, self.CUB)
        return None