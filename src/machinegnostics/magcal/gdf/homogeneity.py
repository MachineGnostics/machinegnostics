"""
Data homogeneity check for GDF (Gnostic Distribution Functions) calculations.

Machine Gnostics
Author: Nirmal Parmar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from typing import Union, Dict, Any, Optional, Tuple, List
from machinegnostics.magcal import ELDF, EGDF

class DataHomogeneity:
    """
    A comprehensive class for checking data homogeneity and performing cluster analysis for GDF objects.
    
    This class analyzes the probability density function (PDF) to determine homogeneity and performs
    advanced cluster detection based on PDF peaks. Each peak represents a potential cluster with
    its own onset and offset boundaries determined by slope analysis.
    
    Key Features:
    - PDF-based peak detection for ALL maxima (global + local maxima)
    - Onset/offset point detection using slope analysis around each maxima
    - Main cluster identification using Z0 (global maxima/gnostic mode)
    - Support for both EGDF and ELDF objects
    - CLB/CUB boundaries from main cluster around global maxima
    
    The main cluster is defined as the cluster containing Z0 (global maxima), with boundaries CLB and CUB
    representing the onset and offset points of that cluster.
    
    Attributes:
        gdf (EGDF or ELDF): The input GDF object containing data and computed PDF
        gdf_type (str): Detected type of the GDF object ('egdf' or 'eldf')
        verbose (bool): Controls detailed output during analysis
        catch (bool): Controls whether results are stored in internal parameters dictionary
        params (dict): Dictionary storing analysis results when catch=True
        _fitted (bool): Indicates whether analysis has been performed
        clusters (list): List of detected clusters with their properties
        main_cluster_idx (int): Index of the main cluster containing Z0
        CLB (float): Cluster Lower Bound of the main cluster
        CUB (float): Cluster Upper Bound of the main cluster
        
        # New attributes for maxima/minima detection
        maxima_indices (np.ndarray): Indices of all maxima in PDF
        minima_indices (np.ndarray): Indices of all minima in PDF
        global_max_idx (int): Index of global maximum (Z0)
        global_min_idx (int): Index of global minimum
        left_boundary_min (int): Index of left boundary minimum around global max
        right_boundary_min (int): Index of right boundary minimum around global max
    """

    def __init__(self, gdf: Union[EGDF, ELDF], verbose=True, catch=True, smoothing_sigma=1.0, 
                 min_height_ratio=0.1, min_distance=None):
        """
        Initialize the DataHomogeneity class.
        
        Parameters:
            gdf (EGDF or ELDF): GDF object containing data and computed PDF.
                               Must be fitted before homogeneity checking.
            verbose (bool, optional): If True, prints detailed information. Defaults to True.
            catch (bool, optional): If True, stores results in params dictionary. Defaults to True.
            smoothing_sigma (float): Gaussian smoothing parameter for peak detection. Defaults to 1.0.
            min_height_ratio (float): Minimum height ratio for peak detection. Defaults to 0.1.
            min_distance (int, optional): Minimum distance between peaks. Auto-calculated if None.
        """
        self.gdf = gdf
        self.gdf_type = self._detect_gdf_type()
        self.verbose = verbose
        self.catch = catch
        self.params = {}
        self._fitted = False

        # Peak/valley detection parameters
        self.smoothing_sigma = smoothing_sigma
        self.min_height_ratio = min_height_ratio
        self.min_distance = min_distance

        # Cluster analysis results
        self.clusters = []  # List of detected clusters
        self.main_cluster_idx = None  # Index of main cluster containing Z0
        self.CLB = None  # Cluster Lower Bound of main cluster
        self.CUB = None  # Cluster Upper Bound of main cluster
        self.z0 = None  # Gnostic mode (global maxima)

        # New attributes for maxima/minima detection
        self.maxima_indices = None
        self.minima_indices = None
        self.global_max_idx = None
        self.global_min_idx = None
        self.left_boundary_min = None
        self.right_boundary_min = None

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
        
        # Check for PDF data - must have pdf_points for proper analysis
        if not (hasattr(self.gdf, 'pdf_points') and self.gdf.pdf_points is not None):
            if hasattr(self.gdf, 'catch') and not self.gdf.catch:
                raise AttributeError(f"{self.gdf_type.upper()} object must have catch=True to generate pdf_points required for homogeneity analysis.")
            else:
                raise AttributeError(f"{self.gdf_type.upper()} object is missing 'pdf_points'. Please ensure catch=True when fitting {self.gdf_type.upper()}.")

    def fit(self, estimate_cluster_bounds=True, plot=False):
        """
        Perform comprehensive homogeneity and cluster analysis on the GDF data.
        
        This method performs:
        1. Homogeneity analysis (peak detection and negative value checking)
        2. Peak-based cluster detection with maxima/minima identification
        3. Main cluster identification using Z0 (global maxima)
        4. Cluster boundary estimation (CLB, CUB) for the main cluster
        
        Parameters:
        -----------
        estimate_cluster_bounds : bool, default=True
            Whether to estimate cluster boundaries and perform detailed cluster analysis
        plot : bool, default=False
            Whether to plot the analysis results
        
        Returns:
        --------
        bool: True if data is homogeneous, False otherwise
        """
        if self.verbose:
            print(f"Starting homogeneity and cluster analysis for {self.gdf_type.upper()} data...")
        
        # Set min_distance if not provided
        if self.min_distance is None:
            pdf_data = self._get_pdf_data()
            self.min_distance = len(pdf_data) // 20
        
        # Perform homogeneity analysis
        is_homogeneous = self._is_homogeneous()
        
        # Perform cluster analysis if requested
        if estimate_cluster_bounds:
            self._find_peaks_and_valleys()
            self._find_cluster_boundaries()
            self._define_clusters()
            self._identify_main_cluster()
            self._set_cluster_bounds()
        
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

    def _smooth_pdf(self):
        """Apply Gaussian smoothing to PDF for better peak detection."""
        pdf_data = self._get_pdf_data()
        return gaussian_filter1d(pdf_data, sigma=self.smoothing_sigma)

    def _find_peaks_and_valleys(self):
        """Find all maxima and minima in the PDF using the new logic."""
        smoothed_pdf = self._smooth_pdf()
        data_points = self._get_data_points()
        
        # Find maxima
        min_height = np.max(smoothed_pdf) * self.min_height_ratio
        maxima_idx, _ = find_peaks(smoothed_pdf, 
                                   height=min_height,
                                   distance=self.min_distance)
        
        # Find minima (peaks in inverted signal)
        inverted_pdf = -smoothed_pdf
        minima_idx, _ = find_peaks(inverted_pdf, distance=self.min_distance)
        
        self.maxima_indices = maxima_idx
        self.minima_indices = minima_idx
        
        # Identify global maximum (Z0)
        z0_value = self._get_z0()
        if z0_value is not None and len(maxima_idx) > 0:
            # Find closest maximum to Z0
            max_positions = data_points[maxima_idx]
            closest_idx = np.argmin(np.abs(max_positions - z0_value))
            self.global_max_idx = maxima_idx[closest_idx]
        elif len(maxima_idx) > 0:
            # Use highest peak as global maximum
            max_heights = smoothed_pdf[maxima_idx]
            self.global_max_idx = maxima_idx[np.argmax(max_heights)]
        
        # Identify global minimum
        if len(minima_idx) > 0:
            min_heights = smoothed_pdf[minima_idx]
            self.global_min_idx = minima_idx[np.argmin(min_heights)]

        if self.verbose:
            print(f"Found {len(maxima_idx)} maxima and {len(minima_idx)} minima in PDF")

    def _find_cluster_boundaries(self):
        """Find boundary minima around global maximum to define main cluster."""
        if self.global_max_idx is None or self.minima_indices is None:
            return
        
        data_points = self._get_data_points()
        global_max_pos = data_points[self.global_max_idx]
        
        # Find boundary minima
        left_minima = []
        right_minima = []
        
        for min_idx in self.minima_indices:
            min_pos = data_points[min_idx]
            if min_pos < global_max_pos:
                left_minima.append((min_idx, min_pos))
            elif min_pos > global_max_pos:
                right_minima.append((min_idx, min_pos))
        
        # Get closest boundary minima
        if left_minima:
            self.left_boundary_min = max(left_minima, key=lambda x: x[1])[0]
        if right_minima:
            self.right_boundary_min = min(right_minima, key=lambda x: x[1])[0]

    def _define_clusters(self):
        """Define main, lower, and upper clusters based on boundary minima."""
        if self.global_max_idx is None:
            return
            
        data_points = self._get_data_points()
        
        # Main cluster boundaries
        left_bound = (data_points[self.left_boundary_min] 
                     if self.left_boundary_min is not None 
                     else data_points[0])
        right_bound = (data_points[self.right_boundary_min] 
                      if self.right_boundary_min is not None 
                      else data_points[-1])
        
        # Store cluster information for compatibility
        self.clusters = [{
            'cluster_id': 0,
            'peak_position': data_points[self.global_max_idx],
            'onset_position': left_bound,
            'offset_position': right_bound,
            'is_main_cluster': True
        }]

    def _is_homogeneous(self):
        """Internal method to check if the data is homogeneous."""
        # First do basic peak detection for homogeneity check
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
        
        return is_homogeneous

    def _get_pdf_data(self):
        """Get PDF points data (only available when catch=True)."""
        if hasattr(self.gdf, 'pdf_points') and self.gdf.pdf_points is not None:
            return self.gdf.pdf_points
        else:
            raise AttributeError("PDF points not available. Ensure GDF object was fitted with catch=True.")

    def _get_data_points(self):
        """Get corresponding data points for PDF data."""
        if hasattr(self.gdf, 'di_points_n') and self.gdf.di_points_n is not None:
            return self.gdf.di_points_n
        else:
            raise AttributeError("Data points (di_points_n) not available. Ensure GDF object was fitted with catch=True.")

    def _get_z0(self):
        """Get Z0 (gnostic mode) from the GDF object - this is the global maxima."""
        if hasattr(self.gdf, 'z0') and self.gdf.z0 is not None:
            return self.gdf.z0
        elif hasattr(self.gdf, 'params') and 'z0' in self.gdf.params:
            return self.gdf.params['z0']
        else:
            # If Z0 not available, use the point with maximum PDF (global maxima)
            pdf_data = self._get_pdf_data()
            data_points = self._get_data_points()
            max_idx = np.argmax(pdf_data)
            if self.verbose:
                print("Warning: Z0 not found in GDF object. Using PDF global maximum as Z0.")
            return data_points[max_idx]

    def _detect_peaks(self):
        """
        Legacy method for backward compatibility - detect peaks for homogeneity check.
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
        
        # Find ALL local maxima (points higher than both neighbors)
        for i in range(1, len(pdf_data) - 1):
            if (pdf_data[i] > pdf_data[i-1] and pdf_data[i] > pdf_data[i+1]):
                is_global = (i == global_max_idx)
                peaks.append({
                    'index': i,
                    'value': pdf_data[i],
                    'position': data_points[i],
                    'is_global_max': is_global
                })
        
        # Check edge cases (first and last points)
        if len(pdf_data) > 1:
            # Check first point
            if pdf_data[0] > pdf_data[1]:
                is_global = (0 == global_max_idx)
                peaks.append({
                    'index': 0,
                    'value': pdf_data[0],
                    'position': data_points[0],
                    'is_global_max': is_global
                })
            
            # Check last point
            last_idx = len(pdf_data) - 1
            if pdf_data[last_idx] > pdf_data[last_idx-1]:
                is_global = (last_idx == global_max_idx)
                peaks.append({
                    'index': last_idx,
                    'value': pdf_data[last_idx],
                    'position': data_points[last_idx],
                    'is_global_max': is_global
                })
        
        # Ensure global maximum is included
        global_max_found = any(peak['is_global_max'] for peak in peaks)
        if not global_max_found:
            peaks.append({
                'index': global_max_idx,
                'value': global_max_value,
                'position': data_points[global_max_idx],
                'is_global_max': True
            })
        
        # Sort peaks by value (highest first), but ensure global max is first
        peaks.sort(key=lambda x: (not x['is_global_max'], -x['value']))
        
        return peaks

    def _identify_main_cluster(self):
        """
        Identify the main cluster containing Z0 (global maxima).
        """
        if not self.clusters:
            if self.verbose:
                print("No clusters detected for main cluster identification")
            return
        
        self.z0 = self._get_z0()
        self.main_cluster_idx = 0  # Always the first (and main) cluster
        
        if self.verbose:
            print(f"Main cluster identified: Contains Z0 ({self.z0:.6f})")

    def _set_cluster_bounds(self):
        """
        Set CLB and CUB based on the main cluster boundaries.
        """
        if not self.clusters or self.global_max_idx is None:
            if self.verbose:
                print("Cannot set cluster bounds: no main cluster identified")
            return
        
        main_cluster = self.clusters[0]
        self.CLB = main_cluster['onset_position']
        self.CUB = main_cluster['offset_position']
        
        # Store results
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
        
        # Write to GDF object's params if GDF has catch=True
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
        Plot the PDF with maxima, minima, and main cluster region highlighted in green.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size for the plot
        title : str, optional
            Title for the plot. Auto-generated if None.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before plotting. Run fit() method first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        pdf_data = self._get_pdf_data()
        data_points = self._get_data_points()
        
        # Plot PDF
        ax.plot(data_points, pdf_data, 'b-', linewidth=2, label='PDF')
        
        # Plot main cluster region (light green fill)
        if self.CLB is not None and self.CUB is not None:
            # Find indices for the main cluster region
            main_mask = (data_points >= self.CLB) & (data_points <= self.CUB)
            if np.any(main_mask):
                ax.fill_between(data_points[main_mask], 
                               pdf_data[main_mask],
                               alpha=0.3, color='lightgreen', 
                               label='Main Cluster')
        
        # Plot maxima
        if self.maxima_indices is not None and len(self.maxima_indices) > 0:
            max_positions = data_points[self.maxima_indices]
            max_values = pdf_data[self.maxima_indices]
            
            # Local maxima (grey solid lines)
            for i, (pos, val) in enumerate(zip(max_positions, max_values)):
                if self.maxima_indices[i] != self.global_max_idx:
                    ax.axvline(pos, color='grey', linestyle='-', linewidth=1, alpha=0.7)
                    ax.plot(pos, val, 'o', color='grey', markersize=6)
            
            # Global maximum (magenta dashed line)
            if self.global_max_idx is not None:
                global_pos = data_points[self.global_max_idx]
                global_val = pdf_data[self.global_max_idx]
                ax.axvline(global_pos, color='magenta', linestyle='--', linewidth=2)
                ax.plot(global_pos, global_val, 'o', color='magenta', markersize=8, 
                       label=f'Global Maximum (Z0={global_pos:.3f})')
        
        # Plot minima
        if self.minima_indices is not None and len(self.minima_indices) > 0:
            min_positions = data_points[self.minima_indices]
            min_values = pdf_data[self.minima_indices]
            
            # Local minima (grey solid lines)
            for i, (pos, val) in enumerate(zip(min_positions, min_values)):
                if self.minima_indices[i] != self.global_min_idx:
                    ax.axvline(pos, color='grey', linestyle='-', linewidth=1, alpha=0.7)
                    ax.plot(pos, val, 's', color='grey', markersize=6)
            
            # Global minimum (magenta dashed line)
            if self.global_min_idx is not None:
                global_pos = data_points[self.global_min_idx]
                global_val = pdf_data[self.global_min_idx]
                ax.axvline(global_pos, color='magenta', linestyle='--', linewidth=2)
                ax.plot(global_pos, global_val, 's', color='magenta', markersize=8,
                       label='Global Minimum')
        
        # Add cluster boundary lines
        if self.CLB is not None:
            ax.axvline(self.CLB, color='red', linestyle=':', linewidth=1.5, alpha=0.8, label=f'CLB={self.CLB:.3f}')
        if self.CUB is not None:
            ax.axvline(self.CUB, color='red', linestyle=':', linewidth=1.5, alpha=0.8, label=f'CUB={self.CUB:.3f}')
        
        ax.set_xlabel('Data Points')
        ax.set_ylabel('PDF Values')
        
        if title is None:
            title = f"{self.gdf_type.upper()} PDF Analysis with Cluster Detection"
        ax.set_title(title)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def get_cluster_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract clustered data based on main cluster bounds (CLB, CUB).
        
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
        
        # Split data based on main cluster bounds
        lower_cluster = data[data < self.CLB]
        main_cluster = data[(data >= self.CLB) & (data <= self.CUB)]
        upper_cluster = data[data > self.CUB]
        
        if self.verbose:
            print(f"Clustered data: Lower={len(lower_cluster)}, Main={len(main_cluster)}, Upper={len(upper_cluster)}")
        
        return lower_cluster, main_cluster, upper_cluster

    def get_all_clusters_data(self) -> List[Dict]:
        """
        Get data for all detected clusters.
        
        Returns:
        --------
        list: List of dictionaries containing cluster data and metadata
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
        Get main cluster bounds as a tuple.
        
        Returns:
        --------
        tuple or None: (CLB, CUB) if bounds have been estimated, None otherwise
        """
        if self.CLB is not None and self.CUB is not None:
            return (self.CLB, self.CUB)
        return None

    @property
    def num_clusters(self) -> int:
        """Get total number of detected clusters."""
        return len(self.clusters) if self.clusters else 0