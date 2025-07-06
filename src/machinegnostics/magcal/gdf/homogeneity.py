"""
Data homogeneity check for GDF (Gnostic Distribution Functions) calculations.

Machine Gnostics
Author: Nirmal Parmar
"""

import numpy as np
import matplotlib.pyplot as plt
from machinegnostics.magcal.mg_weights import GnosticsWeights

class DataHomogeneity:
    """
    Class to check the homogeneity of data for GDF calculations.
    This class is used to ensure that the data is homogeneous before performing GDF calculations.
    """

    def __init__(self, data, params):
        self.data = data
        self.params = params
        self.gw = GnosticsWeights()
        self.analysis_info = {}

    def is_homogeneous(self, prominence_threshold=0.01, distance_threshold=5, filter_data_range=True):
        """
        Check if the data is homogeneous based on Machine Gnostics definition.
        
        A homogeneous data sample is defined as one composed of only one cluster,
        therefore it has only one density maximum. If there are several clusters
        in a sample then each will have a separate maximum in the sample's density function.
        
        This supports infinite data support where all data and Z0 values are strictly positive.
        
        Parameters:
        prominence_threshold (float): Minimum prominence for a peak to be considered significant
        distance_threshold (int): Minimum distance between peaks
        filter_data_range (bool): If True, filter PDF to only include data min-max range
        
        Returns:
        bool: True if homogeneous (single cluster/peak), False if not homogeneous (multiple clusters/peaks)
        dict: Additional information about the homogeneity analysis
        """
        from scipy.signal import find_peaks
        
        # Get PDF from params
        if 'pdf' not in self.params or self.params['pdf'] is None:
            raise ValueError("PDF must be calculated before checking homogeneity.")
        
        pdf = self.params['pdf']
        di_points = self.params.get('di_points', np.arange(len(pdf)))
        
        # Store original data for reference
        pdf_original = pdf.copy()
        di_points_original = di_points.copy()
        
        # Filter PDF data to data min-max range if requested
        if filter_data_range:
            data_min = self.data.min()
            data_max = self.data.max()
            
            # Find indices within data range
            mask = (di_points >= data_min) & (di_points <= data_max)
            
            if np.any(mask):
                # Filter both PDF and di_points to data range
                pdf = pdf[mask]
                di_points = di_points[mask]
                
                # print(f"Filtered PDF to data range [{data_min:.3f}, {data_max:.3f}]")
                # print(f"Original PDF length: {len(pdf_original)}, Filtered PDF length: {len(pdf)}")
            else:
                print("Warning: No PDF points found within data range, using full PDF")
                filter_data_range = False  # Fallback to full range
        
        # Handle infinite data support - ensure all values are strictly positive
        if np.any(pdf <= 0):
            # Add small epsilon to handle zero or negative values
            pdf = pdf + 1e-10
        
        # Normalize PDF for consistent peak detection
        pdf_normalized = pdf / np.max(pdf)
        
        # Find peaks (density maxima) in the PDF
        peaks, properties = find_peaks(
            pdf_normalized, 
            prominence=prominence_threshold,
            distance=distance_threshold
        )
        
        # Number of significant peaks (clusters)
        N0 = len(peaks)
        
        # Homogeneous if exactly one peak (one cluster)
        is_homog = (N0 == 1)
        
        # Get peak locations in original data domain (using filtered di_points)
        peak_locations = []
        if len(peaks) > 0:
            peak_locations = [di_points[idx] for idx in peaks]
        
        # Additional analysis information
        self.analysis_info = {
            'N0': N0,
            'peak_indices': peaks,
            'peak_locations': peak_locations,
            'peak_values': pdf[peaks] if len(peaks) > 0 else [],
            'peak_prominences': properties.get('prominences', []),
            'is_homogeneous': is_homog,
            'homogeneity_statement': f"Sample is homogeneous (single cluster)" if is_homog else f"Sample is not homogeneous ({N0} clusters detected)",
            'clusters_detected': N0,
            'data_filtered': filter_data_range,
            'data_range': [self.data.min(), self.data.max()] if filter_data_range else None,
            'filtered_pdf_length': len(pdf),
            'original_pdf_length': len(pdf_original),
            'filtered_pdf': pdf,
            'filtered_di_points': di_points,
            'original_pdf': pdf_original,
            'original_di_points': di_points_original
        }

        # Update params
        self.params['is_homogeneous'] = is_homog
        self.params['N0'] = N0
        self.params['g_mean'] = peak_locations
        self.params['homogeneity_filtered'] = filter_data_range
        
        return is_homog, self.params

    def analyze_clusters(self):
        """
        Analyze the clusters in the data based on PDF peaks.
        
        Returns:
        dict: Detailed cluster analysis
        """
        if not hasattr(self, 'analysis_info') or not self.analysis_info:
            is_homog, params = self.is_homogeneous()
        
        analysis = self.analysis_info
        cluster_analysis = {
            'homogeneous': analysis['is_homogeneous'],
            'number_of_clusters': analysis['N0'],
            'cluster_centers': analysis['peak_locations'],
            'cluster_densities': analysis['peak_values'].tolist() if len(analysis['peak_values']) > 0 else [],
            'infinite_support': True,  # Machine Gnostics assumption
            'strictly_positive': self._check_positive_support(),
            'data_filtered': analysis.get('data_filtered', False),
            'analysis_range': analysis.get('data_range', None)
        }
        
        return cluster_analysis

    def _check_positive_support(self):
        """
        Check if all data values are strictly positive (infinite data support requirement).
        
        Returns:
        bool: True if all data values are strictly positive
        """
        return np.all(self.data > 0)

    def plot_homogeneity_analysis(self, filter_data_range=True):
        """
        Plot PDF with identified peaks for homogeneity analysis.
        
        Parameters:
        filter_data_range (bool): Whether to show filtered analysis or full range
        """
        if 'pdf' not in self.params:
            raise ValueError("PDF must be calculated before plotting homogeneity analysis.")
        
        # Ensure homogeneity analysis is done with specified filtering
        is_homog, params = self.is_homogeneous(filter_data_range=filter_data_range)
        analysis = self.analysis_info
        
        # Get data for plotting
        pdf_original = analysis['original_pdf']
        di_points_original = analysis['original_di_points']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot full PDF in light color
        ax.plot(di_points_original, pdf_original, 'lightblue', linewidth=1, 
                label='Full PDF', alpha=0.6)
        
        if filter_data_range and analysis['data_filtered']:
            # Plot filtered PDF in main color
            pdf_filtered = analysis['filtered_pdf']
            di_points_filtered = analysis['filtered_di_points']
            
            ax.plot(di_points_filtered, pdf_filtered, 'b-', linewidth=2, 
                    label='Filtered PDF (Data Range)')
            
            # Add vertical lines to show data range
            data_min, data_max = analysis['data_range']
            ax.axvline(x=data_min, color='green', linestyle='--', alpha=0.7, 
                      label=f'Data Min ({data_min:.3f})')
            ax.axvline(x=data_max, color='red', linestyle='--', alpha=0.7, 
                      label=f'Data Max ({data_max:.3f})')
            
            # Shade the filtered region
            ax.axvspan(data_min, data_max, alpha=0.1, color='yellow', 
                      label='Analysis Region')
            
            # Mark peaks (found in filtered data)
            if len(analysis['peak_indices']) > 0:
                peak_x = [di_points_filtered[i] for i in analysis['peak_indices']]
                peak_y = [pdf_filtered[i] for i in analysis['peak_indices']]
                ax.plot(peak_x, peak_y, 'ro', markersize=10, 
                       label=f'Peaks (N0 = {analysis["N0"]})')
        else:
            # Plot full PDF in main color
            ax.plot(di_points_original, pdf_original, 'b-', linewidth=2, label='PDF')
            
            # Mark peaks (found in full data)
            if len(analysis['peak_indices']) > 0:
                peak_x = [di_points_original[i] for i in analysis['peak_indices']]
                peak_y = [pdf_original[i] for i in analysis['peak_indices']]
                ax.plot(peak_x, peak_y, 'ro', markersize=10, 
                       label=f'Peaks (N0 = {analysis["N0"]})')
        
        # Add data points as vertical lines
        for point in self.data:
            ax.axvline(x=point, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        # # Add text annotation for data points
        # ax.text(0.02, 0.95, f'Data points: {list(self.data)}', 
        #         transform=ax.transAxes, verticalalignment='top',
        #         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        ax.set_xlabel('Data Points')
        ax.set_ylabel('PDF')
        
        filter_text = " (Filtered to Data Range)" if (filter_data_range and analysis['data_filtered']) else " (Full Range)"
        ax.set_title(f'Homogeneity Analysis{filter_text}: {analysis["homogeneity_statement"]}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return analysis

    def homogenize(self):
        """
        Perform data homogenization by applying gnostic weights.
        
        Returns:
        np.ndarray: Updated weights after homogenization
        """
        # Get current weights with proper numpy array handling
        current_weights = None
        
        # Check self.weights first
        if hasattr(self, 'weights') and self.weights is not None:
            current_weights = self.weights
        # Check params['weights'] second
        elif 'weights' in self.params and self.params['weights'] is not None:
            current_weights = self.params['weights']
        # Default to ones
        else:
            current_weights = np.ones(len(self.data))
        
        # Get transformed data with proper handling
        z_values = None
        
        # Check self.z first
        if hasattr(self, 'z') and self.z is not None:
            z_values = self.z
        # Check params['z'] second
        elif 'z' in self.params and self.params['z'] is not None:
            z_values = self.params['z']
        else:
            raise ValueError("Transformed data 'z' must be available for homogenization.")
        
        # Calculate homogeneous weights
        homogeneous_weights = self.gw._get_gnostic_weights(z_values)
        
        # Apply homogeneous weights
        new_weights = current_weights * homogeneous_weights
        
        # Update weights
        if hasattr(self, 'weights'):
            self.weights = new_weights
        self.params['weights'] = new_weights
        self.params['homogenization_applied'] = True
        self.params['homogeneous_weights'] = homogeneous_weights
        
        return new_weights

    def get_homogeneity_summary(self):
        """
        Get a comprehensive summary of homogeneity analysis.
        
        Returns:
        dict: Summary of homogeneity analysis
        """
        if not hasattr(self, 'analysis_info') or not self.analysis_info:
            self.is_homogeneous()
        
        cluster_info = self.analyze_clusters()
        
        summary = {
            'data_points': list(self.data),
            'data_range': [float(self.data.min()), float(self.data.max())],
            'strictly_positive': self._check_positive_support(),
            'homogeneous': cluster_info['homogeneous'],
            'number_of_clusters': cluster_info['number_of_clusters'],
            'cluster_centers': cluster_info['cluster_centers'],
            'homogeneity_statement': self.analysis_info['homogeneity_statement'],
            'analysis_filtered': self.analysis_info.get('data_filtered', False),
            'pdf_points_analyzed': self.analysis_info.get('filtered_pdf_length', 0),
            'total_pdf_points': self.analysis_info.get('original_pdf_length', 0)
        }
        
        return summary