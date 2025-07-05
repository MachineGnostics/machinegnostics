"""
Data homogeneity check for GDF (Gnostic Distribution Functions) calculations.

Machine Gnostics
Author: Nirmal Parmar
"""

import numpy as np
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

    def is_homogeneous(self, prominence_threshold=0.01, distance_threshold=5):
        """
        Check if the data is homogeneous based on Machine Gnostics definition.
        
        A homogeneous data sample is defined as one composed of only one cluster,
        therefore it has only one density maximum. If there are several clusters
        in a sample then each will have a separate maximum in the sample's density function.
        
        This supports infinite data support where all data and Z0 values are strictly positive.
        
        Parameters:
        prominence_threshold (float): Minimum prominence for a peak to be considered significant
        distance_threshold (int): Minimum distance between peaks
        
        Returns:
        bool: True if homogeneous (single cluster/peak), False if not homogeneous (multiple clusters/peaks)
        dict: Additional information about the homogeneity analysis
        """
        from scipy.signal import find_peaks
        
        # Get PDF from params
        if 'pdf' not in self.params or self.params['pdf'] is None:
            raise ValueError("PDF must be calculated before checking homogeneity.")
        
        pdf = self.params['pdf']
        
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
        
        # Get peak locations in original data domain
        peak_locations = []
        if 'di_points' in self.params and len(peaks) > 0:
            di_points = self.params['di_points']
            peak_locations = [di_points[idx] for idx in peaks]
        
        # Additional analysis information
        # analysis_info = {
        #     'N0': N0,
        #     'peak_indices': peaks,
        #     'peak_locations': peak_locations,
        #     'peak_values': pdf[peaks] if len(peaks) > 0 else [],
        #     'peak_prominences': properties.get('prominences', []),
        #     'is_homogeneous': is_homog,
        #     'homogeneity_statement': f"Sample is homogeneous (single cluster)" if is_homog else f"Sample is not homogeneous ({N0} clusters detected)",
        #     'clusters_detected': N0
        # }

        # update params
        self.params['is_homogeneous'] = is_homog
        self.params['N0'] = N0
        self.params['g_mean'] = peak_locations
        
        return is_homog, self.params

    def _analyze_clusters(self):
        """
        Analyze the clusters in the data based on PDF peaks.
        
        Returns:
        dict: Detailed cluster analysis
        """
        is_homog, analysis = self.is_homogeneous()
        
        cluster_analysis = {
            'homogeneous': is_homog,
            'number_of_clusters': analysis['N0'],
            'cluster_centers': analysis['peak_locations'],
            'cluster_densities': analysis['peak_values'].tolist() if len(analysis['peak_values']) > 0 else [],
            'infinite_support': True,  # Machine Gnostics assumption
            'strictly_positive': self._check_positive_support()
        }
        
        return cluster_analysis

    def _check_positive_support(self):
        """
        Check if all data values are strictly positive (infinite data support requirement).
        
        Returns:
        bool: True if all data values are strictly positive
        """
        return np.all(self.data > 0)

    def _plot_homogeneity_analysis(self):
        """
        Plot PDF with identified peaks for homogeneity analysis.
        """
        import matplotlib.pyplot as plt
        
        if 'pdf' not in self.params:
            raise ValueError("PDF must be calculated before plotting homogeneity analysis.")
        
        # Ensure homogeneity analysis is done
        is_homog, analysis = self.is_homogeneous()
        
        pdf = self.params['pdf']
        di_points = self.params.get('di_points', range(len(pdf)))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot PDF
        ax.plot(di_points, pdf, 'b-', linewidth=2, label='PDF')
        
        # Mark peaks
        if len(analysis['peak_indices']) > 0:
            peak_x = [di_points[i] for i in analysis['peak_indices']]
            peak_y = [pdf[i] for i in analysis['peak_indices']]
            ax.plot(peak_x, peak_y, 'ro', markersize=8, label=f'Peaks (N0 = {analysis["N0"]})')
        
        ax.set_xlabel('Data Points')
        ax.set_ylabel('PDF')
        ax.set_title(f'Homogeneity Analysis: {analysis["homogeneity_statement"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return analysis
    
    def homogenize(self):
        """
        Perform homogeneity check and analysis.
        
        Returns:
        dict: Homogeneity analysis results
        """
        # Check if PDF is available
        if 'pdf' not in self.params or self.params['pdf'] is None:
            raise ValueError("PDF must be calculated before homogenization.")
        
        gw = GnosticsWeights()
        homogeneous_weights = gw._get_gnostic_weights(self.params['z'])
        self.weights = self.weights * homogeneous_weights