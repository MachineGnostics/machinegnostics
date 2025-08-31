"""
Data homogeneity check for GDF (Gnostic Distribution Functions) calculations.

Machine Gnostics
Author: Nirmal Parmar
"""

import numpy as np
from machinegnostics.magcal.gdf.egdf import EGDF

class DataHomogeneity:
    """
    A class for checking the homogeneity of data for GDF (Gnostic Distribution Functions) calculations.
    
    This class analyzes the probability density function (PDF) of an EGDF object to determine if the 
    underlying data is homogeneous. Data is considered homogeneous when it has:
    1. No negative PDF values (indicating proper probability distribution)
    2. Exactly one peak (indicating a single, coherent data distribution)
    
    The homogeneity check is crucial for ensuring reliable GDF calculations, as heterogeneous data
    can lead to inaccurate or misleading results in machine learning and statistical analysis.
    
    Attributes:
        egdf (EGDF): The input EGDF object containing data and computed PDF
        verbose (bool): Controls detailed output during homogeneity checking
        catch (bool): Controls whether results are stored in internal parameters dictionary
        params (dict): Dictionary storing homogeneity check results when catch=True
    
    Example:
        >>> from machinegnostics.magcal import EGDF
        >>> from machinegnostics.magcal DataHomogeneity
        >>> 
        >>> # Create EGDF object with your data
        >>> egdf = EGDF(data=your_data)
        >>> 
        >>> # Initialize homogeneity checker
        >>> homogeneity_checker = DataHomogeneity(egdf, verbose=True)
        >>> 
        >>> # Check if data is homogeneous
        >>> is_homogeneous = homogeneity_checker.test_homogeneity()
        >>> 
        >>> # Get detailed results
        >>> results = homogeneity_checker.get_homogeneity_params()
        >>> print(f"Homogeneous: {results['is_homogeneous']}")
        >>> print(f"Number of peaks: {results['num_peaks']}")
        >>> print(f"Has negative PDF: {results['has_negative_pdf']}")
    
    Methods:
        test_homogeneity(): Main public method to check data homogeneity
        get_homogeneity_params(): Returns dictionary of stored homogeneity parameters
    
    Notes:
        - Peak detection uses local maxima identification with a significance threshold
        - Only peaks > 1% of maximum PDF value are considered significant
        - Edge cases (arrays < 3 elements) are handled gracefully
        - Error handling ensures robustness against malformed input data
        
    Raises:
        AttributeError: If egdf doesn't have required PDF attribute
        Exception: Various exceptions caught during peak detection (logged if verbose=True)
    """
    def __init__(self, egdf: EGDF, verbose=True, catch=True):
        """
        Initialize the DataHomogeneity class.
        
        Parameters:
            egdf (EGDF): Initial EGDF object containing the data and computed PDF.
                            Must have a 'pdf' attribute containing the probability density function.
            verbose (bool, optional): If True, prints detailed information about the homogeneity 
                                    check process including warnings and results. Defaults to True.
            catch (bool, optional): If True, stores homogeneity check results in the internal 
                                  params dictionary for later retrieval. Defaults to True.
        
        Raises:
            TypeError: If egdf is not an EGDF object
        """
        self.egdf = egdf
        self.verbose = verbose
        self.catch = catch
        self.params = {}
    
    def test_homogeneity(self):
        """        
        This is the main entry point for checking if the EGDF data is homogeneous.
        It performs the complete homogeneity analysis including peak detection
        and negative value checking.
        
        Returns:
            bool: True if data is homogeneous (no negative PDF values and exactly 
                 one peak), False otherwise.
        
        Example:
            >>> import numpy as np
            >>> from machinegnostics.magcal import EGDF
            >>> from machinegnostics.magcal import DataHomogeneity
         
            >>> # Test homogeneous data (single peak, no negatives)
            >>> data = np.array([-13.5, 0,1,2,3,4,5,6,7,8,9,10])
            >>> egdf = EGDF(data)
            >>> checker = DataHomogeneity(egdf, verbose=False)
            >>> result = checker.test_homogeneity()
            >>> print(f"Is homogeneous: {result}")
            Is homogeneous: True
        
        Notes:
            - This method delegates to the internal _is_homogeneous() method
            - Results are automatically stored in params if catch=True
            - Verbose output is controlled by the verbose parameter
        """
        return self._is_homogeneous()
        

    def _is_homogeneous(self):
        """
        Internal method to check if the data is homogeneous.
        
        Performs the actual homogeneity analysis by checking:
        1. Whether PDF has any negative values
        2. Number of peaks in the PDF (should be exactly 1)
        
        Also handles verbose output and parameter storage based on instance settings.
        
        Returns:
            bool: True if homogeneous (no negative PDF values and exactly one peak), 
                 False otherwise.
        """
        num_peaks = self._get_peaks()
        has_negative_pdf = np.any(self.egdf.pdf < 0)

        is_homogeneous = not has_negative_pdf and num_peaks == 1

        if self.verbose:
            if not is_homogeneous:
                reasons = []
                if has_negative_pdf:
                    reasons.append("PDF has negative values")
                if num_peaks > 1:
                    reasons.append(f"multiple peaks [{num_peaks}] detected")
                print(f"Data is not homogeneous: {', '.join(reasons)}.")
            else:
                print("Data is homogeneous: PDF has no negative values and at most one peak detected.")

        if self.catch:
            self.params['is_homogeneous'] = is_homogeneous
            self.params['has_negative_pdf'] = has_negative_pdf
            self.params['num_peaks'] = num_peaks
            
        return is_homogeneous


    def _get_peaks(self):
        """
        Estimate number of significant peaks (local maxima) in the PDF.
        
        Identifies peaks using local maxima detection with a significance threshold.
        A point is considered a peak if it's greater than both neighbors and exceeds
        1% of the maximum PDF value.
        
        Returns:
            int: Number of significant peaks detected. Returns 0 if PDF is unavailable
                or an error occurs.
        """
        if not hasattr(self.egdf, 'pdf') or self.egdf.pdf is None:
            if self.verbose:
                print("Warning: PDF not available for peak detection")
            return 0
        
        try:
            pdf = self.egdf.pdf
            
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
                print(f"Detected {peak_count} peaks in PDF")
            
            return peak_count
            
        except Exception as e:
            if self.verbose:
                print(f"Error in peak detection: {e}")
            return 0
    