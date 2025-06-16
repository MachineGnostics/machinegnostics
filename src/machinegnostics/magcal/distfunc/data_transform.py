import numpy as np
from machinegnostics.magcal.data_conversion import DataConversion

class DataDomainTransformation(DataConversion):
    """
    A class to handle transformations between different data domains and forms
    with improved numerical stability for round-trip transformations.
    """
    
    def __init__(self, data_form=None, lb=None, ub=None, ilb=None, iub=None, epsilon=np.finfo(float).eps):
        """
        Initialize the data domain transformer with improved stability parameters.
        
        Parameters
        ----------
        data_form : str or None, optional
            The form of data: 'a' for additive, 'm' for multiplicative, None for no transformation
        lb : float, optional
            Lower bound for original data
        ub : float, optional
            Upper bound for original data
        ilb : float, optional
            Lower bound for intermediate domain
        iub : float, optional
            Upper bound for intermediate domain
        epsilon : float, optional
            Small value to prevent numerical issues
        """
        self.data_form = data_form
        self.lb = lb
        self.ub = ub
        self.ilb = ilb
        self.iub = iub
        self.epsilon = epsilon
        
        # Intermediate data representations
        self.data = None
        self.z = None
        self.zi = None
        self.zf = None
        
        # Store original data for reference
        self.original_data = None
        
        # Transformation parameters
        self.margin = 0.05  # Margin to avoid boundary effects

    def transform_input(self, data):
        """
        Transform input data from original domain to working domain
        with improved stability and precision.
        
        Uses parent class conversion functions for consistent transformation.
        """
        # Store original data as high precision float
        self.original_data = np.asarray(data, dtype=np.float64)
        self.data = self.original_data.copy()
        
        # Determine bounds if not specified
        if self.lb is None:
            self.lb = np.min(self.data)
        if self.ub is None:
            self.ub = np.max(self.data)
        
        # Calculate adaptive margin for better precision
        range_value = self.ub - self.lb
        # Smaller margin for smaller ranges
        self.margin = min(0.05, max(0.001, 0.01 * np.log10(1 + range_value)))
        
        # Use parent class methods for domain conversion
        if self.data_form == 'a':
            self.z = self._convert_az(self.data, self.lb, self.ub)
        elif self.data_form == 'm':
            self.z = self._convert_mz(self.data, self.lb, self.ub)
        elif self.data_form is None:
            self.z = self.data
        else:
            raise ValueError("Invalid data form specified. Use 'a', 'm', or None.")
        
        # Store original normalized values for comparison
        self.original_z = self.z.copy()
        
        # Set intermediate bounds if not specified
        if self.ilb is None:
            self.ilb = self.z.min()
        if self.iub is None:
            self.iub = self.z.max()
        
        # Convert to infinite domain using parent class method
        if self.data_form == 'a' or self.data_form == 'm':
            self.zi = self._convert_fininf(self.z, self.ilb, self.iub)
        else:
            self.zi = self.z
        
        # Store original working domain values
        self.original_zi = self.zi.copy()
        
        return self.zi
    
    def transform_output(self, data, preserve_original=True):
        """
        Transform output data from working domain back to original domain
        with improved precision and stability.
        
        Parameters
        ----------
        data : array-like
            Data in working domain to transform back
        preserve_original : bool, optional
            If True, for unchanged values attempt to preserve original values
            
        Returns
        -------
        ndarray
            Transformed data in original domain
        """
        # Ensure high precision
        data = np.asarray(data, dtype=np.float64)
                
        # Convert from infinite domain to finite domain using parent class method
        if self.data_form == 'a' or self.data_form == 'm':
            self.zf = self._convert_inffin(data, self.ilb, self.iub)
        else:
            self.zf = data
        
        # Convert back to original domain using parent class methods
        if self.data_form == 'a':
            self.z = self._convert_za(self.zf, self.lb, self.ub)
        elif self.data_form == 'm':
            self.z = self._convert_zm(self.zf, self.lb, self.ub)
        elif self.data_form is None:
            self.z = self.zf
        
        return self.z
    
    def auto_set_bounds(self, data):
        """
        Automatically set bounds based on the provided data.
        """
        data = np.asarray(data)
        if data.size > 0:
            self.lb = np.min(data)
            self.ub = np.max(data)
        return self