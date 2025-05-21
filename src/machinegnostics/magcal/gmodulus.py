'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Description: Calculates Modulus of the sample data Mc, c={i,j}
'''
import numpy as np
from machinegnostics.magcal.sample_characteristics import GnosticCharacteristicsSample

def gmodulus(data:np.ndarray, case:str='i'):
        """
        Public interface to calculate the data sample's modulus.
        Calculate the modulus of the data sample using equation 14.8: M_Z,c = sqrt(F_c^2 - c^2*H_c^2)
        
        Parameters
        ----------
        case : str, default='i'
            The type of modulus to calculate:
            - 'i': Uses irrelevance Hi (estimation case)
            - 'j': Uses irrelevance Hj (quantification case)
            
        Returns
        -------
        float
            The calculated modulus value M_Z,c
            
        Notes
        -----
        This implementation follows Theorem 15 from the reference, which states that
        the modulus of a data sample can be calculated using the relation:
        M_Z,c = sqrt(F_c^2 - c^2*H_c^2)
        
        where:
        - F_c is the relevance function
        - H_c is the irrelevance function
        - c is the case parameter ('i' or 'j')  

        Examples
        --------
        >>> import numpy as np
        >>> from machinegnostics.magcal.gmodulus import gmodulus
        >>> data = np.random.rand(100)
        >>> gmodulus(data, case='i')
                    
        """
        gcs = GnosticCharacteristicsSample(data=data)
        try:
            return gcs._calculate_modulus(case=case)
        except Exception as e:
            raise RuntimeError(f"Error calculating sample modulus: {str(e)}")
