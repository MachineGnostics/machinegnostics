'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.
'''

import numpy as np

class DataConversion:
    """
    A class to convert data between different formats.
    
    converts data from additive to multiplicative and vice versa.
    
    Methods
    -------
    add_to_mult(data)
        Converts additive data to multiplicative format.
    mult_to_add(data)
        Converts multiplicative data to additive format.
    convert_data(data, to_multiplicative=True)
        Converts data between additive and multiplicative formats.
    get_bounds(data)
        Gets the lower and upper bounds of the data.

    """
    
    @staticmethod
    def _convert_az(a, lb=None, ub=None):
        """
        Converts additive data into the finite normalized multiplicative form.

        Parameters:
        ----------
        a : scalar or numpy.ndarray
            Input additive data.
        lb : float
            Lower bound (must be a scalar).
        ub : float
            Upper bound (must be a scalar).

        Returns:
        -------
        z : scalar or numpy.ndarray
            Data converted into finite normalized multiplicative form, 
            same type as 'a'.

        Raises:
        ------
        ValueError:
            If lb or ub is not a scalar.
        """
        if lb is None:
            lb = np.min(a)
        if ub is None:
            ub = np.max(a)

        if not np.isscalar(lb) or not np.isscalar(ub):
            raise ValueError("lb and ub must be scalars")
        eps = 1e-6  # Small value to ensure strict inequality
        if lb >= ub:
            raise ZeroDivisionError("lb must be less than ub")
        
        a = np.asarray(a)
        z = np.exp((2 * a - ub - lb) / (ub - lb) + eps)
        
        if z.size == 1:
            return z.item()  # Return scalar if input was scalar
        return z
    
    @staticmethod
    def _convert_za(z, lb=None, ub=None):
        """
        Converts multiplicative data into the finite normalized additive form.

        Parameters:
        ----------
        z : scalar or numpy.ndarray
            Input multiplicative data.
        lb : float
            Lower bound (must be a scalar).
        ub : float
            Upper bound (must be a scalar).

        Returns:
        -------
        a : scalar or numpy.ndarray
            Data converted into finite normalized additive form, 
            same type as 'z'.

        Raises:
        ------
        ValueError:
            If lb or ub is not a scalar.
        """
        if lb is None:
            lb = np.min(a)
        if ub is None:
            ub = np.max(a)

        if not np.isscalar(lb) or not np.isscalar(ub):
            raise ValueError("lb and ub must be scalars")
        eps = 1e-6  # Small value to ensure strict inequality
        if lb >= ub:
            raise ZeroDivisionError("lb must be less than ub")
        
        z = np.asarray(z)
        a = (np.log(z) * (ub - lb) + lb + ub) / 2 - eps
        
        if a.size == 1:
            return a.item()  # Return scalar if input was scalar
        return a
    
    @staticmethod
    def convert_data(data,to_multiplicative=True):
        """
        Converts data between additive and multiplicative forms.

        Parameters:
        ----------
        data : scalar or numpy.ndarray
            Input data to be converted.
        lb : float
            Lower bound (must be a scalar).
        ub : float
            Upper bound (must be a scalar).
        to_multiplicative : bool
            If True, convert from additive to multiplicative. 
            If False, convert from multiplicative to additive.

        Returns:
        -------
        converted_data : scalar or numpy.ndarray
            Converted data in the desired format.
        """
        # if data is not a numpy array, convert it
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        # Check if data is empty
        if data.size == 0:
            raise ValueError("Input data is empty")
        # Check if data is 1D or 2D
        if data.ndim > 2:
            raise ValueError("Input data must be 1D or 2D")
        # bounds
        lb, ub = DataConversion.get_bounds(data)
        
        if to_multiplicative:
            return DataConversion._convert_az(data, lb, ub)
        else:
            return DataConversion._convert_za(data, lb, ub)
        
    @staticmethod
    def get_bounds(data):
        """
        Get the lower and upper bounds of the data.

        Parameters:
        ----------
        data : scalar or numpy.ndarray
            Input data to get bounds for.

        Returns:
        -------
        lb : float
            Lower bound.
        ub : float
            Upper bound.
        """
        data = np.asarray(data)
        lb = np.min(data)
        ub = np.max(data)
        return lb, ub
    