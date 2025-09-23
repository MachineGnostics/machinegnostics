'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

Author: Nirmal Parmar
'''

import numpy as np

class DataConversion:
    """
    A class to convert data between different data domains.
    converts data domains,
    - from additive to multiplicative,
    - from multiplicative to additive,
    - from finite normalized multiplicative to infinite interval,
    - from infinite interval to finite normalized multiplicative.
    
    Methods
    -------
    add_to_mult(data)
        Converts additive data to multiplicative domain.
    mult_to_add(data)
        Converts multiplicative data to additive domain.
    convert_data(data, to_multiplicative=True)
        Converts data between additive and multiplicative domains.
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
        # if lb >= ub:
        #     raise ZeroDivisionError("lb must be less than ub")
        
        a = np.asarray(a)
        z = np.exp((2 * a - ub - lb) / ((ub - lb) + eps))
        
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
            lb = np.min(z)
        if ub is None:
            ub = np.max(z)

        if not np.isscalar(lb) or not np.isscalar(ub):
            raise ValueError("lb and ub must be scalars")
        eps = 1e-6  # Small value to ensure strict inequality
        if lb >= ub:
            ub = ub + 1e-12
        
        z = np.asarray(z)
        a = (np.log(np.abs(z) + eps) * (ub - lb) + lb + ub) / 2

        if a.size == 1:
            return a.item()  # Return scalar if input was scalar
        return a
    
    @staticmethod
    def _convert_mz(m, lb=None, ub=None):
        """
        Converts multiplicative data into the finite normalized multiplicative form.

        Parameters:
        ----------
        m : scalar or numpy.ndarray
            Input multiplicative data.
        lb : float
            Lower bound (must be a scalar).
        ub : float
            Upper bound (must be a scalar).

        Returns:
        -------
        z : scalar or numpy.ndarray
            Data converted into finite normalized multiplicative form,
            same type as 'm'.

        Raises:
        ------
        ValueError:
            If lb or ub is not a scalar.
        """
        if lb is None:
            lb = np.min(m)
        if ub is None:
            ub = np.max(m)

        if not np.isscalar(lb) or not np.isscalar(ub):
            raise ValueError("lb and ub must be scalars")
        
        m = np.asarray(m)
        a = np.log(m / lb) * (2.0 / np.log(ub / lb)) - 1
        z = np.exp(a)
        
        if z.size == 1:
            return z.item()  # Return scalar if input was scalar
        return z
   
    @staticmethod
    def _convert_zm(z, lb=None, ub=None):
        """
        Converts normalized multiplicative data z back to the original multiplicative form.

        Parameters
        ----------
        z : scalar or numpy.ndarray
            Normalized multiplicative data.
        lb : float
            Lower bound (must be a scalar).
        ub : float
            Upper bound (must be a scalar).

        Returns
        -------
        m : scalar or numpy.ndarray
            Data converted back to multiplicative form, same type as 'z'.

        Raises
        ------
        ValueError:
            If lb or ub is not a scalar.
        """
        if lb is None:
            lb = np.min(z)
        if ub is None:
            ub = np.max(z)

        if not np.isscalar(lb) or not np.isscalar(ub):
            raise ValueError("lb and ub must be scalars")
        v = np.sqrt(ub / lb)
        z = np.asarray(z)
        m = lb * v * z ** np.log(v)
        if m.size == 1:
            return m.item()  # Return scalar if input was scalar
        return m
    
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
    
    @staticmethod
    def _convert_fininf(z_fin, lb=None, ub=None):
        """
        Converts data from the finite normalized multiplicative form into the infinite interval.

        Parameters:
        ----------
        z_fin : scalar or numpy.ndarray
            Input data in finite normalized multiplicative form.
        lb : float
            Lower bound (must be a scalar).
        ub : float
            Upper bound (must be a scalar).

        Returns:
        -------
        z_inf : scalar or numpy.ndarray
            Converted data in infinite interval form, same type as z_fin.

        Raises:
        ------
        ValueError:
            If lb or ub is not a scalar.
        """
        if lb is None:
            lb = np.min(z_fin)
        if ub is None:
            ub = np.max(z_fin)
        
        if not np.isscalar(lb) or not np.isscalar(ub):
            raise ValueError("lb and ub must be scalars")

        # Adjust the logic to ensure the result is strictly less than ub
        epsilon = 1e-6  # Small value to ensure strict inequality
        z_inf = (z_fin -lb) / (1 - (z_fin / ub) + epsilon)
        return z_inf

    @staticmethod
    def _convert_inffin(z_inf, lb=None, ub=None):
        """
        Converts data from the infinite interval into the finite normalized multiplicative form.

        Parameters:
        ----------
        z_inf : scalar or numpy.ndarray
            Input data in infinite interval form.
        lb : float
            Lower bound (must be a scalar).
        ub : float
            Upper bound (must be a scalar).

        Returns:
        -------
        z_fin : scalar or numpy.ndarray
            Data converted into finite normalized multiplicative form, 
            same type as z_inf.

        Raises:
        ------
        ValueError:
            If lb or ub is not a scalar.
        """
        if lb is None:
            lb = np.min(z_inf)
        if ub is None:
            ub = np.max(z_inf)

        if not np.isscalar(lb) or not np.isscalar(ub):
            raise ValueError("lb and ub must be scalars")
        
        z_inf = np.asarray(z_inf)
        z_fin = (z_inf + lb) / (1 + z_inf / ub)
        
        if z_fin.size == 1:
            return z_fin.item()  # Return scalar if input was scalar
        return z_fin
    

    @staticmethod
    def get_zi_bounds(data_form, DL, DU)-> tuple:
        """
        Get the lower and upper bounds of the infinite domain data and perform data conversion.

        Parameters:
        ----------
        data_form : str
            Specifies the data form ('a' for additive, 'm' for multiplicative).
        DLB : float
            Lower bound of the data.
        DUB : float
            Upper bound of the data.
        D : numpy.ndarray
            Input data to be converted.
        C : numpy.ndarray, optional
            Censoring data (default is None).
        B : numpy.ndarray, optional
            Boundary censoring data (default is None).
        ctype : int, optional
            Censoring type (default is 0).

        Returns:
        -------
        sample : dict
            Contains converted data and bounds.
        """

        # Validate bounds
        if not np.isscalar(DL) or not np.isscalar(DU):
            raise ValueError("DL and DU must be scalars.")

        if data_form == "a":  # Additive form
            LB = DL - (DU - DL) / 2
            UB = DU + (DU - DL) / 2

        elif data_form == "m":  # Multiplicative form
            LB = DL / np.sqrt(DU / DL)
            UB = DU * np.sqrt(DU / DL)

        else:
            raise ValueError("Invalid data_form. Use 'a' for additive or 'm' for multiplicative.")

        return LB, UB
    
    