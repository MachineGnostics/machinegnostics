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
            raise ZeroDivisionError("lb must be less than ub")
        
        z = np.asarray(z)
        a = (np.log(z) * (ub - lb) + lb + ub) / 2 - eps
        
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
    
    # def _data_transform_input(self)-> np.ndarray:
    #     """
    #     Transform the input data based on the specified data form.

    #     first from normal domain to standard domain, and then from finite to infinite domain.
    #     """
    #     if self.data_form == 'a':
    #         self.z = self._convert_az(self.data, self.lb, self.ub)
    #     elif self.data_form == 'm':
    #         self.z = self._convert_mz(self.data, self.lb, self.ub)
    #     elif self.data_form is None:
    #         self.z = self.data
    #     else:
    #         raise ValueError("Invalid data form specified. Use 'a', 'm', or None.")
        
    #     # bound checking for infinite domain
    #     if self.ilb is None:
    #         self.ilb = np.min(self.z) if self.z.size > 0 else 0
    #     if self.iub is None:
    #         self.iub = np.max(self.z) if self.z.size > 0 else 1
        
    #     # Convert to infinite domain
    #     if self.data_form == 'a' or self.data_form == 'm':
    #         self.zi = self._convert_fininf(self.z, self.ilb, self.iub)

    #     return self.zi

    # def _data_transform_output(self, data):
    #     """
    #     Transform the output data back to the original domain based on the specified data form.
    #     """
    #     if self.data_form == 'a' or self.data_form == 'm':
    #         # Convert to infinite domain
    #         self.zf = self._convert_fininf(data, self.ilb, self.iub)
    #     else:
    #         self.zf = data

    #     # convert to original domain
    #     if self.data_form == 'a':
    #         self.z = self._convert_za(self.zf, self.lb, self.ub)
    #     elif self.data_form == 'm':
    #         self.z = self._convert_zm(self.zf, self.lb, self.ub)
    #     elif self.data_form is None:
    #         self.z = self.zf

    #     return self.z
    