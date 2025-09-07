"""
# Base class for distribution functions in Machine Gnostics
# This module defines the abstract base class for distribution functions used in the Machine Gnostics framework

Author: Nirmal Parmar
Machine Gnostics
"""

from abc import ABC, abstractmethod

class BaseDistFunc(ABC):
    """
    Abstract base class for distribution functions.
    """

    @abstractmethod
    def fit(self, data):
        """
        Fit MG distribution function to the data.

        Parameters:
        X (array-like): Input features.
        y (array-like): Target values.
        """
        pass

    @abstractmethod
    def plot(self):
        """
        Plot the distribution function.
        """
        pass

    # @abstractmethod
    # def results(self):
    #     """
    #     Return the results of the fitted distribution function.
    #     """
    #     pass