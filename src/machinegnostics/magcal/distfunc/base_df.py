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