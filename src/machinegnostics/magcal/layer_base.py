from abc import ABCMeta, abstractmethod
import logging
from machinegnostics.magcal.util.logging import get_logger


# regression base class
class ModelBase(metaclass=ABCMeta):
    """
    Abstract base class for regression models.

    Abstract Methods:
    ----------------

    - fit(X, y)

    - predict(X)
    """
    def __init__(self):
        # logger
        self.logger = get_logger(self.__class__.__name__, logging.INFO)  # Create a logger for this class
        self.logger.info(f"ModelBase initialized.")

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the regression model to the data.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict using the fitted model.
        """
        pass

    # @abstractmethod
    # def score(self, X, y):
    #     """
    #     Compute the score of the model.
    #     """
    #     pass