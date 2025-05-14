from abc import ABCMeta, abstractmethod

# regression base class
class RegressionBase(metaclass=ABCMeta):
    """
    Abstract base class for regression models.
    """

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