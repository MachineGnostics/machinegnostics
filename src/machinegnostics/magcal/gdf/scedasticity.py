'''
Gnostic - Homoscedasticity and Heteroscedasticity

This module to check for homoscedasticity and heteroscedasticity in data.

Author: Nirmal Parmar
Machine Gnostics
'''
import numpy as np
import logging
from machinegnostics.magcal.util.logging import get_logger

class DataScedasticity:
    """
    Gnostic Scedasticity Test for Homoscedasticity and Heteroscedasticity

    This class provides a method to check for homoscedasticity and heteroscedasticity in data,
    inspired by fundamental principles rather than standard statistical tests. Unlike classical
    approaches, this implementation uses gnostic variance and gnostic linear regression, which are
    based on the Machine Gnostics framework.

    Key Differences from Standard Methods:
    - **Variance Calculation:** The variance used here is the gnostic variance, which may differ in
      definition and properties from classical statistical variance. It is designed to capture
      uncertainty and spread in a way that aligns with gnostic principles.
    - **Regression Model:** The linear regression model employed is a gnostic linear regression,
      not the standard least squares regression. This model is tailored to the gnostic approach and
      may use different loss functions, optimization criteria, or regularization.
    - **Test Philosophy:** This is not a formal statistical test (such as Breusch-Pagan or White's test),
      but rather a diagnostic inspired by the fundamentals of the gnostic framework. The method splits
      residuals based on the median of the independent variable and compares the gnostic variances of
      the squared residuals in each half.

    Usage:
        1. Initialize the class with desired gnostic regression parameters.
        2. Call `fit(x, y)` with your data.
        3. Check the `is_homoscedastic` attribute or returned value to determine if the data is
           homoscedastic (equal gnostic variance across splits) or heteroscedastic.

    Attributes:
        x (np.ndarray): Independent variable data.
        y (np.ndarray): Dependent variable data.
        model (LinearRegressor): Gnostic linear regression model.
        residuals (np.ndarray): Residuals from the fitted model.
        params (dict): Stores calculated variances and variance ratio.
        variance_ratio (float): Ratio of gnostic variances between data splits.
        is_homoscedastic (bool): True if data is homoscedastic under gnostic test, else False.

    Example:
        >>> import numpy as np
        >>> from machinegnostics.magcal import DataScedasticity
        >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> y = np.array([2.1, 4.2, 6.1, 8.3, 10.2, 12.1, 14.2, 16.1, 18.2, 20.1])
        >>> sced = DataScedasticity()
        >>> is_homo = sced.fit(x, y)
        >>> print(f"Is data homoscedastic? {is_homo}")
        >>> print(f"Variance ratio: {sced.variance_ratio}")

    Note:
        This class is intended for users interested in gnostic data analysis. Results and interpretations
        may not align with classical statistical methods. For more details on gnostic variance and regression,
        refer to the Machine Gnostics documentation.
    """

    def __init__(self,
                 scale: str | int | float = 'auto',
                 max_iter: int = 100,
                 tol: float = 0.001,
                 mg_loss: str = 'hi',
                 early_stopping: bool = True,
                 verbose: bool = False,
                 data_form: str = 'a',
                 gnostic_characteristics: bool = True,
                 history: bool = True):
        
        from machinegnostics.models.regression import LinearRegressor
        self.x = None
        self.y = None
        self.model = LinearRegressor(scale=scale,
                                     max_iter=max_iter,
                                     tol=tol,
                                     mg_loss=mg_loss,
                                     early_stopping=early_stopping,
                                     verbose=verbose,
                                     data_form=data_form,
                                     gnostic_characteristics=gnostic_characteristics,
                                     history=history)
        self.residuals = None
        self.params = {}
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.debug(f"{self.__class__.__name__} initialized:")


    def _split_residuals(self):
        """
        Split residuals into two halves based on the median of x. zip x and residuals.
        sorted(zip(x, residuals))
        """
        self.logger.info("Splitting residuals based on median of x.")
        median_x = np.median(self.x)
        left_half = [(xi, ri) for xi, ri in zip(self.x, self.residuals) if xi <= median_x]
        right_half = [(xi, ri) for xi, ri in zip(self.x, self.residuals) if xi > median_x]
        return left_half, right_half
    
    def _variance_ratio(self):
        """
        Calculate the variance ratio of the squared residuals in the two halves.

        Returns:
            float: Variance ratio of the squared residuals.
        """
        from machinegnostics import variance
        self.logger.info("Calculating variance ratio.")
        left_half, right_half = self._split_residuals()
        left_residuals = np.array([ri for xi, ri in left_half])
        right_residuals = np.array([ri for xi, ri in right_half])
        var_left = variance(left_residuals ** 2)
        var_right = variance(right_residuals ** 2)

        self.logger.debug(f"Left variance: {var_left}, Right variance: {var_right}")
        # cap values between [1, 1e-9]
        var_left = float(var_left)
        var_right = float(np.maximum(var_right, 1e-9)) # to avoid division by zero
        if var_right == 0 and var_left == 0:
            variance_ratio = 1.0
        elif var_right == 0:
            variance_ratio = np.inf
        else:
            variance_ratio = var_left / var_right

        # params
        self.logger.info(f"Variance ratio calculated: {variance_ratio}")
        self.params['var_left'] = var_left
        self.params['var_right'] = var_right
        self.params['variance_ratio'] = variance_ratio
        return variance_ratio
        
    
    def _is_homoscedastic(self, threshold: float = 0.001):
        """
        Check if the data is homoscedastic based on the variance ratio.

        Args:
            threshold (float): Threshold to determine homoscedasticity.

        Returns:
            bool: True if homoscedastic, False if heteroscedastic.
        """
        if self.variance_ratio is None:
            self.logger.error("Variance ratio not calculated. Please run fit() first.")
            raise ValueError("Variance ratio not calculated. Please run fit() first.")
        return abs(self.variance_ratio - 1) < threshold

    def fit(self, x: np.ndarray, y: np.ndarray) -> bool:
        """
        Fit the gnostic linear regression model to the data and assess scedasticity.

        This method fits the gnostic linear regression model to the provided data, computes the residuals,
        and evaluates homoscedasticity or heteroscedasticity using the gnostic variance approach. Unlike
        standard statistical tests, this method uses gnostic variance and gnostic regression, which are
        based on the Machine Gnostics framework and may yield different results from classical methods.

        The method splits the data based on the median of the independent variable, calculates the gnostic
        variance of squared residuals in each half, and determines if the data is homoscedastic (equal
        gnostic variance) or heteroscedastic.

        Args:
            x (np.ndarray): Independent variable data.
            y (np.ndarray): Dependent variable data.

        Returns:
            bool: True if data is homoscedastic under the gnostic test, False if heteroscedastic.

        Note:
            This is not a standard statistical test. For details on the gnostic approach, see the
            Machine Gnostics documentation.
        """
        self.logger.info("Fitting DataScedasticity model...")
        self.x = x
        self.y = y

        self.logger.info("Fitting gnostic regression model.")
        self.model.fit(x, y)
        self.logger.debug(f"Model calculations complete.")

        self.logger.info("Calculating residuals.")  
        self.residuals = y - self.model.predict(x)

        # calculate variance ratio
        self.logger.info("Calculating variance ratio.")
        self.variance_ratio = self._variance_ratio()

        # check
        self.logger.info("Checking homoscedasticity.")
        self.is_homoscedastic = self._is_homoscedastic()
        self.logger.info(f"Homoscedasticity check result - is_homoscedastic: {self.is_homoscedastic}")
        return self.is_homoscedastic
