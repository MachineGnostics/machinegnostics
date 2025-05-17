'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
Date: 2025-10-01
Description: Machine Gnostics Robust Regression Machine Learning Model
This module implements a machine learning model for robust regression using the ManGo library.
This model is designed to handle various types of data and is particularly useful for applications in machine gnostics.
'''

from src.magcal import RegressorParamBase

class RobustRegressor(RegressorParamBase):
    """
    ## RobustRegressor: A Polynomial Regression Model Based on Machine Gnostics

    This class implements a robust regression model grounded in the principles of 
    Mathematical Gnostics — a non-statistical, deterministic framework for learning 
    from data. Unlike traditional statistical models that rely on probabilistic 
    assumptions, this approach uses algebraic and geometric structures to model 
    data while maintaining resilience to outliers, noise, and corrupted samples.

    The model fits a polynomial regression function to the input data, adjusting
    the influence of each data point through a gnostically-derived weighting scheme.
    It iteratively optimizes the regression coefficients using a custom criterion
    that minimizes a gnostic loss (e.g., `hi` or `hj`).

    Key Features
    ------------
    - Robust to outliers and heavy-tailed distributions
    - Polynomial feature expansion (up to configurable degree)
    - Gnostic-based iterative loss minimization
    - Custom weighting and scaling strategy
    - Early stopping and convergence control
    - Modular design for extensibility

    Parameters
    ----------
    degree : int, default=2
        Degree of the polynomial used to expand input features. A value of `2` fits
        a quadratic model; higher values increase model flexibility.

    max_iter : int, default=1000
        Maximum number of iterations for the training process.

    tol : float, default=1e-3
        Convergence threshold. Iteration stops if the change in loss or coefficients
        is below this tolerance for `early_stopping` consecutive iterations.

    mg_loss : str, default='hi'\
        Type of gnostic loss to use. Options:
            - `'hi'` : Estimation relevance loss
            - `'hj'` : Joint relevance loss
        Determines how residuals are transformed and weighted during training.

    early_stopping : int or bool, default=True
        Number of iterations over which to check for convergence. If set to `True`, 
        uses a default internal threshold (e.g., 10). If an integer, uses that value
        directly.

    verbose : bool, default=False
        If `True`, prints debug and progress messages during training.

    history : bool, default=False
        If `True`, stores the history of gnostic loss values across training iterations
        in `self._history`.
    
    params : bool, default=False,
        If 'True', store weights, coefficients, and gnostic loss in params

    Attributes
    ----------
    coefficients : ndarray of shape (degree + 1,)
        Final learned polynomial coefficients after training.

    weights : ndarray of shape (n_samples,)
        Final weights assigned to each sample based on gnostic transformations.

    _history : list of float
        List of gnostic loss values recorded at each iteration (if `history=True`).

    Methods
    -------
    fit(X, y)
        Fit the model to training data using polynomial expansion and gnostic loss minimization.

    predict(X)
        Predict output values for new input samples using the trained model.


    Example
    -------
    >>> from mango import RobustRegressor
    >>> model = RobustRegressor(degree=3, mg_loss='hi', verbose=True)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> print(model.coefficients)
    >>> print(model.weights)

    Resource:
    --------
    More information: https://machinegnostics.info/ 

    Github: https://github.com/MachineGnostics/ManGo
    """
    def __init__(self, 
                 degree = 2, 
                 max_iter = 100, 
                 tol = 1e-8, 
                 mg_loss = 'hi', 
                 early_stopping = True, 
                 verbose = False):
        super().__init__(degree, 
                         max_iter, 
                         tol, 
                         mg_loss, 
                         early_stopping, 
                         verbose)
        '''
        Robust Regressor - Machine Gnostics
        
        Initialize the regression model.
        
        Parameters:
        -----------
        degree : int
            Degree of polynomial features
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        mg_loss: str
            Select the gnostic loss
        early_stopping : int
            Number of iterations for early stopping check
        verbose: bool
            To print verbose
        '''
        pass

    def fit(self, X, y):
        '''
        Fit the Robust Regressor model using gnostic weights and polynomial features.

        This method trains the regression model on the given dataset `(X, y)` by iteratively applying
        weighted least squares. The weights are updated at each iteration using a mathematical
        gnostics-based approach that makes the model robust to noise and outliers.

        The process involves:
        - Expanding the input features into a polynomial basis.
        - Solving the weighted least squares problem to estimate model coefficients.
        - Computing residuals and transforming them into a gnostic space (via z-values).
        - Computing gnostic weights to adjust the influence of each data point.
        - Updating and normalizing the weights.
        - Evaluating convergence based on changes in loss and coefficient values.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Input feature values. Can be a 1D array or a single-column 2D array.
        
        y : array-like of shape (n_samples,)
            Target values corresponding to input samples.

        Attributes Updated
        ------------------
        self.coefficients : ndarray of shape (degree + 1,)
            Estimated polynomial regression coefficients after fitting.

        self.weights : ndarray of shape (n_samples,)
            Final sample weights after convergence.

        self.loss_history : list of float
            Gnostic criterion values computed at each iteration (if `history=True`).

        Notes
        -----
        - The method stops iterating when either:
            (1) The change in the recent loss values is below the specified tolerance `tol`, or
            (2) The change in coefficient values is below `tol`.
        - The convergence check uses the last `early_stopping` iterations.
        - The gnostic weighting and loss computations depend on the choice of `mg_loss`
        (e.g., `'hi'` or `'hj'`), which influences the robustness behavior.
        '''
        return self._fit(X, y)
    
                
    def predict(self, X):
        """
        Predict target values using the trained Robust Regressor model.

        This method applies the learned polynomial regression model to new input data
        and returns predicted values. It assumes that the `fit` method has already been
        called to estimate the model coefficients.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Input feature values for which predictions are to be made. Can be a 1D array
            or a single-column 2D array.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values corresponding to the input samples.

        Raises
        ------
        ValueError
            If the model has not been fitted or if the input shape is incompatible.

        Notes
        -----
        - This method expands the input features into the same polynomial basis as used during training.
        - Ensure `fit` has been called before using `predict`, otherwise `self.coefficients` will be `None`.
        - Input `X` will be converted to a NumPy array if it isn't already.
        """
        return self._predict(X)
        