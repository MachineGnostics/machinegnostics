'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar

Description:
This module implements a robust linear regression model using mathematical gnostics principles.

'''

import numpy as np
from machinegnostics.models.regression.layer_io_process_rob_rig import DataProcessRobustRegressor
from machinegnostics.metrics import robr2
from machinegnostics.magcal import disable_parent_docstring

class LinearRegressor(DataProcessRobustRegressor):
    """
    Robust Linear Regression using Mathematical Gnostics principles.

    This regressor fits a linear model to data using robust, gnostic loss functions
    and adaptive sample weights. It is designed to be resilient to outliers and non-Gaussian noise,
    making it suitable for scientific and engineering applications where data quality may vary.

    Key Features
    ------------
    - Robust to outliers: Uses gnostic loss functions and adaptive sample weights.
    - Iterative optimization: Supports early stopping and convergence tolerance.
    - Tracks detailed history: Optionally records loss, weights, entropy, and gnostic characteristics at each iteration.
    - Compatible with numpy arrays for input/output.

    Parameters
    ----------
    scale : {'auto', int, float}, default='auto'
        Scaling method or value for input features.
    max_iter : int, default=100
        Maximum number of optimization iterations.
    tol : float, default=1e-8
        Tolerance for convergence.
    mg_loss : str, default='hi'
        Loss function to use ('hi', 'fi', etc.).
    early_stopping : bool, default=True
        Whether to stop early if convergence is detected.
    verbose : bool, default=False
        If True, prints progress and diagnostics during fitting.
    data_form : str, default='a'
        Internal data representation format.
    gnostic_characteristics : bool, default=True
        If True, computes and records gnostic properties (fi, hi, etc.).
    history : bool, default=True
        If True, records the optimization history for analysis.

    Attributes
    ----------
    coefficients : np.ndarray
        Fitted linear coefficients.
    weights : np.ndarray
        Final sample weights after robust fitting.
    params : list of dict
        List of parameter snapshots (loss, weights, gnostic properties) at each iteration.
    _history : list
        Internal optimization history (if enabled).
    All configuration parameters as set at initialization.

    Methods
    -------
    fit(X, y)
        Fit the linear regressor to input features X and targets y.
    predict(X)
        Predict target values for new input features X.
    score(X, y, case='i')
        Compute the robust R2 score for input features X and true targets y.

    Example
    -------
    >>> from machinegnostics.models.regression import LinearRegressor
    >>> model = LinearRegressor(max_iter=100, verbose=True)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> r2 = model.score(X_test, y_test)

    Notes
    -----
    - This model is part of the Machine Gnostics library, which implements advanced machine learning techniques
      based on mathematical gnostics principles.
    - For more information, visit: https://machinegnostics.info/
    """
    @disable_parent_docstring
    def __init__(
        self, 
        scale: str | int | float = 'auto',
        max_iter: int = 100,
        tol: float = 1e-3,
        mg_loss: str = 'hi',
        early_stopping: bool = True,
        verbose: bool = False,
        data_form: str = 'a',
        gnostic_characteristics: bool = True,
        history: bool = True
    ):
        """
        Initialize a LinearRegressor instance with robust, gnostic regression settings.

        Parameters
        ----------
        scale : {'auto', int, float}, default='auto'
            Scaling method or value for input features.
        max_iter : int, default=100
            Maximum number of optimization iterations.
        tol : float, default=1e-8
            Tolerance for convergence.
        mg_loss : str, default='hi'
            Loss function to use ('hi', 'fi', etc.).
        early_stopping : bool, default=True
            Whether to stop early if convergence is detected.
        verbose : bool, default=False
            If True, prints progress and diagnostics during fitting.
        data_form : str, default='a'
            Internal data representation format.
        gnostic_characteristics : bool, default=True
            If True, computes and records gnostic properties (fi, hi, etc.).
        history : bool, default=True
            If True, records the optimization history for analysis.

        Notes
        -----
        All configuration parameters are stored as attributes for later reference.
        """
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            mg_loss=mg_loss,
            early_stopping=early_stopping,
            verbose=verbose,
            scale=scale,
            data_form=data_form,
            gnostic_characteristics=gnostic_characteristics,
            history=history
        )
        # # Optionally, set self.degree here as well for safety:
        self.degree = 1  # Default to linear regression
        self.max_iter = max_iter
        self.tol = tol
        self.mg_loss = mg_loss
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.scale = scale
        self.data_form = data_form
        self.gnostic_characteristics = gnostic_characteristics
        self._record_history = history
        self.params = []
        # history option
        if history:
            self._history = []
        else:
            self._history = None
        
        # logger
        self.logger.info("LinearRegressor initialized.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the robust linear regressor model to the provided data.

        This method performs robust linear regression using the specified gnostic loss function,
        iteratively optimizing the model coefficients and sample weights to minimize the influence of outliers.
        If history tracking is enabled, it records loss, weights, and gnostic properties at each iteration.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        self : LinearRegressor
            Returns the fitted model instance for chaining or further use.

        Notes
        -----
        - After fitting, the model's coefficients and sample weights are available in the `coefficients` and `weights` attributes.
        - If `history=True`, the optimization history is available in the `params` and `_history` attributes.

        Example
        -------
        >>> model = LinearRegressor(max_iter=100, verbose=True)
        >>> model.fit(X_train, y_train)
        >>> print(model.coefficients)
        >>> print(model.weights)
        """
        self.logger.info("Starting fit process for LinearRegressor.")
        # Call the fit method from DataProcessRobustRegressor
        super()._fit(X, y)
    
    def predict(self, model_input: np.ndarray) -> np.ndarray:
        """
        Predict target values using the fitted linear regressor model.

        Parameters
        ----------
        model_input : np.ndarray
            Input features for prediction, shape (n_samples, n_features).

        Returns
        -------
        y_pred : np.ndarray
            Predicted target values, shape (n_samples,).

        Example
        -------
        >>> model = LinearRegressor(max_iter=100, verbose=True)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
        """
        self.logger.info("Making predictions with LinearRegressor.")
        # Call the predict method from DataProcessRobustRegressor
        return super()._predict(model_input)
    
    def score(self, X: np.ndarray, y: np.ndarray, case:str = 'i') -> float:
        """
        Compute the robust (gnostic) R2 score for the linear regressor model.

        Parameters
        ----------
        X : np.ndarray
            Input features for scoring, shape (n_samples, n_features).
        y : np.ndarray
            True target values, shape (n_samples,).
        case : str, default='i'
            Specifies the case or variant of the R2 score to compute.

        Returns
        -------
        score : float
            Robust R2 score of the model on the provided data.
        
        Example
        -------
        >>> model = LinearRegressor(max_iter=100, verbose=True)
        >>> model.fit(X_train, y_train)
        >>> r2 = model.score(X_test, y_test)
        >>> print(f"Robust R2 score: {r2}")
        """
        self.logger.info("Calculating robust R2 score with LinearRegressor.")
        # prediction
        y_pred = self.predict(X)
        # Call the score method from DataProcessRobustRegressor
        r2 = robr2(y, y_pred, w=self.weights)
        return r2