'''
Machine Gnostics - Machine Gnostics Library
Copyright (C) 2025  Machine Gnostics Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
Date: 2025-01-25

Description:
This module implements a multiclass classification model using mathematical gnostics principles.
'''

import numpy as np
from machinegnostics.models.classification.base_multiclass_classifier_history import HistoryMulticlassClassifierBase
from machinegnostics.models.base_io_models import DataProcessLayerBase
from machinegnostics.metrics import accuracy_score
from machinegnostics.magcal import disable_parent_docstring
from typing import Union
from machinegnostics.magcal.util.narwhals_df import narwhalify

class MulticlassClassifier(HistoryMulticlassClassifierBase, DataProcessLayerBase):
    """
    MulticlassClassifier implements a multiclass classification model based on Mathematical Gnostics principles.

    This class provides a feature-rich multiclass classification implementation using the Machine Gnostic framework.
    It supports polynomial feature expansion, softmax activation, gnostic-based weight estimation for handling
    outliers, early stopping, and detailed training history tracking.

    Key Features:
        - Multiclass classification using softmax activation for probability estimation.
        - Polynomial feature expansion up to a user-specified degree.
        - Gnostic weights for robust handling of outliers and improving model stability.
        - Calculation of gnostic characteristics for advanced model diagnostics.
        - Early stopping based on convergence of cross-entropy loss or residual entropy.
        - Verbose logging for monitoring training progress.
        - Optional scaling and data processing modes.
        - Maintains a history of model parameters and losses for analysis.

    Parameters
    ----------
    degree : int, default=1
        Degree of polynomial features to use for input expansion.
    max_iter : int, default=100
        Maximum number of iterations for the optimization algorithm.
    tolerance : float, default=1e-1
        Tolerance for convergence. Training stops if the change in loss or entropy is below this value.
    early_stopping : bool, default=True
        Whether to stop training early if convergence is detected.
    verbose : bool, default=False
        If True, prints detailed logs during training.
    scale : str | int | float, default='auto'
        Scaling method for gnostic weight calculations. Can be 'auto' or a numeric value.
    data_form : str, default='a'
        Data processing form: 'a' for additive, 'm' for multiplicative.
    gnostic_characteristics : bool, default=False
        If True, calculates and stores gnostic characteristics during training.
    history : bool, default=True
        If True, maintains a history of model parameters and losses.

    Attributes
    ----------
    coefficients : np.ndarray
        Fitted model coefficients after training, shape (n_features, n_classes).
    weights : np.ndarray
        Sample weights used during training.
    num_classes : int
        Number of unique classes in the training data.
    cross_entropy_loss : float
        Cross-entropy loss computed during training.
    _history : list
        List of dictionaries containing training history (loss, coefficients, entropy, etc.).
    params : list
        List of model parameters (for compatibility and inspection).

    Methods
    -------
    fit(X, y)
        Fit the multiclass classifier to the data.
    predict(model_input)
        Predict class labels for new data.
    predict_proba(model_input)
        Predict class probabilities for new data.
    score(X, y)
        Compute the accuracy score of the model on given data.

    Examples
    --------
    >>> from machinegnostics.models import MulticlassClassifier
    >>> model = MulticlassClassifier(degree=2, max_iter=200, verbose=True)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> print("Accuracy:", model.score(X_test, y_test))

    Notes
    -----
    - The model automatically detects the number of classes from the training data.
    - Uses softmax activation for multiclass probability estimation.
    - Gnostic weights help handle outliers and improve robustness of the classifier.
    - More information on gnostic characteristics can be found in the Machine Gnostics documentation.
    - For more information, visit: https://machinegnostics.info/
    """
    
    @disable_parent_docstring
    def __init__(self,
                 degree: int = 1,
                 max_iter: int = 100,
                 tolerance: float = 1e-1,
                 early_stopping: bool = True,
                 verbose: bool = False,
                 scale: 'str | int | float' = 'auto',
                 data_form: str = 'a',
                 gnostic_characteristics: bool = False,
                 history: bool = True):
        """
        Initialize the MulticlassClassifier with specified parameters.

        Parameters:
            - degree: Degree of polynomial features.
            - max_iter: Maximum number of iterations for convergence.
            - tolerance: Tolerance for stopping criteria.
            - early_stopping: Whether to stop training early if convergence is reached.
            - verbose: Whether to print detailed logs during training.
            - scale: Scaling method for gnostic weight calculations.
            - data_form: Form of data processing ('a' for additive, 'm' for multiplicative).
            - gnostic_characteristics: Whether to calculate gnostic characteristics.
            - history: Whether to maintain a history of model parameters and losses.
        
        """
        super().__init__(
            degree=degree,
            max_iter=max_iter,
            tolerance=tolerance,
            early_stopping=early_stopping,
            verbose=verbose,
            scale=scale,
            data_form=data_form,
            gnostic_characteristics=gnostic_characteristics,
            history=history
        )
    
    @narwhalify
    def fit(self, X, y):
        """
        Fit the MulticlassClassifier model to the training data.

        This method trains the multiclass classifier using the provided input features and target labels.
        It supports polynomial feature expansion, softmax activation for probability estimation, gnostic
        weight calculation for robust handling of outliers, and early stopping based on convergence criteria.
        Training history, including loss and coefficients, is stored if enabled.

        Parameters
        ----------
        X : array-like or dataframe
            Input features for training. Accepts NumPy arrays, Pandas DataFrame, or other Narwhals-supported types.
        y : array-like or series
            Target labels for training. Accepts NumPy arrays, Pandas Series/DataFrame column.

        Returns
        -------
        self : MulticlassClassifier
            Returns the fitted model instance for chaining.
        
        Raises
        ------
        ValueError
            If input shapes are incompatible or training fails due to numerical issues.

        Examples
        --------
        >>> model = MulticlassClassifier(degree=2, max_iter=200)
        >>> model.fit(X_train, y_train)
        """
        self.logger.info("Starting fit process for MulticlassClassifier.")

        # check
        Xcheck, ycheck = super()._fit_io(X, y)
        super()._fit(Xcheck, ycheck)
        
        self.coefficients = self.coefficients
        self.weights = self.weights
        return self
    
    @narwhalify
    def predict(self, model_input) -> np.ndarray:
        """
        Predict class labels for new input data.

        This method predicts class labels for the provided input data using the trained model.
        It supports input as NumPy arrays, pandas DataFrames, or PySpark DataFrames (if supported by the parent class).
        The prediction is based on the class with the highest probability from the softmax output.

        Parameters
        ----------
        model_input : array-like or dataframe
            Input data for prediction. Accepts NumPy arrays, Pandas DataFrame, or other Narwhals-supported types.

        Returns
        -------
        array-like
            Predicted class labels (integers). Returns native type (NumPy array or Pandas Series) based on input.

        Examples
        --------
        >>> y_pred = model.predict(X_test)
        """
        self.logger.info("Making predictions with MulticlassClassifier.")
        # check
        model_input_checked = super()._predict_io(model_input)
        return super()._predict(model_input_checked)
    
    @narwhalify
    def predict_proba(self, model_input) -> np.ndarray:
        """
        Predict class probabilities for new input data.

        This method returns the predicted probabilities for each class for the provided input samples.
        It supports input as NumPy arrays, pandas DataFrames, or PySpark DataFrames (if supported by the parent class).
        Probabilities are computed using the softmax activation function applied to the linear predictions.

        Parameters
        ----------
        model_input : array-like or dataframe
            Input data for probability prediction. Accepts NumPy arrays, Pandas DataFrame, or other Narwhals-supported types.

        Returns
        -------
        array-like
            Predicted probabilities for each class, shape (n_samples, n_classes). Returns native type (NumPy array or Pandas DataFrame) based on input. Each row sums to 1.0.

        Examples
        --------
        >>> y_proba = model.predict_proba(X_test)
        >>> print(y_proba[:5])
        """
        self.logger.info("Calculating predicted probabilities with MulticlassClassifier.")
        # check
        model_input_checked = super()._predict_io(model_input)
        return super()._predict_proba(model_input_checked)

    @narwhalify
    def score(self, X, y) -> float:
        """
        Compute the accuracy score of the model on the provided test data.

        This method evaluates the performance of the trained model by computing the accuracy score,
        which is the proportion of correctly classified samples, on the given input features and true labels.

        Parameters
        ----------
        X : array-like or dataframe
            Input features for evaluation.
        y : array-like or series
            True class labels for evaluation.

        Returns
        -------
        float
            Accuracy score of the model predictions on the provided data.

        Examples
        --------
        >>> score = model.score(X_test, y_test)
        >>> print("Accuracy:", score)
        """
        self.logger.info("Calculating accuracy score for MulticlassClassifier.")
        # check
        Xcheck, ycheck = super()._fit_io(X, y)
        y_pred = self.predict(Xcheck)
        return accuracy_score(ycheck, y_pred)

    def __repr__(self):
        """String representation of the MulticlassClassifier model."""
        return (f"MulticlassClassifier(degree={self.degree}, max_iter={self.max_iter}, "
                f"tolerance={self.tolerance}, early_stopping={self.early_stopping}, "
                f"verbose={self.verbose}, scale='{self.scale}', data_form='{self.data_form}', "
                f"gnostic_characteristics={self.gnostic_characteristics})")
