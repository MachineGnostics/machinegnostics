import numpy as np
import pandas as pd
import logging
from machinegnostics.magcal.util.logging import get_logger
try:
    from pyspark.sql import DataFrame as SparkDataFrame
except ImportError:
    SparkDataFrame = None

class DataProcessLayerBase:
    """
    A class to handle input/output processing for machine learning models.

    This class provides methods for data type checking, validation, and conversion
    to ensure that input data is in the correct format for model training and prediction.
    """
    def __init__(self, verbose: bool = False, **kwargs):
        """
        Initialize the DataProcessLayer with optional parameters.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters for configuration.
        """ # To store the type of input for output conversion
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if verbose else logging.WARNING)
        self.logger.info(f"{self.__class__.__name__} initialized:")
        self.logger.info("DataProcessLayerBase initialized.")

    def _identify_and_convert(self, data, is_y=False):
        """
        Identify the type of data and convert it to a numpy array.

        Parameters
        ----------
        data : array-like, pandas DataFrame, or pyspark DataFrame
            Input data to be converted.
        is_y : bool, default=False
            Whether the data is target values (y).

        Returns
        -------
        np.ndarray
            Converted numpy array.
        """
        self.logger.info(f"Identifying and converting data of type: {type(data)}")
        if isinstance(data, np.ndarray):
            arr = data
            self._input_type = 'numpy'
        elif isinstance(data, pd.DataFrame):
            arr = data.values
            self._input_type = 'pandas'
        elif SparkDataFrame is not None and isinstance(data, SparkDataFrame):
            arr = np.array(data.collect())
            self._input_type = 'spark'
        else:
            arr = np.array(data)
            self._input_type = 'unknown'

        if is_y:
            arr = np.ravel(arr)
        return arr

    def _convert_output(self, output, reference_input):
        """
        Convert output numpy array back to the original input format.

        Parameters
        ----------
        output : np.ndarray
            Output data to be converted.
        reference_input : original input data
            The original input data to infer the output format.

        Returns
        -------
        Converted output in the original format.
        """
        self.logger.info(f"Converting output to match reference input type: {type(reference_input)}")
        if isinstance(reference_input, np.ndarray):
            return output
        elif isinstance(reference_input, pd.DataFrame):
            return pd.DataFrame(output, index=reference_input.index, columns=getattr(reference_input, 'columns', None))
        elif SparkDataFrame is not None and isinstance(reference_input, SparkDataFrame):
            # For Spark, convert numpy array to pandas DataFrame, then to Spark DataFrame
            import pyspark.sql
            spark = pyspark.sql.SparkSession.builder.getOrCreate()
            pdf = pd.DataFrame(output)
            return spark.createDataFrame(pdf)
        else:
            return output

    def _check_X(self, X, n_features=None):
        """
        Check if the input X is valid.

        Parameters
        ----------
        X : array-like
            Input features.
        n_features : int, optional
            Expected number of features.

        Raises
        ------
        ValueError
            If X is invalid.
        """
        self.logger.info(f"Checking input X of type: {type(X)}")
        X_arr = self._identify_and_convert(X)

        # if X_qrr is 1 dimensional, reshape it to 2D
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2:
            raise ValueError("X should be a 1D or 2D or nD array-like structure.")
        if n_features is not None and X_arr.shape[1] != n_features:
            raise ValueError(f"X should have {n_features} features, got {X_arr.shape[1]}.")
        if X_arr.shape[0] == 0:
            raise ValueError("X is empty.")
        return X_arr

    def _check_y(self, y, n_samples=None):
        """
        Check if the target y is valid.

        Parameters
        ----------
        y : array-like
            Target values.
        n_samples : int, optional
            Expected number of samples.

        Raises
        ------
        ValueError
            If y is invalid.
        """
        self.logger.info(f"Checking target y of type: {type(y)}")
        y_arr = self._identify_and_convert(y, is_y=True)
        if y_arr.ndim != 1:
            raise ValueError("y should be a 1D array-like structure.")
        if n_samples is not None and y_arr.shape[0] != n_samples:
            raise ValueError(f"y should have {n_samples} samples, got {y_arr.shape[0]}.")
        if y_arr.shape[0] == 0:
            raise ValueError("y is empty.")
        return y_arr

    def _check_X_predict(self, X, n_features=None):
        """
        Check if the input X for prediction is valid.

        Parameters
        ----------
        X : array-like
            Input features for prediction.
        n_features : int, optional
            Expected number of features.

        Raises
        ------
        ValueError
            If X is invalid.
        """
        self.logger.info(f"Checking input X for prediction of type: {type(X)}")
        X = self._check_X(X, n_features=n_features)
        # # output type
        # if self._input_type is None:
        #     self._input_type = 'numpy'
        # elif self._input_type == 'pandas':
        #     X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        # elif self._input_type == 'spark':
        #     import pyspark.sql
        #     spark = pyspark.sql.SparkSession.builder.getOrCreate()
        #     X = spark.createDataFrame(X, schema=[f'feature_{i}' for i in range(X.shape[1])])
        # elif self._input_type == 'unknown':
        #     raise ValueError("Unknown input type. Please provide a valid input format.")
        return X

    def _fit_io(self, X, y):
        """
        Fit the model to the provided data after checking and verifying inputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features for training the model.
        y : array-like of shape (n_samples,)
            Target values corresponding to the input features.

        Returns
        -------
        self : object
            Returns self.
        """
        self.logger.info("Starting fit input/output processing.")
        X_checked = self._check_X(X)
        y_checked = self._check_y(y, n_samples=X_checked.shape[0])
        return X_checked, y_checked

    def _predict_io(self, X):
        """
        Predict using the model after checking and verifying inputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features for prediction.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """
        self.logger.info("Starting predict input/output processing.")
        X_checked = self._check_X_predict(X)
        return X_checked
    
    def _score_io(self, X, y):
        """
        Return the score of the model after checking and verifying inputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            Score of the model.
        """
        self.logger.info("Starting score input/output processing.")
        X_checked = self._check_X_predict(X)
        y_checked = self._check_y(y, n_samples=X_checked.shape[0])
        return self