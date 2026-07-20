"""
Gnostic Neuron Implementation

ManGo - Machine Gnostics Library
Copyright (C) 2026 Nirmal Parmar
"""

import numpy as np
from machinegnostics.magcal import DataConversion
from machinegnostics.magnet.activations import ActivationFunctions
from machinegnostics.magnet.engine.gnostic_engine import GnosticEngine
from machinegnostics.magnet.activations import ActivationFunctions
import logging
from machinegnostics.magcal.util.logging import get_logger

class GnosticNeuron:
    """
    ### Gnostic Neuron Class
    
    This class is designed from the Biological Gnostic Neuron model. It has properties like specificity and irrelevance as the gnostic characteristics. The neuron can be used for regression and classification tasks. It uses the GnosticEngine to compute gnostic weights and activations.

    - Gnostic Neuron has two addional capabilities compared to a standard neuron: 1. Gnostic Weights and 2. Gnostic Activation. These are computed using the GnosticEngine class.
    """

    def __init__(self, learning_rate:float=0.01,
                 epochs:int=100, 
                 scale_param: str|float='auto',
                 early_stopping:bool=False,
                 activation_type:str='fi',
                 gnostic_char_loss:str='fj',
                 history:bool=False,
                 tolerance:float=1e-3,
                 early_stopping_patience:int=5,
                 random_state:int=42,
                 use_gnostic_weights:bool=True,
                 verbose:bool=False):
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.scale_param = scale_param
        self.early_stopping = early_stopping
        self.activation_type = activation_type
        self.gnostic_char_loss = gnostic_char_loss
        self.history = history
        self.tolerance = tolerance
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.use_gnostic_weights = use_gnostic_weights
        self.verbose = verbose

        self._history = {
                        'loss': [],
                        'gnostic_loss': [],
                        'rentropy': [],
                        'gnostic_characteristics': [],
                        } if history else None
        
        self.params = {
            'weights': None,
            'bias': None,
            'gnostic_weights': None,
            'S_local': None,
        }

        # initialize GnosticEngine and DataConversion
        self.gnostic_engine = GnosticEngine(verbose=verbose)
        self.data_conversion = DataConversion()
        self.activation_functions = ActivationFunctions(verbose=verbose)
        self.weights = None
        self.bias = None
        self.gnostic_weights = None

        # fit status
        self._fitted = False

        # validation and checks

        # logger
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if self.verbose else logging.WARNING)
        self.logger.info(f"{self.__class__.__name__} initialized.")

    def fit(self, X, y):
        """
        Fit the Gnostic Neuron model to the training data.

        Parameters:
        - X: Input features (numpy array)
        - y: Target values (numpy array)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.y_z0 = np.median(y)  # Reference value for gnostic characteristics

        # shuffle the data
        np.random.seed(self.random_state)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

        for epoch in range(self.epochs):
            # prediction
            predictions = self.predict(X)

            # errors
            errors = predictions - y
            errors_dc = self.data_conversion._convert_az(errors)

            # gnostic weights
            self.gnostic_weights = self.gnostic_engine._get_gnostic_weights(errors_dc, scale_param=self.scale_param)

            # compute gnostic errors
            gnostic_errors = errors * self.gnostic_weights if self.use_gnostic_weights else errors

            # update weights and bias
            self.weights -= self.learning_rate * (X.T @ gnostic_errors) / n_samples
            self.bias -= self.learning_rate * np.mean(gnostic_errors)

            # compute loss and gnostic loss
            loss = np.mean(errors ** 2)
            gnostic_loss = np.mean(gnostic_errors ** 2)
            # gnsotic characteristics loss [fi, fj]
            gnostic_char_loss = self.compute_gnostic_charc_loss()

            # history tracking
            if self.history:
                self._history['loss'].append(loss)
                self._history['gnostic_loss'].append(gnostic_loss)
                self._history['rentropy'].append(np.mean(self.gnostic_weights))
                self._history['gnostic_characteristics'].append(gnostic_char_loss)

            # print status
            if self.verbose:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss:.6f}, Gnostic Loss: {gnostic_loss:.6f}, Gnostic Char Loss: {gnostic_char_loss:.6f}")

            # convergance check

        self._fitted = True

    def convergance_check(self, history, tolerance, early_stopping_patience):
        """
        Check for convergence based on the loss history.

        Parameters:
        - history: List of loss values
        - tolerance: Tolerance for convergence
        - early_stopping_patience: Number of epochs to wait for improvement

        Returns:
        - converged: Boolean indicating if the model has converged
        """
        if len(history) < early_stopping_patience:
            return False

        recent_losses = history[-early_stopping_patience:]
        if max(recent_losses) - min(recent_losses) < tolerance:
            return True

        return False
    
    def compute_gnostic_charc_loss(self):
        """
        Compute the gnostic characteristics loss based on the specified gnostic_char_loss type.

        Returns:
        - gnostic_char_loss: Computed gnostic characteristics loss (float)
        """
        if self.gnostic_char_loss == 'fi':
            gnostic_char_loss = np.mean(self.gnostic_engine._get_fi())
        elif self.gnostic_char_loss == 'fj':
            gnostic_char_loss = np.mean(self.gnostic_engine._get_fj())
        else:
            raise ValueError(f"Invalid gnostic_char_loss: {self.gnostic_char_loss}. Must be one of ['fi', 'fj'].")
        
        return gnostic_char_loss

    def predict(self, X):
        """
        Predict using the Gnostic Neuron model.

        Parameters:
        - X: Input features (numpy array)

        Returns:
        - predictions: Predicted values (numpy array)
        """
        X = np.asarray(X, dtype=float)

        y_pred = X @ self.weights + self.bias

        # activation
        if self.activation_type in ['fi', 'fj', 'hi', 'hj']:
            predictions = self.gnostic_activation(y_pred, self.y_z0, scale_param=self.scale_param, activation_type=self.activation_type)
        else:
            predictions = self.regular_activation(y_pred, activation_type=self.activation_type)

        return predictions

    def gnostic_activation(self, z, z0, scale_param:str|float, activation_type:str='fi'):
        """
        Compute the gnostic activation for the given input.

        Parameters:
        - z: Input values (numpy array)
        - z0: Reference value (float)
        - scale_param: Scale parameter for gnostic activation (str or float)
        - activation_type: Type of gnostic activation ('fi', 'fj', 'hi', 'hj')

        Returns:
        - activation: Gnostic activation values (numpy array)
        """
        # calculate fi_mean respective to z0 (z0 is the reference value for gnostic characteristics). For activation, y median is used as z0. This can be furthere investigaated.
        z_diff = z - z0
        z_diff_dc = self.data_conversion._convert_az(z_diff)
        _ = self.gnostic_engine._get_gnostic_weights(z_diff_dc, scale_param=scale_param)
        if activation_type == 'fi':
            self.gnostics_charc = self.gnostic_engine._get_fi()
        elif activation_type == 'fj':
            self.gnostics_charc = self.gnostic_engine._get_fj()
        elif activation_type == 'hi':
            self.gnostics_charc = self.gnostic_engine._get_hi()
        elif activation_type == 'hj':
            self.gnostics_charc = self.gnostic_engine._get_hj()
        # calculating gnostics characteristics against z0 to calculate gnostic activation
        gc_diff = self.gnostics_charc - z0
        gc_diff_dc = self.data_conversion._convert_az(gc_diff)
        activation = self.gnostic_engine._get_activation(gc_diff_dc, scale_param=scale_param, activation_type=activation_type)
        return activation

    def regular_activation(self, z, activation_type:str):
        """
        Compute the regular activation (e.g., sigmoid) for the given input.

        Parameters:
        - z: Input values (numpy array)
        - activation_type: Activation name (str)

        Returns:
        - activation: Regular activation values (numpy array)
        """
        activation = self.activation_functions.activate(z, activation_type=activation_type)
        return activation
    
    def score(self, X, y):
        """
        Compute the score of the model on the given data.

        Parameters:
        - X: Input features (numpy array)
        - y: True target values (numpy array)

        Returns:
        - score: Model score (float)
        """
        pass

    def save(self, path):
        """
        Save the trained Gnostic Neuron model to disk.

        Parameters:
        - path: File path to save the model (str)
        """
        pass

    def load(self, path):
        """
        Load a trained Gnostic Neuron model from disk.

        Parameters:
        - path: File path to load the model from (str)
        """
        pass

    def get_params(self):
        """
        Get the parameters of the Gnostic Neuron model.

        Returns:
        - params: Dictionary containing model parameters
        """
        return self.params

    def __repr__(self):
        return f"GnosticNeuron(learning_rate={self.learning_rate}, epochs={self.epochs}, scale_param={self.scale_param})"