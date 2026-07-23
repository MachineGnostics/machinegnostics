"""
Gnostic Neuron Implementation

ManGo - Machine Gnostics Library
Copyright (C) 2026 Nirmal Parmar

- This is gnostic neuron inspired from grandmother neuron concept but with more flexibility and configurability. It can be used as a building block for more complex gnostic neural networks and architectures.

current input X, Y dimensions scope for gnostic neuron:
X: (n_samples, n_features)
Y: (n_samples, 1)
"""

import logging

import numpy as np
from machinegnostics.magcal import GnosticsCharacteristics, DataConversion
from machinegnostics.magnet.engine.gnostic_engine import GnosticEngine
from machinegnostics.magnet.base_neuron import BaseGnosticNeuron
from machinegnostics.magcal.util.logging import get_logger

class BaseGnosticNeuronCalc(BaseGnosticNeuron):
    def __init__(self, 
                 learning_rate:float=0.01,
                 S: str|float='auto', # local scale parameter for gnostic weights; can be 'auto' or a float value in range of [0.01,2]
                 epochs:int=100,
                 verbose:bool=False,
                 threshold:float=1e-2,
                 use_gnostic_weights:bool=True, # NOTE: exploration with E-gw and Q-gw
                 activation:str='fi', # options: 'fi', 'fj', 'hi', 'hj', step, sigmoid, relu, linear, tanh, leaky_relu, elu, softplus, swish, gelu, mish
                 gnostic_weights_type:str='E', # Estimating and Quantifying gnostic weights, options: 'E' for E-gw, 'Q' for Q-gw ['E', 'Q'] or e q
                 gnostic_loss:str='fj', #options: 'fi', 'fj', 'hi', 'hj'
                 random_state:int=42,
                 early_stopping:bool=True,
                 early_stopping_patience:int=5,
                 history:bool=False,
                 flush:bool=False,
                 float_type:str='float32'):  # float bytes option for memory optimization; options: 'float16', 'float32', 'float64'
        super().__init__(verbose=verbose)
        self.weights = None  # Weights will be initialized during training
        self.bias = None
        self.gw = None  # Gnostic weights
        self.lr = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.threshold = threshold
        self.activation = activation.lower() if activation else None
        self.use_gnostic_weights = use_gnostic_weights
        self.gnostic_weights_type = gnostic_weights_type.upper() if gnostic_weights_type else None
        self.gnostic_loss = gnostic_loss.lower() if gnostic_loss else None
        self.history = history
        self.flush = flush
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.ScaleParam = S if isinstance(S, (int, float)) else S.lower()  # if string then lower case, else use as is
        self.random_state = random_state
        self.float_type = float_type 
        self._fitted = False

        if self.history:
            self._history = {
                're': [],
                'loss': [],
                'fidelity': [],
                'irrelevance': [],
                'gnostic_score': [],
                'gw_error': [],
                'errors': []
            }

        self.params = {
            'learning_rate': self.lr,
            'epochs': self.epochs,
            'verbose': self.verbose,
            'threshold': self.threshold,
            'weights': self.weights,
            'bias': self.bias,
            'use_gnostic_weights': self.use_gnostic_weights,
            'activation': self.activation,
            'history': self.history,
            'flush': self.flush,
            'fitted': self._fitted,
            'S': self.ScaleParam,
            'random_state': self.random_state,
            'float_type': self.float_type,
            'errors': {}, # capturing execution errors and exceptions during training and inference
            'warnings': {} # capturing warnings during training and inference, such as convergence warnings, numerical stability issues, etc.
        }

        # logger
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if self.verbose else logging.WARNING)  # Create a logger for this class
        self.logger.info(f"{self.__class__.__name__} initialized with parameters: {self.params}")

        # argument validation
        self._arg_validation()
        self.logger.info(f"{self.__class__.__name__} argument validation passed.")

        
    def _arg_validation(self):
        # Validate activation function
        valid_activations = ['fi', 'fj', 'hi', 'hj', 'step', 'sigmoid', 'relu', 'linear', 'tanh', 'leaky_relu', 'elu', 'softplus', 'swish', 'gelu', 'mish', 'softmax']
        if self.activation not in valid_activations:
            self.logger.error(f"Invalid activation function: {self.activation}. Valid options are: {valid_activations}")
            self.params['warnings']['activation'] = f"Invalid activation function: {self.activation}. Valid options are: {valid_activations}"
            raise ValueError(f"Invalid activation function: {self.activation}. Valid options are: {valid_activations}")
        
        # Validate gnostic loss function
        valid_losses = ['fi', 'fj', 'hi', 'hj', None]
        if self.gnostic_loss not in valid_losses:
            self.logger.error(f"Invalid gnostic loss function: {self.gnostic_loss}. Valid options are: {valid_losses}")
            self.params['warnings']['gnostic_loss'] = f"Invalid gnostic loss function: {self.gnostic_loss}. Valid options are: {valid_losses}"
            raise ValueError(f"Invalid gnostic loss function: {self.gnostic_loss}. Valid options are: {valid_losses}")
        
        # Validate float type
        valid_float_types = ['float16', 'float32', 'float64']
        if self.float_type not in valid_float_types:
            self.logger.error(f"Invalid float type: {self.float_type}. Valid options are: {valid_float_types}")
            self.params['warnings']['float_type'] = f"Invalid float type: {self.float_type}. Valid options are: {valid_float_types}"
            raise ValueError(f"Invalid float type: {self.float_type}. Valid options are: {valid_float_types}")
        
        # scale parameter validation
        if isinstance(self.ScaleParam, (int, float)):
            if not (0.01 <= self.ScaleParam <= 2):
                self.logger.error(f"Invalid scale parameter S: {self.ScaleParam}. It must be in the range [0.01, 2].")
                self.params['warnings']['ScaleParam'] = f"Invalid scale parameter S: {self.ScaleParam}. It must be in the range [0.01, 2]."
                raise ValueError(f"Invalid scale parameter S: {self.ScaleParam}. It must be in the range [0.01, 2].")
        elif self.ScaleParam != 'auto':
            self.logger.error(f"Invalid scale parameter S: {self.ScaleParam}. It must be 'auto' or a float value in the range [0.01, 2].")
            self.params['warnings']['ScaleParam'] = f"Invalid scale parameter S: {self.ScaleParam}. It must be 'auto' or a float value in the range [0.01, 2]."
            raise ValueError(f"Invalid scale parameter S: {self.ScaleParam}. It must be 'auto' or a float value in the range [0.01, 2].")
        
        # to select gnostic loss, gnostic weights must be enabled
        if self.gnostic_loss and not self.use_gnostic_weights:
            self.logger.error(f"Gnostic loss function '{self.gnostic_loss}' cannot be used when gnostic weights are disabled. Please enable gnostic weights to use this loss function.")
            self.params['warnings']['gnostic_loss'] = f"Gnostic loss function '{self.gnostic_loss}' cannot be used when gnostic weights are disabled. Please enable gnostic weights to use this loss function."
            raise ValueError(f"Gnostic loss function '{self.gnostic_loss}' cannot be used when gnostic weights are disabled. Please enable gnostic weights to use this loss function.")
        
        # early stopping validation steps, must be positive integer and greater than 2
        if self.early_stopping:
            if not isinstance(self.early_stopping_patience, int) or self.early_stopping_patience < 2:
                self.logger.error(f"Invalid early_stopping_patience: {self.early_stopping_patience}. It must be an integer greater than or equal to 2.")
                self.params['warnings']['early_stopping_patience'] = f"Invalid early_stopping_patience: {self.early_stopping_patience}. It must be an integer greater than or equal to 2."
                raise ValueError(f"Invalid early_stopping_patience: {self.early_stopping_patience}. It must be an integer greater than or equal to 2.")
            
        # gnostic weights type validation
        valid_gnostic_weights_types = ['E', 'Q']
        if self.gnostic_weights_type not in valid_gnostic_weights_types:
            self.logger.error(f"Invalid gnostic weights type: {self.gnostic_weights_type}. Valid options are: {valid_gnostic_weights_types}")
            self.params['warnings']['gnostic_weights_type'] = f"Invalid gnostic weights type: {self.gnostic_weights_type}. Valid options are: {valid_gnostic_weights_types}"
            raise ValueError(f"Invalid gnostic weights type: {self.gnostic_weights_type}. Valid options are: {valid_gnostic_weights_types}")
        
    def _gnostic_weights_initialization(self, X, y):
        self.weights = np.ones(X.shape[1], dtype=self.float_type)  # Initialize weights to ones
        self.bias = 0.0
        self.gnostic_weights = np.ones(X.shape[0], dtype=self.float_type)

        # logger
        self.logger.info(f"Gnostic weights initialized. Weights shape: {self.weights.shape}, Bias: {self.bias}, Gnostic weights shape: {self.gnostic_weights.shape}")

    def _activation(self, predictions, true_labels, activation=None):
        if activation is None:
            activation = self.activation
        # gnostic activation is different from standard activation functions as it is based on the characteristics of the input data and the error distribution. It can be dynamic and adaptive during training, allowing the neuron to learn complex patterns and relationships in the data. The specific implementation of gnostic activation can vary based on the chosen characteristics and the desired behavior of the neuron.
        # NOTE this still required further study and exploration.
        # logger
        self.logger.debug(f"Calculating gnostic activation. Activation type: {activation}")
        # conditional initiation of gnostic engine
        if activation in ['fi', 'fj', 'hi', 'hj']:
            # init of classes
            gnostic_engine = GnosticEngine()
            data_conv = DataConversion()
            error = predictions - np.median(true_labels)  # Calculate error based on median of true labels for stabilitycharacteristics calculations
            z_error = data_conv._convert_az(error)  # Convert error to az space for gnostic characteristics calculations
            _ = gnostic_engine._get_gnostic_weights(z_error, scale_param=self.ScaleParam)

        if activation == 'fi':
            # Example: gnostic activation based on the fidelity characteristic
            gnostic_chars_mean = np.mean(gnostic_engine._get_fi(), axis=0) # Get the mean of gnostic weights for fi characteristic, keeping dimensions for broadcasting (n_samples, 1)
            # gnostic chars vs predictions
            acti_err = z_error - gnostic_chars_mean  # Calculate activation error based on the difference between gnostic characteristics and the error in az space
            z_acti = data_conv._convert_az(acti_err)
            _ = gnostic_engine._get_gnostic_weights(z_acti, scale_param=self.ScaleParam)
            acti = gnostic_engine._get_fi()  # Get the fi characteristic as the gnostic activation output
            return acti
        elif activation == 'fj':
            # Example: gnostic activation based on the irrelevance characteristic
            gnostic_chars_mean = np.mean(gnostic_engine._get_fj(), axis=0) # Get the mean of gnostic weights for fj characteristic, keeping dimensions for broadcasting (n_samples, 1)
            acti_err = z_error - gnostic_chars_mean  # Calculate activation error based on the difference between gnostic characteristics and the error in az space
            z_acti = data_conv._convert_az(acti_err)
            _ = gnostic_engine._get_gnostic_weights(z_acti, scale_param=self.ScaleParam)
            acti = gnostic_engine._get_fj()  # Get the fj characteristic as the gnostic activation output
            return acti
        elif activation == 'hi':
            # Example: gnostic activation based on the hi characteristic
            gnostic_chars_mean = np.mean(gnostic_engine._get_hi(), axis=0)  # Get the mean of gnostic weights for hi characteristic, keeping dimensions for broadcasting (n_samples, 1)
            acti_err = z_error - gnostic_chars_mean  # Calculate activation error based on the difference between gnostic characteristics and the error in az space
            z_acti = data_conv._convert_az(acti_err)
            _ = gnostic_engine._get_gnostic_weights(z_acti, scale_param=self.ScaleParam)
            acti = gnostic_engine._get_hi()  # Get the hi characteristic as the gnostic activation output
            return acti
        elif activation == 'hj':
            # Example: gnostic activation based on the hj characteristic
            gnostic_chars_mean = np.mean(gnostic_engine._get_hj(), axis=0)  # Get the mean of gnostic weights for hj characteristic, keeping dimensions for broadcasting (n_samples, 1)
            acti_err = z_error - gnostic_chars_mean  # Calculate activation error based on the difference between gnostic characteristics and the error in az space
            z_acti = data_conv._convert_az(acti_err)
            _ = gnostic_engine._get_gnostic_weights(z_acti, scale_param=self.ScaleParam)
            acti = gnostic_engine._get_hj()  # Get the hj characteristic as the gnostic activation output
            return acti
        else:
            # For standard activation functions, you can implement them as needed or use existing implementations from libraries like NumPy or SciPy.
            if activation == 'step':
                return np.where(predictions >= 0, 1, 0)
            elif activation == 'sigmoid':
                return 1 / (1 + np.exp(-predictions))
            elif activation == 'relu':
                return np.maximum(0, predictions)
            elif activation == 'linear':
                return predictions
            elif activation == 'tanh':
                return np.tanh(predictions)
            elif activation == 'leaky_relu':
                return np.where(predictions > 0, predictions, 0.01 * predictions)
            elif activation == 'elu':
                return np.where(predictions > 0, predictions, 0.01 * (np.exp(predictions) - 1))
            elif activation == 'softplus':
                return np.log(1 + np.exp(predictions))
            elif activation == 'swish':
                return predictions / (1 + np.exp(-predictions))
            elif activation == 'gelu':
                return 0.5 * predictions * (1 + np.tanh(np.sqrt(2 / np.pi) * (predictions + 0.044715 * np.power(predictions, 3))))
            elif activation == 'mish':
                return predictions * np.tanh(np.log(1 + np.exp(predictions)))
            elif activation == 'softmax':
                exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))  # Subtract max for numerical stability
                return exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
            else:
                self.logger.error(f"Unsupported activation function: {activation}")
                self.params['warnings']['activation'] = f"Unsupported activation function: {activation}"
                raise ValueError(f"Unsupported activation function: {activation}")
    
    def _predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias  # Linear transformation
        predictions = self._activation(linear_output, self.y)  # Apply activation
        return predictions
    
    def _calculate_gnostic_weights(self, error, data_conv: DataConversion, gnostic_engine: GnosticEngine):
        pass
        
    def _fit(self, X, y):
        # logger
        self.logger.info("Starting training process.")

        # inputs
        self.X = X.astype(self.float_type)
        self.y = y.astype(self.float_type)

        # y reshaping for consistency
        # if len(self.y.shape) == 1:
        # self.y = self.y.reshape(-1, 1)

        # shuffle data
        np.random.seed(self.random_state)
        indices = np.random.permutation(self.X.shape[0])
        self.X = self.X[indices]
        self.y = self.y[indices]

        # Initialize weights and parameters
        self._gnostic_weights_initialization(self.X, self.y)
        self.S_local = 1.0

        # Use config flag for enable/disable, keep gw for runtime tensor
        self.gnostic_engine = GnosticEngine() if self.use_gnostic_weights else None
        data_conv = DataConversion()
        self.logger.info("Gnostic engine initialized. Starting training loop.")

        # Ensure history has all keys used below
        if self.history:
            if not hasattr(self, "_history") or self._history is None:
                self._history = {}
            for key in [
                "re",
                "loss",
                "fidelity",
                "irrelevance",
                "gnostic_score",
                "gw_error",
                "errors",
                "residual_entropy",
                "S_local",
            ]:
                self._history.setdefault(key, [])

        # Training loop
        for epoch in range(self.epochs):
            predictions = self._predict(self.X)

            # residual error
            residual_error = (self.y - predictions)
            z_error = data_conv._convert_az(residual_error)

            # gnostic weights + characteristics
            if self.use_gnostic_weights:
                self.gw = self.gnostic_engine._get_gnostic_weights(
                    z_error, scale_param=self.ScaleParam
                )
                # self.gw = self.gw.reshape(-1, 1)
                # Q gnostic weights calculation
                if self.gnostic_weights_type == 'Q':
                    self.gw = np.clip( 1/ (self.gw + 1e-8), 1e-9, 1e9)  # Avoid division by zero and extreme values, clip to a reasonable range
                elif self.gnostic_weights_type == 'E':
                    # as it is
                    pass

                if self.ScaleParam == "auto":
                    self.S_local = self.gnostic_engine._get_S_local()
                else:
                    self.S_local = self.ScaleParam

                self.fidelity = self.gnostic_engine._get_fi()
                self.irrelevance = self.gnostic_engine._get_hi()
                self.residual_entropy = self.gnostic_engine._get_re()
            else:
                self.gw = np.ones_like(z_error, dtype=self.float_type)
                self.S_local = self.ScaleParam if isinstance(self.ScaleParam, (int, float)) else 1.0
                self.fidelity = np.ones_like(z_error, dtype=self.float_type)
                self.irrelevance = np.zeros_like(z_error, dtype=self.float_type)
                self.residual_entropy = np.ones_like(z_error, dtype=self.float_type)

            # loss
            if self.gnostic_loss in {"fi", "fj", "hi", "hj"} and self.use_gnostic_weights:
                if self.gnostic_loss == "fi":
                    loss = np.mean(self.fidelity)
                elif self.gnostic_loss == "fj":
                    loss = np.mean(self.gnostic_engine._get_fj())
                elif self.gnostic_loss == "hi":
                    loss = np.mean(self.irrelevance)
                else:  # "hj"
                    loss = np.mean(self.gnostic_engine._get_hj())
            else:
                loss = np.mean(residual_error ** 2)

            # update
            gnostic_weighted_error = residual_error * self.gw
            self.weights += self.lr * np.dot(self.X.T, gnostic_weighted_error)
            self.bias += self.lr * np.mean(gnostic_weighted_error)

            # logging
            if self.verbose:
                if self.gnostic_loss in {"fi", "fj", "hi", "hj"} and self.use_gnostic_weights:
                    self.logger.info(
                        f"Epoch {epoch + 1}/{self.epochs} - Gnostic Loss [{self.gnostic_loss}]: {loss:.6f}, Scale Param: {self.S_local:.6f}"
                    )
                else:
                    self.logger.info(f"Epoch {epoch + 1}/{self.epochs} - Loss: {loss:.6f}")

            # history update first, then convergence checks use current epoch included
            if self.history:
                self._history["loss"].append(float(loss))
                self._history["errors"].append(float(np.mean(residual_error ** 2)))
                self._history["fidelity"].append(float(np.mean(self.fidelity)))
                self._history["irrelevance"].append(float(np.mean(self.irrelevance)))
                re_val = float(np.mean(self.residual_entropy))
                self._history["residual_entropy"].append(re_val)
                self._history["re"].append(re_val)
                self._history["gw_error"].append(float(np.mean(gnostic_weighted_error ** 2)))
                self._history["S_local"].append(float(self.S_local))

                if self.gnostic_loss in {"fi", "fj", "hi", "hj"} and self.use_gnostic_weights:
                    self._history["gnostic_score"].append(float(loss))

            # convergence / early stopping
            if self.early_stopping and self.history and (epoch + 1) >= self.early_stopping_patience:
                window = self.early_stopping_patience

                # standard loss mode
                if not (self.gnostic_loss in {"fi", "fj", "hi", "hj"} and self.use_gnostic_weights):
                    recent_losses = self._history["loss"][-window:]
                    if len(recent_losses) == window and (max(recent_losses) - min(recent_losses)) < self.threshold:
                        self.logger.warning(
                            f"Early stopping at epoch {epoch + 1} due to minimal change in loss in the last {window} epochs."
                        )
                        break
                else:
                    # gnostic score stability
                    recent_scores = self._history["gnostic_score"][-window:]
                    if len(recent_scores) == window and (max(recent_scores) - min(recent_scores)) < self.threshold:
                        self.logger.warning(
                            f"Early stopping at epoch {epoch + 1} due to minimal change in {self.gnostic_loss} gnostic loss in the last {window} epochs."
                        )
                        break

                    # residual entropy guard
                    recent_re = self._history["residual_entropy"][-window:]
                    if any(v < 0 for v in recent_re):
                        self.logger.warning(
                            f"Early stopping at epoch {epoch + 1} due to negative residual entropy in the last {window} epochs."
                        )
                        break

        self._fitted = True
        return self