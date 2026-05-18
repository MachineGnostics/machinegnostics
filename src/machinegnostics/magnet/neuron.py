"""
Gnostic Neuron Implementation

ManGo - Machine Gnostics Library
Copyright (C) 2026 Nirmal Parmar
"""

import numpy as np
from machinegnostics.magcal import GnosticsCharacteristics, DataConversion
from machinegnostics.magnet.engine.gnostic_wights import GnosticsWeights

class GnosticNeuron:
    def __init__(self, 
                 learning_rate=0.01,
                 S='auto', # local scale parameter for gnostic weights; can be 'auto' or a float value in range of [0,2]
                 epochs=100,
                 verbose=False,
                 threshold=1e-2,
                 gnostic_weights=True,
                 gnostic_activation='fi', # options: 'fi', 'fj', 'hi', 'hj', step, sigmoid, relu, linear, tanh, leaky_relu, elu, softplus, swish, gelu, mish
                 gnostic_loss='re', #options: 'fi', 'fj', 'hi', 'hj', 're'
                 random_state=42,
                 early_stopping=True,
                 history=False,
                 flush=False):
        self.weights = None  # Weights will be initialized during training
        self.bias = None
        self.gw = None  # Gnostic weights
        self.lr = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.threshold = threshold
        self.gnostic_activation = gnostic_activation.lower() if gnostic_activation else None
        self.gnostic_weights = gnostic_weights
        self.gnostic_loss = gnostic_loss.lower() if gnostic_loss else None
        self.history = history
        self.flush = flush
        self.early_stopping = early_stopping
        self.ScaleParam = S
        self.random_state = random_state

        # input validation
        # for S
        if isinstance(S, str):
            if S.lower() != 'auto':
                raise ValueError("Scale parameter S must be 'auto' or a float value in range [0.01, 2].")
        elif isinstance(S, (int, float)):
            if not (0.01 <= S <= 2):
                raise ValueError("Scale parameter S must be in the range [0.01, 2].")
        else:
            raise ValueError("Scale parameter S must be a string ('auto') or a float value in range [0.01, 2].")
        # for gnostic_activation
        valid_activations = {'fi', 'fj', 'hi', 'hj', 'step', 'sigmoid', 'relu', 'tanh', 'leaky_relu', 'elu', 'softplus', 'swish', 'gelu', 'mish', 'linear', 'softmax'}
        if self.gnostic_activation and self.gnostic_activation not in valid_activations:
            raise ValueError(f"Gnostic activation must be one of {valid_activations}.")
        
        # for gnostic_loss
        valid_losses = {'fi', 'fj', 'hi', 'hj', 're'}
        if self.gnostic_loss and self.gnostic_loss not in valid_losses:
            raise ValueError(f"Gnostic loss must be one of {valid_losses}.")

        self._fitted = False

        if self.history:
            self._history = {
                're': [],
                'loss': [],
                'fidelity': [],
                'irrelevance': [],
                'gnostic_score': [],
                'gw_error': [],
            }

        self.params = {
            'learning_rate': self.lr,
            'epochs': self.epochs,
            'verbose': self.verbose,
            'threshold': self.threshold,
            'weights': self.weights,
            'bias': self.bias,
            'gnostic_weights': self.gnostic_weights,
            'gnostic_activation': self.gnostic_activation,
            'history': self.history,
            'flush': self.flush,
            'fitted': self._fitted,
            'S': self.ScaleParam,
            'random_state': self.random_state
        }

    def _step_activation(self, Z):
        # Vectorized Step Function: applies to the entire array at once
        return np.where(Z >= 0, 1, 0)
    
    def _gnostic_activation(self, z):
        '''Gnostic activation function based on the characteristics of the input z.'''
        z0 = np.median(z)
        z_acti = z0 - np.sum(z)
        dc = DataConversion()
        z_acti = dc._convert_az(z_acti)

        chars = GnosticsCharacteristics(R=z_acti)
        q, q1 = chars._get_q_q1(S=self.S)
        if self.gnostic_activation == 'fi':
            acti = np.sum(chars._fi(q, q1))
        elif self.gnostic_activation == 'fj':
            acti = np.sum(chars._fj(q, q1))
        elif self.gnostic_activation == 'hi':
            acti = np.sum(chars._hi(q, q1))
        elif self.gnostic_activation == 'hj':
            acti = np.sum(chars._hj(q, q1))
        elif self.gnostic_activation == 'step':
            acti = self._step_activation(z)
        elif self.gnostic_activation == 'sigmoid':
            acti = 1 / (1 + np.exp(-z))  # Sigmoid activation
        elif self.gnostic_activation == 'relu':
            acti = np.maximum(0, z)  # ReLU activation
        elif self.gnostic_activation == 'tanh':
            acti = np.tanh(z)
        elif self.gnostic_activation == 'leaky_relu':
            acti = np.where(z > 0, z, 0.01 * z)
        elif self.gnostic_activation == 'elu':
            acti = np.where(z > 0, z, np.exp(z) - 1)
        elif self.gnostic_activation == 'softplus':
            acti = np.log1p(np.exp(np.clip(z, -50, 50)))
        elif self.gnostic_activation == 'softmax':
            exp_z = np.exp(z - np.max(z))  # for numerical stability
            acti = exp_z / np.sum(exp_z)
        elif self.gnostic_activation == 'swish':
            acti = z / (1 + np.exp(-z))
        elif self.gnostic_activation == 'gelu':
            acti = 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * np.power(z, 3))))
        elif self.gnostic_activation == 'mish':
            softplus_z = np.log1p(np.exp(np.clip(z, -50, 50)))
            acti = z * np.tanh(softplus_z)
        elif self.gnostic_activation == 'linear':
            acti = 1  # Linear activation
        else:
            acti = 1  # Fallback to linear-style scaling
        return acti * z

    def predict(self, X):
        """
        Forward Pass:
        X shape: (m, n)
        W shape: (n, 1)
        Z shape: (m, 1)
        """
        Z = X @ self.weights + self.bias  # Linear transformation

        if self.gnostic_activation:
            return self._gnostic_activation(Z)
        else:
            return Z  # No activation, return linear output

    def _is_maximize_mode(self):
        # fi and hj are optimized by maximizing; others are minimized.
        return self.gnostic_loss in {'fi', 'hj'}

    def _has_constant_window(self, values, window=5):
        if len(values) < window:
            return False
        recent = values[-window:]
        return (max(recent) - min(recent)) <= self.threshold

    def fit(self, X, y):
        # Ensure y is a column vector (m, 1) to match predictions
        y = y.reshape(-1, 1)
        # shuffle data at the start of training for better convergence
        np.random.seed(self.random_state)
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        y = y[indices]
        # Initialize weights, bias, and gnostic weights if not already set
        if self.weights is None:
            self.weights = np.zeros((X.shape[1], 1))
        if self.bias is None:
            self.bias = 0.0
        if self.gw is None:
            self.gw = np.ones_like(y).reshape(-1, 1)
        self.S = 1.0  # Initial scale parameter for gnostic weights

        # Early-stopping controller state.
        patience = 5
        maximize_gnostic = self._is_maximize_mode()
        best_gnostic = -np.inf if maximize_gnostic else np.inf
        best_gw_error = np.inf
        no_improve_epochs = 0
        gnostic_window = []
        gw_error_window = []

        # Reuse the engine object to reduce per-epoch setup overhead.
        gw_engine = GnosticsWeights() if self.gnostic_weights else None

        # Training loop
        for epoch in range(self.epochs):
            # 1. Forward pass (Calculate all predictions at once)
            predictions = self.predict(X)
            
            # 2. Calculate error vector
            error = y - predictions

            # gnostic weights from error vector
            z_error = np.exp(error)

            # gnostic weights from transformed error vector
            # gnostic wights switch
            if self.gnostic_weights:
                self.gw = gw_engine._get_gnostic_weights(z_error)
                self.S = gw_engine._get_S_local()
                self.fi = gw_engine._get_fi()
                self.hi = gw_engine._get_hi()
                self.re = gw_engine._get_re()

                # gnostic loss switch
                if self.gnostic_loss == 'fi':
                    err = np.sum(self.fi)
                elif self.gnostic_loss == 'fj':
                    err = np.sum(gw_engine._get_fj())
                elif self.gnostic_loss == 'hi':
                    err = np.sum(self.hi**2)
                elif self.gnostic_loss == 'hj':
                    err = np.sum(gw_engine._get_hj()**2)
                elif self.gnostic_loss == 're':
                    err = np.sum(self.re)
                    # normalize re
                    err = err / (np.sum(np.abs(error)) + 1e-8)  # Avoid division by zero
                else:
                    err = err // np.sum(np.abs(error))  # Default to normalized error if unknown loss is specified
            else:
                self.gw = np.ones_like(error)
                self.S = 1.0
                self.fi = np.ones_like(error)
                self.hi = np.zeros_like(error)
                self.re = np.zeros_like(error)

                err = np.sum(np.abs(error))

            # gnostic weight update
            gw_error = self.gw * error
            self.weights += self.lr * np.dot(X.T, gw_error)
            self.bias += self.lr * np.sum(gw_error)

            # Scalar convergence metrics tracked each epoch.
            gnostic_score = float(np.asarray(err))
            gw_error_score = float(np.sum(np.abs(gw_error)))

            # Verbose logging            
            if self.verbose and self.gnostic_weights:
                print(f"Epoch {epoch+1}/{self.epochs}, Error: {np.sum(np.abs(error))}, Gnostic Error: {gnostic_score}, Mean Residual Entropy: {np.mean(self.re)}")
            elif self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, Error: {np.sum(np.abs(error))}")
            
            # fit status
            self._fitted = True

            # History tracking
            if self.history and self.gnostic_weights:
                self._history['loss'].append(gnostic_score)
                self._history['fidelity'].append(np.sum(self.fi**2))
                self._history['irrelevance'].append(np.sum(self.hi**2))
                self._history['re'].append(err/(np.sum(np.abs(error)) + 1e-8))
                # self._history['gnostic_score'].append(gnostic_score)
                self._history['gw_error'].append(gw_error_score)
            elif self.history:
                self._history['loss'].append(np.sum(np.abs(error)))
                self._history['gnostic_score'].append(gnostic_score)
                self._history['gw_error'].append(gw_error_score)
                # other gnostic characteristics are not tracked when gnostic_weights is False

            # Check for convergence using gnostic objective direction and gw_error minimization.
            # if encounter negative entropy (RE) stop training
            if self.re is not None and np.any(self.re < 0):
                if self.verbose:
                    print(f"Negative residual entropy encountered at epoch {epoch+1}. Stopping training.")
                break
            if self.early_stopping:
                if maximize_gnostic:
                    improved_gnostic = gnostic_score > (best_gnostic + self.threshold)
                else:
                    improved_gnostic = gnostic_score < (best_gnostic - self.threshold)

                improved_gw = gw_error_score < (best_gw_error - self.threshold)

                if improved_gnostic:
                    best_gnostic = gnostic_score
                if improved_gw:
                    best_gw_error = gw_error_score

                if improved_gnostic or improved_gw:
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                gnostic_window.append(gnostic_score)
                gw_error_window.append(gw_error_score)

                constant_5_epochs = (
                    self._has_constant_window(gnostic_window, window=patience)
                    and self._has_constant_window(gw_error_window, window=patience)
                )

                if no_improve_epochs >= patience or constant_5_epochs:
                    if self.verbose:
                        print(f"Convergence reached at epoch {epoch+1}. Stopping training.")
                    break
            
            # update params
            self.params['weights'] = self.weights
            self.params['bias'] = self.bias
            self.params['gnostic_weights'] = self.gw
            self.params['gnostic_activation'] = self.gnostic_activation
            self.params['history'] = self._history if self.history else None
            self.params['S'] = self.S
            self.params['fitted'] = self._fitted

            # Flush history to save memory
            if self.flush and self.history:
                self._history['loss'] = []
                self._history['fidelity'] = []
                self._history['irrelevance'] = []
                self._history['re'] = []
                # self._history['gnostic_score'] = []
                self._history['gw_error'] = []
            
    def __repr__(self):
        """Detailed string representation of the GnosticNeuron instance."""
        fitted = "✓ Fitted" if getattr(self, "_fitted", False) else "✗ Unfitted"
        n_features = ""
        if hasattr(self, "weights") and self.weights is not None:
            n_features = f"(n_features={self.weights.shape[0]})"
        return (
            f"GnosticNeuron(\n"
            f"  model_parameters={{\n"
            f"    'learning_rate': {self.lr},\n"
            f"    'epochs': {self.epochs},\n"
            f"    'threshold': {self.threshold},\n"
            f"    'random_state': {self.random_state},\n"
            f"  }},\n"
            f"  gnostic_config={{\n"
            f"    'gnostic_weights': {self.gnostic_weights},\n"
            f"    'activation': '{self.gnostic_activation}',\n"
            f"    'loss_function': '{self.gnostic_loss}',\n"
            f"    'scale_param': '{self.ScaleParam}',\n"
            f"  }},\n"
            f"  training_config={{\n"
            f"    'early_stopping': {self.early_stopping},\n"
            f"    'history_tracking': {self.history},\n"
            f"    'flush_history': {self.flush},\n"
            f"    'verbose': {self.verbose},\n"
            f"  }},\n"
            f"  status='{fitted}' {n_features}\n"
            f")"
        )