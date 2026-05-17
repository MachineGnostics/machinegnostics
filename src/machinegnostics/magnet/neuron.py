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
                 epochs=100,
                 verbose=False,
                 threshold=1e-2,
                 gnostic_weights=True,
                 gnostic_activation='fi', # options: 'fi', 'fj', 'hi', 'hj', step, None
                 gnostic_loss='re', #options: 'fi', 'fj', 'hi', 'hj', 're'
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

        self._fitted = False

        if self.history:
            self._history = {
                're': [],
                'loss': [],
                'fidelity': [],
                'irrelevance': [],
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
            'S': None
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
        elif self.gnostic_activation == 'linear':
            acti = 1  # Linear activation
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

    def fit(self, X, y):
        # Ensure y is a column vector (m, 1) to match predictions
        y = y.reshape(-1, 1)

        if self.weights is None:
            self.weights = np.zeros((X.shape[1], 1))
        if self.bias is None:
            self.bias = 0.0
        if self.gw is None:
            self.gw = np.ones_like(y).reshape(-1, 1)
        self.S = 1.0  # Initial scale parameter for gnostic weights

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
                gw_engine = GnosticsWeights()
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
            else:
                self.gw = np.ones_like(error)
                self.S = 1.0
                self.fi = np.ones_like(error)
                self.re = np.zeros_like(error)

                err = 0

            # 3. Matrix Weight Update:
            # We multiply the transpose of X by the error to get the total gradient
            # Update Rule: weights = weights + lr * (X_transpose @ error)
            if self.weights is None:
                self.weights = np.zeros((X.shape[1], 1))
            if self.bias is None:
                self.bias = 0.0

            # gnostic weight update
            gw_error = self.gw * error
            self.weights += self.lr * np.dot(X.T, gw_error)
            self.bias += self.lr * np.sum(gw_error)

            # Verbose logging            
            if self.verbose and self.gnostic_weights:
                print(f"Epoch {epoch+1}/{self.epochs}, Error: {np.sum(np.abs(error))}, Gnostic Error: {np.abs(err).sum()}, Mean Residual Entropy: {np.mean(self.re)}")
            elif self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, Error: {np.sum(np.abs(error))}")
            
            # Check for convergence
            # change in last n values of error vector is less than threshold
            # total sum of error vector is less than threshold or approach zero
            # if loss is increasing, stop training, or loss come to a plateau, stop training
            if self.early_stopping:
                STEPS = 5
                PROPERTY = self.gnostic_loss if self.gnostic_weights else 'loss'
                if self.gnostic_weights:
                    if (epoch > 1 and np.abs(self._history[PROPERTY][-1]).sum() >= np.abs(self._history[PROPERTY][-2]).sum()) or \
                    (epoch > STEPS and (np.abs(self._history[PROPERTY][-1]).sum() - np.abs(self._history[PROPERTY][-STEPS]).sum()) < self.threshold):
                        
                        if self.verbose:
                            print(f"Convergence reached at epoch {epoch+1}. Stopping training.")
                        break
                else:
                    if np.sum(np.abs(error)) < self.threshold:
                        if self.verbose:
                            print(f"Convergence reached at epoch {epoch+1}. Stopping training.")
                        break
            
            # fit status
            self._fitted = True

            # History tracking
            if self.history and self.gnostic_weights:
                self._history['loss'].append(np.sum(np.abs(gw_error)))
                self._history['fidelity'].append(np.sum(self.fi**2))
                self._history['irrelevance'].append(np.sum(self.hi**2))
                self._history['re'].append(np.sum(np.abs(self.re)))
                # self._history[self.gnostic_loss].append(np.abs(err).sum())
            elif self.history:
                self._history['loss'].append(np.sum(np.abs(error)))
                # other gnostic characteristics are not tracked when gnostic_weights is False
            
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
            
