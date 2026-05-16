"""
Gnostic Neuron Implementation

ManGo - Machine Gnostics Library
Copyright (C) 2026 Nirmal Parmar
"""

import numpy as np
from machinegnostics.magcal import GnosticsCharacteristics
from machinegnostics.magnet.engine.gnostic_wights import GnosticsWeights

class GnosticNeuron:
    def __init__(self, 
                 learning_rate=0.01, 
                 epochs=100,
                 verbose=False,
                 threshold=1e-5,
                 gnostic_wights=True,
                 gnostic_activation=True):
        self.W = None  # Weights will be initialized during training
        self.b = None
        self.gw = None  # Gnostic weights
        self.lr = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.threshold = threshold
        self.gnostic_activation = gnostic_activation
        self.gnostic_wights = gnostic_wights

    def _step_activation(self, Z):
        # Vectorized Step Function: applies to the entire array at once
        return np.where(Z >= 0, 1, 0)
    
    def _gnostic_activation(self, z):
        z0 = np.median(z)
        z_acti = np.exp(np.sum(z) - z0)
        chars = GnosticsCharacteristics(R=z_acti)
        q, q1 = chars._get_q_q1(S=self.S)
        acti = np.sum(chars._fi(q, q1))
        return acti * z

    def predict(self, X):
        """
        Forward Pass:
        X shape: (m, n)
        W shape: (n, 1)
        Z shape: (m, 1)
        """
        Z = X @ self.W + self.b  # Linear transformation
        
        if self.gnostic_activation:
            Z = self._gnostic_activation(Z)  # Gnostic activation
        else:
            Z = self._step_activation(Z)  # Standard step activation
        return Z

    def fit(self, X, y):
        # Ensure y is a column vector (m, 1) to match predictions
        y = y.reshape(-1, 1)
        if self.W is None:
            self.W = np.zeros((X.shape[1], 1))
        if self.b is None:
            self.b = 0.0
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
            if self.gnostic_wights:
                gw_engine = GnosticsWeights()
                self.gw = gw_engine._get_gnostic_weights(z_error)
                self.S = gw_engine._get_S_local()
                self.fi = gw_engine._get_fi()
                self.re = gw_engine._get_re()

            # gnostic error for convergence check
            err = (1/np.sum(self.fi )+ 1e-6)

            # 3. Matrix Weight Update:
            # We multiply the transpose of X by the error to get the total gradient
            # Update Rule: W = W + lr * (X_transpose @ error)
            if self.W is None:
                self.W = np.zeros((X.shape[1], 1))
            if self.b is None:
                self.b = 0.0

            # gnostic weight update
            gw_error = self.gw * error
            self.W += self.lr * np.dot(X.T, gw_error)
            self.b += self.lr * np.sum(gw_error)

            # Verbose logging            
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, Error: {np.sum(np.abs(error))}, Gnostic Error: {np.abs(err).sum()}, Fidelity: {np.mean(self.fi)}")
            
            # Check for convergence
            # change in last n values of error vector is less than threshold
            # total sum of error vector is less than threshold or approach zero
            if np.sum(np.abs(error)) < self.threshold or np.abs(err).sum() < self.threshold:
                if self.verbose:
                    print(f"Convergence reached at epoch {epoch+1}. Stopping training.")
                break