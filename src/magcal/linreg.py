'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.
'''

import numpy as np

class GnosticRobustRegression:
    """
    A class to perform Gnostic Robust Regression based on the iterative formula
    described in Equation 19.2 of the provided reference.
    """

    def __init__(self, X, y, criterion="E1", max_iter=100, tol=1e-6):
        """
        Initialize the regression model.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            The input feature matrix.
        - y: np.ndarray, shape (n_samples,)
            The target vector.
        - criterion: str
            The gnostic criterion to use (e.g., "Q1", "E1", "Q2", etc.).
        - max_iter: int
            Maximum number of iterations for the regression.
        - tol: float
            Tolerance for convergence.
        """
        self.X = X
        self.y = y
        self.criterion = criterion
        self.max_iter = max_iter
        self.tol = tol
        self.weights = np.ones(X.shape[0])  # Initialize weights to 1
        self.coefficients = None

    def _compute_filtering_weight(self, residuals):
        """
        Compute the filtering weights based on the selected gnostic criterion.

        Parameters:
        - residuals: np.ndarray, shape (n_samples,)
            The residuals of the current iteration.

        Returns:
        - weights: np.ndarray, shape (n_samples,)
            The updated filtering weights.
        """
        if self.criterion == "Q1":
            h_q = residuals ** 2
            f_q = 1 / (h_q + 1e-8)  # Avoid division by zero
            return f_q
        elif self.criterion == "E1":
            h_e = residuals ** 2
            f_e = 1 / (h_e + 1e-8)
            return f_e ** 2
        elif self.criterion == "Q2":
            return np.ones_like(residuals)  # Constant weight for Q2
        elif self.criterion == "E2":
            h_e = residuals ** 2
            f_e = 1 / (h_e + 1e-8)
            return f_e
        elif self.criterion == "Q3":
            h_q = residuals ** 2
            f_q = 1 / (h_q + 1e-8)
            return 1 / np.sqrt(f_q)
        elif self.criterion == "E3":
            h_e = residuals ** 2
            f_e = 1 / (h_e + 1e-8)
            return np.sqrt(f_e)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _update_weights(self, residuals):
        """
        Update the weights based on the residuals and the filtering function.

        Parameters:
        - residuals: np.ndarray, shape (n_samples,)
            The residuals of the current iteration.
        """
        self.weights = self._compute_filtering_weight(residuals)

    def fit(self):
        """
        Fit the regression model using the iterative weighted least squares approach.
        """
        n_samples, n_features = self.X.shape
        X_weighted = self.X.copy()
        y_weighted = self.y.copy()

        # Initialize coefficients
        self.coefficients = np.zeros(n_features)

        for iteration in range(self.max_iter):
            # Compute residuals
            residuals = self.y - np.dot(self.X, self.coefficients)

            # Update weights
            self._update_weights(residuals)

            # Apply weights to the design matrix and target vector
            W = np.sqrt(self.weights)[:, np.newaxis]
            X_weighted = W * self.X
            y_weighted = W.flatten() * self.y

            # Solve the weighted least squares problem
            new_coefficients = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)[0]

            # Check for convergence
            if np.linalg.norm(new_coefficients - self.coefficients) < self.tol:
                print(f"Converged in {iteration + 1} iterations.")
                break

            self.coefficients = new_coefficients
        else:
            print("Maximum iterations reached without convergence.")

    def predict(self, X):
        """
        Predict target values for the given input data.

        Parameters:
        - X: np.ndarray, shape (n_samples, n_features)
            The input feature matrix.

        Returns:
        - y_pred: np.ndarray, shape (n_samples,)
            The predicted target values.
        """
        return np.dot(X, self.coefficients)

    def get_weights(self):
        """
        Get the final weights after fitting the model.

        Returns:
        - weights: np.ndarray, shape (n_samples,)
            The final weights.
        """
        return self.weights

    def get_coefficients(self):
        """
        Get the fitted coefficients of the model.

        Returns:
        - coefficients: np.ndarray, shape (n_features,)
            The fitted coefficients.
        """
        return self.coefficients

