import numpy as np
from itertools import combinations_with_replacement
from machinegnostics.magcal import RegressorBase, DataConversion, GnosticsWeights, GnosticsCharacteristics, ScaleParam

class _LogisticRegressorParamBase(RegressorBase):
    '''
    Base class for Gnostic Logistic Regression (binary classification).
    '''
    def __init__(self,
                 degree: int = 1,
                 max_iter: int = 100,
                 tol: float = 1e-8,
                 verbose: bool = False):
        super().__init__()
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.coefficients = None
        self.weights = None
        self._history = []

    def _generate_polynomial_features(self, X):
        n_samples, n_features = X.shape
        combinations = []
        for degree in range(self.degree + 1):
            combinations += list(combinations_with_replacement(range(n_features), degree))
        X_poly = np.ones((n_samples, len(combinations)))
        for i, comb in enumerate(combinations):
            X_poly[:, i] = np.prod(X[:, comb], axis=1)
        return X_poly

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _gnostic_prob(self, z):
        dc = DataConversion()
        zz = dc._convert_az(z)
        gc = GnosticsCharacteristics(zz)
        scale = ScaleParam()

        # q, q1
        q, q1 = gc._get_q_q1()
        h = gc._hi(q, q1)
        fi = gc._fi(q, q1)
        s = scale._gscale_loc(np.mean(fi))
        q, q1 = gc._get_q_q1(S=s)
        h = gc._hi(q, q1)
        p = gc._idistfun(h)
        return p

    def _process_input(self, X, y=None):
        import numpy as np
        is_pandas = False
        try:
            import pandas as pd
            if isinstance(X, (pd.DataFrame, pd.Series)):
                is_pandas = True
        except ImportError:
            pass
        if is_pandas:
            X_np = X.to_numpy()
        else:
            X_np = np.asarray(X)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        elif X_np.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")
        if y is not None:
            if is_pandas:
                y_np = y.to_numpy().flatten()
            else:
                y_np = np.asarray(y).flatten()
            if y_np.ndim != 1:
                raise ValueError("y must be a 1D array of shape (n_samples,).")
            if X_np.shape[0] != y_np.shape[0]:
                raise ValueError(f"Number of samples in X and y must match. Got {X_np.shape[0]} and {y_np.shape[0]}.")
            return X_np, y_np
        else:
            return X_np

    def fit(self, X, y):
        X, y = self._process_input(X, y)
        self.y = y
        X_poly = self._generate_polynomial_features(X)
        n_samples, n_features = X_poly.shape
        self.coefficients = np.zeros(n_features)
        self.weights = np.ones(n_samples)
        self.log_loss_history = []
        # max float
        # prev_log_loss = np.finfo(np.float).max
        prev_log_loss = np.finfo(np.float64).max

        for it in range(self.max_iter):
            prev_coef = self.coefficients.copy()
            linear_pred = X_poly @ self.coefficients
            # p = self._sigmoid(linear_pred)
            zz = linear_pred - y
            # p = self._gnostic_prob(linear_pred)  # Use gnostic probability instead of sigmoid
            p = self._sigmoid(linear_pred)  # Apply sigmoid to get probabilities
            # W = np.diag(p * (1 - p))
            # Gnostic-style weights 
            dc = DataConversion()
            z = dc._convert_az(zz)
            gw = GnosticsWeights()
            gw = gw._get_gnostic_weights(z)
            sample_weights = self.weights * gw
            W = np.diag(sample_weights)
            # IRLS update
            try:
                XtW = X_poly.T @ W
                XtWX = XtW @ X_poly + 1e-8 * np.eye(n_features)
                XtWy = XtW @ (linear_pred + (y - p) / (p * (1 - p) + 1e-8))
                self.coefficients = np.linalg.solve(XtWX, XtWy)
            except np.linalg.LinAlgError:
                self.coefficients = np.linalg.pinv(XtWX) @ XtWy

            # --- Log loss calculation ---
            proba_pred = np.clip(self._gnostic_prob(y - (X_poly @ self.coefficients)), 1e-8, 1-1e-8)
            log_loss = -np.mean(y * np.log(proba_pred) + (1 - y) * np.log(1 - proba_pred))
            self.log_loss_history.append(log_loss)

            # Convergence check: coefficients and log loss
            coef_converged = np.all(np.abs(self.coefficients - prev_coef) < self.tol)
            logloss_converged = np.abs(log_loss - prev_log_loss) < self.tol
            if coef_converged or logloss_converged:
                if self.verbose:
                    print(f"Converged at iteration {it+1}: coef change={np.max(np.abs(self.coefficients - prev_coef)):.4e}, log loss change={np.abs(log_loss - prev_log_loss):.4e}")
                break
            prev_log_loss = log_loss

            # Verbose output
            if self.verbose:
                print(f"Iteration {it+1}, coef change: {np.max(np.abs(self.coefficients - prev_coef)):.4e}, log loss: {log_loss:.6f}")

        return self

    def predict_proba(self, X):
        if self.coefficients is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict_proba'.")
        X = self._process_input(X)
        X_poly = self._generate_polynomial_features(X)
        linear_pred = X_poly @ self.coefficients
        gproba = self._gnostic_prob(linear_pred)  # Only use linear_pred!
        proba = self._sigmoid(linear_pred)  # Apply sigmoid to get probabilities
        return proba

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)