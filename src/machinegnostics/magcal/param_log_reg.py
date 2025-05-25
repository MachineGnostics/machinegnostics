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
                 verbose: bool = False,
                 scale: [str, float, int] = 'auto', # if auto then automatically select scale based on the data else user can give float value between 0 to 2
                 early_stopping: bool = True,
                 history: bool = True,
                 proba: str = 'gnostic',
                 data_form:str = 'a'):
        super().__init__()
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.coefficients = None
        self.weights = None
        self.scale = scale
        self.early_stopping = early_stopping
        self.proba = proba
        self.data_form = data_form

        if self.proba not in ['gnostic', 'sigmoid']:
            raise ValueError("proba must be either 'gnostic' or 'sigmoid'.")
        # degree check
        if not isinstance(self.degree, int) or self.degree < 0:
            raise ValueError("degree must be a non-negative integer.")
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        # --- Scale input handling ---
        if isinstance(scale, str):
            if scale != 'auto':
                raise ValueError("scale must be 'auto' or a float between 0 and 2.")
            self.scale_value = 'auto'
        elif isinstance(scale, (int, float)):
            if not (0 <= scale <= 2):
                raise ValueError("scale must be 'auto' or a float between 0 and 2.")
            self.scale_value = float(scale)
        else:
            raise ValueError("scale must be 'auto' or a float between 0 and 2.")
        # data form check additive or multiplicative
        if self.data_form not in ['a', 'm']:
            raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative.")
        
        # history option
        if history:
            self._history = []
            # default history content
            self._history.append({
                'iteration': 0,
                'log_loss': None,
                'coefficients': None,
                'information': None,
                'rentropy': None,
                'converged': False
            })
        else:
            self._history = None

    def _early_stopping_check(self, rentropy):
        '''
        First check and give priority to residual entropy (rentropy) change.

        if rentropy increasing then it's previous values, then stop the iteration and it is converged.
        '''
        if self._history:
            prev_rentropy = self._history[-1]['rentropy']
            if prev_rentropy is not None and np.mean(rentropy) > np.mean(prev_rentropy):
                return True
        return False
        

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
        if self.data_form == 'a':
            zz = dc._convert_az(z)
        elif self.data_form == 'm':
            zz = dc._convert_mz(z)
        else:
            raise ValueError("data_form must be 'a' for additive or 'm' for multiplicative.")
        gc = GnosticsCharacteristics(zz)

        # q, q1
        q, q1 = gc._get_q_q1()
        h = gc._hi(q, q1)
        fi = gc._fi(q, q1)

        # Scale handling
        if self.scale_value == 'auto':
            scale = ScaleParam()
            s = scale._gscale_loc(np.mean(fi))
        else:
            s = self.scale_value
            
        q, q1 = gc._get_q_q1(S=s)
        h = gc._hi(q, q1)
        fi = gc._fi(q, q1)
        fj = gc._fj(q, q1)
        p = gc._idistfun(h)
        info = gc._info_i(p)
        re = gc._rentropy(fi, fj)
        return p, info, re

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
        self._history = []
        prev_log_loss = np.finfo(np.float64).max

        for it in range(self.max_iter):
            prev_coef = self.coefficients.copy()
            linear_pred = X_poly @ self.coefficients
            zz = linear_pred - y
            # Gnostic-style weights 
            dc = DataConversion()
            z = dc._convert_az(zz)
            gw = GnosticsWeights()
            gw = gw._get_gnostic_weights(z)
            sample_weights = self.weights * gw
            W = np.diag(sample_weights)

            # gnostic probabilities
            if self.proba == 'gnostic':
                # Gnostic probability calculation
                p, info, re = self._gnostic_prob(z)
            elif self.proba == 'sigmoid':
                # Sigmoid probability calculation
                p = self._sigmoid(linear_pred)
                _, info, re = self._gnostic_prob(z)

            # IRLS update
            try:
                XtW = X_poly.T @ W
                XtWX = XtW @ X_poly + 1e-8 * np.eye(n_features)
                XtWy = XtW @ (linear_pred + (y - p) / (p * (1 - p) + 1e-8))
                self.coefficients = np.linalg.solve(XtWX, XtWy)
            except np.linalg.LinAlgError:
                self.coefficients = np.linalg.pinv(XtWX) @ XtWy

            # --- Log loss calculation ---
            proba_pred = np.clip(p, 1e-8, 1-1e-8)
            log_loss = -np.mean(y * np.log(proba_pred) + (1 - y) * np.log(1 - proba_pred))
            
            # history update for gnostic vs sigmoid
            re = np.mean(re)
            info = np.mean(info)

            self._history.append({
                'iteration': it + 1,
                'log_loss': log_loss,
                'coefficients': self.coefficients.copy(),
                'information': info,
                'rentropy': re,
            })
            
            # --- Unified convergence check: stop if mean rentropy change is within tolerance ---
            if it > 0 and self.early_stopping:
                prev_re = self._history[-2]['rentropy'] if len(self._history) > 1 else None
                curr_re = np.mean(re)
                prev_re_val = np.mean(prev_re) if prev_re is not None else None
                if prev_re_val is not None and np.abs(curr_re - prev_re_val) < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {it + 1} (early stop): mean rentropy change below tolerance.")
                    break
            if self.verbose:
                print(f"Iteration {it + 1}, Log Loss: {log_loss:.6f}, mean residual entropy: {np.mean(re):.6f}")

           
        return self

    def predict_proba(self, X):
        if self.coefficients is None:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict_proba'.")
        X = self._process_input(X)
        X_poly = self._generate_polynomial_features(X)
        linear_pred = X_poly @ self.coefficients
        # gnostic vs sigmoid probability calculation
        if self.proba == 'gnostic':
            # Gnostic probability calculation
            proba, info, re = self._gnostic_prob(-linear_pred)
        elif self.proba == 'sigmoid':
            # Sigmoid probability calculation
            proba = self._sigmoid(linear_pred)
        else:
            raise ValueError("Invalid probability method. Must be 'gnostic' or 'sigmoid'.")
        return proba

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)