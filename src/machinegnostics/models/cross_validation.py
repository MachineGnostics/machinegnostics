import numpy as np
import logging
from machinegnostics.magcal.util.logging import get_logger

class CrossValidator:
    """
    A custom implementation of k-Fold Cross-Validation for evaluating machine learning models.

    Parameters
    ----------
    model : object
        A machine learning model that implements `fit(X, y)` and `predict(X)` methods.
        
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,)
        Target labels.

    k : int, default=5
        Number of folds to use in cross-validation.

    shuffle : bool, default=True
        Whether to shuffle the dataset before splitting into folds.

    random_seed : int or None, default=None
        Seed used to shuffle the data. Ignored if `shuffle=False`.
    
    verbose : bool, default=False
        If True, enables detailed logging.

    Attributes
    ----------
    folds : list of tuple
        List of (train_indices, test_indices) for each fold.

    Example
    -------
    >>> from machinegnostics.models import CrossValidator
    >>> from machinegnostics.models import LinearRegressor
    >>> from sklearn.metrics import mean_squared_error
    >>> X = np.random.rand(100, 10)
    >>> y = np.random.rand(100)
    >>> model = LinearRegression()
    >>> cv = CrossValidator(model, X, y, k=5, shuffle=True, random_seed=42)
    >>> scores = cv.evaluate(mean_squared_error)
    >>> print("Cross-Validation Scores:", scores)
    >>> print("Mean Score:", np.mean(scores))
    """

    def __init__(self, model , X:np.ndarray, y:np.ndarray, k=5, shuffle=True, random_seed=None, verbose: bool = False):
        self.model = model
        self.X = np.array(X)
        self.y = np.array(y)
        self.k = k
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.verbose = verbose

        self.logger = get_logger('CrossValidator', level=logging.WARNING if not verbose else logging.INFO)

    def split(self):
        """
        Split the dataset into k folds.

        Returns
        -------
        folds : list of tuple
            A list of (train_indices, test_indices) for each fold.
        """
        self.logger.info("Starting k-Fold split...")
        n_samples = len(self.X)
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(indices)

        fold_sizes = np.full(self.k, n_samples // self.k, dtype=int)
        fold_sizes[:n_samples % self.k] += 1

        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            folds.append((train_idx, test_idx))
            current = stop
        self.logger.info("Completed k-Fold split.")
        return folds

    def evaluate(self, scoring_func):
        """
        Perform k-fold cross-validation and return the evaluation scores.

        Parameters
        ----------
        scoring_func : callable
            A function that takes `y_true` and `y_pred` and returns a numeric score (e.g., accuracy_score).

        Returns
        -------
        scores : list of float
            Evaluation scores for each fold.
        """
        self.logger.info("Starting cross-validation evaluation...")
        scores = []
        for train_idx, test_idx in self.split():
            X_train, y_train = self.X[train_idx], self.y[train_idx]
            X_test, y_test = self.X[test_idx], self.y[test_idx]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            score = scoring_func(y_test, y_pred)
            scores.append(score)
        self.logger.info("Completed cross-validation evaluation.")
        return scores
