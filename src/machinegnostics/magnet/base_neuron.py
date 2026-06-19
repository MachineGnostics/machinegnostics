"""
Gnostic Neuron Implementation
Base Class

ManGo - Machine Gnostics Library
Copyright (C) 2026 Nirmal Parmar
"""

from abc import ABCMeta, abstractmethod
import logging
import os
import joblib
from machinegnostics.magcal.util.logging import get_logger


# base class for gnostic neuron
class BaseGnosticNeuron(metaclass=ABCMeta):
    """
    Abstract base class for gnostic neurons.

    Abstract Methods:
    ----------------

    - fit(X, y)

    - predict(X)
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        # logger
        self.logger = get_logger(self.__class__.__name__, logging.DEBUG if self.verbose else logging.WARNING)  # Create a logger for this class
        self.logger.info(f"{self.__class__.__name__} initialized.")

    # @abstractmethod
    # def fit(self, X, y):
    #     """
    #     Fit the gnostic neuron to the data.
    #     """
    #     pass

    # @abstractmethod
    # def predict(self, X):
    #     """
    #     Predict using the fitted gnostic neuron.
    #     """
    #     pass

    # @abstractmethod
    # def score(self, X, y):
    #     """
    #     Compute the score of the gnostic neuron.
    #     """
    #     pass

    # @classmethod
    # def load(cls, path:str):
    #     """Load a trained gnostic neuron instance from disk using joblib.

    #     Parameters:
    #     - path: str
    #         Directory containing the saved model artifact.

    #     Returns:
    #     - BaseGnosticNeuron
    #         An instance of `cls` loaded from `path/model.pkl`.

    #     Raises:
    #     - FileNotFoundError
    #         If `path/model.pkl` does not exist.
    #     - Exception
    #         Any exception propagated by `joblib.load` (e.g., deserialization errors).

    #     Notes:
    #     - The model is stored as a single pickle file named "model.pkl".
    #     - Use `save(path)` on a trained model to create this artifact.
    #     - Compatible with subclasses of `BaseGnosticNeuron`; this method restores the full object state.

    #     Examples:
    #     >>> # Save and load a trained gnostic neuron
    #     >>> model = GnosticNeuron()
    #     >>> model.fit(X, y)
    #     >>> model.save("/tmp/my_model")
    #     >>> loaded = GnosticNeuron.load("/tmp/my_model")
    #     >>> isinstance(loaded, GnosticNeuron)
    #     True
    #     """
    #     return joblib.load(os.path.join(path, "model.pkl"))
        
    def save(self, path):
        """Persist the trained gnostic neuron instance to disk using joblib.

        Parameters:
        - path: str
            Directory where the artifact will be written. Created if it does not exist.

        Creates:
        - path/model.pkl
            A single pickle file containing the entire gnostic neuron instance.

        Raises:
        - Exception
            Any exception propagated by filesystem operations or `joblib.dump`.

        Notes:
        - Overwrites an existing `model.pkl` if present.
        - Ensures `path` exists (`os.makedirs(path, exist_ok=True)`).
        - Use `load(path)` to restore the saved instance later.

        Examples:
        >>> model = GnosticNeuron()
        >>> model.fit(X, y)
        >>> model.save("/tmp/my_model")
        >>> # Later or elsewhere
        >>> loaded = GnosticNeuron.load("/tmp/my_model")
        >>> isinstance(loaded, GnosticNeuron)
        True
        """
        self.logger.info(f"Saving gnostic neuron to {path}.")
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, "model.pkl"))

    def get_params(self):
        """Return a dictionary of the gnostic neuron's public parameters.

        Returns:
        - dict
            A shallow copy of the instance's public attributes (keys that do not
            start with an underscore). Typical entries include configuration
            options (e.g., `degree`, `max_iter`, `tol`), flags (e.g.,
            `early_stopping`, `verbose`), and any learned attributes that are
            stored as public fields.

        Notes:
        - Private/internal attributes (prefixed with `_`) are excluded.
        - This method is intended for inspection, logging, or serialization of
          configuration; it does not guarantee only hyperparameters are returned
          if your subclass stores learned values in public fields.
        - For stable export/import, prefer using `save()`/`load()` which persist
          the entire object.

        Examples:
        >>> model = GnosticNeuron()
        >>> model.fit(X, y)
        >>> params = model.get_params()
        >>> isinstance(params, dict)
        True
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}