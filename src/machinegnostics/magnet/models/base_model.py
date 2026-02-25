'''
BaseModel: Common Model API for Magnet. Base class providing user-facing methods for saving and loading model weights. In future, may include additional common functionality for all models (e.g., device management, common training utilities). Currently focused on weight persistence via `save` and `load` methods.

Author: Nirmal Parmar
'''

import pickle

class BaseModel:
    """
    BaseModel: Common Model API for Magnet
    ======================================
    Provides user-facing methods for saving and loading model weights.

    Usage
    -----
    Subclass this in your model (e.g., Sequential) to enable:

        model.save("path.pkl")      # Save weights to file
        model.load("path.pkl")      # Load weights from file (in-place)

    Optionally, you can implement a classmethod `load_from` in subclasses for full model restoration.

    Methods
    -------
    save(path):
        Save model weights to a file.
    load(path):
        Load model weights from a file (in-place).
    """
    def save(self, path):
        """
        Save model weights to a file.

        Parameters
        ----------
        path : str
            Path to the file where weights will be saved (e.g., 'model_weights.pkl').

        Example
        -------
        >>> model = Sequential([...])
        >>> model.save('my_weights.pkl')

        Notes
        -----
        Only the model's parameters (weights) are saved. To restore weights, use `model.load(path)` on a model with the same architecture.
        """
        weights = [p.data.copy() for p in self.parameters()]
        with open(path, "wb") as f:
            pickle.dump(weights, f)

    def load(self, path):
        """
        Load model weights from a file (in-place).

        Parameters
        ----------
        path : str
            Path to the file from which weights will be loaded (e.g., 'model_weights.pkl').

        Example
        -------
        >>> model = Sequential([...])
        >>> model.load('my_weights.pkl')

        Notes
        -----
        The model architecture must match the saved weights. This method loads weights into the current model instance.
        """
        with open(path, "rb") as f:
            weights = pickle.load(f)
        for p, w in zip(self.parameters(), weights):
            p.data[...] = w
