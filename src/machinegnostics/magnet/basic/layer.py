class Layer:
    """
    Base class for every layer in the network.
    Subclasses must implement:
    forward(self, x, training=True) -> output
    backward(self, grad_output) -> grad_input
    Subclasses that own learnable parameters should populate
    self.params and self.grads dictionaries with matching keys,
    e.g. self.params = {"W": W, "b": b}, self.grads = {"W": dW, "b": db}.
    """
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.params = {} # e.g. {"W": ndarray, "b": ndarray}
        self.grads= {} # e.g. {"W": ndarray, "b": ndarray}
        self.trainable = True
        self._training= True
        self.input= None # cached for backward pass
        
    def forward(self, x, training=True):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError
    
    def get_params_and_grads(self):
        """Yield (param_array, grad_array) pairs for the optimizer."""
        for key in self.params:
            yield self.params[key], self.grads[key]

    def set_mode(self, training: bool):
        self._training= training

    def __call__(self, x, training=True):
        return self.forward(x, training=training)

    def __repr__(self):
        n_params = sum(p.size for p in self.params.values())
        return f"<{self.name}: {n_params} params>"