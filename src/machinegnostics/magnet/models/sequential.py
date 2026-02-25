'''
Sequential model class for Magnet. This is a simple container that allows stacking layers in order.

Author: Nirmal Parmar
'''

class Sequential:
    def __init__(self, layers=None):
        self.layers = layers if layers else []

    def add(self, layer):
        """Allows Keras-style model.add(Dense(...))"""
        self.layers.append(layer)

    def __call__(self, x):
        """The forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def parameters(self):
        """
        Collects all Tensors from all layers so the 
        optimizer can find them.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params
    
    def zero_grad(self):
        """Resets gradients for all parameters before a new batch."""
        for p in self.parameters():
            p.grad.fill(0)