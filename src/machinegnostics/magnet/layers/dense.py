import numpy as np
from .base_layer import Layer
from ..tensor import Tensor

class Dense(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Xavier/Glorot Initialization for weights
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.w = Tensor(np.random.uniform(-limit, limit, (input_dim, output_dim)), name="weight")
        self.b = Tensor(np.zeros((output_dim,)), name="bias")
        
        self.params = [self.w, self.b]

    def forward(self, inputs):
        # Logic: Y = XW + B
        # Our Tensor class handles the grad tracking and broadcasting!
        return (inputs @ self.w) + self.b