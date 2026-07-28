from .layer import Layer

class Flatten(Layer):
    """Reshapes (N, C, H, W) -> (N, C*H*W). Used between conv and dense blocks."""
    def forward(self, x, training=True):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)