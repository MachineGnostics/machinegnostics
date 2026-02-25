'''
Base Layer class for Magnet. All layers should inherit from this class.
Author: Nirmal Parmar
'''

class Layer:
    def __init__(self):
        self.params = []  # List of Tensors (Weights, Biases)
        self.training = True

    def __call__(self, inputs):
        # This makes Layer(inputs) work like a function
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError
    
    def get_params(self):
        return self.params