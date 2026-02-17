"""
Base layer interface, aligned with project's logging style.
"""
import numpy as np
import logging
from machinegnostics.magcal.util.logging import get_logger

class BaseLayer:
    def __init__(self, name: str = None, neuron_type: str = 'E'):
        self.name = name or self.__class__.__name__
        self.built = False
        self.params = {}
        self.grads = {}
        self.neuron_type = neuron_type  # 'E' or 'Q'
        self.logger = get_logger(self.__class__.__name__, logging.WARNING)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.built = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError

    def get_params(self):
        return self.params

    def get_grads(self):
        return self.grads
