"""MAGNET (Machine Gnostics Neural Networks) public package exports.

This is the flat user-facing MAGNET namespace. The package runs on a
hidden PyTorch backend, but the public API stays in MAGNET terms:
tensors, layers, activations, losses, optimizers, and runtime helpers.
"""

from .core import (
    Callback,
    EarlyStopping,
    History,
    Tensor,
    configure,
    get_runtime,
    get_torch_device,
    get_torch_dtype,
    to_numpy,
    to_torch,
    unbroadcast,
)
from .models import Model, Sequential, GnosticNeuron, get_model
from .initializers import (
    GlorotNormal,
    GlorotUniform,
    HeNormal,
    HeUniform,
    Initializer,
    Normal,
    Ones,
    RandomNormal,
    Uniform,
    XavierNormal,
    XavierUniform,
    Zeros,
    get_initializer,
)
from .activations import (
    Activation,
    ELU,
    Ei,
    Fi,
    Fj,
    Hi,
    Hj,
    LeakyReLU,
    ReLU,
    Sigmoid,
    Softmax,
    Softplus,
    Square,
    Step,
    Swish,
    Tanh,
    ei,
    fi,
    fj,
    get_activation,
    hi,
    hj,
)
from .losses import (
    BinaryCrossEntropy,
    FidelityLoss,
    ISSLoss,
    InfidelityLoss,
    InformationLoss,
    Loss,
    MSE,
    RSSLoss,
    ResidualEntropyLoss,
    fidelity_loss,
    get_loss,
    gnostic_characteristic_loss,
    gnostic_weighted_mse,
    gnostic_weighted_rmse,
    infidelity_loss,
    irrelevance_loss,
    relevance_loss,
)
from .optimizers import Adagrad, Adam, RMSprop, SGD, get_optimizer
from .layers import BatchNorm, Dense, Flatten, GnosticBatchNorm, Layer, iDense, jDense
from .activations.gn_activations import ActivationFunctions

__all__ = [
    'Tensor', 'configure', 'get_runtime', 'get_torch_device', 'get_torch_dtype',
    'to_numpy', 'to_torch', 'unbroadcast', 'History', 'get_initializer',
    'Initializer', 'GlorotUniform', 'GlorotNormal', 'HeUniform', 'HeNormal',
    'Normal', 'Uniform', 'Zeros', 'Ones', 'RandomNormal', 'XavierUniform',
    'XavierNormal', 'get_activation', 'fi', 'fj', 'hi', 'hj', 'ei', 'Activation',
    'ReLU', 'Step', 'LeakyReLU', 'ELU', 'Sigmoid', 'Softplus', 'Tanh', 'Swish',
    'Softmax', 'Square', 'Ei', 'Fi', 'Fj', 'Hi', 'Hj', 'get_loss', 'Loss', 'MSE',
    'BinaryCrossEntropy', 'FidelityLoss', 'InfidelityLoss', 'RSSLoss', 'ISSLoss',
    'ResidualEntropyLoss', 'InformationLoss', 'fidelity_loss', 'infidelity_loss',
    'irrelevance_loss', 'relevance_loss', 'gnostic_weighted_mse',
    'gnostic_weighted_rmse', 'gnostic_characteristic_loss', 'get_optimizer',
    'Adagrad', 'SGD', 'Adam', 'RMSprop', 'Callback', 'EarlyStopping', 'Layer',
    'Dense', 'iDense', 'jDense', 'BatchNorm', 'GnosticBatchNorm', 'Flatten',
    'Model', 'Sequential', 'ActivationFunctions', 'GnosticNeuron', 'get_model',
]
