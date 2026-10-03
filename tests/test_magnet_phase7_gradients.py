"""Phase 7 numerical gradient checks for MAGNET."""

from __future__ import annotations

from typing import Type

import numpy as np

from machinegnostics.magnet import (
    Dense,
    Ei,
    Fi,
    Fj,
    FidelityLoss,
    Hi,
    Hj,
    ISSLoss,
    InfidelityLoss,
    InformationLoss,
    RSSLoss,
    ResidualEntropyLoss,
    Sequential,
    Tensor,
    configure,
)
from machinegnostics.magnet.activations.base import Activation
from machinegnostics.magnet.losses.base import Loss


GRADIENT_EPS = 1e-6
RELATIVE_TOLERANCE = 1e-4


def _configure_gradient_runtime() -> None:
    """Use CPU float64 to make finite-difference comparisons reliable."""
    configure(device="cpu", dtype="float64", seed=17, deterministic=True)
    np.random.seed(17)


def _relative_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute the maximum elementwise relative error."""
    denominator = np.maximum(1e-12, np.abs(reference) + np.abs(candidate))
    return float(np.max(np.abs(reference - candidate) / denominator))


def _scale_to_raw(scale: float) -> float:
    bounded = np.clip(scale / 2.0, 1e-6, 1.0 - 1e-6)
    return float(np.log(bounded / (1.0 - bounded)))


def _raw_to_scale(raw: float) -> float:
    return float(2.0 / (1.0 + np.exp(-raw)))


def _activation_scalar_output(
    activation_type: Type[Activation],
    inputs: np.ndarray,
    scale: float,
    center: float,
) -> float:
    """Evaluate the sum of one activation's outputs for finite differences."""
    layer = activation_type(initial_S=scale, initial_z0=center)
    return float(layer(Tensor(inputs)).sum())


def _autograd_for_activation(activation_type: Type[Activation]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute autograd gradients for one gnostic activation."""
    layer = activation_type(initial_S=1.3, initial_z0=-0.2)
    inputs = Tensor(np.array([-0.7, 0.1, 1.2], dtype=np.float64), requires_grad=True)
    output = layer(inputs).sum()
    output.backward()
    return inputs.grad.copy(), layer.S.grad.copy(), layer.z0.grad.copy()


def _numerical_input_gradient(activation_type: Type[Activation]) -> np.ndarray:
    """Approximate the activation gradient with respect to input values."""
    base_inputs = np.array([-0.7, 0.1, 1.2], dtype=np.float64)
    numerical_gradient = np.zeros_like(base_inputs)
    for index in range(base_inputs.size):
        plus_inputs = base_inputs.copy()
        minus_inputs = base_inputs.copy()
        plus_inputs[index] += GRADIENT_EPS
        minus_inputs[index] -= GRADIENT_EPS
        plus_value = _activation_scalar_output(activation_type, plus_inputs, 1.3, -0.2)
        minus_value = _activation_scalar_output(activation_type, minus_inputs, 1.3, -0.2)
        numerical_gradient[index] = (plus_value - minus_value) / (2.0 * GRADIENT_EPS)
    return numerical_gradient


def _numerical_scale_gradient(activation_type: Type[Activation]) -> np.ndarray:
    """Approximate the gradient with respect to the raw gated scale parameter."""
    base_inputs = np.array([-0.7, 0.1, 1.2], dtype=np.float64)
    raw_scale = _scale_to_raw(1.3)
    plus_value = _activation_scalar_output(activation_type, base_inputs, _raw_to_scale(raw_scale + GRADIENT_EPS), -0.2)
    minus_value = _activation_scalar_output(activation_type, base_inputs, _raw_to_scale(raw_scale - GRADIENT_EPS), -0.2)
    return np.array([(plus_value - minus_value) / (2.0 * GRADIENT_EPS)], dtype=np.float64)


def _numerical_center_gradient(activation_type: Type[Activation]) -> np.ndarray:
    """Approximate the activation gradient with respect to its center parameter."""
    base_inputs = np.array([-0.7, 0.1, 1.2], dtype=np.float64)
    plus_value = _activation_scalar_output(activation_type, base_inputs, 1.3, -0.2 + GRADIENT_EPS)
    minus_value = _activation_scalar_output(activation_type, base_inputs, 1.3, -0.2 - GRADIENT_EPS)
    return np.array([(plus_value - minus_value) / (2.0 * GRADIENT_EPS)], dtype=np.float64)


def _loss_scalar_output(loss_type: Type[Loss], predictions: np.ndarray) -> float:
    targets = np.array([[0.2, -0.1], [0.4, 0.3]], dtype=np.float64)
    loss = loss_type(S=1.25)(Tensor(predictions), Tensor(targets))
    return float(loss)


def _autograd_for_loss(loss_type: Type[Loss]) -> np.ndarray:
    predictions = Tensor(np.array([[0.5, -0.3], [0.1, 0.9]], dtype=np.float64), requires_grad=True)
    targets = Tensor(np.array([[0.2, -0.1], [0.4, 0.3]], dtype=np.float64))
    loss = loss_type(S=1.25)(predictions, targets)
    loss.backward()
    return predictions.grad.copy()


def _numerical_prediction_gradient(loss_type: Type[Loss]) -> np.ndarray:
    base_predictions = np.array([[0.5, -0.3], [0.1, 0.9]], dtype=np.float64)
    numerical_gradient = np.zeros_like(base_predictions)
    for index in np.ndindex(base_predictions.shape):
        plus_predictions = base_predictions.copy()
        minus_predictions = base_predictions.copy()
        plus_predictions[index] += GRADIENT_EPS
        minus_predictions[index] -= GRADIENT_EPS
        plus_value = _loss_scalar_output(loss_type, plus_predictions)
        minus_value = _loss_scalar_output(loss_type, minus_predictions)
        numerical_gradient[index] = (plus_value - minus_value) / (2.0 * GRADIENT_EPS)
    return numerical_gradient


def test_all_gnostic_activation_input_gradients_match_finite_difference() -> None:
    """Fi, Hi, Fj, Hj, and Ei input gradients should match central differences."""
    _configure_gradient_runtime()
    for activation_type in (Fi, Hi, Fj, Hj, Ei):
        autograd_input, _, _ = _autograd_for_activation(activation_type)
        numerical_input = _numerical_input_gradient(activation_type)
        assert _relative_error(autograd_input, numerical_input) < RELATIVE_TOLERANCE, activation_type.__name__


def test_all_gnostic_activation_scale_gradients_match_finite_difference() -> None:
    """Every gnostic activation should produce correct raw-scale gradients."""
    _configure_gradient_runtime()
    for activation_type in (Fi, Hi, Fj, Hj, Ei):
        _, autograd_scale, _ = _autograd_for_activation(activation_type)
        numerical_scale = _numerical_scale_gradient(activation_type)
        assert _relative_error(autograd_scale, numerical_scale) < RELATIVE_TOLERANCE, activation_type.__name__


def test_all_gnostic_activation_center_gradients_match_finite_difference() -> None:
    """Every gnostic activation should produce correct gradients for its center z0."""
    _configure_gradient_runtime()
    for activation_type in (Fi, Hi, Fj, Hj, Ei):
        _, _, autograd_center = _autograd_for_activation(activation_type)
        numerical_center = _numerical_center_gradient(activation_type)
        assert _relative_error(autograd_center, numerical_center) < RELATIVE_TOLERANCE, activation_type.__name__


def test_all_gnostic_loss_prediction_gradients_match_finite_difference() -> None:
    """All gnostic loss functions should expose correct prediction gradients."""
    _configure_gradient_runtime()
    for loss_type in (FidelityLoss, InfidelityLoss, RSSLoss, ISSLoss, ResidualEntropyLoss, InformationLoss):
        autograd_gradient = _autograd_for_loss(loss_type)
        numerical_gradient = _numerical_prediction_gradient(loss_type)
        assert _relative_error(autograd_gradient, numerical_gradient) < RELATIVE_TOLERANCE, loss_type.__name__


def test_dense_weight_and_bias_gradients_match_finite_difference() -> None:
    """Dense layer W and b gradients should agree with finite differences."""
    _configure_gradient_runtime()
    inputs = np.array([[0.2, -0.4], [0.5, 1.2]], dtype=np.float64)
    weights = np.array([[0.3, -0.2, 0.1], [0.4, 0.7, -0.5]], dtype=np.float64)
    bias = np.array([0.1, -0.3, 0.2], dtype=np.float64)

    def scalar_output(test_weights: np.ndarray, test_bias: np.ndarray) -> float:
        layer = Dense(2, 3)
        layer.W.data = test_weights
        layer.b.data = test_bias
        return float(layer(Tensor(inputs)).sum())

    layer = Dense(2, 3)
    layer.W.data = weights
    layer.b.data = bias
    output = layer(Tensor(inputs)).sum()
    output.backward()

    numerical_weight_gradient = np.zeros_like(weights)
    for row in range(weights.shape[0]):
        for column in range(weights.shape[1]):
            plus_weights = weights.copy()
            minus_weights = weights.copy()
            plus_weights[row, column] += GRADIENT_EPS
            minus_weights[row, column] -= GRADIENT_EPS
            plus_value = scalar_output(plus_weights, bias)
            minus_value = scalar_output(minus_weights, bias)
            numerical_weight_gradient[row, column] = (plus_value - minus_value) / (2.0 * GRADIENT_EPS)

    numerical_bias_gradient = np.zeros_like(bias)
    for index in range(bias.size):
        plus_bias = bias.copy()
        minus_bias = bias.copy()
        plus_bias[index] += GRADIENT_EPS
        minus_bias[index] -= GRADIENT_EPS
        plus_value = scalar_output(weights, plus_bias)
        minus_value = scalar_output(weights, minus_bias)
        numerical_bias_gradient[index] = (plus_value - minus_value) / (2.0 * GRADIENT_EPS)

    assert _relative_error(layer.W.grad, numerical_weight_gradient) < RELATIVE_TOLERANCE
    assert _relative_error(layer.b.grad, numerical_bias_gradient) < RELATIVE_TOLERANCE


def test_dense_activation_stack_parameter_gradients_match_finite_difference() -> None:
    """A Dense+Fi stack should expose correct W, b, S_raw, and z0 gradients together."""
    _configure_gradient_runtime()
    inputs = np.array([[0.1, -0.2], [0.3, 0.7]], dtype=np.float64)
    weights = np.array([[0.4], [-0.6]], dtype=np.float64)
    bias = np.array([0.2], dtype=np.float64)
    scale = 1.1
    center = -0.15

    def scalar_output(test_weights: np.ndarray, test_bias: np.ndarray, test_scale: float, test_center: float) -> float:
        model = Sequential([Dense(2, 1), Fi(initial_S=test_scale, initial_z0=test_center)])
        dense_layer = model.layers[0]
        dense_layer.W.data = test_weights
        dense_layer.b.data = test_bias
        return float(model(Tensor(inputs)).sum())

    model = Sequential([Dense(2, 1), Fi(initial_S=scale, initial_z0=center)])
    dense_layer = model.layers[0]
    activation_layer = model.layers[1]
    dense_layer.W.data = weights
    dense_layer.b.data = bias

    output = model(Tensor(inputs)).sum()
    output.backward()

    plus_weights = weights.copy()
    minus_weights = weights.copy()
    plus_weights[0, 0] += GRADIENT_EPS
    minus_weights[0, 0] -= GRADIENT_EPS
    numerical_weight = np.array(
        [
            (scalar_output(plus_weights, bias, scale, center) - scalar_output(minus_weights, bias, scale, center))
            / (2.0 * GRADIENT_EPS)
        ]
    )

    plus_bias = bias.copy()
    minus_bias = bias.copy()
    plus_bias[0] += GRADIENT_EPS
    minus_bias[0] -= GRADIENT_EPS
    numerical_bias = np.array(
        [
            (scalar_output(weights, plus_bias, scale, center) - scalar_output(weights, minus_bias, scale, center))
            / (2.0 * GRADIENT_EPS)
        ]
    )

    raw_scale = _scale_to_raw(scale)
    numerical_scale = np.array(
        [
            (
                scalar_output(weights, bias, _raw_to_scale(raw_scale + GRADIENT_EPS), center)
                - scalar_output(weights, bias, _raw_to_scale(raw_scale - GRADIENT_EPS), center)
            )
            / (2.0 * GRADIENT_EPS)
        ]
    )
    numerical_center = np.array(
        [
            (scalar_output(weights, bias, scale, center + GRADIENT_EPS) - scalar_output(weights, bias, scale, center - GRADIENT_EPS))
            / (2.0 * GRADIENT_EPS)
        ]
    )

    assert _relative_error(np.array([dense_layer.W.grad[0, 0]]), numerical_weight) < RELATIVE_TOLERANCE
    assert _relative_error(np.array([dense_layer.b.grad[0]]), numerical_bias) < RELATIVE_TOLERANCE
    assert _relative_error(activation_layer.S.grad, numerical_scale) < RELATIVE_TOLERANCE
    assert _relative_error(activation_layer.z0.grad, numerical_center) < RELATIVE_TOLERANCE
