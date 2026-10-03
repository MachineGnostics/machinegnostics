"""Comprehensive Phase 7 integration tests for MAGNET."""

from __future__ import annotations

from typing import Type

import numpy as np

from machinegnostics.magnet import (
    Adagrad,
    Adam,
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
    MSE,
    RMSprop,
    RSSLoss,
    ResidualEntropyLoss,
    SGD,
    Sequential,
    configure,
)
from machinegnostics.magnet.activations.base import Activation
from machinegnostics.magnet.losses.base import Loss
from machinegnostics.magnet.optimizers.base import Optimizer


XOR_INPUTS = np.array(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    dtype=np.float64,
)
XOR_TARGETS = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float64)


def _configure_training_runtime() -> None:
    """Select a deterministic CPU runtime for training-focused tests."""
    configure(device="cpu", dtype="float32", seed=7, deterministic=True)
    np.random.seed(7)


def _make_two_moons(n_samples: int = 120, noise: float = 0.08, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic two-moons dataset without external dependencies."""
    rng = np.random.default_rng(seed)
    outer_count = n_samples // 2
    inner_count = n_samples - outer_count

    outer_theta = np.linspace(0.0, np.pi, outer_count)
    inner_theta = np.linspace(0.0, np.pi, inner_count)

    outer_arc = np.column_stack([np.cos(outer_theta), np.sin(outer_theta)])
    inner_arc = np.column_stack([1.0 - np.cos(inner_theta), 1.0 - np.sin(inner_theta) - 0.5])

    features = np.vstack([outer_arc, inner_arc])
    features += rng.normal(scale=noise, size=features.shape)
    labels = np.vstack([np.zeros((outer_count, 1)), np.ones((inner_count, 1))])
    return features.astype(np.float64), labels.astype(np.float64)


def _build_binary_model(hidden_activation: Type[Activation]) -> Sequential:
    """Build a small sequential network with a gnostic hidden activation."""
    return Sequential(
        [
            Dense(2, 6),
            hidden_activation(initial_S=1.0),
            Dense(6, 1),
            Fi(initial_S=1.0),
        ]
    )


def _train_xor_model(hidden_activation: Type[Activation]) -> tuple[float, float, np.ndarray]:
    """Train a single-hidden-layer model on XOR and return training facts."""
    _configure_training_runtime()
    model = _build_binary_model(hidden_activation)
    model.compile(loss=MSE(), optimizer=Adam(lr=0.03))
    initial_loss = model.evaluate(XOR_INPUTS, XOR_TARGETS)
    history = model.fit(XOR_INPUTS, XOR_TARGETS, epochs=100, batch_size=4, shuffle=True)
    predictions = model.predict(XOR_INPUTS)
    return initial_loss, history["loss"][-1], predictions


def _assert_xor_pattern(predictions: np.ndarray) -> None:
    """Assert that XOR negatives stay low and positives stay high."""
    negatives = predictions[[0, 3], 0]
    positives = predictions[[1, 2], 0]
    assert np.all(negatives < 0.1)
    assert np.all(positives > 0.9)


def test_sequential_with_fi_reduces_loss_on_xor() -> None:
    """Fi hidden activations should learn the XOR pattern end-to-end."""
    initial_loss, final_loss, predictions = _train_xor_model(Fi)
    assert final_loss < initial_loss * 0.01
    _assert_xor_pattern(predictions)


def test_sequential_with_hi_reduces_loss_on_xor() -> None:
    """Hi hidden activations should integrate correctly during XOR training."""
    initial_loss, final_loss, predictions = _train_xor_model(Hi)
    assert final_loss < initial_loss * 0.01
    _assert_xor_pattern(predictions)


def test_sequential_with_fj_reduces_loss_on_xor() -> None:
    """Fj hidden activations should learn XOR while preserving stable training."""
    initial_loss, final_loss, predictions = _train_xor_model(Fj)
    assert final_loss < initial_loss * 0.01
    _assert_xor_pattern(predictions)


def test_sequential_with_hj_reduces_loss_on_xor() -> None:
    """Hj hidden activations should learn XOR without breaking backpropagation."""
    initial_loss, final_loss, predictions = _train_xor_model(Hj)
    assert final_loss < initial_loss * 0.01
    _assert_xor_pattern(predictions)


def test_sequential_with_ei_reduces_loss_on_xor() -> None:
    """Ei hidden activations should learn XOR in a realistic training loop."""
    initial_loss, final_loss, predictions = _train_xor_model(Ei)
    assert final_loss < initial_loss * 0.01
    _assert_xor_pattern(predictions)


def test_sequential_with_fi_learns_two_moons_pattern() -> None:
    """A Fi-based sequential model should separate a noisy two-moons dataset."""
    _configure_training_runtime()
    features, labels = _make_two_moons()
    model = Sequential([Dense(2, 8), Fi(initial_S=1.0), Dense(8, 1), Fi(initial_S=1.0)])
    model.compile(loss=MSE(), optimizer=Adam(lr=0.02))

    initial_loss = model.evaluate(features, labels)
    history = model.fit(features, labels, epochs=60, batch_size=16, shuffle=True)
    predictions = model.predict(features)
    accuracy = ((predictions >= 0.5).astype(np.int64) == labels.astype(np.int64)).mean()

    assert history["loss"][-1] < initial_loss * 0.05
    assert accuracy >= 0.95


def test_all_gnostic_activation_loss_combinations_reduce_objective() -> None:
    """Every gnostic activation-loss combination should train without divergence."""
    _configure_training_runtime()
    activation_types: tuple[Type[Activation], ...] = (Fi, Hi, Fj, Hj, Ei)
    loss_types: tuple[Type[Loss], ...] = (
        FidelityLoss,
        InfidelityLoss,
        RSSLoss,
        ISSLoss,
        ResidualEntropyLoss,
        InformationLoss,
    )

    for activation_type in activation_types:
        for loss_type in loss_types:
            model = Sequential([Dense(2, 4), activation_type(initial_S=1.0), Dense(4, 1), Fi(initial_S=1.0)])
            model.compile(loss=loss_type(), optimizer=Adam(lr=0.01))
            initial_loss = model.evaluate(XOR_INPUTS, XOR_TARGETS)
            history = model.fit(XOR_INPUTS, XOR_TARGETS, epochs=10, batch_size=4, shuffle=True)
            final_loss = history["loss"][-1]
            predictions = model.predict(XOR_INPUTS)

            assert np.isfinite(initial_loss)
            assert np.isfinite(final_loss)
            assert final_loss < initial_loss, (activation_type.__name__, loss_type.__name__)
            assert np.all(np.isfinite(predictions))


def test_all_magnet_optimizers_train_a_gnostic_sequential_model() -> None:
    """Sequential, Dense, Fi, MSE, and every bundled optimizer should integrate cleanly."""
    _configure_training_runtime()
    rng = np.random.default_rng(11)
    features = rng.normal(size=(64, 2))
    targets = ((features[:, :1] - 0.5 * features[:, 1:2]) > 0.0).astype(np.float64)
    optimizers: tuple[Optimizer, ...] = (SGD(lr=0.05), Adam(lr=0.02), Adagrad(lr=0.05), RMSprop(lr=0.02))

    for optimizer in optimizers:
        model = Sequential([Dense(2, 6), Fi(initial_S=1.0), Dense(6, 1), Fi(initial_S=1.0)])
        model.compile(loss=MSE(), optimizer=optimizer)
        initial_loss = model.evaluate(features, targets)
        history = model.fit(features, targets, epochs=20, batch_size=16, shuffle=True)
        final_loss = history["loss"][-1]

        assert final_loss < initial_loss * 0.25, type(optimizer).__name__
