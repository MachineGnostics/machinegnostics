"""Phase 7 edge-case tests for MAGNET numerical behavior."""

from __future__ import annotations

import numpy as np

from machinegnostics.magnet import Dense, Fi, Hi, MSE, Sequential, Tensor, configure
from machinegnostics.magnet.activations.ei import ei
from machinegnostics.magnet.activations.fi import fi
from machinegnostics.magnet.activations.fj import fj
from machinegnostics.magnet.activations.hi import hi
from machinegnostics.magnet.activations.hj import hj


def _configure_cpu_runtime() -> None:
    """Use a stable CPU runtime for edge-case assertions."""
    configure(device="cpu", dtype="float32", seed=13, deterministic=True)
    np.random.seed(13)


def test_functional_activations_handle_million_element_inputs() -> None:
    """Vectorized activation functions should stay finite on million-element inputs."""
    inputs = np.linspace(-3.0, 3.0, 1_000_000, dtype=np.float64)

    fi_values = fi(inputs, S=1.5, z0=0.25)
    hi_values = hi(inputs, S=1.5, z0=0.25)
    fj_values = fj(inputs, S=1.5, z0=0.25)
    hj_values = hj(inputs, S=1.5, z0=0.25)
    ei_values = ei(inputs, S=1.5, z0=0.25, case="i")

    assert fi_values.shape == (1_000_000,)
    assert np.all(np.isfinite(fi_values))
    assert np.all(np.isfinite(hi_values))
    assert np.all(np.isfinite(fj_values))
    assert np.all(np.isfinite(hj_values))
    assert np.all(np.isfinite(ei_values))


def test_functional_activations_preserve_precision_for_small_scales() -> None:
    """Small-valued inputs should preserve the Fi/Hi conservation identity."""
    inputs = np.array([-3e-6, -1e-6, 0.0, 1e-6, 3e-6], dtype=np.float64)
    fi_values = fi(inputs, S=1e-6, z0=0.0)
    hi_values = hi(inputs, S=1e-6, z0=0.0)

    np.testing.assert_allclose(fi_values**2 + hi_values**2, np.ones_like(fi_values), atol=1e-6)


def test_sequential_training_with_zero_inputs_keeps_loss_finite() -> None:
    """Training on all-zero features and labels should not produce NaNs or infs."""
    _configure_cpu_runtime()
    features = np.zeros((32, 2), dtype=np.float64)
    targets = np.zeros((32, 1), dtype=np.float64)

    model = Sequential([Dense(2, 4), Fi(initial_S=1.0), Dense(4, 1), Fi(initial_S=1.0)])
    model.compile(loss=MSE(), optimizer="adam")
    history = model.fit(features, targets, epochs=12, batch_size=8, shuffle=True)
    predictions = model.predict(features)

    assert np.isfinite(history["loss"][-1])
    assert np.all(np.isfinite(predictions))


def test_dense_returns_identical_outputs_for_identical_samples() -> None:
    """Identical samples should produce identical Dense outputs before learning."""
    _configure_cpu_runtime()
    layer = Dense(3, 2)
    features = np.ones((5, 3), dtype=np.float64)
    outputs = layer(features).data

    repeated_rows = np.repeat(outputs[:1], repeats=outputs.shape[0], axis=0)
    np.testing.assert_allclose(outputs, repeated_rows, atol=1e-6)


def test_gnostic_activations_clip_infinities_and_preserve_nan_signals() -> None:
    """Layer activations should clip infinities while leaving NaN inputs observable."""
    _configure_cpu_runtime()
    inputs = Tensor(np.array([np.nan, -np.inf, 0.0, np.inf], dtype=np.float64))

    fi_values = Fi(learnable_S=False, learnable_z0=False)(inputs).data
    hi_values = Hi(learnable_S=False, learnable_z0=False)(inputs).data
    fj_values = fj(np.array([np.nan, -np.inf, 0.0, np.inf], dtype=np.float64))
    hj_values = hj(np.array([np.nan, -np.inf, 0.0, np.inf], dtype=np.float64))
    ei_values = ei(np.array([np.nan, -np.inf, 0.0, np.inf], dtype=np.float64), case="i")

    assert np.isnan(fi_values[0])
    assert np.isnan(hi_values[0])
    assert np.isnan(fj_values[0])
    assert np.isnan(hj_values[0])
    assert np.isnan(ei_values[0])
    assert np.all(np.isfinite(fi_values[1:]))
    assert np.all(np.isfinite(hi_values[1:]))
    assert np.all(np.isfinite(fj_values[1:]))
    assert np.all(np.isfinite(hj_values[1:]))
    assert np.all(np.isfinite(ei_values[1:]))


def test_dense_backward_stays_finite_for_singular_weight_matrix() -> None:
    """Singular Dense weights should still yield finite forward and backward values."""
    _configure_cpu_runtime()
    layer = Dense(3, 2)
    layer.W.data = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=np.float64)
    layer.b.data = np.array([0.5, -0.5], dtype=np.float64)

    features = np.ones((4, 3), dtype=np.float64)
    outputs = layer(features)
    loss = MSE()(outputs, np.zeros((4, 2), dtype=np.float64))
    loss.backward()

    assert np.all(np.isfinite(outputs.data))
    assert np.all(np.isfinite(layer.W.grad))
    assert np.all(np.isfinite(layer.b.grad))
