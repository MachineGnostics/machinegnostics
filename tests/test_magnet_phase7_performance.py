"""Phase 7 performance-oriented tests for MAGNET."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from machinegnostics.magnet import Adam, Dense, Fi, MSE, Sequential, configure


def _configure_benchmark_runtime() -> None:
    """Pin benchmarks to a deterministic CPU runtime."""
    configure(device="cpu", dtype="float32", seed=19, deterministic=True)
    np.random.seed(19)


def _make_benchmark_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Create a small binary dataset for benchmark training loops."""
    rng = np.random.default_rng(19)
    features = rng.normal(size=(100, 2))
    targets = ((features[:, :1] + 0.5 * features[:, 1:2]) > 0.0).astype(np.float64)
    return features.astype(np.float64), targets.astype(np.float64)


def test_sequential_training_loop_completes_within_budget() -> None:
    """Training 100 samples for 10 epochs should finish within a generous budget."""
    _configure_benchmark_runtime()
    features, targets = _make_benchmark_dataset()
    model = Sequential([Dense(2, 8), Fi(initial_S=1.0), Dense(8, 1), Fi(initial_S=1.0)])
    model.compile(loss=MSE(), optimizer=Adam(lr=0.02))

    start = time.perf_counter()
    history = model.fit(features, targets, epochs=10, batch_size=16, shuffle=True)
    elapsed = time.perf_counter() - start

    assert elapsed < 3.0
    assert history["loss"][-1] < history["loss"][0]


def test_model_initialization_has_bounded_memory_growth() -> None:
    """Building a representative sequential model should not allocate excessive Python memory."""
    _configure_benchmark_runtime()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    _ = Sequential([Dense(32, 64), Fi(initial_S=1.0), Dense(64, 16), Fi(initial_S=1.0)])
    after = tracemalloc.take_snapshot()
    total_growth = sum(stat.size_diff for stat in after.compare_to(before, "lineno"))
    tracemalloc.stop()

    assert total_growth < 10 * 1024 * 1024


def test_dense_forward_pass_average_latency_is_reasonable() -> None:
    """Dense forward propagation should stay fast for common batch sizes."""
    _configure_benchmark_runtime()
    layer = Dense(32, 32)
    inputs = np.ones((256, 32), dtype=np.float64)

    start = time.perf_counter()
    for _ in range(200):
        layer(inputs)
    average_latency = (time.perf_counter() - start) / 200.0

    assert average_latency < 0.01


def test_adam_step_average_latency_is_reasonable() -> None:
    """Optimizer step plus gradient reset should remain inexpensive per batch."""
    _configure_benchmark_runtime()
    features, targets = _make_benchmark_dataset()
    model = Sequential([Dense(2, 8), Fi(initial_S=1.0), Dense(8, 1), Fi(initial_S=1.0)])
    model.compile(loss=MSE(), optimizer=Adam(lr=0.02))
    params = model.params

    loss = MSE()(model(features), targets)
    loss.backward()

    start = time.perf_counter()
    for _ in range(100):
        model.optimizer.step(params)
        model.optimizer.zero_grad(params)
        loss = MSE()(model(features), targets)
        loss.backward()
    average_step_latency = (time.perf_counter() - start) / 100.0

    assert average_step_latency < 0.02
