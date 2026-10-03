from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_package(package_name: str, package_dir: Path):
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1] / "src" / "machinegnostics" / "magnet"
INITIALIZERS = _load_package("phase56_initializers", ROOT / "initializers")
UTILS_PACKAGE = types.ModuleType("phase56_utils")
UTILS_PACKAGE.__path__ = [str(ROOT / "utils")]
sys.modules["phase56_utils"] = UTILS_PACKAGE
_load_module("phase56_utils.misc", ROOT / "utils" / "misc.py")
DATA = _load_module("phase56_utils.data", ROOT / "utils" / "data.py")
NORMALIZATION = _load_module("phase56_utils.normalization", ROOT / "utils" / "normalization.py")

GlorotNormal = INITIALIZERS.GlorotNormal
GlorotUniform = INITIALIZERS.GlorotUniform
HeNormal = INITIALIZERS.HeNormal
HeUniform = INITIALIZERS.HeUniform
Normal = INITIALIZERS.Normal
Ones = INITIALIZERS.Ones
Uniform = INITIALIZERS.Uniform
Zeros = INITIALIZERS.Zeros
get_initializer = INITIALIZERS.get_initializer

create_batches = DATA.create_batches
normalize = DATA.normalize
shuffle_data = DATA.shuffle_data
split_train_test = DATA.split_train_test

ensure_2d = sys.modules["phase56_utils.misc"].ensure_2d
set_random_seed = sys.modules["phase56_utils.misc"].set_random_seed

min_max_scale = NORMALIZATION.min_max_scale
standardize = NORMALIZATION.standardize


def test_glorot_uniform_produces_correct_shape():
    weights = GlorotUniform(seed=7)((4, 3))
    assert weights.shape == (4, 3)


def test_glorot_uniform_has_correct_bounds():
    shape = (5, 2)
    weights = GlorotUniform(seed=11)(shape)
    limit = np.sqrt(6.0 / sum(shape))
    assert np.all(weights >= -limit)
    assert np.all(weights <= limit)


def test_he_uniform_suitable_for_relu():
    shape = (8, 4)
    weights = HeUniform(seed=13)(shape)
    limit = np.sqrt(6.0 / shape[0])
    assert weights.shape == shape
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= -limit)
    assert np.all(weights <= limit)


def test_other_initializers_produce_expected_values_and_shapes():
    assert GlorotNormal(seed=3)((3, 2)).shape == (3, 2)
    assert HeNormal(seed=5)((3, 2)).shape == (3, 2)
    assert Normal(mean=1.5, stddev=0.0, seed=0)((2, 2)).tolist() == [[1.5, 1.5], [1.5, 1.5]]
    assert Uniform(low=-0.5, high=-0.5, seed=0)((2, 2)).tolist() == [[-0.5, -0.5], [-0.5, -0.5]]
    np.testing.assert_array_equal(Zeros()((2, 3)), np.zeros((2, 3)))
    np.testing.assert_array_equal(Ones()((2, 3)), np.ones((2, 3)))


def test_create_batches_yields_correct_sizes():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    batches = list(create_batches(X, y, batch_size=4, shuffle=False))
    assert [len(xb) for xb, _ in batches] == [4, 4, 2]
    np.testing.assert_array_equal(batches[0][0], X[:4])
    np.testing.assert_array_equal(batches[0][1], y[:4])


def test_normalize_produces_zero_mean():
    X = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    normalized, mean, std = normalize(X)
    np.testing.assert_allclose(mean, np.array([3.0, 6.0]))
    np.testing.assert_allclose(np.mean(normalized, axis=0), np.zeros(2), atol=1e-7)
    np.testing.assert_allclose(np.std(normalized, axis=0), np.ones(2), atol=1e-7)
    np.testing.assert_allclose(std, np.std(X, axis=0), atol=1e-12)


def test_split_train_test_proportions():
    X = np.arange(30).reshape(10, 3)
    y = np.arange(10)
    X_train, y_train, X_test, y_test = split_train_test(X, y, test_size=0.3, shuffle=False)
    assert len(X_train) == 7
    assert len(X_test) == 3
    np.testing.assert_array_equal(X_test, X[:3])
    np.testing.assert_array_equal(y_test, y[:3])


def test_all_initializers_accessible_via_registry():
    names_and_types = {
        "glorot_uniform": GlorotUniform,
        "glorot_normal": GlorotNormal,
        "he_uniform": HeUniform,
        "he_normal": HeNormal,
        "normal": Normal,
        "uniform": Uniform,
        "zeros": Zeros,
        "ones": Ones,
        "xavieruniform": GlorotUniform,
        "randomnormal": Normal,
    }
    for name, initializer_type in names_and_types.items():
        assert isinstance(get_initializer(name), initializer_type)


def test_utility_helpers_cover_shuffling_and_scaling():
    X = np.arange(12).reshape(6, 2)
    y = np.arange(6)
    set_random_seed(5)
    shuffled_X, shuffled_y = shuffle_data(X, y)
    assert shuffled_X.shape == X.shape
    assert shuffled_y.shape == y.shape
    assert set(map(tuple, shuffled_X.tolist())) == set(map(tuple, X.tolist()))
    assert set(shuffled_y.tolist()) == set(y.tolist())

    np.testing.assert_allclose(ensure_2d(np.array([1.0, 2.0, 3.0])).shape, (3, 1))
    scaled = min_max_scale(X)
    assert np.all(scaled >= 0.0)
    assert np.all(scaled <= 1.0)
    standardized = standardize(X)
    np.testing.assert_allclose(np.mean(standardized, axis=0), np.zeros(X.shape[1]), atol=1e-7)
