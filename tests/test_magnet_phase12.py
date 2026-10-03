import numpy as np

from machinegnostics.magnet import (
    Ei,
    Fi,
    Fj,
    FidelityLoss,
    Hi,
    Hj,
    ISSLoss,
    InfidelityLoss,
    InformationLoss,
    ResidualEntropyLoss,
    RSSLoss,
    SGD,
    Tensor,
    get_activation,
    get_loss,
)


def test_fi_hi_conservation_identity_holds():
    x = Tensor(np.linspace(-3.0, 3.0, 21), requires_grad=True)
    fi = Fi(learnable_S=False, learnable_z0=False, initial_S=1.3)(x)
    hi = Hi(learnable_S=False, learnable_z0=False, initial_S=1.3)(x)
    np.testing.assert_allclose(fi.data**2 + hi.data**2, np.ones_like(fi.data), atol=1e-6)


def test_complementary_characteristics_are_consistent():
    x = Tensor(np.array([-2.0, -0.5, 0.0, 0.5, 2.0]), requires_grad=True)
    fi = Fi(learnable_S=False, learnable_z0=False, initial_S=0.75)(x)
    fj = Fj(learnable_S=False, learnable_z0=False, initial_S=0.75)(x)
    hi = Hi(learnable_S=False, learnable_z0=False, initial_S=0.75)(x)
    hj = Hj(learnable_S=False, learnable_z0=False, initial_S=0.75)(x)
    np.testing.assert_allclose(fi.data * fj.data, np.ones_like(fi.data), atol=1e-6)
    np.testing.assert_allclose(hj.data / fj.data, hi.data, atol=1e-6)


def test_gnostic_activations_propagate_gradients_to_s_and_z0():
    layer = Fi(initial_S=1.5, initial_z0=0.25)
    x = Tensor(np.array([-1.0, 0.0, 2.0]), requires_grad=True)
    loss = layer(x).mean()
    loss.backward()
    assert layer.S.grad is not None
    assert layer.z0.grad is not None
    assert np.all(np.isfinite(layer.S.grad))
    assert np.all(np.isfinite(layer.z0.grad))


def test_entropy_activation_is_stable_for_large_inputs():
    x = Tensor(np.array([-1e6, 0.0, 1e6]), requires_grad=True)
    output = Ei(learnable_S=False, learnable_z0=False, initial_S=2.0)(x)
    assert np.all(np.isfinite(output.data))
    assert np.all(output.data >= 0.0)


def test_losses_are_correct_at_perfect_alignment():
    y_pred = Tensor(np.zeros((4, 2)), requires_grad=True)
    y_true = Tensor(np.zeros((4, 2)))
    fidelity = float(FidelityLoss()(y_pred, y_true))
    assert np.isclose(fidelity, 0.0), f"Expected FidelityLoss=0.0 at perfect alignment, got {fidelity}"

    infidelity = float(InfidelityLoss()(y_pred, y_true))
    assert np.isclose(infidelity, 1.0), f"Expected InfidelityLoss=1.0 at perfect alignment, got {infidelity}"

    assert float(RSSLoss()(y_pred, y_true)) == 0.0
    assert float(ISSLoss()(y_pred, y_true)) == 0.0
    assert float(ResidualEntropyLoss()(y_pred, y_true)) == 0.0


def test_information_loss_is_finite_and_backward_safe():
    y_pred = Tensor(np.array([[1.0, -1.0], [0.2, -0.3]]), requires_grad=True)
    y_true = Tensor(np.zeros((2, 2)))
    loss = InformationLoss()(y_pred, y_true)
    assert np.isfinite(float(loss))
    loss.backward()
    assert y_pred.grad is not None
    assert np.all(np.isfinite(y_pred.grad))
    assert np.max(np.abs(y_pred.grad)) <= 1e6 + 1e-6


def test_sgd_automatically_scales_special_parameters_more_conservatively():
    optimizer = SGD(lr=0.1, gradient_scale_factor=10.0)
    special = Tensor(np.array([1.0]), requires_grad=True, name="Fi_S")
    regular = Tensor(np.array([1.0]), requires_grad=True, name="W")
    special.grad = np.array([10.0])
    regular.grad = np.array([10.0])
    optimizer.step([special, regular])
    special_step = 1.0 - special.data[0]
    regular_step = 1.0 - regular.data[0]
    assert special_step < regular_step


def test_registry_exposes_new_activation_and_losses():
    assert isinstance(get_activation("ei"), Ei)
    assert isinstance(get_loss("fidelity_loss"), FidelityLoss)
    assert isinstance(get_loss("infidelity-loss"), InfidelityLoss)
