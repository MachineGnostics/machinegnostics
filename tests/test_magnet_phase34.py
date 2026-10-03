import numpy as np

from machinegnostics.magnet import Adam, Dense, GnosticNeuron, Layer, MSE, Sequential, Tensor, get_model


class ScaleLayer(Layer):
	def __init__(self, factor: float):
		super().__init__()
		self.factor = factor
		self.trainable = False

	def forward(self, x, training=True):
		x = x if isinstance(x, Tensor) else Tensor(x)
		return x * self.factor

	def backward(self, grad_output):
		raise NotImplementedError


class ShiftLayer(Layer):
	def __init__(self, bias: float):
		super().__init__()
		self.bias = bias
		self.trainable = False

	def forward(self, x, training=True):
		x = x if isinstance(x, Tensor) else Tensor(x)
		return x + self.bias

	def backward(self, grad_output):
		raise NotImplementedError


def test_sequential_model_stacks_layers_correctly():
	model = Sequential([ScaleLayer(2.0), ShiftLayer(3.0)])
	output = model.forward(np.array([[1.0, 2.0]]))
	np.testing.assert_allclose(output.data, np.array([[5.0, 7.0]]))


def test_dense_layer_computes_correct_output_shape():
	layer = Dense(4, 3)
	output = layer(np.ones((5, 4)))
	assert output.shape == (5, 3)


def test_model_backward_pass_propagates_gradients():
	model = Sequential([Dense(2, 3), Dense(3, 1)])
	x = np.array([[0.5, -1.0], [1.5, 2.0]], dtype=np.float64)
	y = np.array([[1.0], [0.0]], dtype=np.float64)
	loss = MSE()(model(x), y)
	loss.backward()
	for param in model.params:
		assert param.grad is not None
		assert np.all(np.isfinite(param.grad))


def test_sequential_training_reduces_loss():
	rng = np.random.default_rng(7)
	x = rng.normal(size=(64, 2))
	y = (2.0 * x[:, :1]) - (0.5 * x[:, 1:2]) + 0.25

	model = Sequential([Dense(2, 1)])
	model.compile(loss=MSE(), optimizer=Adam(lr=0.05))
	initial_loss = model.evaluate(x, y)
	history = model.fit(x, y, epochs=120, batch_size=16, shuffle=True)
	final_loss = history["loss"][-1]

	assert final_loss < initial_loss
	assert final_loss < 0.05


def test_gnostic_neuron_model_works():
	neuron = GnosticNeuron(2, 1, activation="sigmoid")
	neuron.W.data = np.array([[1.0], [-1.0]], dtype=np.float64)
	neuron.b.data = np.array([0.5], dtype=np.float64)

	x = np.array([[2.0, 1.0], [0.0, 0.0]], dtype=np.float64)
	output = neuron(x).data
	expected = 1.0 / (1.0 + np.exp(-((x @ neuron.W.data) + neuron.b.data)))
	np.testing.assert_allclose(output, expected, atol=1e-7)


def test_model_registry_access():
	assert get_model("sequential") is Sequential
	assert get_model("gnostic_neuron") is GnosticNeuron
	assert get_model("gnostic-neuron") is GnosticNeuron
	assert get_model("missing-model") is None
