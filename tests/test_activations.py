import pytest
from diy_neural_net import get_numpy
from diy_neural_net.Activations import ReLU, Sigmoid, Tanh

np = get_numpy()


def test_relu_forward():
    activation = ReLU()
    Z = np.array([[-1.0, 0.0, 1.0]])
    expected = np.array([[0.0, 0.0, 1.0]])

    output = activation.forward(Z)
    assert np.allclose(output, expected)


def test_relu_backward():
    activation = ReLU()
    Z = np.array([[-5.0, 5.0]])
    activation.forward(Z, train=True)

    dA = np.array([[1.0, 1.0]])
    # Gradient should be 0 where input was negative, 1 where positive
    expected = np.array([[0.0, 1.0]])

    dZ = activation.backward(dA)
    assert np.allclose(dZ, expected)


def test_sigmoid_range():
    activation = Sigmoid()
    Z = np.random.randn(10, 10)
    output = activation.forward(Z)
    assert np.all(output >= 0) and np.all(output <= 1)


def test_tanh_range():
    activation = Tanh()
    Z = np.random.randn(10, 10)
    output = activation.forward(Z)
    assert np.all(output >= -1) and np.all(output <= 1)
