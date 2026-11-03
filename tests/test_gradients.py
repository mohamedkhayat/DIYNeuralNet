"""Numerical gradient checks: analytic backward passes must match finite differences."""

import numpy as onp

from diy_neural_net import BatchNorm, CrossEntropyLossWithLogits, Dense, NeuralNetwork, ReLU, get_numpy

np = get_numpy()


def _check(model, X, y, eps=1e-5, tol=1e-4):
    out = model.forward(X, train=True)
    model.criterion(y, out)  # sets batch_size, as the training loop does before backward
    model.backprop(model.criterion.backward(y, out))
    for layer in model.layers:
        for name, param in getattr(layer, "params", {}).items():
            grad = layer.grads["d" + name]
            for idx in [(0, 0), (param.shape[0] - 1, param.shape[1] - 1)]:
                old = float(param[idx])
                param[idx] = old + eps
                lp = float(model.criterion(y, model.forward(X, train=True)))
                param[idx] = old - eps
                lm = float(model.criterion(y, model.forward(X, train=True)))
                param[idx] = old
                numeric = (lp - lm) / (2 * eps)
                assert abs(numeric - float(grad[idx])) < tol * max(1.0, abs(numeric)), (type(layer).__name__, name, idx)


def test_dense_cross_entropy_gradients():
    onp.random.seed(0)
    model = NeuralNetwork.Sequential([Dense(5, 4), ReLU(), Dense(4, 3, initializer="glorot")])
    model.set_loss(CrossEntropyLossWithLogits())
    X = np.asarray(onp.random.randn(5, 16))
    y = np.asarray(onp.eye(3)[:, onp.random.randint(0, 3, 16)])
    _check(model, X, y)


def test_batchnorm_gradients():
    onp.random.seed(1)
    model = NeuralNetwork.Sequential([Dense(4, 6), BatchNorm(6), ReLU(), Dense(6, 3, initializer="glorot")])
    model.set_loss(CrossEntropyLossWithLogits())
    X = np.asarray(onp.random.randn(4, 32))
    y = np.asarray(onp.eye(3)[:, onp.random.randint(0, 3, 32)])
    _check(model, X, y)


def test_cross_entropy_matches_softmax():
    logits = np.asarray(onp.array([[2.0, 1000.0], [0.0, 0.0], [-1.0, 0.0]]))
    y = np.asarray(onp.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))
    loss = float(CrossEntropyLossWithLogits()(y, logits))
    p0 = onp.exp(2.0) / (onp.exp(2.0) + 1 + onp.exp(-1.0))
    assert abs(loss - (-onp.log(p0) + 0.0) / 2) < 1e-6  # second column: huge logit, prob ≈ 1, no overflow
