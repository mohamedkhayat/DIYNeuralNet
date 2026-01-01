import pytest
from diy_neural_net import get_numpy
from diy_neural_net.Layers import Dense, Dropout, BatchNorm

np = get_numpy()


class TestDense:
    def test_dense_output_shape(self):
        batch_size = 32
        input_size = 10
        output_size = 5

        layer = Dense(input_size, output_size)
        X = np.random.randn(input_size, batch_size)

        output = layer.forward(X)
        assert output.shape == (output_size, batch_size)

    def test_dense_backward_shape(self):
        batch_size = 32
        input_size = 10
        output_size = 5

        layer = Dense(input_size, output_size)
        X = np.random.randn(input_size, batch_size)

        # Forward pass to cache input
        layer.forward(X)

        # Fake gradient coming back
        dZ = np.random.randn(output_size, batch_size)
        dA_prev = layer.backward(dZ)

        assert dA_prev.shape == (input_size, batch_size)
        assert layer.grads["dW"].shape == layer.params["W"].shape
        assert layer.grads["db"].shape == layer.params["b"].shape


class TestDropout:
    def test_dropout_training_randomness(self):
        layer = Dropout(keep_prob=0.5)
        A = np.ones((100, 100))

        out1 = layer.forward(A, train=True)
        out2 = layer.forward(A, train=True)

        # Outputs should differ due to randomness
        assert not np.allclose(out1, out2)
        # Roughly 50% should be zero
        assert np.abs(np.mean(out1 == 0) - 0.5) < 0.1

    def test_dropout_inference_pass_through(self):
        layer = Dropout(keep_prob=0.5)
        A = np.ones((10, 10))
        out = layer.forward(A, train=False)
        # Should be identical during inference
        assert np.allclose(A, out)


class TestBatchNorm:
    def test_batchnorm_output_shape(self):
        layer = BatchNorm(input_size=10)
        X = np.random.randn(10, 32)
        out = layer.forward(X, train=True)
        assert out.shape == X.shape

    def test_batchnorm_normalization(self):
        # Create data with mean=10, std=5
        X = np.random.randn(10, 1000) * 5 + 10
        layer = BatchNorm(input_size=10)

        out = layer.forward(X, train=True)

        # Output should have roughly mean=0, std=1
        # Convert to numpy for assertion if on GPU
        out_np = out.get() if hasattr(out, "get") else out

        assert np.allclose(np.mean(out_np, axis=1), 0, atol=0.1)
        assert np.allclose(np.std(out_np, axis=1), 1, atol=0.1)
