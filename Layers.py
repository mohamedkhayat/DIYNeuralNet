from abc import ABC, abstractmethod
from DeviceSelector import *
from InputValidation import InputValidator

np = get_numpy()


class Layer(ABC):
    def __init__(self):
        self.input = None
        self.params = {}
        self.grads = {}

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size, initializer="he"):
        super().__init__()

        self.input_size = InputValidator.validate_number_units(input_size)
        self.output_size = InputValidator.validate_number_units(output_size)

        self.params["W"] = np.random.randn(output_size, input_size)

        if initializer == "he":
            self.params["W"] *= np.sqrt(2 / input_size)

        elif initializer == "glorot":
            limit = np.sqrt(6) / (np.sqrt(input_size + output_size))
            self.params["W"] = np.random.uniform(
                -limit, limit, size=(output_size, input_size)
            )

        elif initializer != "random":
            print("Not a valid Initialization method, using random init")

        self.params["b"] = np.zeros((output_size, 1))

        self.grads["dW"] = None
        self.grads["db"] = None

    def forward(self, X, train=True):
        if train:
            self.input = X

        z = self.params["W"] @ X + self.params["b"]

        return z

    def backward(self, dZ):
        batch_size = self.input.shape[1]

        self.grads["dW"] = dZ @ self.input.T / batch_size
        self.grads["db"] = np.sum(dZ, axis=1, keepdims=True) / batch_size

        dA_prev = self.params["W"].T @ dZ

        return dA_prev

    def zero_grad(self):
        self.grads["dW"] = np.zeros_like(self.params["W"])
        self.grads["db"] = np.zeros_like(self.params["b"])

    def get_params(self):
        return self.params

    def get_grads(self):
        return self.grads


class Dropout(Layer):
    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = InputValidator.validate_keep_prob(keep_prob)
        self.mask = None

    def forward(self, A, train=True):
        if train:
            self.mask = (
                np.random.rand(A.shape[0], A.shape[1]) < self.keep_prob
            ).astype(float)
            A = A * self.mask / self.keep_prob

        else:
            self.mask = None

        return A

    def backward(self, dA):
        return dA * self.mask / self.keep_prob
