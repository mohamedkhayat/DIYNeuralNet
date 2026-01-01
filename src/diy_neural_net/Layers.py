from abc import ABC, abstractmethod
from typing import Dict, Optional
from .DeviceSelector import get_numpy, ArrayType
from .InputValidation import InputValidator

np = get_numpy()


class Layer(ABC):
    """Abstract base class for neural network layers.

    This class defines the interface that all neural network layers must implement,
    including forward and backward pass methods.
    """

    def __init__(self) -> None:
        """Initialize layer with empty parameters and gradients."""
        self.input: Optional[ArrayType] = None
        self.params: Dict[str, ArrayType] = {}
        self.grads: Dict[str, ArrayType] = {}

    @abstractmethod
    def forward(self, X: ArrayType, train: bool = True) -> ArrayType:
        """Forward pass through the layer.

        Args:
            X: Input data with shape (input_size, batch_size)
            train: Whether the layer is in training mode

        Returns:
            Output of the layer with appropriate shape
        """
        pass

    @abstractmethod
    def backward(self, dA: ArrayType) -> ArrayType:
        """Backward pass through the layer.

        Args:
            dA: Gradient of loss with respect to layer output

        Returns:
            Gradient of loss with respect to layer input
        """
        pass


class Dense(Layer):
    """Fully connected (dense) neural network layer.

    This layer performs a linear transformation of the input data followed by
    an optional bias addition: output = W @ input + b
    """

    def __init__(
        self, input_size: int, output_size: int, initializer: str = "he"
    ) -> None:
        """Initialize dense layer with specified dimensions.

        Args:
            input_size: Number of input features
            output_size: Number of output features
            initializer: Weight initialization method ("he", "glorot", or "random")
        """
        super().__init__()

        self.input_size: int = InputValidator.validate_number_units(input_size)
        self.output_size: int = InputValidator.validate_number_units(output_size)

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

    def forward(self, X: ArrayType, train: bool = True) -> ArrayType:
        """Forward pass through dense layer.

        Args:
            X: Input data with shape (input_size, batch_size)
            train: Whether to store input for backward pass

        Returns:
            Linear transformation result with shape (output_size, batch_size)
        """
        if train:
            self.input = X

        z = self.params["W"] @ X + self.params["b"]

        return z

    def backward(self, dZ: ArrayType) -> ArrayType:
        """Backward pass through dense layer.

        Args:
            dZ: Gradient of loss with respect to layer output

        Returns:
            Gradient of loss with respect to layer input
        """
        batch_size = self.input.shape[1]

        self.grads["dW"] = dZ @ self.input.T / batch_size
        self.grads["db"] = np.sum(dZ, axis=1, keepdims=True)

        dA_prev = self.params["W"].T @ dZ

        return dA_prev

    def zero_grad(self) -> None:
        """Reset gradients to zero."""
        self.grads["dW"] = np.zeros_like(self.params["W"])
        self.grads["db"] = np.zeros_like(self.params["b"])

    def get_params(self) -> Dict[str, ArrayType]:
        """Get layer parameters.

        Returns:
            Dictionary containing weights and biases
        """
        return self.params

    def get_grads(self) -> Dict[str, ArrayType]:
        """Get layer gradients.

        Returns:
            Dictionary containing parameter gradients
        """
        return self.grads


class Dropout(Layer):
    """Dropout regularization layer.

    During training, randomly sets input elements to zero with probability
    (1 - keep_prob) and scales remaining elements by 1/keep_prob.
    During inference, passes input unchanged.
    """

    def __init__(self, keep_prob: float) -> None:
        """Initialize dropout layer.

        Args:
            keep_prob: Probability of keeping each element during training (0 < keep_prob <= 1)
        """
        super().__init__()
        self.keep_prob: float = InputValidator.validate_keep_prob(keep_prob)
        self.mask: Optional[ArrayType] = None

    def forward(self, A: ArrayType, train: bool = True) -> ArrayType:
        """Forward pass through dropout layer.

        Args:
            A: Input activations with shape (input_size, batch_size)
            train: Whether to apply dropout (True) or pass through unchanged (False)

        Returns:
            Dropout-applied activations with same shape as input
        """
        if train:
            self.mask = (
                np.random.rand(A.shape[0], A.shape[1]) < self.keep_prob
            ).astype(float)
            A = A * self.mask / self.keep_prob

        else:
            self.mask = None

        return A

    def backward(self, dA: ArrayType) -> ArrayType:
        """Backward pass through dropout layer.

        Args:
            dA: Gradient of loss with respect to layer output

        Returns:
            Gradient of loss with respect to layer input
        """
        return dA * self.mask / self.keep_prob


class BatchNorm(Layer):
    """Batch normalization layer.

    Normalizes inputs by maintaining running statistics of mean and variance.
    Applies learnable scale (gamma) and shift (beta) parameters.
    """

    def __init__(self, input_size: int, momentum: float = 0.1) -> None:
        """Initialize batch normalization layer.

        Args:
            input_size: Number of input features
            momentum: Momentum factor for running statistics update
        """
        super().__init__()
        self.momentum: float = momentum
        self.input_size: int = input_size
        self.output_size: int = input_size
        self.running_mean: ArrayType = np.zeros((input_size, 1))
        self.running_var: ArrayType = np.ones((input_size, 1))
        self.batch_size: Optional[int] = None

        self.params["gamma"] = np.ones((input_size, 1))
        self.params["beta"] = np.zeros((input_size, 1))

        self.grads["dgamma"] = None
        self.grads["dbeta"] = None

        self.cache: Dict[str, ArrayType] = {}

    def forward(self, X: ArrayType, train: bool = True) -> ArrayType:
        """Forward pass through batch normalization layer.

        Args:
            X: Input data with shape (input_size, batch_size)
            train: Whether to use batch statistics (True) or running statistics (False)

        Returns:
            Normalized and scaled activations with same shape as input
        """
        epsilon = 1e-8
        self.batch_size = X.shape[1]
        # X : n_features, n_samples
        if train:
            mean = np.mean(X, axis=1, keepdims=True)
            var = np.var(X, axis=1, keepdims=True)

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )

            X_centred = X - mean
            std = np.sqrt(var + epsilon)
            X_norm = X_centred / std

            self.cache = {"X_norm": X_norm, "std": std}
        else:
            X_centred = X - self.running_mean
            std = np.sqrt(self.running_var + epsilon)
            X_norm = X_centred / std

        output = X_norm * self.params["gamma"] + self.params["beta"]
        return output

    def backward(self, dA: ArrayType) -> ArrayType:
        """Backward pass through batch normalization layer.

        Args:
            dA: Gradient of loss with respect to layer output

        Returns:
            Gradient of loss with respect to layer input
        """
        gamma = self.params["gamma"]
        X_norm = self.cache["X_norm"]
        std = self.cache["std"]

        self.grads["dgamma"] = np.sum(dA * X_norm, axis=1, keepdims=True)
        self.grads["dbeta"] = np.sum(dA, axis=1, keepdims=True)

        dX_norm = dA * gamma

        dA_prev = (
            (1.0 / self.batch_size)
            * (1.0 / std)
            * (
                self.batch_size * dX_norm
                - np.sum(dX_norm, axis=1, keepdims=True)
                - X_norm * np.sum(dX_norm * X_norm, axis=1, keepdims=True)
            )
        )

        return dA_prev

    def get_params(self) -> Dict[str, ArrayType]:
        """Get layer parameters.

        Returns:
            Dictionary containing gamma and beta parameters
        """
        return self.params

    def get_grads(self) -> Dict[str, ArrayType]:
        """Get layer gradients.

        Returns:
            Dictionary containing parameter gradients
        """
        return self.grads
