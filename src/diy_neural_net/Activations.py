from abc import ABC, abstractmethod
from typing import Optional
from .DeviceSelector import get_numpy, ArrayType

np = get_numpy()


class Activation(ABC):
    """Abstract base class for activation functions.

    This class defines the interface that all activation functions must implement,
    including forward and backward pass methods.
    """

    def __init__(self) -> None:
        """Initialize activation function."""
        super().__init__()

    @abstractmethod
    def forward(self, Z: ArrayType, train: bool = True) -> ArrayType:
        """Forward pass through activation function.

        Args:
            Z: Input data with shape (input_size, batch_size)
            train: Whether the activation is in training mode

        Returns:
            Activated output with same shape as input
        """
        pass

    @abstractmethod
    def backward(self, dA: ArrayType) -> ArrayType:
        """Backward pass through activation function.

        Args:
            dA: Gradient of loss with respect to activation output

        Returns:
            Gradient of loss with respect to activation input
        """
        pass


class ReLU(Activation):
    """Rectified Linear Unit (ReLU) activation function.

    Applies element-wise ReLU: f(x) = max(0, x)
    """

    def __init__(self) -> None:
        """Initialize ReLU activation."""
        super().__init__()
        self.input: Optional[ArrayType] = None

    def forward(self, Z: ArrayType, train: bool = True) -> ArrayType:
        """Forward pass through ReLU.

        Args:
            Z: Input data with shape (input_size, batch_size)
            train: Whether to store input for backward pass

        Returns:
            ReLU-activated output with same shape as input
        """
        if train:
            self.input = Z

        return np.maximum(0, Z)

    def backward(self, dA: ArrayType) -> ArrayType:
        """Backward pass through ReLU.

        Args:
            dA: Gradient of loss with respect to activation output

        Returns:
            Gradient of loss with respect to activation input
        """
        return np.where(self.input > 0, 1, 0) * dA


class Sigmoid(Activation):
    """Sigmoid activation function.

    Applies element-wise sigmoid: f(x) = 1 / (1 + exp(-x))
    """

    def __init__(self) -> None:
        """Initialize Sigmoid activation."""
        super().__init__()
        self.input: Optional[ArrayType] = None
        self.output: Optional[ArrayType] = None

    def forward(self, Z: ArrayType, train: bool = True) -> ArrayType:
        """Forward pass through Sigmoid.

        Args:
            Z: Input data with shape (input_size, batch_size)
            train: Whether to store input and output for backward pass

        Returns:
            Sigmoid-activated output with same shape as input
        """
        output = 1 / (1 + np.exp(-Z))
        if train:
            self.input = Z
            self.output = output

        return output

    def backward(self, dA: ArrayType) -> ArrayType:
        """Backward pass through Sigmoid.

        Args:
            dA: Gradient of loss with respect to activation output

        Returns:
            Gradient of loss with respect to activation input
        """
        return (self.output * (1 - self.output)) * dA


class Tanh(Activation):
    """Hyperbolic tangent (Tanh) activation function.

    Applies element-wise tanh: f(x) = tanh(x)
    """

    def __init__(self) -> None:
        """Initialize Tanh activation."""
        super().__init__()
        self.output: Optional[ArrayType] = None

    def forward(self, Z: ArrayType, train: bool = True) -> ArrayType:
        """Forward pass through Tanh.

        Args:
            Z: Input data with shape (input_size, batch_size)
            train: Whether to store output for backward pass

        Returns:
            Tanh-activated output with same shape as input
        """
        output = np.tanh(Z)
        if train:
            self.output = output
        return output

    def backward(self, dA: ArrayType) -> ArrayType:
        """Backward pass through Tanh.

        Args:
            dA: Gradient of loss with respect to activation output

        Returns:
            Gradient of loss with respect to activation input
        """
        grad = 1 - np.square(self.output)
        return grad * dA
