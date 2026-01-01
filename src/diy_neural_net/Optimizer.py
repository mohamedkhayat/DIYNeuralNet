from abc import ABC, abstractmethod
from typing import Dict, List, Any, TYPE_CHECKING
from .DeviceSelector import get_numpy, ArrayType

if TYPE_CHECKING:
    from .Network import NeuralNetwork

np = get_numpy()


class Optimizer(ABC):
    """Abstract base class for optimization algorithms.

    This class defines the interface for all optimization algorithms used
    to update neural network parameters based on computed gradients.
    """

    def __init__(self, model: "NeuralNetwork", lr: float) -> None:
        """Initialize optimizer with model and learning rate.

        Args:
            model: Neural network model to optimize
            lr: Learning rate for parameter updates
        """
        self.layers: List[Any] = [
            layer for layer in model.layers if hasattr(layer, "params")
        ]
        self.lr: float = lr

    @abstractmethod
    def step(self) -> None:
        """Perform one optimization step to update model parameters.

        This method should be implemented by all concrete optimizer classes
        to define how parameters are updated based on their gradients.
        """
        raise NotImplementedError


class GradientDescent(Optimizer):
    """Standard Gradient Descent optimizer.

    Updates parameters using the rule: param = param - lr * gradient
    """

    def __init__(self, model: "NeuralNetwork", lr: float = 1e-1) -> None:
        """Initialize gradient descent optimizer.

        Args:
            model: Neural network model to optimize
            lr: Learning rate (default: 0.1)
        """
        super().__init__(model, lr)

    def step(self) -> None:
        """Perform one gradient descent step.

        Updates all layer parameters by subtracting the learning rate
        times the gradient.
        """
        for layer in self.layers:
            for param in layer.params:
                layer.params[param] -= self.lr * layer.grads["d" + param]


class GradientDescentWithMomentum(Optimizer):
    """Gradient Descent with Momentum optimizer.

    Uses momentum to accelerate gradients in relevant direction and
    dampens oscillations. Helps escape local minima and speeds up convergence.
    """

    def __init__(
        self, model: "NeuralNetwork", lr: float = 1e-1, momentum: float = 0.9
    ) -> None:
        """Initialize gradient descent with momentum optimizer.

        Args:
            model: Neural network model to optimize
            lr: Learning rate (default: 0.1)
            momentum: Momentum factor (default: 0.9)
        """
        super().__init__(model, lr)
        self.momentum: float = momentum
        self.velocity: Dict[int, Dict[str, ArrayType]] = {}
        for i, layer in enumerate(self.layers):
            self.velocity[i] = {k: np.zeros_like(v) for k, v in layer.params.items()}

    def step(self) -> None:
        """Perform one gradient descent step with momentum.

        Updates velocity using momentum and current gradient, then
        updates parameters using the velocity.
        """
        for i, layer in enumerate(self.layers):
            for param in layer.params:
                self.velocity[i][param] = (
                    self.momentum * self.velocity[i][param]
                    + (1 - self.momentum) * layer.grads["d" + param]
                )
                layer.params[param] -= self.lr * self.velocity[i][param]


class RMSProp(Optimizer):
    """RMSProp (Root Mean Square Propagation) optimizer.

    Adapts the learning rate for each parameter by dividing by a running
    average of the magnitudes of recent gradients. Helps deal with sparse
    gradients and non-stationary objectives.
    """

    def __init__(
        self,
        model: "NeuralNetwork",
        lr: float = 1e-3,
        decay_rate: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize RMSProp optimizer.

        Args:
            model: Neural network model to optimize
            lr: Learning rate (default: 0.001)
            decay_rate: Decay rate for moving average of squared gradients (default: 0.99)
            epsilon: Small constant for numerical stability (default: 1e-8)
        """
        super().__init__(model, lr)
        self.decay_rate: float = decay_rate
        self.epsilon: float = epsilon
        self.squared_grads: Dict[int, Dict[str, ArrayType]] = {}
        for i, layer in enumerate(self.layers):
            self.squared_grads[i] = {
                k: np.zeros_like(v) for k, v in layer.params.items()
            }

    def step(self) -> None:
        """Perform one RMSProp optimization step.

        Updates the moving average of squared gradients and then updates
        parameters with adaptive learning rates.
        """
        for i, layer in enumerate(self.layers):
            for param in layer.params:
                self.squared_grads[i][param] = self.decay_rate * self.squared_grads[i][
                    param
                ] + (1 - self.decay_rate) * (layer.grads["d" + param] ** 2)
                layer.params[param] -= (
                    self.lr
                    * layer.grads["d" + param]
                    / (np.sqrt(self.squared_grads[i][param]) + self.epsilon)
                )


class Adam(Optimizer):
    """Adam (Adaptive Moment Estimation) optimizer.

    Combines the advantages of both AdaGrad and RMSProp by computing adaptive
    learning rates for each parameter from estimates of first and second moments
    of the gradients.
    """

    def __init__(
        self,
        model: "NeuralNetwork",
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize Adam optimizer.

        Args:
            model: Neural network model to optimize
            lr: Learning rate (default: 0.001)
            beta1: Exponential decay rate for first moment estimates (default: 0.9)
            beta2: Exponential decay rate for second moment estimates (default: 0.999)
            epsilon: Small constant for numerical stability (default: 1e-8)
        """
        super().__init__(model, lr)
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon
        self.moment_1: Dict[int, Dict[str, ArrayType]] = {}
        self.moment_2: Dict[int, Dict[str, ArrayType]] = {}
        self.t: int = 0
        for i, layer in enumerate(self.layers):
            self.moment_1[i] = {k: np.zeros_like(v) for k, v in layer.params.items()}
            self.moment_2[i] = {k: np.zeros_like(v) for k, v in layer.params.items()}

    def step(self) -> None:
        """Perform one Adam optimization step.

        Updates biased first and second moment estimates, applies bias correction,
        and then updates parameters with adaptive learning rates.
        """
        self.t += 1

        for i, layer in enumerate(self.layers):
            for param in layer.params:
                self.moment_1[i][param] = (
                    self.beta1 * self.moment_1[i][param]
                    + (1 - self.beta1) * layer.grads["d" + param]
                )
                self.moment_2[i][param] = self.beta2 * self.moment_2[i][param] + (
                    1 - self.beta2
                ) * (layer.grads["d" + param] ** 2)

                bias_correction_1 = self.moment_1[i][param] / (1 - self.beta1**self.t)
                bias_correction_2 = self.moment_2[i][param] / (1 - self.beta2**self.t)

                layer.params[param] -= (
                    self.lr
                    * bias_correction_1
                    / (np.sqrt(bias_correction_2) + self.epsilon)
                )
