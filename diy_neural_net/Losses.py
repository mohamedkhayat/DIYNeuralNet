from abc import ABC, abstractmethod
from typing import Optional
from .InputValidation import InputValidator
from .DeviceSelector import get_numpy, ArrayType

np = get_numpy()


class Loss(ABC):
    """Abstract base class for loss functions.

    This class defines the interface that all loss functions must implement,
    including forward pass (loss computation) and backward pass (gradient computation).
    """

    def __init__(self) -> None:
        """Initialize loss function."""
        super().__init__()
        self.batch_size: Optional[int] = None

    @abstractmethod
    def __call__(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        """Compute the loss between true and predicted values.

        Args:
            y_true: True labels
            y_pred: Predicted values

        Returns:
            Scalar loss value
        """
        pass

    @abstractmethod
    def backward(self, y_true: ArrayType, y_pred: ArrayType) -> ArrayType:
        """Compute gradient of loss with respect to predictions.

        Args:
            y_true: True labels
            y_pred: Predicted values

        Returns:
            Gradient of loss with respect to predictions
        """
        pass


class BCELoss(Loss):
    """Binary Cross-Entropy Loss function.

    Computes the binary cross-entropy loss between true binary labels
    and predicted probabilities.
    """

    def __init__(self) -> None:
        """Initialize BCE loss."""
        super().__init__()

    def __call__(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        """Compute binary cross-entropy loss.

        Args:
            y_true: True binary labels with shape (1, batch_size)
            y_pred: Predicted probabilities with shape (1, batch_size)

        Returns:
            Average binary cross-entropy loss
        """
        # NEED TO ADD L2
        y_true, y_pred = InputValidator.validate_same_shape(y_true, y_pred)
        self.batch_size = y_true.shape[1]

        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def backward(self, y_true: ArrayType, y_pred: ArrayType) -> ArrayType:
        """Compute gradient of BCE loss.

        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities

        Returns:
            Gradient of loss with respect to predictions
        """
        y_true, y_pred = InputValidator.validate_same_shape(y_true, y_pred)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / ((y_pred * (1 - y_pred) + epsilon) * self.batch_size)


class CrossEntropyLossWithLogits(Loss):
    """Cross-Entropy Loss function with logits.

    Computes cross-entropy loss between true class labels and predicted
    class probabilities. Used for multi-class classification.
    """

    def __init__(self) -> None:
        """Initialize cross-entropy loss."""
        super().__init__()
        self.batch_size: Optional[int] = None

    def __call__(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        """Compute cross-entropy loss.

        Args:
            y_true: True class labels (one-hot encoded) with shape (num_classes, batch_size)
            y_pred: Predicted class probabilities with shape (num_classes, batch_size)

        Returns:
            Average cross-entropy loss
        """
        y_true, y_pred = InputValidator.validate_same_shape(y_true, y_pred)

        self.batch_size = y_true.shape[1]

        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        log_pred = np.log(y_pred)
        selected_log_preds = np.sum(log_pred * y_true, axis=0)

        loss = -np.mean(selected_log_preds)

        return loss

    def backward(self, y_true: ArrayType, y_pred: ArrayType) -> ArrayType:
        """Compute gradient of cross-entropy loss.

        Args:
            y_true: True class labels (one-hot encoded)
            y_pred: Predicted class probabilities

        Returns:
            Gradient of loss with respect to predictions
        """
        y_true, y_pred = InputValidator.validate_same_shape(y_true, y_pred)
        return (y_pred - y_true) / self.batch_size


class MSELoss(Loss):
    """Mean Squared Error Loss function.

    Computes the mean squared error between true and predicted values.
    Commonly used for regression tasks.
    """

    def __init__(self) -> None:
        """Initialize MSE loss."""
        super().__init__()
        self.batch_size: Optional[int] = None

    def __call__(self, y_true: ArrayType, y_pred: ArrayType) -> float:
        """Compute mean squared error loss.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Mean squared error loss
        """
        y_true, y_pred = InputValidator.validate_same_shape(y_true, y_pred)

        self.batch_size = y_true.shape[1]

        return np.mean(np.square(y_true - y_pred))

    def backward(self, y_true: ArrayType, y_pred: ArrayType) -> ArrayType:
        """Compute gradient of MSE loss.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Gradient of loss with respect to predictions
        """
        y_true, y_pred = InputValidator.validate_same_shape(y_true, y_pred)

        return 2 * (y_pred - y_true) / self.batch_size
