from numpy import inf
from .InputValidation import InputValidator


class EarlyStopping:
    """Early stopping callback to prevent overfitting during training.

    Monitors validation loss and stops training when the loss stops improving
    for a specified number of epochs (patience).
    """

    def __init__(self, patience: int, delta: float = 0) -> None:
        """Initialize early stopping callback.

        Args:
            patience: Number of epochs with no improvement after which training stops
            delta: Minimum change in validation loss to qualify as improvement
        """
        self.patience: int = InputValidator.validate_patience(patience)
        self.delta: float = InputValidator.validate_delta(delta)
        self.counter: int = 0
        self.best_val_loss: float = inf
        self.current_epoch: int = 0
        self.done: bool = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop based on validation loss.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop, False otherwise
        """
        self.current_epoch += 1

        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.counter = 0

        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(
                f"Early stopping triggered during epoch : {self.current_epoch}\nbest val_loss = {self.best_val_loss:.4f}"
            )
            self.done = True

        return self.done
