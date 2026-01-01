import time
from typing import List, Union, Optional, Tuple, Dict, Any
from .Activations import Activation
from .DeviceSelector import get_numpy, ArrayType
from .EarlyStopping import EarlyStopping
from .utils import create_mini_batches
from .Layers import Dense, Layer
from .Losses import CrossEntropyLossWithLogits, BCELoss, Loss, MSELoss
from .InputValidation import InputValidator

np = get_numpy()


class NeuralNetwork:
    """Neural network class for building and training deep learning models.

    This class provides a complete neural network implementation with support
    for various layers, activation functions, loss functions, and training procedures.
    """

    def __init__(self, layers: List[Union[Layer, Activation, Loss]]) -> None:
        """Initialize neural network with layers.

        Args:
            layers: List of layers, activations, and loss functions
        """
        self.layers: List[Union[Layer, Activation, Loss]] = (
            InputValidator.validate_layers(layers)
        )
        self.training: bool = True
        self.criterion: Optional[Loss] = None

    @staticmethod
    def Sequential(layers: List[Union[Layer, Activation, Loss]]) -> "NeuralNetwork":
        """Create a sequential neural network model.

        Args:
            layers: List of layers, activations, and loss functions

        Returns:
            NeuralNetwork instance with sequential architecture
        """
        layers = InputValidator.validate_layers(layers)
        model = NeuralNetwork(layers)
        return model

    def set_loss(self, loss: Loss) -> None:
        """Set the loss function for the network.

        Args:
            loss: Loss function to use for training

        Returns:
            Self for method chaining
        """
        self.criterion = InputValidator.validate_criterion(loss)

    def forward(self, X: ArrayType, train: Optional[bool] = None) -> ArrayType:
        """Forward pass through the network.

        Args:
            X: Input data with shape (input_size, batch_size)
            train: Whether to run in training mode (None uses self.training)

        Returns:
            Network output with appropriate shape
        """
        if train is None:
            train = self.training

        output = X

        for layer in self.layers:
            output = layer.forward(output, train)

        return output

    def backprop(self, dA: ArrayType) -> None:
        """Backward pass through the network.

        Args:
            dA: Gradient of loss with respect to network output
        """
        for layer in reversed(list(self.layers)):
            dA = layer.backward(dA)

    def zero_grad(self) -> None:
        """Reset gradients of all trainable layers to zero."""
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.zero_grad()

    def optimize(self) -> None:
        """Apply gradients to update parameters (deprecated - use optimizer instead).

        Note:
            This method is deprecated. Use external optimizers instead.
        """
        for layer in self.layers:
            if hasattr(layer, "params"):
                for param in layer.params:
                    layer.params[param] -= self.learning_rate * layer.grads["d" + param]

    def fit(
        self,
        X_train: ArrayType,
        y_train: ArrayType,
        optimizer: Any,
        epochs: int = 30,
        batch_size: int = 64,
        shuffle: bool = True,
        validation_data: Optional[Tuple[ArrayType, ArrayType]] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_delta: float = 0,
    ) -> Dict[str, Any]:
        """Train the neural network.

        Args:
            X_train: Training features with shape (n_features, n_samples)
            y_train: Training labels with shape (n_targets, n_samples)
            optimizer: Optimizer instance for parameter updates
            epochs: Number of training epochs (default: 30)
            batch_size: Size of mini-batches (default: 64)
            shuffle: Whether to shuffle data each epoch (default: True)
            validation_data: Optional tuple of (X_val, y_val) for validation
            early_stopping_patience: Number of epochs to wait before stopping (default: None)
            early_stopping_delta: Minimum improvement required (default: 0)

        Returns:
            Dictionary containing training history with keys:
            - 'Train_losses': List of training losses per epoch
            - 'Test_losses': List of validation losses per epoch
            - 'Train_accuracy': List of training accuracies per epoch
            - 'Test_accuracy': List of validation accuracies per epoch
            - 'Time_Elapsed': Total training time in seconds
        """
        History = {}

        train_losses = []
        test_losses = []

        train_accuracies = []
        test_accuracies = []

        start_time = time.time()

        if early_stopping_patience is not None and early_stopping_patience >= 1:
            er = EarlyStopping(early_stopping_patience, early_stopping_delta)

        for epoch in range(epochs):
            avg_train_loss, avg_train_accuracy = self.train(
                X_train, y_train, optimizer, batch_size, shuffle
            )

            train_losses.append(float(avg_train_loss))
            train_accuracies.append(float(avg_train_accuracy))

            if validation_data is not None:
                X_test, y_test = validation_data

                test_loss, test_accuracy = self.evaluate(X_test, y_test, batch_size)

                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)

            if epoch % 10 == 0:
                print(f"Epoch : {epoch}")
                print(
                    f"Train Loss : {float(avg_train_loss):.4f} Test Loss : {float(test_loss):.4f}"
                )

            if early_stopping_patience is not None and er(test_loss):
                break

        end_time = time.time()

        History = {
            "Train_losses": train_losses,
            "Test_losses": test_losses,
            "Train_accuracy": train_accuracies,
            "Test_accuracy": test_accuracies,
            "Time_Elapsed": end_time - start_time,
        }

        return History

    def set_to_train(self) -> None:
        """Set network to training mode."""
        self.training = True

    def set_to_eval(self) -> None:
        """Set network to evaluation mode."""
        self.training = False

    def train(
        self,
        X_train: ArrayType,
        y_train: ArrayType,
        optimizer: Any,
        batch_size: int,
        shuffle: bool,
    ) -> Tuple[float, float]:
        """Perform one training epoch.

        Args:
            X_train: Training features with shape (n_features, n_samples)
            y_train: Training labels with shape (n_targets, n_samples)
            optimizer: Optimizer instance for parameter updates
            batch_size: Size of mini-batches
            shuffle: Whether to shuffle data

        Returns:
            Tuple of (average_loss, average_accuracy) for the epoch
        """
        epoch_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_samples = 0

        self.set_to_train()

        mini_batches = create_mini_batches(
            X_train, y_train, batch_size=batch_size, shuffle=shuffle, drop_last=True
        )

        for X_batch, y_batch in mini_batches:
            self.zero_grad()
            y_pred = self.forward(X_batch, train=True)

            loss = self.criterion(y_batch, y_pred)
            epoch_loss += float(loss)

            if isinstance(self.criterion, BCELoss):
                y_pred_labels = (y_pred > 0.5).astype(int)
                batch_correct = np.sum(y_pred_labels == y_batch)

                dA = self.criterion.backward(y_batch, y_pred)
                self.backprop(dA)

            elif isinstance(self.criterion, CrossEntropyLossWithLogits):
                y_pred_labels = np.argmax(y_pred, axis=0)
                y_true_labels = np.argmax(y_batch, axis=0)
                batch_correct = np.sum(y_pred_labels == y_true_labels)

                dA = self.criterion.backward(y_batch, y_pred)
                self.backprop(dA)

            elif isinstance(self.criterion, MSELoss):
                dA = self.criterion.backward(y_batch, y_pred)
                self.backprop(dA)

            # self.optimize()
            optimizer.step()

            if not (isinstance(self.criterion, MSELoss)):
                correct_predictions += int(batch_correct)
                total_samples += y_batch.shape[1]

            num_batches += 1

        if not isinstance(self.criterion, MSELoss):
            avg_train_accuracy = correct_predictions / total_samples
        else:
            avg_train_accuracy = 0

        avg_train_loss = epoch_loss / num_batches

        return avg_train_loss, avg_train_accuracy

    def evaluate(
        self, X_test: ArrayType, y_test: ArrayType, batch_size: int
    ) -> Tuple[float, float]:
        """Evaluate the model on test/validation data.

        Args:
            X_test: Test features with shape (n_features, n_samples)
            y_test: Test labels with shape (n_targets, n_samples)
            batch_size: Size of mini-batches for evaluation

        Returns:
            Tuple of (average_loss, average_accuracy) on test data
        """
        self.set_to_eval()

        # Create batches for test data too
        test_batches = create_mini_batches(
            X_test, y_test, batch_size=batch_size, shuffle=False, drop_last=False
        )

        test_loss = 0.0
        test_correct = 0
        test_total = 0
        test_num_batches = 0

        for X_batch_test, y_batch_test in test_batches:
            y_pred_test = self.forward(X_batch_test, train=False)
            batch_test_loss = self.criterion(y_batch_test, y_pred_test)
            test_loss += float(batch_test_loss)

            if isinstance(self.criterion, BCELoss):
                y_pred_test_labels = (y_pred_test > 0.5).astype(int)
                test_correct += np.sum(y_pred_test_labels == y_batch_test)

            elif isinstance(self.criterion, CrossEntropyLossWithLogits):
                y_pred_test_labels = np.argmax(y_pred_test, axis=0)
                y_true_test = np.argmax(y_batch_test, axis=0)
                test_correct += np.sum(y_pred_test_labels == y_true_test)

            test_total += y_batch_test.shape[1]
            test_num_batches += 1

        if not isinstance(self.criterion, MSELoss):
            test_accuracy = test_correct / test_total
        else:
            test_accuracy = 0

        test_loss = test_loss / test_num_batches  # Average loss over batches

        return test_loss, test_accuracy

    def predict(self, X: ArrayType) -> ArrayType:
        """Make predictions on input data.

        Args:
            X: Input features with shape (n_features, n_samples) or (n_features,)

        Returns:
            Predictions based on the loss function type:
            - Binary classification: 0 or 1
            - Multi-class classification: class indices
            - Regression: continuous values
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        predictions = self.forward(X)

        if isinstance(self.criterion, BCELoss):
            return (predictions > 0.5).astype(int)

        elif isinstance(self.criterion, CrossEntropyLossWithLogits):
            return np.argmax(predictions, axis=0)

        else:
            return predictions

    def accuracy_score(self, y_pred: ArrayType, y_true: ArrayType) -> float:
        """Calculate accuracy between predictions and true labels.

        Args:
            y_pred: Predicted labels
            y_true: True labels with shape (n_targets, n_samples)

        Returns:
            Accuracy score as a float between 0 and 1
        """
        batch_size = y_true.shape[1]

        if isinstance(self.criterion, CrossEntropyLossWithLogits):
            y_true = np.argmax(y_true, axis=0)

        correct = np.sum(y_pred == y_true)
        return float(correct / batch_size)
