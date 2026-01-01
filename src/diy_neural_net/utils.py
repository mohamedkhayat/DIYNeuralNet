from random import randint
from typing import Tuple, Generator, Any
from .DeviceSelector import get_numpy, is_gpu_available, ArrayType
import pathlib
import matplotlib.pyplot as plt
import numpy as np_cpu

np = get_numpy()

np.random.seed(42)


def train_test_split(
    X: ArrayType, y: ArrayType, test_size: float = 0.2
) -> Tuple[ArrayType, ArrayType, ArrayType, ArrayType]:
    """Split data into training and testing sets.

    Args:
        X: Feature data with shape (n_features, n_samples)
        y: Target data with shape (n_targets, n_samples)
        test_size: Fraction of data to use for testing (default: 0.2)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    m = X.shape[1]

    indices = np.random.permutation(m)

    test_size = int(m * test_size)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_test, y_test = X[:, test_indices], y[:, test_indices]
    X_train, y_train = X[:, train_indices], y[:, train_indices]

    return X_train, X_test, y_train, y_test


def generate_xor_data(
    n_samples: int, np: Any, noise: float = 0.01
) -> Tuple[ArrayType, ArrayType]:
    """Generate XOR dataset for binary classification.

    Args:
        n_samples: Number of samples to generate
        np: Numpy-like library (numpy or cupy)
        noise: Amount of noise to add to the data (default: 0.01)

    Returns:
        Tuple of (X, y) where X is features and y is binary labels
    """
    X = np.random.rand(2, n_samples) * 2 - 1  # Centered around 0
    y = np.logical_xor(X[0, :] > 0, X[1, :] > 0).astype(int).reshape(1, -1)
    X += np.random.normal(0, noise, X.shape)  # Add noise
    return X, y


def generate_regression_data(
    n_samples: int = 1000, n_features: int = 1, noise: float = 0.1, np: Any = np
) -> Tuple[ArrayType, ArrayType]:
    """Generate synthetic regression dataset.

    Args:
        n_samples: Number of samples to generate (default: 1000)
        n_features: Number of features (default: 1)
        noise: Standard deviation of noise (default: 0.1)
        np: Numpy-like library (numpy or cupy)

    Returns:
        Tuple of (X, y) where X is features and y is continuous targets
    """
    # Generate random features
    X = np.random.randn(n_features, n_samples)

    # Generate weights (1 to n_features)
    weights = np.arange(1, n_features + 1).reshape(-1, 1)

    # Calculate y = w1*x1 + w2*x2 + ... + wn*xn
    y = np.sum(weights * X, axis=0, keepdims=True)

    # Add noise
    y += noise * np.random.randn(1, n_samples)

    return X, y


def plot_image(
    X: ArrayType,
    model: Any,
    n_images: int,
    original_image_shape: Tuple[int, int] = (28, 28),
    n_classes: int = 1,
) -> None:
    """Plot sample images with model predictions.

    Args:
        X: Image data with shape (n_pixels, n_samples)
        model: Trained model with predict method
        n_images: Number of images to plot
        original_image_shape: Shape to reshape images to (default: (28, 28))
        n_classes: Number of classes for labeling (default: 1 for binary)
    """
    plt.figure(figsize=(6, 6))

    indices = [randint(0, len(X)) for _ in range(n_images)]

    HEIGHT, WIDTH = original_image_shape

    for i, idx in enumerate(indices):
        test_example = X[:, idx]

        if len(test_example.shape) == 1:
            test_example = test_example.reshape(-1, 1)

        test_pred = model.predict(test_example)

        plt.subplot(2, (n_images + 1) // 2, i + 1)

        test_example = test_example.reshape(HEIGHT, WIDTH) * 255.0

        if n_classes == 1:
            plt.title("One" if test_pred.item() == 1 else "not a One")

        else:
            plt.title(str(test_pred.item()))

        plt.imshow(
            to_cpu(test_example) if is_gpu_available() else test_example, cmap="gray"
        )
        plt.axis("off")

    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"Plotting image failed due to : {e}, saving image instead")
        plt.savefig("example_preds.png")


def load_binary_mnist() -> Tuple[ArrayType, ArrayType]:
    """Load binary MNIST dataset (1s vs non-1s).

    Returns:
        Tuple of (X, y) where X is image data and y is binary labels
    """
    data = np_cpu.loadtxt(
        pathlib.Path("Data", "balanced_mnist_1.csv"), delimiter=",", skiprows=1
    )
    X = data[:, 1:].transpose()
    y = data[:, 0].reshape(1, -1)
    return np.asarray(X), np.asarray(y)


def load_mnist() -> Tuple[ArrayType, ArrayType]:
    """Load full MNIST dataset.

    Returns:
        Tuple of (X, y) where X is image data and y is digit labels
    """
    data = np_cpu.loadtxt(pathlib.Path("Data", "train.csv"), delimiter=",", skiprows=1)

    X = data[:, 1:].T / 255.0
    y = data[:, 0].reshape(1, -1)

    y = y.flatten().astype(np.int64)

    n_classes = len(np.unique(y))
    n_samples = len(y)

    one_hot = np.zeros((n_classes, n_samples))

    one_hot[y, np.arange(n_samples)] = 1

    y = one_hot

    return np.asarray(X), np.asarray(y)


def to_cpu(data):
    if hasattr(data, "get"):
        return data.get().copy()

    if isinstance(data, list):
        return np_cpu.array([x.get() if hasattr(x, "get") else x for x in data])

    return np_cpu.array(data, copy=True)


def plot_metrics(History: dict) -> None:
    """Plot training metrics from training history.

    Args:
        History: Dictionary containing training metrics with keys:
                'Train_losses', 'Test_losses', 'Train_accuracy', 'Test_accuracy'
    """
    try:
        train_accuracy = to_cpu(History["Train_accuracy"])
        test_accuracy = to_cpu(History["Test_accuracy"])
        train_losses = to_cpu(History["Train_losses"])
        test_losses = to_cpu(History["Test_losses"])

        plt.figure(1)
        plt.clf()
        plt.title("Loss per Epoch")
        plt.plot(to_cpu(train_losses), label="Train loss", c="r")
        plt.plot(to_cpu(test_losses), label="Test loss", c="b")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        y_train_max = max(train_losses)
        y_train_min = min(train_losses)

        y_test_max = max(test_losses)
        y_test_min = min(test_losses)

        y_min = min(y_test_min, y_train_min)
        y_max = max(y_train_max, y_test_max)

        plt.axis([0, len(train_losses), y_min - y_min * 0.1, y_max + y_max * 0.1])
        plt.grid(True)
        try:
            plt.show()
        except Exception as e:
            print(f"Plotting image failed due to : {e}, saving image instead")
            plt.savefig("losses.png")

        plt.figure(2)
        plt.clf()
        plt.title("Accuracy per Epoch")
        plt.plot(to_cpu(train_accuracy), label="Train accuracy", c="r")
        plt.plot(to_cpu(test_accuracy), label="Test accuracy", c="b")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        y_train_max = max(train_accuracy)
        y_train_min = min(train_accuracy)

        y_test_max = max(test_accuracy)
        y_test_min = min(test_accuracy)

        y_min = min(y_test_min, y_train_min)
        y_max = max(y_train_max, y_test_max)

        plt.axis([0, len(train_accuracy), y_min - y_min * 0.1, y_max + y_max * 0.1])
        plt.grid(True)
        try:
            plt.show()
        except Exception as e:
            print(f"Plotting image failed due to : {e}, saving image instead")
            plt.savefig("accuracies.png")

    except Exception as e:
        print(
            f"Error : {e}, PS : this function expects you chose to input validation data during fit if you chose not to that could be the source of the issue"
        )


def create_mini_batches(
    X: ArrayType,
    y: ArrayType,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = True,
) -> Generator[Tuple[ArrayType, ArrayType], None, None]:
    """Create mini-batches for training.

    Args:
        X: Feature data with shape (n_features, n_samples)
        y: Target data with shape (n_targets, n_samples)
        batch_size: Size of each batch (default: 64)
        shuffle: Whether to shuffle data before batching (default: True)
        drop_last: Whether to drop the last incomplete batch (default: True)

    Yields:
        Tuples of (X_batch, y_batch) for each mini-batch
    """
    num_samples = X.shape[1]
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)

        if drop_last and end_idx - start_idx < batch_size:
            break

        batch_indices = indices[start_idx:end_idx]

        yield X[:, batch_indices], y[:, batch_indices]
