"""DIY Neural Network Library.

A neural network library built from scratch with NumPy/CuPy support for both CPU and GPU computation.

This package provides:
- Various layer types (Dense, Dropout, BatchNorm)
- Activation functions (ReLU, Sigmoid, Tanh, Softmax)
- Loss functions (BCE, CrossEntropy, MSE)
- Optimizers (SGD, SGD with Momentum, RMSProp, Adam)
- Automatic GPU/CPU device selection
- Training utilities and early stopping

Example:
    >>> from diy_neural_net import NeuralNetwork, Dense, ReLU, Adam, BCELoss
    >>> model = NeuralNetwork.Sequential([
    ...     Dense(784, 128),
    ...     ReLU(),
    ...     Dense(128, 1)
    ... ])
    >>> model.set_loss(BCELoss())
    >>> optimizer = Adam(model, lr=0.001)
"""

__version__ = "0.1.0"
__author__ = "Mohamed"

# Core components
from .Network import NeuralNetwork
from .DeviceSelector import get_numpy, is_gpu_available, ArrayType

# Layers
from .Layers import Layer, Dense, Dropout, BatchNorm

# Activations
from .Activations import Activation, ReLU, Sigmoid, Tanh

# Loss functions
from .Losses import Loss, BCELoss, CrossEntropyLossWithLogits, MSELoss

# Optimizers
from .Optimizer import (
    Optimizer,
    GradientDescent,
    GradientDescentWithMomentum,
    RMSProp,
    Adam,
)

# Utilities
from .EarlyStopping import EarlyStopping
from . import utils

# Input validation
from .InputValidation import InputValidator

__all__ = [
    # Core
    "NeuralNetwork",
    "get_numpy",
    "is_gpu_available",
    "ArrayType",
    # Layers
    "Layer",
    "Dense",
    "Dropout",
    "BatchNorm",
    # Activations
    "Activation",
    "ReLU",
    "Sigmoid",
    "Tanh",
    # Loss functions
    "Loss",
    "BCELoss",
    "CrossEntropyLossWithLogits",
    "MSELoss",
    # Optimizers
    "Optimizer",
    "GradientDescent",
    "GradientDescentWithMomentum",
    "RMSProp",
    "Adam",
    # Utilities
    "EarlyStopping",
    "utils",
    "InputValidator",
]
