# Neural Network Implementation from Scratch

A pure NumPy/CuPy implementation of a deep neural network with modern features including dropout regularization, mini-batch gradient descent, and He/Glorot initialization. This project is designed to demonstrate how deep learning models work under the hood, offering both flexibility and performance.

---

## Features

- **Pure NumPy/CuPy Implementation**: Built using NumPy for CPU-based computations and CuPy for GPU acceleration.
- **Configurable Architecture**:
  - Fully customizable layer configurations.
  - Support for both shallow and deep architectures.
- **Advanced Features**:
  - Dropout regularization with configurable rates.
  - He and Glorot weight initialization.
  - Mini-batch gradient descent for efficient training.
  - Early stopping to prevent overfitting.
  - CUDA support via CuPy for GPU acceleration.
- **Task Support**:
  - **Binary Classification**: Sigmoid activation with Binary Cross-Entropy Loss (BCELoss).
  - **Multi-Class Classification**: Softmax activation with Cross-Entropy Loss.
  - **Regression**: Mean Squared Error (MSE) Loss (ongoing development).
- **Visualizations**:
  - Loss and accuracy curves for training and validation.
  - Training time and device usage statistics.

---

## Installation

### Create and activate a Conda environment

```bash
conda create -n neuralnet python=3.8.20
conda activate neuralnet
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Clone the repository

```bash
git clone https://github.com/mohamedkhayat/DIYNeuralNet.git
cd DIYNeuralNet
python main.py
```

---

## Quick Start

1. **Import required modules**:

```python
from Network import NeuralNetwork
from Layers import Dense, Dropout
from Activations import ReLU, Sigmoid, Softmax
from Losses import BCELoss, CrossEntropyLoss, MSELoss
from Utils import *  # Import all utility functions
from DeviceSelector import *  # For selecting CPU/GPU device
np = get_numpy()
```

2. **Define network architecture**:

```python
layers = [
    Dense(input_size=n_features, output_size=64, initializer='he'),  # Input layer with He initialization
    ReLU(),
    Dense(input_size=64, output_size=64, initializer='he'),  # Hidden layer 1
    ReLU(),
    Dropout(keep_prob=0.8),  # Dropout layer with 80% keep probability
    Dense(input_size=64, output_size=32, initializer='he'),  # Hidden layer 2
    ReLU(),
    Dense(input_size=32, output_size=32, initializer='he'),  # Hidden layer 3
    ReLU(),
    Dropout(keep_prob=0.8),  # Dropout layer
    Dense(input_size=32, output_size=n_classes, initializer='glorot'),  # Output layer with Glorot initialization
    Softmax()  # Softmax activation for multi-class classification
]
```

3. **Initialize the model**:

```python
model = NeuralNetwork(
    n_classes=n_classes,  # Number of classes for classification
    layers=layers,
    learning_rate=0.01,
    criterion=CrossEntropyLoss()  # Use BCELoss for binary classification or MSELoss for regression
)
```

4. **Train the Model**:

To train the model, ensure that the input data `X` has the shape `(n_features, n_samples)` and the target labels `y` have the shape `(n_classes, n_samples)`.

```python
history = model.fit(
    X_train=X_train,            # Training features (shape: n_features x n_samples)
    y_train=y_train,            # Training labels (shape: n_classes x n_samples)
    epochs=100,                 # Number of training epochs
    batch_size=32,              # Batch size for training
    validation_data=(X_val, y_val),  # Validation data for evaluation during training
    early_stopping_patience=10  # Patience for early stopping (stopping training if no improvement)
)
```

5. **Plot Metrics**:

```python
plot_metrics(history)
```

---

## Future Improvements

This implementation lays the groundwork for a fully functional neural network framework. Here's what's already implemented and what's coming next:

### **Implemented Features**:
- **Binary Classification**: Sigmoid activation with Binary Cross-Entropy Loss (BCELoss).
- **Multi-Class Classification**: Softmax activation with Cross-Entropy Loss.
- **Regression**: Mean Squared Error (MSE) Loss (ongoing development).
- **Advanced Features**: Dropout, He/Glorot initialization, mini-batch gradient descent, and early stopping.
- **Visualizations**: Loss and accuracy curves, training time, and device usage statistics.

### **Planned Features**:
1. **Additional Loss Functions**:
   - Support for more regression loss functions (e.g., Mean Absolute Error, Huber Loss).
2. **Optimizers**:
   - Implementation of advanced optimizers like Adam, RMSprop, and SGD with momentum.
3. **Regularization**:
   - L2 regularization to prevent overfitting.
4. **Advanced Layers**:
   - Batch normalization for faster and more stable training.
   - Convolutional layers for image-based tasks.
5. **Improved Usability**:
   - Save and load functionality for model parameters.
   - Detailed logging and visualization dashboards.
   
---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
