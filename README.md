# Neural Network Implementation from Scratch

A pure NumPy/CuPy implementation of a deep neural network with modern features including dropout regularization, mini batch gradient descent, and He/Glorot initialization. This project is designed to demonstrate how deep learning models work under the hood, offering both flexibility and performance.

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
from Activations import ReLU, Sigmoid
from Losses import BCELoss
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
    Sigmoid()  # Sigmoid activation for binary classification
]
```

3. **Initialize the model**:

```python
model = NeuralNetwork(
    n_classes=1,  # Binary classification
    layers=layers,
    learning_rate=0.01,
    criterion=BCELoss()
)
```

4. **Train the model**:

```python
history = model.fit(
    X_train=X_train,
    y_train=y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    early_stopping_patience=10
)
```

5. **Plot Metrics**:

```python
plot_metrics(History)
```

---

## Future Improvements

This implementation lays the groundwork for a fully functional neural network framework. Potential future enhancements include:

1. **Additional Features**:
   - Support for multi-class classification with softmax activation and regression.
   - Implementation of L2 regularization.
   - Adding support for optimizers like Adam, RMSprop, and SGD with momentum.
   - Adding other activation functions
2. **Advanced Layers**:
   - Batch normalization for faster and more stable training.
   - Convolutional layers for image-based tasks.
3. **Improved Usability**:
   - Save and load functionality for model parameters.
   - Detailed logging and visualization dashboards.

This roadmap ensures the project remains a valuable learning resource while gradually evolving into a robust deep learning library.



