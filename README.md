# DIY Neural Network Library

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green)
![Status](https://img.shields.io/badge/status-active-success)
![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)

A modular deep learning framework built entirely from scratch using **NumPy** and **CuPy**. 

**This project is first and foremost an educational tool.** It is designed to bridge the gap between reading equations in a textbook and seeing them run on a GPU. It implements modern features like **Adam Optimization** and **Batch Normalization** without the "black box" magic of Autograd engines like PyTorch or TensorFlow.

---

## 🎯 Goal & Philosophy

The primary goal of this project is to provide a **readable, approachable codebase** for beginners who want to understand *exactly* how Deep Learning works under the hood.

### 1. A "Glass Box" Playground
This library offers the best of both worlds:
- **For the Beginner:** The complex training loop is abstracted away. You can treat the library like a playground—rapidly swapping Optimizers (SGD vs Adam), tweaking Hyperparameters (Learning Rate, Momentum), and changing Weight Initializations (He vs Glorot) to build intuition on how they affect convergence.
- **For the Curious:** Unlike big frameworks where the training logic is hidden behind C++ bindings, here the `fit()` loop is written in pure, readable Python. You can click "Go to Definition" and understand exactly how gradients are applied and metrics are tracked.

### 2. An Extensible Blueprint
This project serves as a template for anyone wanting to build their own deep learning components. It provides a standardized architecture:
- Want to add **LeakyReLU**? Subclass `Activation`.
- Want to implement **Huber Loss**? Subclass `Loss`.
- Want to try **Adagrad**? Subclass `Optimizer`.

You simply inherit from the base class and implement `__init__`, `forward`, and `backward` (or `step`). The modular design handles the rest, removing the intimidation factor of contributing to complex libraries.

### 3. No Magic, Just Math
- **No C++ / CUDA Code Required:** The heavy lifting is done by `numpy` (or `cupy`), but the logic is pure Python.
- **Math $\leftrightarrow$ Code:** The code is written to look as close to the mathematical derivation as possible.
- **No Autograd Magic:** Gradients are calculated via explicit backward pass methods, forcing you to understand the Chain Rule rather than relying on automatic differentiation.

## ⚡ Features

- **Hardware Acceleration**: Automatic backend switching. Uses **CuPy** if a GPU is detected, falls back to **NumPy** on CPU.
- **Modern Optimizers**:
  - **Adam** (Adaptive Moment Estimation) with bias correction.
  - **RMSProp** & **SGD** with Momentum.
- **Advanced Layers**:
  - **Batch Normalization**: Full forward/backward implementation with running statistics for inference.
  - **Dropout**: Inverted dropout scaling for training stability.
  - **Dense**: Fully connected layers with He/Glorot initialization.
- **Robust Training Loop**:
  - Mini-batch processing.
  - Early Stopping with patience.
  - Real-time metric tracking.
- **Tasks**:
  - Binary Classification (`BCELoss`)
  - Multi-Class Classification (`CrossEntropyLossWithLogits`)
  - Regression (`MSELoss`)

---

## 🛠️ Installation

### Option A: Docker (Recommended for GPU)
The provided Docker image is optimized for the latest NVIDIA hardware (RTX 5090 / CUDA 12.8) and handles all compiler dependencies for CuPy.

1. **Build the image:**
   ```bash
   docker build -t diy-neural-net .
   ```

2. **Run the container:**
   ```bash
   docker run -it --rm --gpus all -v "$(pwd)":/workspace diy-neural-net bash
   ```

### Option B: Local Installation (pip)
If you have a CUDA environment set up locally:

```bash
# Clone the repo
git clone https://github.com/mohamedkhayat/DIYNeuralNet.git
cd DIYNeuralNet

# Install in editable mode with GPU support
pip install -e ".[gpu]"

# Or for CPU only
pip install -e .
```

---

## 🚀 Quick Start

Here is how to train a model on MNIST using the library.

### 1. Define the Architecture
The API mimics the Sequential style of modern frameworks but requires explicit understanding of input/output dimensions.

```python
from diy_neural_net import NeuralNetwork
from diy_neural_net.Layers import Dense, BatchNorm, Dropout
from diy_neural_net.Activations import ReLU
from diy_neural_net.Losses import CrossEntropyLossWithLogits
from diy_neural_net.Optimizer import Adam

# 784 Inputs (MNIST) -> 10 Outputs (Digits)
model = NeuralNetwork.Sequential([
    Dense(784, 512, initializer="he"),
    BatchNorm(512),
    ReLU(),
    Dropout(keep_prob=0.8),
    
    Dense(512, 128, initializer="he"),
    BatchNorm(128),
    ReLU(),
    
    Dense(128, 10, initializer="glorot") 
    # Note: Softmax is handled numerically inside CrossEntropyLossWithLogits
])
```

### 2. Configure & Train
```python
# Load Data
from diy_neural_net import utils
X, y = utils.load_mnist()
X_train, X_test, y_train, y_test = utils.train_test_split(X, y)

# Set Loss & Optimizer
model.set_loss(CrossEntropyLossWithLogits())
optimizer = Adam(model, lr=0.001)

# Train
history = model.fit(
    X_train, y_train,
    optimizer=optimizer,
    epochs=20,
    batch_size=256,
    validation_data=(X_test, y_test),
    early_stopping_patience=5
)

# Visualize
utils.plot_metrics(history)
```

---

## 📂 Project Structure

```
├── src/diy_neural_net/
│   ├── Network.py          # Main model class (Sequential)
│   ├── Layers.py           # Dense, Dropout, BatchNorm
│   ├── Optimizer.py        # SGD, RMSProp, Adam
│   ├── Activations.py      # ReLU, Sigmoid, Tanh
│   ├── Losses.py           # BCE, MSE, CrossEntropy
│   ├── DeviceSelector.py   # CPU/GPU logic
│   └── utils.py            # Data loading & Plotting
├── tests/                  # Unit and Integration tests
├── Data/                   # Datasets (MNIST, etc.)
├── notebooks/              # Jupyter notebooks for demos
├── pyproject.toml          # Package configuration
└── Dockerfile              # CUDA 12.8 environment
```

---

## 📊 Benchmarks

On the MNIST dataset (35,000 images):

| Optimizer | Accuracy | Time/Epoch (RTX 5090) |
|-----------|----------|-----------------------|
| GradientDescent | 92.4% | ~0.8s |
| **Adam** | **97.8%** | **~0.9s** |

---

## 🤝 Contributing

Contributions are welcome!
---

## License

MIT License. See [LICENSE](LICENSE) for details.