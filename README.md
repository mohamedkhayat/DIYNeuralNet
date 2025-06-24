# Neural Network Implementation from Scratch

A pure NumPy/CuPy implementation of a deep neural network with modern features including dropout regularization, mini-batch gradient descent, and He/Glorot initialization. This project is designed to demonstrate how deep learning models work under the hood, offering both flexibility and performance.

---

## Features

- **Pure NumPy/CuPy Implementation**: Built using NumPy for CPU-based computations and CuPy for GPU acceleration when available.
- **Configurable Architecture**:
  - Fully customizable layer configurations using Sequential API.
  - Support for both shallow and deep architectures.
- **Advanced Features**:
  - Dropout regularization with configurable keep probabilities.
  - He and Glorot weight initialization schemes.
  - Mini-batch gradient descent with configurable batch sizes.
  - Early stopping with patience and delta thresholds to prevent overfitting.
  - Automatic CUDA support via CuPy for GPU acceleration.
- **Task Support**:
  - **Binary Classification**: Sigmoid activation with Binary Cross-Entropy Loss (BCELoss).
  - **Multi-Class Classification**: Softmax activation with Cross-Entropy Loss.
  - **Regression**: Mean Squared Error (MSE) Loss with linear output.
- **Training Features**:
  - Data shuffling and train/validation splitting.
  - Comprehensive training history tracking.
  - Real-time loss and accuracy monitoring.
- **Visualizations**:
  - Loss and accuracy curves for training and validation.
  - Sample predictions visualization for image classification.
  - Training time and device usage statistics.

---

## Installation

### 1. Clone the Repository

First, clone the project repository to your local machine:
```bash
git clone https://github.com/mohamedkhayat/DIYNeuralNet.git
cd DIYNeuralNet
```

### 2. Set Up the Environment

Choose **one** of the following methods to set up the required environment.

#### Option A: Docker Installation (Recommended)

This method uses Docker to create an isolated, pre-configured environment with all dependencies, including CUDA and CuPy for GPU support. It is the most reliable way to ensure reproducibility.

**Prerequisites:**
- [Docker Engine](https://docs.docker.com/get-docker/) installed and running.
- For GPU support: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

**Using VS Code Dev Containers (Easiest Method):**
1. Install [Visual Studio Code](https://code.visualstudio.com/) and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
2. Open the cloned `DIYNeuralNet` folder in VS Code.
3. A notification will appear in the bottom-right corner: **"Reopen in Container"**. Click it.
4. VS Code will automatically build the Docker image, start the container, and connect to it. You will now have a terminal open inside the fully configured environment.

**Using Docker from the Command Line:**
1. Build the Docker image from the project root:
   ```bash
   docker build -t diy-neural-net -f .devcontainer/Dockerfile .
   ```
2. Run the container. This command starts an interactive `bash` session inside the container with the project directory mounted.
   ```bash
   # With GPU support (recommended)
   docker run -it --rm --gpus all -v "$(pwd)":/workspace diy-neural-net bash

   # For CPU-only
   docker run -it --rm -v "$(pwd)":/workspace diy-neural-net bash
   ```

#### Option B: Local Installation (using Conda)

This method sets up the environment directly on your machine.

1. **Create and activate a Conda environment:**
   ```bash
   conda create -n neuralnet python=3.10.2
   conda activate neuralnet
   ```
2. **Install dependencies:**
   For GPU acceleration, ensure you have a CUDA-compatible NVIDIA GPU and drivers installed. The `requirements.txt` file includes `cupy`, which will be used if a GPU is detected.
   ```bash
   pip install -r requirements.txt
   ```

### 3. Download Data and Run

These steps are required regardless of the installation method you chose.

**Download MNIST Data (for multi-class classification):**
1. Go to the [Kaggle Digit Recognizer Competition](https://www.kaggle.com/competitions/digit-recognizer/data).
2. Download `train.csv`.
3. Place the downloaded file into the `Data/` directory in the project.

*(Note: The `balanced_mnist_1.csv` file for binary classification is already included in the repository.)*

**Run the example:**
From your terminal (either the Conda environment or the Docker container session), execute:
```bash
python3 src/main.py
```

---

## Quick Start

### 1. Import required modules

```python
from Network import NeuralNetwork
from Layers import Dense, Dropout
from Activations import ReLU, Sigmoid, Softmax
from Losses import BCELoss, CrossEntropyLoss, MSELoss
from DeviceSelector import get_numpy

np = get_numpy()  # Automatically selects CuPy if GPU available, otherwise NumPy
```

### 2. Define network architecture using Sequential API

```python
model = NeuralNetwork.Sequential([
    Dense(input_size=784, output_size=512, initializer='he'),
    ReLU(),
    Dense(input_size=512, output_size=256, initializer='he'),
    ReLU(),
    Dropout(keep_prob=0.85),
    Dense(input_size=256, output_size=128, initializer='he'),
    ReLU(),
    Dropout(keep_prob=0.85),
    Dense(input_size=128, output_size=32, initializer='he'),
    ReLU(),
    Dense(input_size=32, output_size=10, initializer='glorot'),  # 10 classes for MNIST
    Softmax()
])
```

### 3. Compile the model

```python
model.compile(
    learning_rate=0.999,
    criterion=CrossEntropyLoss()
)
```

### 4. Train the model

Ensure your data has the correct shape:
- `X`: `(n_features, n_samples)` - features are rows, samples are columns
- `y`: `(n_classes, n_samples)` - one-hot encoded labels

```python
history = model.fit(
    X_train=X_train,
    y_train=y_train,
    epochs=200,
    batch_size=384,  # Automatically handles mini-batching
    shuffle=True,
    validation_data=(X_val, y_val),
    early_stopping_patience=15,
    early_stopping_delta=0.001
)
```

### 5. Evaluate and visualize

```python
# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = model.accuracy_score(predictions, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training history
utils.plot_metrics(history)

# Visualize predictions (for image data)
utils.plot_image(X_test, model, n_images=6, 
                original_image_shape=(28, 28), n_classes=10)
```

---

## Architecture Overview

### Core Components

- **`Network.py`**: Main neural network class with Sequential API
- **`Layers.py`**: Dense (fully connected) and Dropout layer implementations
- **`Activations.py`**: ReLU, Sigmoid, Tanh, and Softmax activation functions
- **`Losses.py`**: Binary Cross-Entropy, Cross-Entropy, and MSE loss functions
- **`DeviceSelector.py`**: Automatic CPU/GPU device selection
- **`EarlyStopping.py`**: Early stopping implementation with patience
- **`InputValidation.py`**: Comprehensive input validation utilities
- **`utils.py`**: Data loading, preprocessing, and visualization utilities

### Supported Problem Types

The `main.py` demonstrates three problem types:
1. **Binary Classification** (problem=1): Binary MNIST classification
2. **Multi-Class Classification** (problem=2): Full MNIST digit recognition
3. **Regression** (problem=3): Synthetic regression data

---

## Key Features Explained

### Weight Initialization
- **He Initialization**: `std = sqrt(2/fan_in)` - optimal for ReLU activations
- **Glorot Initialization**: `limit = sqrt(6/(fan_in + fan_out))` - optimal for sigmoid/tanh
- **Random Initialization**: Standard normal distribution (fallback)

### Dropout Regularization
- Applied during training only
- Scales remaining activations by `1/keep_prob` to maintain expected values
- Automatically disabled during inference

### Mini-Batch Training
- Supports configurable batch sizes
- Automatic data shuffling
- Handles partial batches appropriately
- Memory-efficient for large datasets

### GPU Acceleration
- Automatic detection of CUDA-capable GPUs
- Seamless fallback to CPU if GPU unavailable
- All operations transparently accelerated when using CuPy

---

## Performance Tips

1. **GPU Usage**: Install CuPy for significant speedup on CUDA-enabled GPUs
2. **Batch Size**: Larger batches (256-512) often train faster but use more memory
3. **Learning Rate**: Start with 0.1-1.0, reduce if loss explodes
4. **Early Stopping**: Use patience=10-20 to prevent overfitting
5. **Dropout**: 0.8-0.9 keep_prob works well for most problems

---

## Example Results

On MNIST digit classification:
- **Architecture**: 784→512→256→128→32→10
- **Training Time**: ~30 seconds (GPU) / ~2 minutes (CPU)
- **Test Accuracy**: ~97-98%
- **GPU Speedup**: 3-5x faster than CPU

---

## Future Improvements

### Planned Features
1. **Changing Shape Convention**: Use `(n_samples, n_features/n_classes)`
2. **Advanced Optimizers**: Adam, RMSprop, SGD with momentum
3. **Regularization**: L1/L2 weight regularization
4. **Batch Normalization**: For faster and more stable training
5. **Convolutional Layers**: For image processing tasks
6. **Model Persistence**: Save/load trained models
7. **Learning Rate Scheduling**: Adaptive learning rate strategies

### Potential Extensions
- Different activation functions (Swish, GELU)
- Gradient clipping
- Data augmentation utilities
- Model architecture visualization

---

## Educational Value

This implementation is perfect for:
- **Students** learning how neural networks actually work
- **Educators** teaching deep learning fundamentals
- **Practitioners** understanding the math behind frameworks

The code prioritizes clarity and educational value while maintaining practical performance.

---

## Contributing

Contributions are welcome! Areas of interest:
- Additional activation functions
- New optimization algorithms
- Performance improvements
- Documentation enhancements
- Bug fixes and testing

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Built as an educational tool to demystify deep learning. Special thanks to the NumPy and CuPy communities for providing the computational foundation.