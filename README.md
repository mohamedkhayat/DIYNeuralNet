# Neural Network Implementation from Scratch

A NumPy and CuPy implementation of a deep neural network with modern features including dropout regularization, L2 regularization, and He initialization. The architecture is fully configurable - you can specify any number of layers and units per layer to suit your needs. The code supports both CPU and GPU execution for enhanced performance.

## Features

- **Pure NumPy and CuPy Implementation**: Built using NumPy for CPU-based computations and CuPy for GPU acceleration.
- **Flexible Architecture**:
  - Define any number of layers
  - Specify any number of units per layer
  - Example: `[2, 128, 64, 32, 1]` creates a network with 2 input units, three hidden layers (128, 64, and 32 units), and 1 output unit
- **Configurable Dropout**:
  - Set custom dropout rates for each layer using a list
  - Values between 0 and 1 (1 = keep all units, 0.5 = drop half the units)
  - Example: `{'2':0.8, '3':0.8, '4':0.9}` applies 20% dropout to the first two hidden layers and 10% to the third
- **He Weight Initialization**: Proper initialization for ReLU networks
- **Dropout Regularization**: Configurable dropout rates per layer
- **L2 Regularization**: Weight decay to prevent overfitting
- **Binary Classification**: Using sigmoid activation and binary cross-entropy loss
- **Train-Test Split**: Built-in data splitting functionality
- **CUDA Support (CuPy)**: Supports both CPU and GPU computation, enabling faster training on compatible hardware.

## Requirements

```
python 3.8.20
numpy 1.19.5
cupy 8.3.0
```

GPU acceleration only works with Nvidia GPUs and will automatically be turned on if an nvidia graphic's card is detected, otherwise it will default to CPU

## Instructions

```
conda create -n cupy python=3.8.20 numpy=1.19.5
conda activate cupy
conda install cupy
git clone https://github.com/mohamedkhayat/DIYNeuralNet.git
cd DIYNeuralNet
```
You can play around with the code, change the architechture, hyper paramaters, activations,
dataset, once you want to train the model save your changes and then run
```
python Train.py
```

## Usage

### Basic Example

```python
# Example of how to define and train the model

# Define network architecture
layer_dims = [n_features, 64, 64, 32, n_classes]  # Input layer, hidden layers, and output layer

# Dropout rates for layers (layer '2' will have 10% dropout, layer '3' will have 15% dropout)
dropout_rates = {'2': 0.9, '3': 0.85}

# Learning rate and L2 regularization parameter
learning_rate = 0.1
lamb = 0.01

# Initialize the Neural Network model
model = NeuralNetwork(n_classes, layer_dims, dropout_rates, learning_rate, lamb)

# Train the model on the data (X, y) for 300 epochs
model.fit(X, y, 300)

# Predict on the test set
y_pred_test = model.predict(X_test)

# Calculate the test accuracy
test_accuracy = model.accuracy_score(y_pred_test, y_test)
print(f"Test Accuracy: {float(test_accuracy):.4f}")
```

### Customizing Network Architecture

You can easily modify the network architecture by changing the `layer_dims` list and `dropout_rates` dictionary:

```python
# Wide network with aggressive dropout
layer_dims = [2, 256, 256, 1]  # Input layer, two hidden layers, output layer
dropout_rates = {1: 0.9, 2: 0.5, 3: 0.5}  # Keep 90% in input, 50% dropout on both hidden layers

# Deep network with varying dropout
layer_dims = [2, 64, 64, 32, 32, 16, 1]  # Input layer, 5 hidden layers, output layer
dropout_rates = {2: 0.8, 3: 0.8, 4: 0.7, 5: 0.7, 6: 0.9}  # Varying dropout rates

# Pyramid architecture with graduated dropout
layer_dims = [2, 128, 64, 32, 16, 1]  # Input layer, 4 hidden layers, output layer
dropout_rates = {2: 0.7, 3: 0.8, 4: 0.9}  # Gradually decreasing dropout
```

## Implementation Details

### Key Components

1. **Weight Initialization**
   - He initialization for weights

2. **Forward Propagation**
   - ReLU activation for hidden layers
   - Sigmoid activation for output layer
   - Optional dropout during training

3. **Backward Propagation**
   - Gradient computation with dropout masks
   - L2 regularization gradients

4. **Training**
   - Binary cross-entropy loss
   - Basic gradient descent optimization
   - GPU support for faster training (using CuPy)

## Future Improvements

- [ ] Add mini-batches
- [ ] Implement More Optimizers
- [ ] Add more Weight Initialization methods
- [ ] Add batch normalization
- [ ] Implement gradient clipping 
- [ ] Improve the model customization (layers and dropout selection)
- [ ] Implement model saving and loading functionality
- [ ] Add more activation functions like LeakyReLU, Tanh, and Softmax
- [ ] Add support for multi-class classification with categorical cross-entropy
- [ ] Add support for regression tasks with MSE and RMSE loss functions
- [ ] Add a UI
- [ ] add Learning Rate Scheduling
- [ ] Implement autograd engine
- [ ] add Hyper paramater Tuning
- [ ] Add More metrics
- [ ] Add Learning Rate Schedulers

## License

MIT License
