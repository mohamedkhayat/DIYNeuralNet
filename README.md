# Neural Network Implementation from Scratch

A NumPy implementation of a deep neural network with modern features including dropout regularization, L2 regularization, and He initialization. The architecture is fully configurable - you can specify any number of layers and units per layer to suit your needs.

## Features

- **Pure NumPy Implementation**: Built using only NumPy for numerical computations
- **Flexible Architecture**: 
  - Define any number of layers
  - Specify any number of units per layer
  - Example: `[2, 128, 64, 32, 1]` creates a network with 2 input units, three hidden layers (128, 64, and 32 units), and 1 output unit
- **He Weight Initialization**: Proper initialization for ReLU networks
- **Dropout Regularization**: Configurable dropout rates per layer
- **L2 Regularization**: Weight decay to prevent overfitting
- **Binary Classification**: Using sigmoid activation and binary cross-entropy loss
- **Train-Test Split**: Built-in data splitting functionality

## Requirements

```
numpy
```

## Usage

### Basic Example

```python
# Generate sample XOR data
X, y = generate_xor_data(n_samples=2000, noise=0.2)

# Define custom network architecture
layer_dims = [2, 128, 64, 32, 1]  # [input_dim, hidden1, hidden2, hidden3, output_dim]
dropout_rates = [1, 0.8, 0.8, 1, 1]  # Dropout keep probabilities per layer

# Initialize weights
weights = he_initialization(layer_dims)

# Train the model
weights = train(X_train, y_train, 
                weights,
                epochs=1000,
                learning_rate=0.01,
                lamb=0.01,  # L2 regularization parameter
                dropout=dropout_rates)

# Make predictions
y_pred, _ = forward_prop(X_test, weights, training=False)
accuracy = evaluate(y_pred, y_test)
```

### Customizing Network Architecture

You can easily modify the network architecture by changing the `layer_dims` list:
```python
# Wide network
layer_dims = [2, 256, 256, 1]  # Two large hidden layers

# Deep network
layer_dims = [2, 64, 64, 32, 32, 16, 1]  # Many smaller layers

# Pyramid architecture
layer_dims = [2, 128, 64, 32, 16, 1]  # Gradually decreasing units
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

## Future Improvements

- [ ] Add mini-batch gradient descent
- [ ] Implement Adam optimizer
- [ ] Add early stopping
- [ ] Potentially add batch normalization
- [ ] Implement gradient clipping if needed
- [ ] Improve the model customization (layers and dropout selection)
- [ ] Potentially add a UI
      

## License

MIT License
