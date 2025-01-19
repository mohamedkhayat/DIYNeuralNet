from utils import *
from DeviceSelector import *
from Network import NeuralNetwork
from Losses import BCELoss

from Layers import Dense,Dropout
from Activations import ReLU,Sigmoid

np = get_numpy()
#import numpy as np

_GPU_AVAILABLE = is_gpu_available()

# Setting random seed for reproducibility

np.random.seed(42)

# Loading Mnist data
if __name__ == "__main__":
  try:
    X, y = load_mnist()
    
  except:
    
    #Falling back to generating XOR in case of errors
    
    n_samples = 2000
    X, y = generate_xor_data(n_samples, np)

  # Extracting n_features and n_classes from X and y

  n_features = X.shape[0]
  n_classes = y.shape[0]

  # specifying percantage of data to be used for validation

  ratio = 0.2

  # if GPU is available, transform X and y to cupy ndarray
  # effectivaly loading the data into the gpu

  if(_GPU_AVAILABLE):
    X, y = np.asarray(X), np.asarray(y)

  # split data into train and validation data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio)

  # How Big/Small your weight updates are, essentially, how fast your model learns 
  learning_rate = 0.03
  
  # Initializing our Loss function, We use BCELoss because its a binary classification problem
  loss = BCELoss()

  layers = [
    Dense(n_features, 64, 'he'), # Input layer, input size = n_features, output_size (n of units) = 64, HE init because it uses ReLU
    ReLU(), # ReLU Activation Function
    Dense(64, 64, 'he'), # First hidden layer, input size = 64, output size = 64, he init too because it uses ReLU
    ReLU(), # ReLU again
    Dropout(keep_prob = 0.8), # Dropout layer, turns off 10% of units
    Dense(64, 32, 'he'), # Second Hidden layer, input size = 64, output size = 32, he init again because it uses ReLU
    ReLU(), # relu again
    Dense(32, 32, 'he'), # Third Hidden layer input size = 32, output size = 32 he init again
    ReLU(), # relu again
    Dropout(keep_prob = 0.8), # Dropout layer, turns off 10% of units
    Dense(32, n_classes, 'glorot'), # Output layer, input size = 32, output size = n_classes (1), glorot init because it uses sigmoid
    Sigmoid() # Sigmoid Activation function because we are using BCELoss (it's a binary classification problem, predicting if an image is 1 or not 1)
  ]

  model = NeuralNetwork(n_classes = n_classes, # Needed
                        layers = layers, # Needed
                        learning_rate = learning_rate, # Needed
                        criterion = loss, # Needed
                        )

  # Training the model for 300 iterations

  History = model.fit(X_train = X_train, # Needed
                      y_train = y_train, # Needed
                      batch_size = 64*8, # Optional, defaults to 64
                      shuffle = True, # Optional, defaults to True
                      epochs = 100, # Needed
                      validation_data = (X_test, y_test), # Optional if you dont need plotting
                      early_stopping_patience = 15, # Optional
                      early_stopping_delta = 0.001 # Optional
            )

  # Print Time Elapsed and Device used to train

  print(f"Time Elapsed : {History['Time_Elapsed']:.2F} seconds on : {'GPU' if _GPU_AVAILABLE else 'CPU'}")

  plot_metrics(History)

  # using the model to make predictions on the train set

  y_pred_train = model.predict(X_train)

  # using predictions to calculate model's accuracy on the train set

  train_accuracy = model.accuracy_score(y_pred_train, y_train)
  print(f"Train Accuracy : {float(train_accuracy):.4f}")

  # using the model to make predictions on the test set 

  y_pred_test = model.predict(X_test)

  # using predictions to calculate model's accuracy on the test set

  test_accuracy = model.accuracy_score(y_pred_test, y_test)
  print(f"Test Accuracy : {float(test_accuracy):.4f}")

  #Plotting random n_images from the test set with their predictions
  plot_image(X = X_test, model = model, n_images = 6, original_image_shape = (28, 28))