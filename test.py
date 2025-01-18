from utils import *
from DeviceSelector import *
from OOPNet import NeuralNetwork
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


  print(f"Shape of X : {X.shape}")

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

  learning_rate = 0.03

  loss = BCELoss()

  layers = [
    Dense(n_features, 64, 'he'),
    ReLU(),
    Dense(64, 64, 'he'),
    ReLU(),
    Dropout(keep_prob = 0.85),
    Dense(64, 32, 'he'),
    ReLU(),
    Dense(32, 32, 'he'),
    ReLU(),
    Dropout(keep_prob = 0.85),
    Dense(32, n_classes, 'glorot'),
    Sigmoid()
  ]

  model = NeuralNetwork(n_classes = n_classes, # Needed
                        layers = layers, # Needed
                        learning_rate = learning_rate, # Needed
                        criterion = loss, # Needed
                        )

  # Training the model for 300 iterations

  History = model.fit(X_train = X_train,
            y_train = y_train,
            epochs = 10000,
            validation_data = (X_test, y_test),
            EarlyStopping_Patience = 10,
            EarlyStoppingDelta = 0.001
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