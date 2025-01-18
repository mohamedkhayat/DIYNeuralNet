from utils import *
from DeviceSelector import *
from OOPNet import NeuralNetwork
from Losses import BCELoss

from Layers import Dense
from Activations import ReLU,Sigmoid

from random import randint
#np = get_numpy()
import numpy as np

_GPU_AVAILABLE = is_gpu_available()

# Setting random seed for reproducibility

np.random.seed(42)

# Loading Mnist data

try:
  X,y = load_mnist()
  
except:
  
  #Falling back to generating XOR in case of errors
  
  n_samples = 2000
  X, y = generate_xor_data(n_samples,np)

# Extracting n_features and n_classes from X and y


print(f"Shape of X : {X.shape}")

n_features = X.shape[0]
n_classes = y.shape[0]

# specifying percantage of data to be used for validation

ratio = 0.2

# if GPU is available, transform X and y to cupy ndarray
# effectivaly loading the data into the gpu

if(_GPU_AVAILABLE):
  X,y = np.asarray(X),np.asarray(y)

# split data into train and validation data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=ratio)

# Here we specify the architechture of our MLP, these are all hyperparamaters you can play with
"""
layer_dims = [n_features, # Input size (number of features)
              64, # Hidden layer 1 Number of units
              64, # Hidden layer 2 Number of units
              32, # Hidden layer 3 Number of units
              32, # Hidden layer 3 Number of units
              #.... you can add more hidden layers, by adding more elements to the list
              n_classes # Output size (number of labels)
              ]
"""

learning_rate = 0.03

loss = BCELoss()

layers = [
  Dense(n_features,64,'he'),
  ReLU(),
  Dense(64,32,'he'),
  ReLU(),
  Dense(32,n_classes,'he'),
  Sigmoid()
]


model = NeuralNetwork(n_classes, # Needed
                      layers, # Needed
                      learning_rate, # Needed
                      loss
                      )

# Training the model for 300 iterations

History = model.fit(X_train,
          y_train,
          100,
          validation_data=(X_test,y_test)
          )

# Print Time Elapsed and Device used to train

print(f"Time Elapsed : {History['Time_Elapsed']:.2F} seconds on : {'GPU' if _GPU_AVAILABLE else 'CPU'}")
plot_metrics(History)
