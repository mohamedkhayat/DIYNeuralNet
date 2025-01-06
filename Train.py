from utils import *
from DeviceSelector import *
from NeuralNet import NeuralNetwork

np = get_numpy()
  
_GPU_AVAILABLE = is_gpu_available()
# Setting random seed for reproducibility

np.random.seed(42)

# Loading Mnist data

try:
  X,y = load_mnist()
  
except:
  n_samples = 2000
  X, y = generate_xor_data(n_samples,np)

# Extracting n_features and n_classes from X and y

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

# Number of hidden layers and units
layer_dims = [n_features, # Input size (number of features)
              64, # Hidden layer 1 Number of units
              64, # Hidden layer 2 Number of units
              32, # Hidden layer 3 Number of units
              #.... you can add more hidden layers, by adding more elements to the list
              n_classes # Output size (number of labels)
              ]

# Dropout Dict, keys represents which layer to apply to, value is the keep prob, not drop

dropout_rates = {
  '2':0.9, # dropout applied to hidden layer 2, dropping (1 - 0.9) = 0.1 or 10 % of units
  '3':0.85 # dropout applied to hidden layer 3, dropping (1 - 0.85) = 0.15 or 15 % of units
  }
  
# Learning rate, controls how drastic weight updates are

learning_rate = 0.03

# controls the intensity of L2 Regularization,
# higher values mean more intense regularization -> less over fitting but risk of under fitting
# lower values mean less intense regularization -> less under fitting but risk of over fitting        

lamb = 0.01

#initializing the model

model = NeuralNetwork(n_classes, # Needed
                      layer_dims, # Needed
                      dropout_rates, # Optional, put None for no dropout
                      learning_rate, # Needed
                      lamb # Optional, put None for no L2
                      )

# Training the model for 300 iterations

History = model.fit(X,
          y,
          1000,
          validation_data=(X_test,y_test),
          EarlyStopping_Patience= 10 # Early Stopping Patience, if not specified, no Early Stopping is used
                                     # Else, Training stops if Val Loss does not improve during n = Patience of steps
          )

# Print Time Elapsed and Device used to train

print(f"Time Elapsed : {History['Time_Elapsed']:.2F} seconds on : {'GPU' if _GPU_AVAILABLE else 'CPU'}")

# Plot Loss and Accuracy for Train and Validation set

plot_metrics(History)

# using the model to make predictions on the train set

y_pred_train= model.predict(X_train)

# using predictions to calculate model's accuracy on the train set

train_accuracy = model.accuracy_score(y_pred_train,y_train)
print(f"Train Accuracy : {float(train_accuracy):.4f}")

# using the model to make predictions on the test set 

y_pred_test = model.predict(X_test)

# using predictions to calculate model's accuracy on the test set

test_accuracy = model.accuracy_score(y_pred_test,y_test)
print(f"Test Accuracy : {float(test_accuracy):.4f}")
