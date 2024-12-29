from utils import *
from DeviceSelector import *
from NeuralNet import NeuralNetwork

np = get_numpy()

_GPU_AVAILABLE = is_gpu_available()

np.random.seed(42)

n_samples = 2000
n_features = 2
n_classes = 1
ratio = 0.2
X, y = generate_xor_data(n_samples,np)

if(_GPU_AVAILABLE):
  X,y = np.asarray(X),np.asarray(y)
  
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=ratio)

layer_dims = [n_features,64,64,32,32,16,16,n_classes]  
dropout_rates = [1 , 0.9 , 0.85  ,0.85,0.85,0.9,1]
learning_rate = 0.01
lamb = 0.1

model = NeuralNetwork(n_classes,layer_dims,dropout_rates,learning_rate,lamb)

model.fit(X,y,1000)

y_pred_train= model.predict(X_train)

train_accuracy = model.accuracy_score(y_pred_train,y_train)

print(f"Train Accuracy : {float(train_accuracy):.4f}")

y_pred_test = model.predict(X_test)

test_accuracy = model.accuracy_score(y_pred_test,y_test)

print(f"Test Accuracy : {float(test_accuracy):.4f}")
