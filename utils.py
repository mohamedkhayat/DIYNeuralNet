from random import randint
from DeviceSelector import *
np = get_numpy()
import pathlib
import matplotlib.pyplot as plt

#import numpy as np
def train_test_split(X,y,test_size=0.2):
  """
  Takes in matrices X (features) and y (target),and test size, which represents ratio of data to be used
  for evaluation, and returns X_train,X_test,y_train,y_test split according to the ratio specified 
  """
  m = X.shape[1]
  
  indices = np.random.permutation(m)
  
  test_size = int(m * test_size)
  
  test_indices = indices[:test_size]
  train_indices = indices[test_size:]
  
  X_test,y_test = X[:,test_indices],y[:,test_indices]
  X_train,y_train = X[:,train_indices],y[:,train_indices]
  
  return X_train,X_test,y_train,y_test
  
def generate_xor_data(n_samples,np ,noise=0.01):
    """Generates XOR-like data."""
    X = np.random.rand(2, n_samples) * 2 - 1  # Centered around 0
    y = np.logical_xor(X[0, :] > 0, X[1, :] > 0).astype(int).reshape(1, -1)
    X += np.random.normal(0, noise, X.shape)  # Add noise
    return X, y
  
def plot_image(X,model,n_images,original_image_shape = (28,28),n_classes=1):

  plt.figure(figsize=(6,6))
  
  indices = [randint(0,len(X)) for _ in range(n_images)]
  
  HEIGHT,WIDTH = original_image_shape
  
  for i,idx in enumerate(indices):
  
    test_example = X[:,idx]

    if(len(test_example.shape) == 1):
      test_example = test_example.reshape(-1,1)
      
    test_pred = model.predict(test_example)
   
    if(is_gpu_available() == True):
      test_example = test_example.get()
  
    plt.subplot(2,(n_images + 1)// 2, i + 1)
    
    test_example = test_example.reshape(HEIGHT,WIDTH) * 255.
    
    if n_classes == 1:
      plt.title("One" if test_pred.item() == 1 else "not a One")

    else:
      plt.title(str(test_pred.item()))
    
    plt.imshow(test_example,cmap='gray')
    plt.axis('off')
    
  plt.tight_layout()
  plt.show()

def load_binary_mnist():
  """
  Loads a transformed mnist dataset, Only keeping labels 0 and 1 for binary classification,
  and using under sampling for the dataset to be balanded
  """
  data = numpy.loadtxt(pathlib.Path('Data','balanced_mnist_1.csv'),delimiter=',',skiprows=1)
  X = data[:,1:].transpose()
  y = data[:,0].reshape(1,-1)
  return np.asarray(X),np.asarray(y)

def load_mnist():
  data = numpy.loadtxt(pathlib.Path('Data','train.csv'),delimiter=',',skiprows=1)

  X = data[:,1:].T / 255.
  y = data[:,0].reshape(1,-1)

  y = y.flatten().astype(np.int64)

  n_classes = len(np.unique(y))
  n_samples = len(y)
  
  one_hot = np.zeros((n_classes, n_samples))
  
  one_hot[y,np.arange(n_samples)] = 1
  
  y = one_hot
  
  return np.asarray(X),np.asarray(y)

def plot_metrics(History):
  try:
    train_accuracy = History['Train_accuracy']
    train_losses = History['Train_losses']
    test_losses,test_accuracy = History['Test_losses'],History['Test_accuracy']
  
    plt.clf()
    
    plt.figure(1)
    plt.clf()
    plt.title('Loss per Epoch')
    plt.plot(list(train_losses),label="Train loss",c='r')
    plt.plot(list(test_losses),label="Test loss",c='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    y_train_max = max(train_losses)
    y_train_min = min(train_losses)
    
    y_test_max = max(test_losses)
    y_test_min = min(test_losses)

    y_min = min(y_test_min,y_train_min)
    y_max = max(y_train_max,y_test_max)
    
    plt.axis([0,len(train_losses),y_min - y_min * 0.1 , y_max + y_max * 0.1])
    plt.grid(True)
    plt.show()
    
    plt.figure(2)
    plt.clf()
    plt.title('Accuracy per Epoch')
    plt.plot(train_accuracy,label="Train accuracy",c='r')
    plt.plot(test_accuracy,label="Test accuracy",c='b')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    y_train_max = max(train_accuracy)
    y_train_min = min(train_accuracy)
    
    y_test_max = max(test_accuracy)
    y_test_min = min(test_accuracy)

    y_min = min(y_test_min,y_train_min)
    y_max = max(y_train_max,y_test_max)
    
    plt.axis([0,len(train_accuracy),y_min - y_min * 0.1 , y_max + y_max * 0.1])
    plt.grid(True)
    plt.show()
  
  except Exception as e:
    print(f"Error : {e}, PS : this function expects you chose to input validation data during fit if you chose not to that could be the source of the issue")

def create_mini_batches(X, y, batch_size = 64,
                        shuffle = True, drop_last = True):
                          
    num_samples = X.shape[1]
    indices = np.arange(num_samples)
    
    if shuffle == True:
      np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
      end_idx = min(start_idx + batch_size, num_samples)
      
      if drop_last == True and end_idx - start_idx < batch_size:
        break
      
      batch_indices = indices[start_idx:end_idx]
      
      yield X[:, batch_indices], y[:, batch_indices]
 