from DeviceSelector import *
import pathlib
import matplotlib.pyplot as plt
np = get_numpy()

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
  
def generate_xor_data(n_samples,np ,noise=0.2):
    """Generates XOR-like data."""
    X = np.random.rand(2, n_samples) * 2 - 1  # Centered around 0
    y = np.logical_xor(X[0, :] > 0, X[1, :] > 0).astype(int).reshape(1, -1)
    X += np.random.normal(0, noise, X.shape)  # Add noise
    return X, y
  
def load_mnist():
  """
  Loads a transformed mnist dataset, Only keeping labels 0 and 1 for binary classification,
  and using under sampling for the dataset to be balanded
  """
  data = numpy.loadtxt(pathlib.Path('Data','balanced_mnist_1.csv'),delimiter=',',skiprows=1)
  X = data[:,1:].reshape(784,-1)
  y = data[:,0].reshape(1,-1)
  return X,y

def plot_metrics(History):
  
  try:
    train_accuracy = History['Train_accuracy']
    train_losses = History['Train_losses']
    test_losses,test_accuracy = History['Test_losses'],History['Test_accuracy']

    plt.figure(figsize=(10,8))
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
    
    plt.figure(figsize=(10,8))
    plt.title('Accuracy per Epoch')
    plt.plot(train_accuracy,label="Train accuracy",c='r')
    plt.plot(test_accuracy,label="Test accuracy",c='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
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