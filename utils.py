from DeviceSelector import *

np = get_numpy()

def train_test_split(X,y,test_size=0.2):
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
  