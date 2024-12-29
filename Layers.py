from DeviceSelector import *

np = get_numpy()

def he_initialization(layer_dims):
  """
    He initilization, takes a list [input_dim,hidden1_dim,hidden2_dim,....,hiddenn_dim]
    and returns a dict of weights in correct shapes using He initilization
  """

  n = len(layer_dims)
  weights = {}

  for i in range(1,n):

    weights['W'+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1]) * np.sqrt(2/layer_dims[i-1])
    weights['b'+str(i)] = np.zeros((layer_dims[i],1))
  
  return weights

def layer_forward(W,b,A_prev):
  """
  Takes in Weight matrix i, bias i, and A[i-1] to calculate Z[i], used during forward prop
  """
  
  Z = np.dot(W,A_prev) + b

  return Z

def dropout_forward(A,keep_prob):
  """
  Applies Dropout, takes in Activation matrix A and applies a mask of 1 - keep_prob % zeros
  effectivaly "turning off" a certain percentage of units for this layer, returns the new A
  with dropout applied and the mask itself for later use during backprop
  """
  
  mask = (np.random.rand(A.shape[0],A.shape[1]) < keep_prob).astype(float)
  A = A * mask / keep_prob
  
  return A,mask

def dropout_backprop(dA,mask,keep_prob):
  """
  is used during backprop to make sure the gradients are only calculated based on the units that participated
  in making the prediction
  """
  
  dA = dA * mask / keep_prob  
  return dA

