from DeviceSelector import *

np = get_numpy()

def relu_forward(Z):
  """
  Relu activation function,takes in logits Z  if Z > 0 returns x else returns 0
  """

  return np.maximum(0,Z)

def relu_backward(Z):
  """
  Derivative of ReLU, takes logits Z, if Z > 0 returns 1 else 0 
  """

  return np.where(Z>0,1,0)

def sigmoid_forward(Z):
  """
  Sigmoid activation function, takes in Z and returns into a value between 0 and 1, used for
  Binary classification
  """
  return 1 / (1+ np.exp(-Z)+1e-8)

def sigmoid_backward(A):
  """
  Derivative of the sigmoid activation function, takes in A and returns A * (1 - A) with A being the sigmoid(Z)
  Used during backprop to calculate gradients
  """
  
  return A * (1 - A)

def activation_forward(Z,activation):
  """
  Takes in logits Z and an activation function, and returns the activation function applied to 
  the logits
  """
  
  if activation == "relu":
    
    return relu_forward(Z)
    
  elif activation == 'sigmoid':
    
    return sigmoid_forward(Z)
    
  else:

    raise ValueError("relu or sigmoid only")



