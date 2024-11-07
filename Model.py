import numpy as np

def he_initialization(list):
  """
    He initilization, takes a list [input_dim,hidden1_dim,hidden2_dim,....,hiddenn_dim]
    and returns a dict of weights in correct shapes using He initilization
  """

  n = len(list)
  weights = {}
  for i in range(1,n):
    weights['W'+str(i)] = np.random.randn(list[i],list[i-1]) * np.sqrt(2/list[i-1])
    weights['b'+str(i)] = np.zeros((list[i],1))
  
  return weights

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
  return 1 / (1+ np.exp(-Z + 1e-8))

def sigmoid_backward(Z):
  g = sigmoid_forward(Z)
  return g * (1 - g)

def binary_cross_entropy(yhat,y,weights=None,lamb=0.01):
  n = yhat.shape[0]
  epsilon = 1e-8
  loss = - np.sum(y * np.log(yhat+epsilon) + (1 - y) * np.log(1 - yhat+epsilon))
  loss /=n
  
  if weights is not None:
    weights = np.concatenate([w.flatten() for w in weights.values()])
    L2_reg = (lamb/(2*n)) *  np.sum(weights ** 2)
    loss += L2_reg
  return loss 

def binary_cross_entropy_backward(yhat,y):
  derivitave = (yhat - y) / (yhat * (1 - yhat))
  return derivitave

def layer_forward(W,b,A_prev):
  Z = np.dot(W,A_prev) + b
  return Z

def activation_forward(Z,activation):
  if activation =="relu":
    return relu_forward(Z)
  elif activation == 'sigmoid':
    return sigmoid_forward(Z)
  else:
    print("relu or sigmoid only")

def dropout_forward(A_prev,prob):
  mask = np.random.rand(A_prev.shape[0],A_prev.shape[1])
  mask = mask > prob
  A = A_prev * mask
  return A,mask

def dropout_backprop(dA,mask,prob):
  dA = dA * mask
  dA /= prob
  return dA

def forward_prop(X,weights,training,dropout=None):
  
  """
  dropout is a dict, with key being layer to implement dropout and value being the lose prob
  """

  L = len(weights.values()) // 2
  cache = {}

  A = X.copy()
  for l in range(L-1):
    Z = layer_forward(weights['W'+str(l)],weights['b'+str(l)],A)
    A = activation_forward(Z,'relu')
    if l in dropout:
      if training:
        A,mask = dropout_forward(A,dropout[l])
        cache['Mask'+str(l)] = mask
      else:
        A*= dropout[l]
    cache['Z'+str(l)] = Z
    cache['A'+str(l)] = A
    
  Z = layer_forward(weights['W'+str(L-1)],weights['b'+str(L-1)],A)
  A = activation_forward(Z,'sigmoid')
  
  return A,cache

