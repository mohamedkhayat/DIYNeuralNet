
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
  """
  Sigmoid activation function, takes in Z and returns into a value between 0 and 1, used for
  Binary classification
  """
  return 1 / (1+ np.exp(-Z + 1e-8))

def sigmoid_backward(Z):
  """
  Derivative of the sigmoid activation function, takes in Z and returns sigmoid(Z) * (1 - sigmoid(Z))
  Used during backprop to calculate gradients
  """
  g = sigmoid_forward(Z)

  return g * (1 - g)

def binary_cross_entropy(yhat,y,weights=None,lamb=0.01):
  """
  loss function, takes in predictions yhat, true labels y, weights and scaling factor
  lambda for L2 reg and returns Negative Log likelyhood
  """
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
  """
  Derivative of the loss function with respect to yhat, takes predictions yhat and true labels y
  used during backprop
  """
  derivative = (yhat - y) / (yhat * (1 - yhat))

  return derivative

def layer_forward(W,b,A_prev):
  """
  Takes in Weight matrix i, bias i, and A[i-1] to calculate Z[i], used during forward prop
  """
  Z = np.dot(W,A_prev) + b

  return Z

def activation_forward(Z,activation):
  """
  Takes in logits Z and an activation function, and returns the activation function applied to 
  the logits
  """
  if activation =="relu":
    
    return relu_forward(Z)
    
  elif activation == 'sigmoid':
    
    return sigmoid_forward(Z)
    
  else:

    print("relu or sigmoid only")

def dropout_forward(A,prob):
  """
  Applies Dropout, takes in Activation matrix A and applies a mask of prob % zeros
  effectivaly "turning off" a certain percentage of units for this layer, returns the new A
  with dropout applied and the mask itself for later use during backprop
  """
  mask = np.random.rand(A.shape[0],A.shape[1])
  mask = mask > prob
  A = (A * mask)/(1 - prob)
  
  return A,mask.astype(float)

def dropout_backprop(dA,mask):
  """
  is used during backprop to make sure the gradients are only calculated based on the units that participated
  in making the prediction
  """
  dA = dA * mask
  
  return dA

def forward_prop(X,weights,training,dropout=None):
  
  """
  dropout is a dict, with key being layer to implement drop["W" + str(l + 1)].T, dZ)out and value being the lose prob
  training being a boolean to know if applying dropout is needed or not since dropout is only applied
  during training and not inference
  this function is used to make a prediction yhat for an input x 
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

      #else:
        #A*= (1 - dropout[l])
        
    cache['Z'+str(l)] = Z
    cache['A'+str(l)] = A
    
  Z = layer_forward(weights['W'+str(L-1)],weights['b'+str(L-1)],A)
  A = activation_forward(Z,'sigmoid')
  
  return A,cache

def backprop(x,y,weights,cache,dropout=None,lamb=None):
  """
  backpropagation, takes in input x used for calculating the gradients for the first layer, y used for calculating
  the gradients for the last layer,cache that has the mask and the weights for each layer, and dropout to know which
  layers we need to multiply by their mask
  
  THIS SHOULD BE INCORRECT AND NEEDS FIXING
  
  """
  derivatives = {}

  m = len(y)
  L = len(weights) // 2
  
  aL = cache["A"+str(L)] 
  dZL= aL - y
  dWL = (1 / m) * np.dot(dZL,cache["A"+str(L-1)].T)
  if lamb:
    dWL += (lamb / m) * cache["W"+str(L)]
  dbL = (1 / m ) * np.sum(dZL,axis=1,keepdims=True)

  derivatives["dWL"] = dWL 
  derivatives["dbL"] = dbL 

  dZl = dZL
  dbl = dbL

  for l in range(L -1,0,-1):

    dAl = np.dot(weights["W"+str(l+1)].T,dZl)
    
    if dropout and "Mask"+str(l) in cache:
      dAl = dropout_backprop(dAl,cache['Mask'+str(l)])
    relu_derivative = relu_backward(cache['Z'+str(l)])
    
    dZl = dAl * relu_derivative
      
    dWl = (1 / m) * np.dot(dZl,cache["A"+str(l)].T)
    if lamb:
      dWl += (lamb / m) * cache["W"+str(l)]
    dbl = (1 / m) * np.sum(dZl,axis=1,keepdims=True)
    
    derivatives["dW"+str(l)] = dWl 
    derivatives["db"+str(l)] = dbl 
  
  
  dA0 = np.dot(weights["W1"].T,dZl)
  relu_derivative_0 = relu_backward(cache['Z0'])

  dZ0 = dA0 * relu_derivative_0
  dW0 = (1 / m) * np.dot(dZ0,x.T)
  if lamb:
    dW0 += (lamb / m) * cache["W0"]
  db0 = (1 / m) * np.sum(dZ0,axis=1,keepdims=True)

  derivatives["dW0"] = dW0
  derivatives["db0"] = db0

  return derivatives
