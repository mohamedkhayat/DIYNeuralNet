import numpy as np

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

def sigmoid_backward(A):
  """
  Derivative of the sigmoid activation function, takes in A and returns A * (1 - A) with A being the sigmoid(Z)
  Used during backprop to calculate gradients
  """
  
  return A * (1 - A)

def binary_cross_entropy(yhat,y,weights=None,lamb=0.01):
  """
  loss function, takes in predictions yhat, true labels y, weights and scaling factor
  lambda for L2 reg and returns Negative Log likelyhood
  """
  
  n = yhat.shape[1]
  epsilon = 1e-8
  
  loss = - np.sum(y * np.log(yhat + epsilon) + (1 - y) * np.log(1 - yhat + epsilon))
  loss /=n
  
  if lamb is not None and weights is not None:

    L2_reg = 0

    for l in range(1,len(weights)//2 + 1):
      L2_reg += np.sum(weights["W"+str(l)]**2)

    L2_reg = (lamb/(2*n)) *  L2_reg
    loss += L2_reg
    
  return loss 

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
  
  if activation == "relu":
    
    return relu_forward(Z)
    
  elif activation == 'sigmoid':
    
    return sigmoid_forward(Z)
    
  else:

    print("relu or sigmoid only")

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

def forward_prop(X,weights,training,dropout=None):
  
  """
  dropout is a dict, with key being layer to implement dropout and value being the keep prob
  training being a boolean to know if applying dropout is needed or not since dropout is only applied
  during training and not inference
  this function is used to make a prediction yhat for an input x 
  """

  L = len(weights) // 2 
  
  cache = {}

  A_prev = X.copy()
  
  cache['A0'] = A_prev
  
  for l in range(1,L):

    Z = layer_forward(weights['W'+str(l)],weights['b'+str(l)],A_prev)
    
    A = activation_forward(Z,'relu')

    if training and dropout is not None and str(l) in dropout:

      A,mask = dropout_forward(A,dropout[str(l)])
        
      cache['Mask'+str(l)] = mask

    cache['Z'+str(l)] = Z
    cache['A'+str(l)] = A
    
    A_prev = A

  Z = layer_forward(weights['W'+str(L)],weights['b'+str(L)],A_prev)
  A = activation_forward(Z,'sigmoid')

  cache['Z'+str(L)] = Z
  cache['A'+str(L)] = A
  
  return A,cache

def backprop(y,weights,cache,dropout=None,lamb=None):
  """
  backpropagation, takes in input x used for calculating the gradients for the first layer, y used for calculating
  the gradients for the last layer,cache that has the mask and the weights for each layer, and dropout to know which
  layers we need to multiply by their mask
  
  THIS SHOULD BE INCORRECT AND NEEDS FIXING
  """
  
  derivatives = {}

  m = y.shape[1]
  
  L = len(weights) // 2
  
  A = cache["A"+str(L)] 
  
  dZL= A - y
  
  A_prev = cache['A'+str(L-1)]
  
  dWL = (1 / m) * np.dot(dZL,A_prev.T)
  
  if lamb:
    
    dWL += (lamb / m) * weights["W"+str(L)]
    
  dbL = (1 / m ) * np.sum(dZL,axis=1,keepdims=True)

  derivatives["dW"+str(L)] = dWL 
  derivatives["db"+str(L)] = dbL 

  dA_prev = np.dot(weights['W'+str(L)].T,dZL)
  
  for l in reversed(range(1,L)):

    if dropout is not None and str(l) in dropout and  "Mask"+str(l) in cache:
      dA_prev = dropout_backprop(dA_prev,cache['Mask'+str(l)],dropout[str(l)])

    dZl = dA_prev * relu_backward(cache['Z'+str(l)])
    A_prev = cache['A'+str(l-1)]
    dWl = (1 / m) * np.dot(dZl,A_prev.T)

    if lamb:
      dWl += (lamb / m) * weights["W"+str(l)]
      
    dbl = (1 / m) * np.sum(dZl,axis=1,keepdims=True)
    
    derivatives["dW"+str(l)] = dWl 
    derivatives["db"+str(l)] = dbl 
  
    dA_prev = np.dot(weights['W'+str(l)].T,dZl)

  return derivatives

def optimize(weights,derivatives,learning_rate=0.01):
  
  for paramater in weights.keys():
    weights[paramater] -= learning_rate * derivatives["d"+paramater]

  return weights

def train(X,y,weights,epochs=30,learning_rate=0.01,lamb=None,dropout=None):
  
  for epoch in range(epochs):

    yhat,cache = forward_prop(X,weights,True,dropout)      
    
    loss = binary_cross_entropy(yhat,y,weights,lamb)
    
    derivatives = backprop(y,weights,cache,dropout,lamb)
    
    weights = optimize(weights,derivatives,learning_rate)

    print(f"Epoch : {epoch} : Loss : {float(loss):.4f}")

  return weights

def evaluate(yhat,y):
  predictions = (yhat>0.5).astype(int)
  correct = np.sum(predictions == y,axis=1)
  accuracy = correct / y.shape[1]
  return accuracy[0]

def train_test_split(X,y,test_size=0.2):
  m = X.shape[1]
  
  indices = np.random.permutation(m)
  
  test_size = int(m * test_size)
  
  test_indices = indices[:test_size]
  train_indices = indices[test_size:]
  
  X_test,y_test = X[:,test_indices],y[:,test_indices]
  X_train,y_train = X[:,train_indices],y[:,train_indices]
  
  return X_train,X_test,y_train,y_test
  
def generate_xor_data(n_samples, noise=0.2):
    """Generates XOR-like data."""
    X = np.random.rand(2, n_samples) * 2 - 1  # Centered around 0
    y = np.logical_xor(X[0, :] > 0, X[1, :] > 0).astype(int).reshape(1, -1)
    X += np.random.normal(0, noise, X.shape)  # Add noise
    return X, y
  
np.random.seed(42)

n_samples = 2000
n_features = 2
n_classes = 1
ratio = 0.2
layer_dims = [n_features,64*2,32*2,n_classes]  
dropout_rates = [1 , 0.8 , 1  , 1]

X, y = generate_xor_data(n_samples)

#print(X.shape,y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=ratio)

#print(X_train.shape,y_train.shape)
#print(X_test.shape,y_test.shape)

weights = he_initialization(layer_dims=layer_dims)

weights = train(X_train,y_train,weights,epochs=1000,learning_rate=0.01,lamb=0.01,dropout=dropout_rates) 

y_pred_train,mask = forward_prop(X_train,weights,training=False)

train_accuracy = evaluate(y_pred_train,y_train)

print(f"Train Accuracy : {float(train_accuracy):.4f}")

y_pred_test,mask = forward_prop(X_test,weights,training=False)

test_accuracy = evaluate(y_pred_test,y_test)

print(f"Test Accuracy : {float(test_accuracy):.4f}")
