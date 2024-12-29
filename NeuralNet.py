from DeviceSelector import *

np = get_numpy()

from Layers import layer_forward,dropout_backprop,dropout_forward,he_initialization
from Activations import relu_backward,activation_forward
from Losses import binary_cross_entropy

class NeuralNetwork():
  
  def __init__(self,n_classes,layers_dim,dropout,learning_rate,lamb,):
    self.n_classes = n_classes
    self.layers_dim = layers_dim
    self.dropout = dropout
    self.learning_rate = learning_rate
    self.lamb = lamb
    self.weights = he_initialization(self.layers_dim)
    
  def forward_prop(self,X,training):
    
    """
    dropout is a dict, with key being layer to implement dropout and value being the keep prob
    training being a boolean to know if applying dropout is needed or not since dropout is only applied
    during training and not inference
    this function is used to make a prediction yhat for an input x 
    """

    L = len(self.weights) // 2 
    
    cache = {}

    A_prev = X.copy()
    
    cache['A0'] = A_prev
    
    for l in range(1,L):

      Z =  layer_forward(self.weights['W'+str(l)],self.weights['b'+str(l)],A_prev)
      
      A =  activation_forward(Z,'relu')

      if training and self.dropout is not None and str(l) in self.dropout:

        A,mask =  dropout_forward(A,self.dropout[str(l)])
          
        cache['Mask'+str(l)] = mask

      cache['Z'+str(l)] = Z
      cache['A'+str(l)] = A
      
      A_prev = A

    Z =  layer_forward(self.weights['W'+str(L)],self.weights['b'+str(L)],A_prev)
    A =  activation_forward(Z,'sigmoid')

    cache['Z'+str(L)] = Z
    cache['A'+str(L)] = A
    
    return A,cache

  def backprop(self,y,cache):
    """
    backpropagation, takes in input x used for calculating the gradients for the first layer, y used for calculating
    the gradients for the last layer,cache that has the mask and the weights for each layer, and dropout to know which
    layers we need to multiply by their mask
    
    """
    
    derivatives = {}

    m = y.shape[1]
    
    L = len(self.weights) // 2
    
    A = cache["A"+str(L)] 
    
    dZL= A - y
    
    A_prev = cache['A'+str(L-1)]
    
    dWL = (1 / m) * np.dot(dZL,A_prev.T)
    
    if self.lamb:
      
      dWL += (self.lamb / m) * self.weights["W"+str(L)]
      
    dbL = (1 / m ) * np.sum(dZL,axis=1,keepdims=True)

    derivatives["dW"+str(L)] = dWL 
    derivatives["db"+str(L)] = dbL 

    dA_prev = np.dot(self.weights['W'+str(L)].T,dZL)
    
    for l in reversed(range(1,L)):

      if self.dropout is not None and str(l) in self.dropout and  "Mask"+str(l) in cache:
        dA_prev = dropout_backprop(dA_prev,cache['Mask'+str(l)],self.dropout[str(l)])

      dZl = dA_prev * relu_backward(cache['Z'+str(l)])
      A_prev = cache['A'+str(l-1)]
      dWl = (1 / m) * np.dot(dZl,A_prev.T)

      if self.lamb:
        dWl += (self.lamb / m) * self.weights["W"+str(l)]
        
      dbl = (1 / m) * np.sum(dZl,axis=1,keepdims=True)
      
      derivatives["dW"+str(l)] = dWl 
      derivatives["db"+str(l)] = dbl 
    
      dA_prev = np.dot(self.weights['W'+str(l)].T,dZl)

    return derivatives

  def optimize(self,derivatives):
    
    for paramater in self.weights.keys():
      self.weights[paramater] -= self.learning_rate * derivatives["d"+paramater]

  def fit(self,X,y,epochs=30):
    
    for epoch in range(epochs):

      yhat,cache = self.forward_prop(X,True)      
      
      loss = binary_cross_entropy(yhat,y,self.weights,self.lamb)
      
      derivatives = self.backprop(y,cache)
      
      self.optimize(derivatives)
      if epoch % (epochs // 10) == 0:
        print(f"Epoch : {epoch} : Loss : {float(loss):.4f}")
  
  def predict(self,X):

    predictions,_ = self.forward_prop(X,False)

    return (predictions>0.5).astype(int)
  
  def accuracy_score(self,y_pred,y_true):
    
    correct = np.sum(y_pred == y_true,axis=1)
    
    return correct / y_true.shape[1]

