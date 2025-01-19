from DeviceSelector import *
from abc import ABC,abstractmethod
np = get_numpy()

"""
def binary_cross_entropy(yhat,y,weights=None,lamb=None):
  loss function, takes in predictions yhat, true labels y, weights and scaling factor
  lambda for L2 reg and returns Negative Log likelyhood
  
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

"""
class Loss(ABC):
  def __init__(self):
    super().__init__()
  @abstractmethod
  def __call__(self,y_true,y_pred):
    pass

  @abstractmethod
  def backward(self,loss):
    pass

class BCELoss(Loss):
  #NEED TO ADD L2 
  def __init__(self):

    self.batch_size = None
  
  def __call__(self,y_true,y_pred):
    self.batch_size = y_pred.shape[1]

    epsilon = 1e-7
    
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)       

    loss = - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
      
    return loss 

  def backward(self,y_true,y_pred):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) + epsilon)
