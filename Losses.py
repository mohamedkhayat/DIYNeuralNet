from DeviceSelector import *

np = get_numpy()

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

