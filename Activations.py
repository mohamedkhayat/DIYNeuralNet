from DeviceSelector import *
from Layers import Layer
from abc import ABC,abstractmethod

#import numpy as np
np = get_numpy()
class Activation(ABC):

  def __init__(self):
    super().__init__()
  
  @abstractmethod
  def forward(self,Z):
    pass

  @abstractmethod
  def backward(self,dZ):
    pass
class ReLU(Activation):
  def __init__(self):
    super().__init__()

  def forward(self,Z,train=True):
    if train:
      self.input = Z
    return np.maximum(0,Z)

  def backward(self,dA):
    return np.where(self.input>0,1,0) * dA

class Sigmoid(Activation):
  def __init__(self):
    super().__init__()

  def forward(self,Z,train=True):
    output = 1 /(1+np.exp(-Z))
    if train:
      self.input = Z
      self.output = output
    return output

  def backward(self,dA):
    return (self.output * (1 - self.output)) * dA


