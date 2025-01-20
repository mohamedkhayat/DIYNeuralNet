from Layers import Layer,Dropout
from Activations import Activation
from Losses import Loss

class InputValidator:
  
  @staticmethod
  def validate_n_classes(n_classes):
    if not isinstance(n_classes, int):
      raise TypeError("n_classes needs to be an int")
      
    if n_classes < 1:
      raise ValueError("n_classes needs to be strictly positive")
      
    return n_classes
  
  @staticmethod
  def validate_layers(layers):
    if not isinstance(layers,(list,tuple)) or len(layers) < 1:
      raise TypeError("Layers needs to be a non empty list or a tuple")
    
    if not all(isinstance(layer,(Layer,Activation)) for layer in layers):
      raise TypeError("Layers must consist of Layers or Activations only")
    
    layers_only = [layer for layer in layers if isinstance(layer,Layer) and not isinstance(layer,Dropout)]
    for i in range(len(layers_only) - 1):
      if not layers_only[i].output_size == layers_only[i+1].input_size:
        raise ValueError("Matrix shapes do not match")

    return layers

  @staticmethod
  def validate_learning_rate(learning_rate):
    if not isinstance(learning_rate, (float,int)):
      raise TypeError("Learning Rate needs to be a float or an int")
    
    if learning_rate <= 0:
      raise ValueError("Learning rate needs to be strictly positive")
    
    return learning_rate

  @staticmethod
  def validate_criterion(criterion):
    if not isinstance(criterion,Loss):
      raise ValueError("Loss used must be a valid Loss")
    
    return criterion
    
  @staticmethod
  def validate_():
    pass