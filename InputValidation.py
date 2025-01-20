
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
    from Layers import Layer,Dropout
    from Activations import Activation

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
    from Losses import Loss
    if not isinstance(criterion,Loss):
      raise ValueError("Loss used must be a valid Loss")
    
    return criterion
    
  @staticmethod
  def validate_keep_prob(keep_prob):
    if not isinstance(keep_prob, (int,float)):
      raise TypeError("keep prob should be an int or a float")
    
    if keep_prob <= 0 or keep_prob > 1:
      raise ValueError("keep_prob needs to be between 0 exclusive and 1 inclusive")

    return keep_prob

  @staticmethod
  def validate_number_units(number_of_units):
    if not isinstance(number_of_units, int):
      raise TypeError("Number of units needs to be an int")
    
    if number_of_units <= 0:
      raise ValueError("Number of units needs to greater than 0")
    
    return number_of_units

  @staticmethod
  def validate_delta(delta):
    if not isinstance(delta, (float,int)):
      raise TypeError('Delta needs to be a float or an int')

    if delta < 0:
      raise ValueError('Delta needs to positive')

    return delta
    
  @staticmethod
  def validate_patience(patience):
    if not isinstance(patience,int):
        raise TypeError('Patience needs to be an int')

    if patience < 0:
      raise ValueError('Patience needs to positive')

    return patience
  
  @staticmethod
  def validate_same_shape(a,b):
    if not (type(a) == type(b)):
      raise TypeError("paramater 1 and 2 are two different types, they need to be of same type")
    
    if a.shape != b.shape:
      raise ValueError("paramater 1 and 2's shapes do not match")
    
    return a,b