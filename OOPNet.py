from DeviceSelector import *
import time
from EarlyStopping import EarlyStopping

#np = get_numpy()

import numpy as np
from Layers import layer_forward,dropout_backprop,dropout_forward,he_initialization

from Activations import relu_backward,activation_forward
from Losses import binary_cross_entropy

class NeuralNetwork():
  
  def __init__(self,n_classes,layers,learning_rate,loss,dropout=None,lamb=None):

    self.n_classes = n_classes
    self.layers = layers
    self.learning_rate = learning_rate
    self.loss = loss
    
  def forward(self,X,train=True):
    
    """
    dropout is a dict, with key being layer to implement dropout and value being the keep prob
    training being a boolean to know if applying dropout is needed or not since dropout is only applied
    during training and not inference
    this function is used to make a prediction yhat for an input x 
    """
    output = X
    for layer in self.layers:
      output = layer.forward(output,train)

    return output

  def backprop(self,dA):
    """
    backpropagation, takes in input x used for calculating the gradients for the first layer, y used for calculating
    the gradients for the last layer,cache that has the mask and the weights for each layer, and dropout to know which
    layers we need to multiply by their mask
    """

    for layer in reversed(self.layers):
      dA = layer.backward(dA)


  def optimize(self):
    
    for layer in self.layers:
      if hasattr(layer,"params"):
        for param in layer.params:
          layer.params[param] -= self.learning_rate * layer.grads['d'+param]


  def fit(self,X_train,y_train,epochs=30,validation_data=None,EarlyStopping_Patience = None):
    
    History = {}
    
    train_losses = []
    test_losses = []
    
    train_accuracies = []
    test_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(epochs):


      y_pred = self.forward(X_train,train=True)      
      
      loss = self.loss(y_train,y_pred)

      train_losses.append(loss.tolist())

      y_train_pred_labels = (y_pred > 0.5).astype(int)
      train_accuracy = self.accuracy_score(y_train_pred_labels,y_train)
      train_accuracies.append(train_accuracy.item())
      
      if validation_data is not None:
        
        X_test,y_test = validation_data
       
        y_test_pred_prob = self.forward(X_test,train=False)     
        
        test_loss = self.loss(y_test,y_test_pred_prob)      
        
        test_losses.append(test_loss.tolist())
        
        y_test_pred_labels = (y_test_pred_prob > 0.5).astype(int)
        test_accuracy = self.accuracy_score(y_test_pred_labels,y_test)
        test_accuracies.append(test_accuracy.item())

      dA = self.loss.backward(y_train,y_pred)

      self.backprop(dA)
      
      self.optimize()
      
      if epoch % 50 == 0:
        print(f"Epoch : {epoch} : Loss : {float(loss):.4f}")
    
        
    end_time = time.time()
    
    History = {'Train_losses':train_losses,
               'Test_losses':test_losses,
               'Train_accuracy':train_accuracies,
               'Test_accuracy':test_accuracies,
               'Time_Elapsed':end_time - start_time
               }
               
    return History
    
  def predict(self,X):

    if(len(X.shape) == 1):
      X = X.reshape(-1,1)
    predictions = self.forward(X)

    return (predictions>0.5).astype(int)
  
  def accuracy_score(self,y_pred,y_true):
    
    correct = np.sum(y_pred == y_true,axis=1)
    
    return correct / y_true.shape[1]

