from DeviceSelector import *
import time
from EarlyStopping import EarlyStopping
from utils import create_mini_batches
from Layers import Dropout,Dense
from Activations import Softmax 
from Losses import CrossEntropyLoss,BCELoss
np = get_numpy()

#import numpy as np
class NeuralNetwork():
  
  def __init__(self, n_classes, layers, learning_rate,
               criterion, lamb=None):

    self.n_classes = n_classes
    self.layers = layers
    self.learning_rate = learning_rate
    self.criterion = criterion
    
  def forward(self,X,train=None):
    
    """
    dropout is a dict, with key being layer to implement dropout and value being the keep prob
    training being a boolean to know if applying dropout is needed or not since dropout is only applied
    during training and not inference
    this function is used to make a prediction yhat for an input x 
    """
    
    if train is None:
      train = self.training
      
    output = X
    for layer in self.layers:
      
      if self.training == False and isinstance(layer,Dropout):
        continue
      
      output = layer.forward(output,train)

    return output

  def backprop(self,dA):
    """
    backpropagation, takes in input x used for calculating the gradients for the first layer, y used for calculating
    the gradients for the last layer,cache that has the mask and the weights for each layer, and dropout to know which
    layers we need to multiply by their mask
    """

    for i,layer in reversed(list(enumerate(self.layers))):
      if (isinstance(layer, Softmax) and isinstance(self.criterion, CrossEntropyLoss)):
        dA = layer.backward(dA)
      else:
        dA = layer.backward(dA)

  def zero_grad(self):
    
    for layer in self.layers:
      if isinstance(layer,Dense):
        layer.zero_grad()
      
  def optimize(self):
    
    for layer in self.layers:
      if hasattr(layer,"params"):
        
        for param in layer.params:
          layer.params[param] -= self.learning_rate * layer.grads['d'+param]

  def fit(self, X_train, y_train, epochs = 30, 
          batch_size = 64, shuffle = True, validation_data = None, 
          early_stopping_patience = None, early_stopping_delta = 0):
    
    History = {}
    
    train_losses = []
    test_losses = []
    
    train_accuracies = []
    test_accuracies = []
    
    start_time = time.time()
    
    if early_stopping_patience is not None and early_stopping_patience >= 1:
      
      er = EarlyStopping(early_stopping_patience,early_stopping_delta)

    for epoch in range(epochs):

      self.train()
      epoch_loss = 0.0
      #epoch_train_accuracy = 0
      num_batches = 0
      correct_predictions = 0
      total_samples = 0
      mini_batches =  create_mini_batches(X_train,y_train, batch_size = batch_size,
                                          shuffle = shuffle, drop_last = True)
        
      for X_batch, y_batch in mini_batches:
        self.zero_grad()
        y_pred = self.forward(X_batch, train=True)      
        
        loss = self.criterion(y_batch, y_pred)
        epoch_loss += loss
        
        dA = self.criterion.backward(y_batch, y_pred)

        if isinstance(self.criterion, BCELoss):
          y_pred_labels = (y_pred > 0.5).astype(int)
          batch_correct = np.sum(y_pred_labels == y_batch)
          self.backprop(dA)
          
        elif(isinstance(self.criterion, CrossEntropyLoss)):
          y_pred_labels = np.argmax(y_pred, axis = 0)
          y_true_labels = np.argmax(y_batch, axis=0)  
          batch_correct = np.sum(y_pred_labels == y_true_labels)
          self.backprop(y_batch)
        
        self.optimize()
        
        correct_predictions += batch_correct
        total_samples += y_batch.shape[1]
        num_batches += 1
        
      avg_train_loss =  epoch_loss / num_batches
      avg_train_accuracy = correct_predictions / total_samples
      
      train_losses.append(float(avg_train_loss))
      train_accuracies.append(float(avg_train_accuracy))

      if validation_data is not None:
        
        X_test, y_test = validation_data
        #test_loss, test_accuracy = self.evaluate(X_test, y_test, batch_size)
        
        y_pred_test = self.forward(X_test, train = False)
        test_loss = self.criterion(y_test, y_pred_test)
        test_losses.append(test_loss.tolist())
        
        
        if isinstance(self.criterion, BCELoss):
          y_pred_test_labels = (y_pred_test > 0.5).astype(int)
          test_correct = np.sum(y_pred_test_labels == y_test)
          
        elif(isinstance(self.criterion, CrossEntropyLoss)):
          y_pred_test_labels = np.argmax(y_pred_test, axis = 0)
          y_test_labels = np.argmax(y_test, axis = 0)
          test_correct = np.sum(y_test_labels == y_pred_test_labels)
        
        test_accuracy = test_correct / y_test.shape[1] 
        test_accuracies.append(float(test_accuracy))
     
      if epoch % 10 == 0:
        print(f"Epoch : {epoch} : Loss : {float(test_loss):.4f}")

      if early_stopping_patience is not None and er(test_loss):
        break
        
    end_time = time.time()
    
    History = {'Train_losses':train_losses,
               'Test_losses':test_losses,
               'Train_accuracy':train_accuracies,
               'Test_accuracy':test_accuracies,
               'Time_Elapsed':end_time - start_time
               }
               
    return History
  
  def evaluate(self, X, y, batch_size=64):
    self.eval()  
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    mini_batches = create_mini_batches(X, y, batch_size=batch_size,
                                     shuffle=False, drop_last=False)
    
    for X_batch, y_batch in mini_batches:
        y_pred = self.forward(X_batch)
        loss = self.criterion(y_batch, y_pred)
        total_loss += loss
        
        y_pred_labels = (y_pred > 0.5).astype(int)
        batch_accuracy = self.accuracy_score(y_pred_labels, y_batch)
        total_accuracy += batch_accuracy
        num_batches += 1
    
    return total_loss / num_batches, total_accuracy / num_batches

  
  def train(self):
    self.training = True
    
  def eval(self):
    self.training = False
  
  def predict(self,X):

    if(len(X.shape) == 1):
      X = X.reshape(-1,1)

    predictions = self.forward(X)
    if isinstance(self.criterion, BCELoss):
      return (predictions>0.5).astype(int)
    elif isinstance(self.criterion, CrossEntropyLoss):
      return np.argmax(predictions, axis = 0)
    else:
      return predictions
  
  def accuracy_score(self,y_pred,y_true):
    if isinstance(self.criterion, CrossEntropyLoss):
      y_pred = np.argmax(y_pred, axis = 0)
      y_true= np.argmax(y_true, axis = 0)
    elif isinstance(self.criterion, BCELoss):
      y_pred = (y_pred > 0.5)
    correct = np.sum(y_pred == y_true)
    return correct / y_true.shape[1]

