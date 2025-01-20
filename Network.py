import time
from DeviceSelector import *
np = get_numpy()
from EarlyStopping import EarlyStopping
from utils import create_mini_batches
from Layers import Dropout,Dense,Layer
from Activations import Softmax,Activation
from Losses import CrossEntropyLoss,BCELoss,Loss,MSELoss
from InputValidation import InputValidator
class NeuralNetwork():
  
  def __init__(self, n_classes, layers, learning_rate,
               criterion):
               
    self.n_classes = InputValidator.validate_n_classes(n_classes)
    self.layers = InputValidator.validate_layers(layers)
    self.learning_rate = InputValidator.validate_learning_rate(learning_rate)
    self.criterion = InputValidator.validate_criterion(criterion)
    
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
      """
      if self.training == False and isinstance(layer,Dropout):
        continue
      """
      output = layer.forward(output,train)

    return output

  def backprop(self,dA):
    """
    backpropagation, takes in input x used for calculating the gradients for the first layer, y used for calculating
    the gradients for the last layer,cache that has the mask and the weights for each layer, and dropout to know which
    layers we need to multiply by their mask
    """

    for i,layer in reversed(list(enumerate(self.layers))):
      #what is the point of this if statement ? need to check again, seems pointless
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

      avg_train_loss,avg_train_accuracy = self.train(X_train,y_train,batch_size,shuffle)
      
      train_losses.append(float(avg_train_loss))
      train_accuracies.append(float(avg_train_accuracy))

      if validation_data is not None:
        X_test, y_test = validation_data

        test_loss,test_accuracy = self.evaluate(X_test,y_test,batch_size)

        test_losses.append(float(test_loss))
        test_accuracies.append(float(test_accuracy))
        
      if epoch % 10 == 0:
        print(f"Epoch : {epoch}")
        print(f"Train Loss : {float(avg_train_loss):.4f} Test Loss : {float(test_loss):.4f}")

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

  def set_to_train(self):
    self.training = True
    
  def set_to_eval(self):
    self.training = False
  
  def train(self,X_train,y_train,batch_size,shuffle):
    epoch_loss = 0.0
    num_batches = 0
    correct_predictions = 0
    total_samples = 0

    self.set_to_train()
    
    mini_batches =  create_mini_batches(X_train,y_train, batch_size = batch_size,
                                        shuffle = shuffle, drop_last = True)
      
    for X_batch, y_batch in mini_batches:
      self.zero_grad()
      y_pred = self.forward(X_batch, train=True)      
      
      loss = self.criterion(y_batch, y_pred)
      epoch_loss += float(loss)
      
      if isinstance(self.criterion, BCELoss):
        y_pred_labels = (y_pred > 0.5).astype(int)
        batch_correct = np.sum(y_pred_labels == y_batch)
        
        dA = self.criterion.backward(y_batch, y_pred)
        self.backprop(dA)
        
      elif(isinstance(self.criterion, CrossEntropyLoss)):
        y_pred_labels = np.argmax(y_pred, axis = 0)
        y_true_labels = np.argmax(y_batch, axis = 0)  
        batch_correct = np.sum(y_pred_labels == y_true_labels)
        
        self.backprop(y_batch)
      
      elif(isinstance(self.criterion, MSELoss)):

        dA = self.criterion.backward(y_batch, y_pred)
        self.backprop(dA)

      self.optimize()
      
      if not (isinstance(self.criterion,MSELoss)):
        correct_predictions += int(batch_correct)
        total_samples += y_batch.shape[1]

      num_batches += 1

    if not isinstance(self.criterion, MSELoss): 
      avg_train_accuracy = correct_predictions / total_samples
    else:
      avg_train_accuracy = 0  

    avg_train_loss =  epoch_loss / num_batches
    
    return avg_train_loss,avg_train_accuracy
  
  
  def evaluate(self,X_test,y_test,batch_size):
    self.set_to_eval()
      
      # Create batches for test data too
    test_batches = create_mini_batches(X_test, y_test, batch_size=batch_size,
                                    shuffle=False, drop_last=False)
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_num_batches = 0
    
    for X_batch_test, y_batch_test in test_batches:
        y_pred_test = self.forward(X_batch_test, train=False)
        batch_test_loss = self.criterion(y_batch_test, y_pred_test)
        test_loss += float(batch_test_loss)
        
        if isinstance(self.criterion, BCELoss):
          y_pred_test_labels = (y_pred_test > 0.5).astype(int)
          test_correct += np.sum(y_pred_test_labels == y_batch_test)
          
        elif(isinstance(self.criterion, CrossEntropyLoss)):
          y_pred_test_labels = np.argmax(y_pred_test, axis=0)
          y_true_test = np.argmax(y_batch_test, axis=0)
          test_correct += np.sum(y_pred_test_labels == y_true_test)
        
        test_total += y_batch_test.shape[1]
        test_num_batches += 1
    
    test_loss = test_loss / test_num_batches  # Average loss over batches
    if not isinstance(self.criterion, MSELoss):
      test_accuracy = test_correct / test_total
    else:
      test_accuracy = 0

    return test_loss,test_accuracy
  
  
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
    batch_size = y_true.shape[1]

    if isinstance(self.criterion, CrossEntropyLoss):
      y_true= np.argmax(y_true, axis = 0)

    correct = np.sum(y_pred == y_true)
    return float(correct / batch_size)

