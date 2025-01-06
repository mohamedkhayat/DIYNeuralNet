from numpy import inf


class EarlyStopping:
  def __init__(self,patience):
    self.patience = patience
    self.counter = 0
    self.best_val_loss = inf
    self.current_epoch = 0
    self.done = False
    
  def __call__(self,val_loss):
    self.current_epoch += 1

    if(val_loss < self.best_val_loss):
      self.best_val_loss = val_loss
      self.counter = 0
      
    else:
      self.counter += 1

    if(self.counter >= self.patience):
      print(f"Early stopping triggered during epoch : {self.current_epoch}\nbest val_loss = {self.best_val_loss.item():.4f}")
      self.done = True
    
    return self.done   
    