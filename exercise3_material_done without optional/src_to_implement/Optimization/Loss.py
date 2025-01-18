import numpy as np

class CrossEntropyLoss:
    # No need to inheritance as it is not a layer
    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):   # one-hot-encoded true labels. Shape: (batch_size, num_classes).
        self.prediction_tensor = prediction_tensor    
        self.label_tensor = label_tensor
        epsilon = 1e-15
        self.loss = -np.sum(label_tensor * np.log(prediction_tensor + np.finfo(float).eps))  #solving numerical instability as well 
        return np.mean(self.loss)  # average loss across all samples in the batch
    
    def backward(self, label_tensor):
        return -(label_tensor / (self.prediction_tensor + np.finfo(float).eps))        
       
       