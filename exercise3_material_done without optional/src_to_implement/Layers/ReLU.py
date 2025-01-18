import numpy as np
from Layers import Base

class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor): #local variable 
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)
    
    def backward(self, error_tensor):
        grad_input_tensor = np.where(self.input_tensor > 0, error_tensor, 0)
        return grad_input_tensor
    