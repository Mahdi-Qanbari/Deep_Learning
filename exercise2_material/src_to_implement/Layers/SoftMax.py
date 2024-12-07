import numpy as np
from Optimization import Optimizers
from Layers import Base



class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        temp = np.exp(input_tensor - input_tensor.max(axis=1, keepdims=True))
        self.output_tensor = temp / temp.sum(axis=1, keepdims=True)

        return self.output_tensor
    
    def backward(self, error_tensor):
         return self.output_tensor * (error_tensor - (error_tensor * self.output_tensor).sum(axis = 1, keepdims= True))