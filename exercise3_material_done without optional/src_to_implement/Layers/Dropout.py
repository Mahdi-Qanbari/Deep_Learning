#Exe 3.1.3
import numpy as np
from Layers import Base


class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        else:
            temp = np.random.rand(*input_tensor.shape)
            self.mask = (temp < self.probability).astype(float)
            self.mask /= self.probability
            return input_tensor * self.mask

    def backward(self, error_tensor):
        return error_tensor * self.mask

    