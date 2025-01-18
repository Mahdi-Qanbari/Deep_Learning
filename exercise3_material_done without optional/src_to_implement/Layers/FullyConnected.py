import numpy as np
from Layers import Base


class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size + 1, output_size)
        self.bias = np.zeros((1, output_size)) 
        self.trainable = True
        self.input_tensor = None
        self.optimizer = None
        self.gradient_weights = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(
            (self.input_size + 1, self.output_size),
            fan_in=self.input_size,
            fan_out=self.output_size
        )
        
        self.bias = bias_initializer.initialize(
            (1, self.output_size),
            fan_in=self.input_size,
            fan_out=self.output_size
        )

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        augmented_input = np.concatenate([input_tensor, np.ones((input_tensor.shape[0], 1))], axis=1)
        output = np.dot(augmented_input, self.weights)
        return output

    def backward(self, error_tensor):
        augmented_input = np.concatenate([self.input_tensor, np.ones((self.input_tensor.shape[0], 1))], axis=1)
        self.gradient_weights = np.dot(augmented_input.T, error_tensor)
        gradient_bias = np.sum(error_tensor, axis=0, keepdims=True)

        grad_input = np.dot(error_tensor, self.weights.T)
        grad_input = grad_input[:, :-1]  
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, gradient_bias)

        return grad_input
