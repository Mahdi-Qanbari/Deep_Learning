import numpy as np
from Layers import Base
from Optimization import Optimizers


class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size) : 
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = None  # Weights will be initialized using an initializer
        #self.weights = np.random.rand(input_size + 1, output_size) # Initialize weights randomly
        self.trainable = True
        self.gradient_weights = None
        self._optimizer = None    # protected Variable
    
     # Getter for optimizer
    @property
    def optimizer(self):
        """Returns the optimizer of this layer."""
        return self._optimizer

    # Setter for optimizer
    @optimizer.setter
    def optimizer(self, value):
        """Sets the optimizer for this layer."""
        self._optimizer = value



    def forward(self, input_tensor):
        # self.input_tensor = input_tensor
        # self.output_tensor = np.dot(input_tensor, self.weights) + self.bias
        # return self.output_tensor
     
        # Add a column of ones to the input tensor to account for the bias
        ones = np.ones((input_tensor.shape[0], 1))
        augmented_input = np.hstack((input_tensor, ones))
        self.input_tensor = augmented_input
        self.output_tensor = np.dot(augmented_input, self.weights)
        return self.output_tensor

    def backward(self, error_tensor):
    # Compute gradients
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        self.gradientinput_tensor = np.dot(error_tensor, self.weights[:-1, :].T)  # Exclude bias row for input gradient

        # Update weights using the optimizer
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)

        return self.gradientinput_tensor






      #From Exe 01 append:

    def initialize(self, weights_initializer, bias_initializer):
        """
        Reinitialize weights and biases using the provided initializers.
        """
        # Initialize weights (excluding bias)
        weights = weights_initializer.initialize(
            weights_shape=(self.input_size, self.output_size),
            fan_in=self.input_size,
            fan_out=self.output_size
        )

        # Initialize biases separately
        biases = bias_initializer.initialize(
            weights_shape=(1, self.output_size),
            fan_in=self.input_size,
            fan_out=self.output_size
        )

        # Combine weights and biases into a single weights matrix
        self.weights = np.vstack([weights, biases])  # Add biases as the last row


    
    