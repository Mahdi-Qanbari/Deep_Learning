import numpy as np
from Layers import Base

class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        if isinstance(stride_shape, int):
            self.stride_shape = (stride_shape, stride_shape)
        else:
            self.stride_shape = stride_shape

        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        batch_size, channels, input_height, input_width = input_tensor.shape
        pool_height, pool_width = self.pooling_shape
        stride_height, stride_width = self.stride_shape

        output_height = (input_height - pool_height) // stride_height + 1
        output_width = (input_width - pool_width) // stride_width + 1

        output_tensor = np.zeros((batch_size, channels, output_height, output_width))

        self.max_indices = {}  #To store max indices for backward pass

        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * stride_height
                        start_j = j * stride_width
                        end_i = start_i + pool_height
                        end_j = start_j + pool_width

                        region = input_tensor[b, c, start_i:end_i, start_j:end_j]
                        max_val = np.max(region)
                        max_idx = np.unravel_index(np.argmax(region), region.shape)

                        output_tensor[b, c, i, j] = max_val
                        self.max_indices[(b, c, i, j)] = (start_i + max_idx[0], start_j + max_idx[1])

        return output_tensor

    def backward(self, error_tensor):
        grad_input = np.zeros_like(self.input_tensor)

        for (b, c, i, j), (max_i, max_j) in self.max_indices.items():
            grad_input[b, c, max_i, max_j] += error_tensor[b, c, i, j]

        return grad_input
