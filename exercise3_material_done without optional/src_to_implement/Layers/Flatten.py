from Layers import Base
class Flatten(Base.BaseLayer) :
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_tensor.shape)
    