import copy

class NeuralNetwork():
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None       

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        # Pass the input through all the layers
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        # The output of the last layer (loss layer) is returned
        return self.loss_layer.forward(input_tensor, self.label_tensor)
    
    def backward(self):
        # Calculate the error tensor by comparing the output with the label
        error_tensor = self.loss_layer.backward(self.label_tensor)
        
        # Backward propagate the error tensor through all the layers
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
  
    def append_layer(self, layer):
        self.layers.append(layer)

        # Ensure trainable layers get their own optimizer
        if getattr(layer, 'trainable', False):
            layer.optimizer = copy.deepcopy(self.optimizer)
       
    def train(self, iterations):
        for epoch in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
