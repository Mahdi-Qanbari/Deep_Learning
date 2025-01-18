import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self._phase = None
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None

        self.weights_initializer = copy.deepcopy(weights_initializer)
        self.bias_initializer = copy.deepcopy(bias_initializer)
    
    #Exe 3.1.1
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     del state['data_layer']
    #     return state
    
    # def __setstate__(self, state):
    #     self.__dict__ = state


    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, testing_phase):
        self._phase = testing_phase
    
    # Updated according Exe 3.1.2
    def forward(self):
        input_tensor, self.label_tensor = copy.deepcopy(self.data_layer.next())
        reg_loss = 0
        for layer in self.layers:
            layer.testing_phase = False
            input_tensor = layer.forward(input_tensor)
            if self.optimizer.regularizer:
                reg_loss += self.optimizer.regularizer.norm(layer.weights)
        
        data_loss = self.loss_layer.forward(input_tensor, copy.deepcopy(self.label_tensor))
        return data_loss + reg_loss
        
    def backward(self):
        error_tensor = self.loss_layer.backward(copy.deepcopy(self.label_tensor))

        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.initialize(self.weights_initializer, self.bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = 'train'
        for epoch in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        self.phase = 'test'
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)
        return input_tensor


