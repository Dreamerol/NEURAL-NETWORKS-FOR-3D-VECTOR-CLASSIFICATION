import numpy as np

#Activation function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def Relu(x):
    if x >= 0:
        return x
    else:
        return 0
    
def Relu_deriv(x):
    return 1

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1/np.cos(x)**2

activation_functions = {
    sigmoid : sigmoid_derivative,
    Relu: Relu_deriv,
    tanh: tanh_deriv
}
   
class Neuron:
    def __init__(self, n_inputs, activation):
        self.activation = activation
        self.activation_derivative = activation_functions[activation]
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()
        self.output = 0
        self.delta = 0

    def forward(self, inputs):
        self.inputs = np.array(inputs)
        self.output = np.dot(self.weights, self.inputs) + self.bias
        self.output = self.activation(self.output)
        return self.output
    
    def compute_delta(self, target=None, next_weights=None, next_deltas=None):
        if target is not None:
            # Output layer
            self.delta = (self.output - target) * self.activation_derivative(self.output)
        else:
            # when we are in the hidden layer we get the deltas from the nth layer
            # and multiply it by the matrix with weights - so to see the contribution
            # of each neuron to the loss
            self.delta = self.activation_derivative(self.output)*np.dot(next_weights, next_deltas)
    
    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.delta * self.inputs
        self.bias -= learning_rate * self.delta


class DenseLayer:
    def __init__(self, n_neurons, n_inputs, activation):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]
    
    def forward(self, inputs):
        outputs = [neuron.forward(inputs) for neuron in self.neurons]
        self.outputs = outputs
        return outputs
    
    def compute_deltas(self, targets=None, next_layer=None):
        if targets is not None:
            for i, neuron in enumerate(self.neurons):
                neuron.compute_delta(target=targets[i])
        else:
            for i, neuron in enumerate(self.neurons):
                next_weights = np.array([n.weights[i] for n in next_layer.neurons])
                next_deltas = np.array([n.delta for n in next_layer.neurons])
                neuron.compute_delta(next_weights=next_weights, next_deltas=next_deltas)

    def update_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)

class NeuralNetwork:
    def __init__(self, layers_info, n_input):
        self.layers = []
        input_size = n_input
        for n_neurons, activation in layers_info:
            layer = DenseLayer(n_neurons, input_size, activation)
            self.layers.append(layer)
            input_size = n_neurons

    def forwardNN(self, inputs):
        result = inputs
        for layer in self.layers:
            result = layer.forward(result)
        return result
    
    def backpropagation(self, targets, learning_rate):
        self.layers[-1].compute_deltas(targets=targets)
        for i in (range(len(self.layers) - 2,-1, -1)):
            self.layers[i].compute_deltas(next_layer=self.layers[i+1])
        for layer in self.layers:
            layer.update_weights(learning_rate)

samples_inputs = [[0,0,1], [1,1,1]]
samples_outputs = [[0], [1]]
NN = NeuralNetwork([
    (2, Relu),
    (1, sigmoid)
], n_input=3)

learning_rate = 0.5
epochs = 200
        
for epoch in range(epochs):
    for x, y in zip(samples_inputs, samples_outputs):
        output = NN.forwardNN(x)
        loss = 0.5 * sum((np.array(output) - np.array(y))**2)
        NN.backpropagation(y, learning_rate)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

