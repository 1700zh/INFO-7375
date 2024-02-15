import numpy as np

class ActivationFunction:
    def relu(x):
        return np.maximum(0, x)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

class Layer:
    def __init__(self, input_dim, output_dim, activation_func):
        self.weights = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim, 1)
        self.activation_func = activation_func
        self.output = None
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation_func(np.dot(self.weights, input_data) + self.bias)
        return self.output

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_dim, output_dim, activation_func):
        layer = Layer(input_dim, output_dim, activation_func)
        self.layers.append(layer)

    def forward_propagation(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def compute_loss(self, predicted_output, true_output):
        return np.mean(np.square(true_output - predicted_output))
