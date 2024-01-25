import numpy as np

class Perceptron:
    def __init__(self, num_inputs, activation_function):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand(1)
        self.activation_function = activation_function

    def forward_pass(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(z)

    def compute_loss(self, output, target):
        return (target - output) ** 2  # Mean squared error

    def backpropagate(self, inputs, output, target):
        error = target - output
        d_weights = -2 * inputs * error  # Derivative with respect to weights
        d_bias = -2 * error  # Derivative with respect to bias
        return d_weights, d_bias

    def update_parameters(self, d_weights, d_bias, learning_rate):
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

    def train(self, training_data, targets, learning_rate):
        for inputs, target in zip(training_data, targets):
            output = self.forward_pass(inputs)
            d_weights, d_bias = self.backpropagate(inputs, output, target)
            self.update_parameters(d_weights, d_bias, learning_rate)

# Sigmoid as an activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Instantiate the Perceptron
perceptron = Perceptron(num_inputs=400, activation_function=sigmoid)

# Train the Perceptron
perceptron.train(training_data, targets, learning_rate=0.01)

