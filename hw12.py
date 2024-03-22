import numpy as np

def softmax(Z):
    exp_values = np.exp(Z - np.max(Z))  # Subtracting np.max(Z) for numerical stability
    probabilities = exp_values / np.sum(exp_values)
    return probabilities
