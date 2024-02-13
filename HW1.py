class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        # Multiply inputs by their corresponding weights and sum them
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs))
        
        # Apply the threshold function
        if weighted_sum >= self.threshold:
            return 1
        else:
            return 0
