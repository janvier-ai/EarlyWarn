# Cost values for early warning systems
# Define costs associated with different types of errors
# And costs of delays in warnings
# Combined into a time-dependent cost function

class Cost:
    def __init__(self, misclassification, delay, combination):
        self.misclassification = misclassification
        self.delay = delay
        self.combination = combination
    
    def episode_cost(self, y_true, y_pred, t):
        # Calculate misclassification cost
        misclass_cost = self.misclassification[y_true][y_pred]
        # Calculate delay cost
        if self.delay["type"] == "linear":
            slope = self.delay["slope"]
            delay_cost = slope * t
        else:
            delay_cost = 0
        # Combine costs based on combination method
        if self.combination["method"] == "tradeoff":
            alpha = self.combination["alphas"][t]
            total_cost = alpha * misclass_cost + (1 - alpha) * delay_cost
        else:
            total_cost = misclass_cost + delay_cost
        return total_cost