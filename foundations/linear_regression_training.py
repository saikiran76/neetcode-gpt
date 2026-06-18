import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        curr_weights = np.array(initial_weights, dtype=np.float64)
        # model_predictions = self.get_model_prediction(X, curr_weights)
        # grad = get_derivative(X, model_predictions, )
        # for i in range(len(curr_weights)):
        #     gradient = self.get_derivative(model_predictions, Y, len(X), X, i)
        #     curr_weights[i] = curr_weights[i] - self.learning_rate*gradient
        N = len(X)

        for _ in range(num_iterations): # epochs
            # 2. Re-calculate predictions at the start of every iteration with the updated weights
            model_predictions = self.get_model_prediction(X, curr_weights)
            
            # 3. Calculate gradients and update weights
            for i in range(len(curr_weights)):
                gradient = self.get_derivative(model_predictions, Y, N, X, i)
                curr_weights[i] = curr_weights[i] - self.learning_rate * gradient
        
        return np.round(curr_weights, 5)


        # for _ in range(iterations):
        #     gradient = get_gradient()
        #     curr_weights = curr_weights - learning_rate*

