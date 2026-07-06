import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2/N * np.sum((ground_truth - model_prediction)*(X[:, desired_weight]))
       
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        Y = np.dot(X, weights)
        return np.squeeze(Y)
        

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
        weights = np.array(initial_weights, dtype=np.float64)
        N = len(X)
        

        for i in range(num_iterations):
            pred = self.get_model_prediction(X, weights)
            gradients = np.zeros_like(weights)

            for j in range(len(weights)):
                gradients[j] = self.get_derivative(pred, Y, N, X, j)
            for j in range(len(weights)):
                weights[j] -= self.learning_rate*gradients[j]
        return np.round(weights, 5)

        
        
        


        

