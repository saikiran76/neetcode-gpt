import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # Extract number of samples and features
        n_samples, n_features = X.shape
        
        # Initialize weights to zeros and bias to 0
        w = np.zeros(n_features)
        b = 0.0
        
        # Training loop
        for _ in range(epochs):
            # 1. Forward pass: compute predictions
            y_hat = np.dot(X, w) + b
            
            # Compute the difference between predictions and actual targets
            error = y_hat - y
            
            # 2. Backward pass: compute gradients
            # dw = (2/n) * X^T @ (y_hat - y)
            dw = (2 / n_samples) * np.dot(X.T, error)
            
            # db = (2/n) * sum(y_hat - y)
            db = (2 / n_samples) * np.sum(error)
            
            # 3. Update weights and bias
            w -= lr * dw
            b -= lr * db
            
        # Round the final weights and bias to 5 decimal places as requested
        return (np.round(w, 5), round(float(b), 5))