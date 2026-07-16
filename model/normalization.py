import numpy as np
from numpy.typing import NDArray

class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # eps = 1e-5
        eps = 1e-5
        
        # Calculate mean and variance across the feature vector
        mean = np.mean(x)
        var = np.var(x)
        
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        x_hat = (x - mean) / np.sqrt(var + eps)
        
        # Scale and shift: out = gamma * x_hat + beta
        out = gamma * x_hat + beta
        
        # return np.round(your_answer, 5)
        return np.round(out, 5)