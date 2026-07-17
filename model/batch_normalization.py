import numpy as np
from typing import Tuple, List

class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        
        # Convert lists to numpy arrays for element-wise and matrix operations
        X = np.array(x)
        G = np.array(gamma)
        B = np.array(beta)
        r_mean = np.array(running_mean)
        r_var = np.array(running_var)
        
        if training:
            # 1. Calculate mean and variance across the batch (axis=0) for each feature
            batch_mean = np.mean(X, axis=0)
            batch_var = np.var(X, axis=0)
            
            # 2. Normalize using batch statistics
            X_hat = (X - batch_mean) / np.sqrt(batch_var + eps)
            
            # 3. Update running statistics
            r_mean = (1 - momentum) * r_mean + momentum * batch_mean
            r_var = (1 - momentum) * r_var + momentum * batch_var
        else:
            # During inference: Normalize using the running statistics
            X_hat = (X - r_mean) / np.sqrt(r_var + eps)
            
        # Apply affine transformation: y = gamma * x_hat + beta
        Y = G * X_hat + B
        
        # Round the outputs to 4 decimal places and convert back to Python lists
        Y_out = np.round(Y, 4).tolist()
        r_mean_out = np.round(r_mean, 4).tolist()
        r_var_out = np.round(r_var, 4).tolist()
        
        return Y_out, r_mean_out, r_var_out