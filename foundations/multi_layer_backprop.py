import numpy as np
from typing import List

class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        
        # Convert lists to numpy arrays for matrix operations
        x_arr = np.array(x)
        W1_arr = np.array(W1)
        b1_arr = np.array(b1)
        W2_arr = np.array(W2)
        b2_arr = np.array(b2)
        y_true_arr = np.array(y_true)

        # --- Forward Pass ---
        # Layer 1
        z1 = np.dot(W1_arr, x_arr) + b1_arr
        a1 = np.maximum(0, z1)  # ReLU activation (element-wise max)
        
        # Layer 2
        z2 = np.dot(W2_arr, a1) + b2_arr
        
        # Loss (MSE)
        n = len(y_true_arr)
        loss = np.mean(np.square(z2 - y_true_arr))

        # --- Backward Pass ---
        # 1. Output gradient (dL/dz2)
        dz2 = (2.0 / n) * (z2 - y_true_arr)
        
        # 2. Layer 2 gradients
        dW2 = np.outer(dz2, a1)
        db2 = dz2
        
        # 3. Gradient through ReLU
        da1 = np.dot(dz2, W2_arr)
        dz1 = da1 * (z1 > 0)  # ReLU derivative mask
        
        # 4. Layer 1 gradients
        dW1 = np.outer(dz1, x_arr)
        db1 = dz1

        # --- Formatting Output ---
        return {
            'loss': float(np.round(loss, 4)),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }