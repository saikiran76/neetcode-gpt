import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x_np = np.array(x)
        rms_x = np.sqrt(np.mean(np.square(x_np)) + eps)
        x_hat = x_np/rms_x
        out = np.array(gamma)*x_hat
        return np.round(out, 4)

