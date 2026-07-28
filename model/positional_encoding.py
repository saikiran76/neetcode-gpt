import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # Initialize output array of shape (seq_len, d_model)
        pe = np.zeros((seq_len, d_model), dtype=np.float64)
        
        # Positions: column vector of shape (seq_len, 1)
        position = np.arange(seq_len)[:, np.newaxis]
        
        # Dimension indices for even positions (2i): vector of shape (d_model // 2,)
        i = np.arange(0, d_model, 2)
        
        # Division term: 10000^(2i / d_model)
        div_term = np.power(10000.0, i / d_model)
        
        # Compute angles using broadcasting -> shape (seq_len, d_model // 2)
        angles = position / div_term
        
        # Assign sine to even indices and cosine to odd indices
        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles)
        
        # Round to 5 decimal places as required
        return np.round(pe, 5)