import numpy as np
from numpy.typing import NDArray

class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        # Use NumPy advanced indexing to fetch the rows corresponding to token_ids
        result = embeddings[token_ids]
        
        # Round the result to 5 decimal places as requested
        return np.round(result, 5)